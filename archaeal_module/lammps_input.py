
from dataclasses import dataclass
from typing import TextIO

from .ssp import *


@dataclass
class type_Atom:
    name: str
    mass: float
    diameter: float

def write_lammps_data(f: TextIO, ssp: SharedSimulationParams, atom_style: str):
    def _atom2s(atom: tuple):
        return "{} {} {} {} {} {}".format(atom[0], atom[3], atom[1], *atom[2])

    def _list2s(l: list[str] | list[int]):
        # return string with objects from list
        return " ".join(map(str, l))

    def _atomtype2s(id: int, at: type_Atom):
        return f"{id} {at.mass}"

    f.write("LAMMPS data file\n")
    f.write("\n")

    for i, a in enumerate("xyz"):
        f.write(f"{ssp.origin[i]} {ssp.origin[i]+ssp.lbox[i]} {a}lo {a}hi\n")

    if ssp.atoms:
        f.write(f"{len(ssp.atoms)} atoms\n")
    if ssp.bonds:
        f.write(f"{len(ssp.bonds)} bonds\n")
    if ssp.angles:
        f.write(f"{len(ssp.angles)} angles\n")
    if ssp.dihedrals:
        f.write(f"{len(ssp.dihedrals)} dihedrals\n")

    f.write("\n")

    if ssp.atomtypes:
        f.write(f"{len(ssp.atomtypes)} atom types\n")
    if ssp.bondtypes:
        f.write(f"{len(ssp.bondtypes)} bond types\n")
    if ssp.angletypes:
        f.write(f"{len(ssp.angletypes)} angle types\n")
    if ssp.dihedraltypes:
        f.write(f"{len(ssp.dihedraltypes)} dihedral types\n")

    f.write("\n")

    if ssp.atomtypes:
        f.write("\n")
        f.write("Masses\n")
        f.write("\n")
        for i, t in enumerate(ssp.atomtypes):
            atID = int(i + 1)
            f.write("\t" + _atomtype2s(atID, t) + "\n")
        f.write("\n")

    if ssp.atoms:
        f.write(f"Atoms # {atom_style}\n")
        f.write("\n")
        for a in ssp.atoms:
            a2 = [int(x + 1) if isinstance(x, int) else x for x in a]
            f.write("\t" + _atom2s(a2) + "\n")
        f.write("\n")

    if ssp.bonds:
        f.write("Bonds \n")
        f.write("\n")
        for b in ssp.bonds:
            b2 = [int(x + 1) for x in b]
            f.write("\t" + _list2s(list(b2)) + "\n")
        f.write("\n")

    if ssp.angles:
        f.write("Angles \n")
        f.write("\n")
        for ang in ssp.angles:
            ang2 = [int(x + 1) for x in ang]
            f.write("\t" + _list2s(list(ang2)) + "\n")
        f.write("\n")

    if ssp.dihedrals:
        f.write("Dihedrals \n")
        f.write("\n")
        for dih in ssp.dihedrals:
            dih2 = [int(x + 1) for x in dih]
            f.write("\t" + _list2s(list(dih2)) + "\n")
        f.write("\n")

@dataclass
class LammpsCoeffs:
    """
    Helper class to setup potentials in Lammps given a SharedSimulationParams

    ssp : SharedSimulationParams
    atom_style:str

    """
    ssp: SharedSimulationParams
    atom_style:str
    

    def _atomtype2s(self, id: int, at: type_Atom):
        return f"{id} {at.mass}"

    def _atom2s(self, atom: tuple):
        return "{} {} {} {} {} {}".format(atom[0], atom[3], atom[1], *atom[2])

    def _list2s(self, l: list[str] | list[int]):
        # return string with objects from list
        return " ".join(map(str, l))

    def _pairCosSq(self, paircoeff: paircoeff_CosSq):
        # returns ordered list of cossq coefficient params
        return [
            "cosine/squared",
            paircoeff.epsilon,
            paircoeff.sigma,
            paircoeff.r_cut,
            "wca",
        ]

    def _pairLJ(self, paircoeff: paircoeff_LJ):
        # returns ordered list of lj coefficient params
        return ["lj/cut", paircoeff.epsilon, paircoeff.sigma, paircoeff.r_cut]

    def _bondHarmonic(self, bondcoeff: bondcoeff_Harmonic):
        # returns ordered list of harmonic bond params
        return ["harmonic", bondcoeff.k, bondcoeff.r0]

    def _bondFENE(self, bondcoeff: bondcoeff_FENE):
        # returns ordered list of harmonic bond params
        return ["fene", bondcoeff.k, bondcoeff.r0, bondcoeff.epsilon, bondcoeff.sigma]

    def _angleHarmonic(self, anglecoeff: anglecoeff_Harmonic):
        # returns ordered list of harmonic angle params
        angle = 180 * anglecoeff.t0 / np.pi
        return ["harmonic", anglecoeff.k, angle]

    def _dihedralHarmonic(self, dihedralcoeff: dihedralcoeff_Harmonic):
        # returns ordered list of harmonic dihedral params
        # phi0 not integrated in Lammps
        return ["harmonic", dihedralcoeff.k, dihedralcoeff.d, dihedralcoeff.n]

    def _topotype2s(
        self,
        topocoeff: Any,
        topoid: int | None = None,
        pair: tuple[int, int] | None = None,
    ):
        # convert bondtuple
        strtopotype = type(topocoeff).__name__

        # add underscore between topo_coeff and remove style
        kind = strtopotype.split("coeff")[0]
        topo = kind + "_coeff"

        kind_to_store = {
            "pair": self.ssp.pairstyles,
            "angle": self.ssp.anglestyles,
            "bond": self.ssp.bondstyles,
            "dihedral": self.ssp.dihedralstyles,
        }

        # determine style
        styles = {
            "paircoeff_LJ": self._pairLJ,
            "paircoeff_CosSq": self._pairCosSq,
            "bondcoeff_Harmonic": self._bondHarmonic,
            "bondcoeff_FENE": self._bondFENE,
            "anglecoeff_Harmonic": self._angleHarmonic,
            "dihedralcoeff_Harmonic": self._dihedralHarmonic,
        }
        if strtopotype not in styles.keys():
            raise ValueError(f"Topotype {strtopotype} not known")

        if topo == "pair_coeff" and pair is not None:
            c = [topo, min(pair), max(pair)]
        elif topo != "pair_coeff" and topoid is not None:
            c = [topo, topoid]
        else:
            raise ValueError(f"TopoId or Pair not given for {topo}")

        args = styles[strtopotype](topocoeff)
        if len(kind_to_store[kind]) == 1:
            args = args[1:]
        c.extend(args)

        return self._list2s(c)

    def write_lammps_interaction(self,writeLine):
        """
        Translates coefficients in SharedSimulationParams to lmp commands and
        adds to simulation.
        """
        ssp=self.ssp

        def styletolmp(s):
            return s.split("_")[1].lower()

        if ssp.pairstyles:
            pairstyles = ["pair_style"]
            if len(ssp.pairstyles) > 1:
                pairstyles.append("hybrid")
            clstolmp = {
                "paircoeff_CosSq": "cosine/squared",
                "paircoeff_LJ": "lj/cut",
            }
            for style, rc in ssp.pairstyles.items():
                pairstyles.append(clstolmp[style])
                pairstyles.append(rc)
            writeLine(self._list2s(pairstyles))

        if ssp.bondstyles:
            cmd = ["bond_style"]
            styles = ssp.bondstyles
            if len(styles) > 1:
                cmd.append("hybrid")
            for s in styles:
                cmd.append(styletolmp(s))
            writeLine(self._list2s(cmd))

        if ssp.anglestyles:
            cmd = ["angle_style"]
            styles = ssp.anglestyles
            if len(styles) > 1:
                cmd.append("hybrid")
            for s in styles:
                cmd.append(styletolmp(s))
            writeLine(self._list2s(cmd))

        if ssp.dihedralstyles:
            cmd = ["dihedral_style"]
            styles = ssp.dihedralstyles
            if len(styles) > 1:
                cmd.append("hybrid")
            for s in styles:
                cmd.append(styletolmp(s))
            writeLine(self._list2s(cmd))

        for pair in ssp.pairtypes:
            pair2 = [(x + 1) if isinstance(x, int) else x for x in pair]
            writeLine(self._topotype2s(pair2[1], pair=tuple(pair2[2:])))
        for id, bondtype in enumerate(ssp.bondtypes):
            id2 = (id + 1)
            writeLine(self._topotype2s(bondtype, topoid=id2))

        for id, angletype in enumerate(ssp.angletypes):
            id2 = (id + 1)
            writeLine(self._topotype2s(angletype, topoid=id2))

        for id, dihedraltype in enumerate(ssp.dihedraltypes):
            id2 = (id + 1)
            writeLine(self._topotype2s(dihedraltype, topoid=id2))