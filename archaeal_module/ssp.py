from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Union

import numpy as np


def qphi(phi, axis):
    axis = axis / np.linalg.norm(axis)
    return (
        np.cos(phi / 2),
        axis[0] * np.sin(phi / 2),
        axis[1] * np.sin(phi / 2),
        axis[2] * np.sin(phi / 2),
    )


def quatquat(q1, q2):
    q3 = np.zeros(4)
    q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[3] - q1[3] * q2[3]
    q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[2] - q1[3] * q2[2]
    q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return q3


def quatrot(vec, q):
    qvec = np.array([0, vec[0], vec[1], vec[2]])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])

    tmp = quatquat(q, qvec)
    qrot = quatquat(tmp, qc)
    return qrot[1:]

@dataclass
class WriteParams:
    dump_time_delta: float
    thermo_time_delta: float
    restart_time_delta: float


@dataclass
class type_Atom:
    name: str
    mass: float
    diameter: float


@dataclass
class type_Rigid:
    # all lists need to have length len(sattypes)
    comtype: int
    sattypes: tuple[int, ...]
    positions: np.ndarray
    masses: tuple[int, ...]

    def _moment_inertia(self):
        Ixx, Iyy, Izz = 0, 0, 0
        for i in range(self.nsats):
            Ixx += self.masses[i] * (
                self.positions[i, 1] ** 2 + self.positions[i, 2] ** 2
            )
            Iyy += self.masses[i] * (
                self.positions[i, 0] ** 2 + self.positions[i, 2] ** 2
            )
            Izz += self.masses[i] * (
                self.positions[i, 0] ** 2 + self.positions[i, 1] ** 2
            )
        self.moment_inertia = (Ixx, Iyy, Izz)

    def __post_init__(self):
        self.nsats = len(self.sattypes)
        assert self.positions.shape[0] == self.nsats
        assert len(self.masses) == self.nsats
        self._moment_inertia()
        self.orientations = self.nsats * [(1, 0, 0, 0)]
        self.diameters = self.nsats * [1.0]
        self.charges = self.nsats * [0.0]


import numba as nb


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64)])
def rclip(x, a, w):
    if w != 0:
        l = (x - a) / w
        if l > 1.0:
            return 1.0
        elif l < 0.0:
            return 0.0
        else:
            return l
    else:
        return 1.0 if x >= a else 0.0


@dataclass
class bondcoeff_Harmonic:
    k: float
    r0: float


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.boolean)])
def lj(r: float, epsilon: float, sigma: float, r_cut: float, shift: bool):
    b = r_cut / sigma
    a = r / sigma
    x = 0.0 if a < 0.0 else (b if a > b else a)
    if x == 0:
        return np.inf
    else:
        if x >= b:
            return 0.0
        else:
            v = x ** (-12) - x ** (-6)
            if shift:
                v -= b ** (-12) - b ** (-6)
            return 4 * epsilon * v


def wca(r: float, epsilon: float, rc: float):
    x = np.clip(r / rc, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return np.where(x == 0.0, np.inf, epsilon * (x ** (-12) - 2 * x ** (-6) + 1))


def cossq(r: float, epsilon: float, rm: float, w: float):
    return -epsilon * np.cos(np.pi / 2 * rclip(r, rm, w)) ** 2


def fene(r: float, r0: float, k: float):
    x = np.array(r / r0)
    with np.errstate(divide="ignore"):
        x = np.log(1 - np.clip(x, -np.inf, 1.0) ** 2)
    return -(0.5 * k * r0**2) * x


@dataclass
class bondcoeff_FENE:
    """
    k:
        Elastic term pre-factor
    r0:
        Elastic term maximum extension
    epsilon:
        Lennard-jones pre-factor
    sigma:
        Lennard-Jones zero-crossing
    """

    k: float
    r0: float
    epsilon: float
    sigma: float

    def energy(self, r: float):
        return fene(r, self.r0, self.k) + wca(
            r, self.epsilon, self.sigma * 2 ** (1 / 6)
        )


@dataclass
class anglecoeff_Harmonic:
    # t0 in radians
    k: float
    t0: float

    def energy(self, theta):
        return self.k * (theta - self.t0) ** 2


@dataclass
class dihedralcoeff_Harmonic:
    # phi0 in radians
    # d in [-1,1]
    k: float
    d: int
    n: int
    phi0: float


@dataclass
class paircoeff_LJ:
    sigma: float
    epsilon: float
    r_cut: float

    def energy(self, r: float, shift: bool = False):
        return lj(r, self.epsilon, self.sigma, self.r_cut, shift)


@dataclass
class paircoeff_CosSq:
    sigma: float
    epsilon: float
    r_cut: float

    def energy(self, r: float):
        v = wca(r, self.epsilon, self.sigma)
        w = self.r_cut - self.sigma
        if w != 0:
            v += cossq(r, self.epsilon, self.sigma, w)
        return v


@dataclass
class paircoeff_Table:
    V: np.ndarray
    F: np.ndarray
    r_min: float
    r_cut: float
    name: str


@dataclass
class method_NVE:
    pass


@dataclass
class method_NVT:
    kT: float
    tau: float


@dataclass
class method_Langevin:
    kT: float
    alpha: float


@dataclass
class method_NPT:
    kT: float
    tau: float
    S: float | tuple[float, ...]
    tauS: float
    box_dof: tuple[bool, ...]
    couple: str


@dataclass
class method_NPH:
    S: float | tuple[float, ...]
    tauS: float
    box_dof: tuple[bool, ...]
    couple: str


@dataclass
class SharedSimulationParams:
    """
    SharedSimulationParams is a class that holds information on topology data
    without specification of simulation software package.

    Non-Topology
        lbox: box dimensions
        origin : box origin
    Atoms
        atomtypes: list of type_Atom objects for each atomtype
        atoms: list of tuples for each particle instance holding information on
            atomtype, moleculeid, position, rigidbody, orientation and inertia
        type2id: lookup table of atomtype.name strings to index in atomtypes
    Molecules
        nmolecules: integer to keep track of number of moleculetypes
    Bonds
        bondtypes: list of bondcoeff_XX objects for each bondtype with params
        bonds: list of tuples holding id, typeindex and participating atoms
        bondstyles: list of bondcoeff_XX typenames that are being used
    Angles
        angletypes: list of anglecoeff_XX objects for each angletype with
            params
        angles: list of tuples holding id, typeindex and participating atoms
        anglestyles: list of anglecoeff_XX typenames that are being used
    Dihedrals
        dihedraltypes: list of dihedralcoeff_XX objects for each dihedraltype
            with params
        dihedrals: list of tuples holding id, typeindex and participating atoms
        dihedralstyles: list of dihedralcoeff_XX typenames that are being used
    Rigids
        rigidtypes: list of type_Rigid objects for each rigidtype, object
            contains info on satellite atoms and moment of inertia
        rigids: list with tuples with indices of center-of-mass and satellites
            adding to this list automatically generates satellites
    Pairs
        pairtypes: list of tuples with id, paircoeff_XX object and
            participating atomtype-indices.
        pairstyles: list of paircoeff_XX typenams that are being used
        cutoff_graph: dict with values for cutoff
    Groups
        groups: dict with {name: atomtypeindexlist} pairs that can be used
            as filters in simulation
    """

    # SIM PARAMS

    lbox: tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0, 1.0))
    origin: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    special_bonds: dict[str, bool] = field(default_factory=dict)
    boundary: tuple[Literal["p", "f"], Literal["p", "f"], Literal["p", "f"]] = field(
        default=("p", "p", "p")
    )

    # ATOMS
    # atomtypes stores a list of type_ATOM objects containing atom info
    # atoms stores a list of atoms with type index, molecule id, position
    # body is used in HOOMD for the definition of rigid bodies
    atomtypes: list[dict[str, Any]] = field(default_factory=list, repr=False)
    atoms: list[tuple[int, int, np.ndarray, int, int, tuple, tuple]] = field(
        default_factory=list, repr=False
    )
    type2id: dict[str, int] = field(default_factory=dict)

    def atomtype(self, atomcoeff):
        id = len(self.atomtypes)
        self.atomtypes.append(atomcoeff)
        self.type2id[atomcoeff.name] = id
        return id

    def atom(
        self,
        t: int,
        p: np.ndarray,
        mid: int = None,
        body: bool = -1,
        q: tuple[int, int, int, int] = (1, 0, 0, 0),
        moment: tuple[int, int, int] = (0, 0, 0),
    ):
        id = len(self.atoms)
        mid = self.molecule() if mid is None else mid
        # Take numeric value if given, take id if true, -1 if false
        bodyid = body if body >= 0 else (id if body == -2 else body)
        self.atoms.append((id, t, p, mid, bodyid, q, moment))
        return id

    def atoms2frame(self):
        import pandas as pd
        return pd.DataFrame(
            self.atoms,
            columns=["id", "type", "pos", "mol", "rigidbody", "orientation", "inertia"],
        ).set_index("id")

    # MOLECULES
    # molecules stores number of molecules
    nmolecules: int = field(default=0)

    def molecule(self):
        id = self.nmolecules
        self.nmolecules += 1
        return id

    # BONDS
    # bondtypes stores a list of bondcoeff_XXX objects containing bondparams
    # bonds stores the index of this class with the two interacting atoms
    bondtypes: list[Any] = field(default_factory=list)
    bonds: list[tuple[int, int, int, int]] = field(default_factory=list, repr=False)
    bondstyles: list[str] = field(default_factory=list)

    def bondtype(self, bondcoeff):
        id = len(self.bondtypes)
        self.bondtypes.append(bondcoeff)
        self.bondstyles.append(type(bondcoeff).__name__)
        self.bondstyles = list(set(self.bondstyles))
        return id

    def bond(self, t: int, a: int, b: int):
        id = len(self.bonds)
        self.bonds.append((id, t, a, b))
        return id

    # ANGLES
    # angletypes stores a list of anglecoeff_XXX objects containing angleparams
    # angles stores the index of this class with the three interacting atoms
    angletypes: list[Any] = field(default_factory=list)
    angles: list[tuple] = field(default_factory=list, repr=False)
    anglestyles: list[str] = field(default_factory=list)

    def angletype(self, anglecoeff):
        id = len(self.angletypes)
        self.angletypes.append(anglecoeff)
        self.anglestyles.append(type(anglecoeff).__name__)
        self.anglestyles = list(set(self.anglestyles))
        return id

    def angle(self, t: int, a: int, b: int, c: int):
        id = len(self.angles)
        self.angles.append((id, t, a, b, c))
        return id

    # DIHEDRALS
    # dihedraltypes stores a list of dihedralcoeff_XXX objects containing dihedralparams
    # dihedrals stores the index of this class with the four interacting atoms
    dihedraltypes: list[Any] = field(default_factory=list)
    dihedrals: list[tuple] = field(default_factory=list, repr=False)
    dihedralstyles: list[str] = field(default_factory=list)

    def dihedraltype(self, dihedralcoeff):
        id = len(self.dihedraltypes)
        self.dihedraltypes.append(dihedralcoeff)
        self.dihedralstyles.append(type(dihedralcoeff).__name__)
        self.dihedralstyles = list(set(self.dihedralstyles))
        return id

    def dihedral(self, t: int, a: int, b: int, c: int, d: int):
        id = len(self.dihedrals)
        self.dihedrals.append((id, t, a, b, c, d))
        return id

    # RIGID COMS
    rigidtypes: list[type_Rigid] = field(default_factory=list)
    rigids: list[tuple] = field(default_factory=list, repr=False)

    def rigidtype(self, rigidcoeff):
        # COMTYPES AND SATTYPES MUST EXIST IN SELF.ATOMTYPES
        id = len(self.rigidtypes)
        self.rigidtypes.append(rigidcoeff)
        return id

    def rigid(self, t: int, position: np.ndarray, orientation: tuple):
        # Orientation and position of the central particle sets positions of sattellites
        # Rigid() creates atoms for you!
        id = len(self.rigids)
        rigidbody = self.rigidtypes[t]
        commolid = self.molecule()
        satmolid = self.molecule()

        comid = self.atom(
            rigidbody.comtype,
            position,
            commolid,
            -2,
            orientation,
            rigidbody.moment_inertia,
        )
        rb = (
            id,
            t,
            comid,
        )
        for i in range(len(rigidbody.sattypes)):
            relpos = quatrot(rigidbody.positions[i], orientation)
            satid = self.atom(rigidbody.sattypes[i], position + relpos, satmolid, comid)
            rb += (satid,)
        self.rigids.append(rb)
        return id

    # PAIRSTYLES
    pairtypes: list[Any] = field(default_factory=list)
    pairstyles: dict[str, int] = field(default_factory=dict)
    cutoff_graph: dict[tuple, int] = field(default_factory=dict)

    # As opposed to bonds, angles and dihedrals type indices are stored, not atom indices
    def pairtype(self, paircoeff, typeA, typeB):
        id = len(self.pairtypes)
        self.pairtypes.append((id, paircoeff, typeA, typeB))

        name = type(paircoeff).__name__
        if name in self.pairstyles.keys():
            rc = self.pairstyles[name]
            self.pairstyles[name] = max(rc, paircoeff.r_cut)
        else:
            self.pairstyles[name] = paircoeff.r_cut

        self.cutoff_graph[(typeA, typeB)] = max(
            self.pairtypes[id][1].r_cut, self.cutoff_graph.get((typeA, typeB), 0)
        )

        return id

    # GROUPS
    groups: dict[str, list[int]] = field(default_factory=dict)

    def group(self, name: str, types: list):
        id = len(self.groups)
        self.groups[name] = tuple(types)
        return id

    def __str__(self):
        s = "System with bounding box [{} {} {}] contains:\n".format(*self.lbox)

        # ATOMS
        s += "ATOMS\n{:>8d}    atomtypes: {}\n".format(
            len(self.atomtypes), [c.name for c in self.atomtypes]
        )
        s += f"{len(self.atoms):>8d}    atoms\n"
        for i, at in enumerate(self.atomtypes):
            s += "{:>8d}    atoms of type {} ('{}') and mass {}\n".format(
                sum([1 if a[1] == i else 0 for a in self.atoms]), i, at.name, at.mass
            )

        # MOLECULES
        s += f"\nMOLECULES\n{self.nmolecules:>8d}    molecules\n"

        # BONDS
        s += f"\nBONDS\n{len(self.bondtypes):>8d}    bondtypes\n"
        s += f"{len(self.bonds):>8d}    bonds\n"
        for i, bt in enumerate(self.bondtypes):
            s += "{:>8d}    bonds of type {} with style {} and params {}\n".format(
                sum([1 if b[1] == i else 0 for b in self.bonds]),
                i,
                type(bt).__name__.split("_")[1],
                asdict(bt),
            )

        # ANGLES
        s += f"\nANGLES\n{len(self.angletypes):>8d}    angletypes\n"
        s += f"{len(self.angles):>8d}    angles\n"
        for i, at in enumerate(self.angletypes):
            s += "{:>8d}    angles of type {} with style {} and params {}\n".format(
                sum([1 if a[1] == i else 0 for a in self.angles]),
                i,
                type(at).__name__.split("_")[1],
                asdict(at),
            )

        # DIHEDRALS
        s += f"\nDIHEDRALS\n{len(self.dihedraltypes):>8d}    dihedraltypes\n"
        s += f"{len(self.dihedrals):>8d}    dihedrals\n"
        for i, dt in enumerate(self.dihedraltypes):
            s += "{:>8d}    dihedrals of type {} with style {} and params {}\n".format(
                sum([1 if d[1] == i else 0 for d in self.dihedrals]),
                i,
                type(dt).__name__.split("_")[1],
                asdict(dt),
            )

        # PAIRTYPES
        def _vars(paircoeff):
            # Return parameter dict from paircoeff object without r_cut
            return {
                key: value
                for key, value in vars(paircoeff).items()
                if key not in ["V", "F"]
            }

        s += f"\nPAIRTYPES\n{len(self.pairtypes):>8d}    pairtypes\n"
        s += f"{len(self.pairtypes):>8d}    pairs\n"
        for i, pt in enumerate(self.pairtypes):
            s += "            pair between {} and {} of type {} with style {} and params {}\n".format(
                self.pairtypes[i][2],
                self.pairtypes[i][3],
                i,
                type(pt[1]).__name__.split("_")[1],
                _vars(pt[1]),
            )

        # RIGIDBODIES
        s += f"\nRIGIDTYPES\n{len(self.rigidtypes):>8d}    rigidtypes\n"
        s += f"{len(self.rigids):>8d}    rigid bodies\n"
        for i, rt in enumerate(self.rigidtypes):
            s += "{:>8d}    rigid bodies of type {} with {} satellites\n".format(
                sum([1 if r[1] == i else 0 for r in self.rigids]), i, rt.nsats
            )

        return s
