# minimal script to setup a planar membrane of bilayer/bolalipids, run a energy minimization, and then an ensemble in nph
import pprint
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Literal, Union

import numpy as np
import numpy.typing as npt

from .lammps_input import LammpsCoeffs

NDfloat = npt.NDArray[np.floating]
NDbool = npt.NDArray[np.bool_]
NDint = npt.NDArray[np.integer]
NDint64 = npt.NDArray[np.int64]
NDnumber = npt.NDArray[Union[np.integer, np.floating]]

from .ssp import (
    SharedSimulationParams,
    anglecoeff_Harmonic,
    bondcoeff_FENE,
    paircoeff_CosSq,
    type_Atom,
)


def calc_r_specie_coords(
    seed: int,
    scale: float | tuple[float, float],
    nx: int,
    ny: int,
    o: int,
    u_a: int,
    u_b: int,
    d_a: int,
    d_b: int,
):
    """
    nx,ny : discrete grid dimensions where to place lipids
    o : # of bolalipids
    x_a,x_b : upper/lower leaflet
    u_x,d_x : # of u-shaped bolalipid/bilayer lipids
    """
    rng = np.random.default_rng(seed=seed)

    if isinstance(scale, float):
        scales = (scale, scale)
    else:
        scales = scale

    def uod_choice(rng, n, o, u_a, u_b, d_a, d_b):
        """
        outputs array n,2 with indices from 1 to (o+u_a+u_b+d_a+d_b)
        0 means empty
        """

        n_o = rng.choice(n, size=o, replace=False)

        def f(u_x, d_x):
            n_u_d = np.delete(np.arange(n), n_o)
            n_u_d_a = rng.choice(n_u_d, size=u_x + d_x, replace=False)
            s = rng.permutation(n_u_d_a)
            n_u_x = s[:u_x]
            n_d_x = s[u_x:]
            return n_u_x, n_d_x

        n_u_a, n_d_a = f(u_a, d_a)
        n_u_b, n_d_b = f(u_b, d_b)
        id2n = np.concatenate(
            [
                n_o,
                n_u_a,
                n_u_b,
                n_d_a,
                n_d_b,
            ]
        )
        id2s = np.concatenate(
            [
                np.repeat(1, o),
                np.repeat(2, u_a + u_b),
                np.repeat(3, d_a + d_b),
            ]
        )
        id2l = np.concatenate(
            [
                np.repeat(1, o),
                np.repeat(1, u_a),
                np.repeat(2, u_b),
                np.repeat(1, d_a),
                np.repeat(2, d_b),
            ]
        )
        return np.stack([id2s, id2n, id2l], axis=-1)

    id2s_n_l = uod_choice(rng, n=nx * ny, o=o, u_a=u_a, u_b=u_b, d_a=d_a, d_b=d_b)

    id2s, id2n, id2l = id2s_n_l.T
    id2x = id2n % nx
    id2y = id2n // nx

    id2pos = np.stack(
        [
            ((id2x + 0.5) / nx - 0.5) * (nx * scales[0]),
            ((id2y + 0.5) / ny - 0.5) * (ny * scales[1]),
        ],
        axis=-1,
    )

    bond_r0 = 2 ** (1 / 6)

    def pos_s(kind, pos, layer):
        if kind == 1 or kind == 2:
            # o,u shape
            phi = rng.random() * 2 * np.pi
            s0, s1 = (layer, -layer) if kind == 1 else (layer, layer)
            delta = (
                0.0
                if s0 != s1
                else np.array([np.cos(phi), np.sin(phi)]) * bond_r0 / 2.0
            )
            return np.array(
                [[*(pos - delta), s0 * (0.5 + i)] for i in (2, 1, 0)]
                + [[*(pos + delta), s1 * (0.5 + i)] for i in (0, 1, 2)]
            )
        elif kind == 3:
            # d shape
            return np.array([[*pos, layer * (0.5 + i)] for i in (2, 1, 0)])
        else:
            raise ValueError

    r_specie_coords: list[tuple[int, NDfloat]] = []
    for s, pos, l in zip(id2s, id2pos, id2l):
        r_specie_coords.append(
            (0 if s in (1, 2) else 1, pos_s(s, pos, 1 if l == 1 else -1))
        )

    return r_specie_coords

@dataclass
class Params:
    seed: int = 0

    e_angle: float = 0.0
    membrane_pair_w: float = 1.5
    membrane_pair_eps: float = 1 / 1.1
    membrane_intra_align_eps: float = 5.0
    
    membrane_intra_bond_k: float = 30.0
    membrane_intra_bond_r0: float = 1.5
    membrane_intra_bond_eps: float = 1.0

    membrane_o: int = 0
    membrane_u_a: int = 0
    membrane_u_b: int = 0
    @property
    def membrane_u(self):
        return self.membrane_u_a+self.membrane_u_b
    membrane_d_a: int = 25 * 25
    membrane_d_b: int = 25 * 25

    lx: float = 25.0
    ly: float = 25.0
    lz: float = 50.0

    # thermostat_temperature: float = 1.0
    # thermostat_damp: float = 1.0
    # thermostat_gjf: Literal["no", "vfull", "vhalf"] = "no"

    # barostat_pressure: float = 0.0
    # barostat_damp: float = 10.0
    # timestep: float = 1e-2

    # skiprun: bool = False

    # thermo_time_delta: float = 1.0
    # dump_time_delta: float = 1e1
    # restart_time_delta: float = 5e2

    # init_duration: float = 1e3
    # init_relax: bool = True
    # equib_duration: float = 10e3

    # ntasks: int = 1
    # check_process: bool = True
    def to_ssp(self):
        params = self
        sim = SharedSimulationParams()
        gen = np.random.default_rng(params.seed)
        sim.special_bonds = {"1-2": False, "1-3": True, "1-4": True}

        two6 = 2 ** (1 / 6)
        lipid_sigma = 1.0

        lxy = np.array(
            [
                params.lx,
                params.ly,
            ]
        )

        def calc_nx_ny(y_over_x: float, h: float):
            # must be big enough to place the largest leaflet
            Y = (y_over_x * h) ** 0.5
            X = h / Y
            nx, ny = int(np.ceil(X)), int(np.ceil(Y))
            return nx, ny

        nx, ny = calc_nx_ny(
            y_over_x=lxy[1] / lxy[0],
            h=params.membrane_o
            + max(
                params.membrane_u_a + params.membrane_d_a,
                params.membrane_u_b + params.membrane_d_b,
            ),
        )

        r_specie_coords = calc_r_specie_coords(
            params.seed,
            (lxy[0] / nx, lxy[1] / ny),
            nx,
            ny,
            params.membrane_o,
            params.membrane_u_a,
            params.membrane_u_b,
            params.membrane_d_a,
            params.membrane_d_b,
        )
        x_s = np.array([c for s, c in r_specie_coords if s == 0])
        x_d = np.array([c for s, c in r_specie_coords if s == 1])


        boxsize = np.array([*lxy, params.lz])
        sim.origin = tuple(-boxsize / 2)
        sim.lbox = tuple(boxsize)

        # Make atomtypes
        atomH = sim.atomtype(type_Atom("H", 1.0, 0.95 * lipid_sigma))
        atomT = sim.atomtype(type_Atom("T", 1.0, lipid_sigma))

        sim.group("membrane", [atomH, atomT])

        # BONDS & ANGLES
        membrane_intra_bond = sim.bondtype(
            bondcoeff_FENE(
                self.membrane_intra_bond_k,
                self.membrane_intra_bond_r0,
                self.membrane_intra_bond_eps,
                lipid_sigma,
            )
        )

        if params.membrane_intra_align_eps == 0.0:
            membrane_intra_align = None
            add_align = lambda t, i, j, k: 0
        else:
            make_align_t = lambda v: sim.angletype(anglecoeff_Harmonic(v, np.pi))
            add_align = lambda t, i, j, k: (
                sim.angle(t, i, j, k) if t is not None else None
            )
        
            membrane_intra_align = (
                make_align_t(params.membrane_intra_align_eps)
                if params.membrane_intra_align_eps != 0
                else None
            )

        membrane_mono = params.membrane_u + params.membrane_o
        membrane_monolayer_align = sim.angletype(
            anglecoeff_Harmonic(params.e_angle, np.pi)
        )
        if membrane_mono > 0 and params.e_angle != params.membrane_intra_align_eps:
            if params.e_angle != 0:
                membrane_monolayer_align = sim.angletype(
                    anglecoeff_Harmonic(params.e_angle, np.pi)
                )
            else:
                membrane_monolayer_align = None
        else:
            membrane_monolayer_align = membrane_intra_align

        def topo_bilayer_lipid(xs):
            mi = sim.molecule()
            unit = {}
            for j in range(3):
                unit[j] = sim.atom(atomH if j == 0 else atomT, xs[j], mi)
            for j in range(2):
                sim.bond(membrane_intra_bond, unit[j], unit[j + 1])
            for j in range(1):
                add_align(membrane_intra_align, unit[j], unit[j + 1], unit[j + 2])
            return mi, unit

        def topo_monolayer_lipid(xs):
            mi = sim.molecule()
            unit = {}
            for j in range(6):
                unit[j] = sim.atom(atomH if (j == 0 or j == 5) else atomT, xs[j], mi)
            for j in range(5):
                sim.bond(membrane_intra_bond, unit[j], unit[j + 1])
            for j in range(4):
                add_align(
                    membrane_intra_align if j in (0, 3) else membrane_monolayer_align,
                    unit[j],
                    unit[j + 1],
                    unit[j + 2],
                )
            return mi, unit

        def add_pos_noise(pos):
            return pos + np.ones_like(pos) * gen.random(pos.shape) * 0.005

        x_s = add_pos_noise(x_s)
        x_d = add_pos_noise(x_d)

        for xs in x_s:
            topo_monolayer_lipid(xs)

        for xs in x_d:
            topo_bilayer_lipid(xs)

        # PAIRS
        ht_sigma = 0.95 * two6 * lipid_sigma
        tt_sigma = two6 * lipid_sigma
        rc_tt = tt_sigma + params.membrane_pair_w

        sim.pairtype(
            paircoeff_CosSq(ht_sigma, params.membrane_pair_eps, ht_sigma), atomH, atomH
        )
        sim.pairtype(
            paircoeff_CosSq(ht_sigma, params.membrane_pair_eps, ht_sigma), atomH, atomT
        )
        sim.pairtype(
            paircoeff_CosSq(tt_sigma, params.membrane_pair_eps, rc_tt), atomT, atomT
        )

        return sim
    

    

def main():
    import argparse
    parser=argparse.ArgumentParser('make_input',
                            description="Generate input topology file for generating topology and initial configuration for archaeal membranes with lammps",
                            epilog="TODO add link to paper")
    parser.add_argument('-fbi',default=0.,type=float,help='fraction of head beads that belong to bilayer lipids')
    parser.add_argument('-kbola',default=0.,type=float,help='bolalipid rigidity')
    parser.add_argument('-lx',default=25.,type=float,help='box length in x')
    parser.add_argument('-ly',default=25.,type=float,help='box length in y')
    parser.add_argument('-lz',default=50.,type=float,help='box length in z')
    parser.add_argument('-aph',default=1.2,type=float,help='initial area per head')
    parser.add_argument('-teff',default=1.2,type=float,help='Effective Temperature')
    parser.add_argument('-ufraction',default=0.5,type=float,help='initial u-shaped bolalipid fraction')
    parser.add_argument('-t','--target',default='.',type=str,help='Destination directory')
    args=parser.parse_args()

    with open('args.txt','wt') as f:
        pprint.pprint(args._get_kwargs(),f)
    
    N=2*int((args.lx*args.ly)/args.aph)
    
    No=int(N*(1-args.fbi))
    No=2*(No//2)
    m=No//2
    u=int(m*args.ufraction)
    o=m-u
    u_a=u//2
    u_b=u-u_a
    
    d=N-No
    d_a=d//2
    d_b=d-d_a

    print(f"initial aph:",2*(args.lx*args.ly)/(2*(u_a+u_b+o)+(d_a+d_b)))

    # params.membrane_o
    #         + max(
    #             params.membrane_u_a + params.membrane_d_a,
    #             params.membrane_u_b + params.membrane_d_b,
    #         ),
    #     )
    
    params=Params(
        e_angle=args.kbola,
        membrane_pair_eps=1/args.teff,
        membrane_o=o,
        membrane_u_a=u_a,
        membrane_u_b=u_b,
        membrane_d_a=d_a,
        membrane_d_b=d_b,

        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        
        )
    ssp=params.to_ssp()
    from .lammps_input import write_lammps_data
    path=Path(args.target)
    with (path/'in.data').open('wt') as f:
        write_lammps_data(f,ssp,atom_style='angle')

    with (path/'in.interaction.lmp').open('wt') as f:
        def writeLine(x):
            f.write(x+'\n')
        LammpsCoeffs(ssp,atom_style='angle').write_lammps_interaction(writeLine)
    
    path_in=(path/'in.lmp')
    if path_in.exists():
        pass
    else:
        with (path/'in.lmp').open('wt') as f:
            f.write(dedent(f"""\
            processors * * 1

            units           lj
            atom_style      angle
            dimension       3
            boundary        p p p


            read_data ./in.data
            special_bonds lj 0.0 1.0 1.0
            comm_style tiled
            neighbor 1. bin
            neigh_modify every 1 delay 1
            special_bonds lj 0 1 1


            include ./in.interaction.lmp

            # initial relaxation

            dump dump all atom 10 out.0.relax.traj.lammpstrj
            thermo 10
            thermo_style custom step dt time cpu fmax fnorm temp ke pe ebond eangle epair pxx pyy pzz
            thermo_modify flush yes

            min_style cg
            minimize 1e-4 1e-6 1000 10000

            fix langevin all langevin 1.0 1.0 1.0 1 zero yes
            velocity all create 1.0 1
            velocity all zero linear
            timestep        0.001
            fix f_nve all nve/limit 0.01
            run 10
            unfix f_nve
            unfix langevin
            undump dump

            # main run
            timestep        0.01
            dump dump all atom 1000 out.1.main.traj.lammpstrj
            fix langevin all langevin 1.0 1.0 1.0 1 zero yes
            fix f_nve all nve
            thermo 100
            run 100000
            """))

if __name__=="__main__":
    main()