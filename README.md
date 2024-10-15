# Archaeal membranes : code & examples
<p align="center">
    <img src="https://github.com/user-attachments/assets/36bb5261-fd3a-416e-8eb2-bff4eb069bf9" width="50%" height="50%"/>
</p>

<p align="center">
   flexible bolalipid ($k_\mathrm{bola}=0$) membrane patch 
</p>
<!-- Paper ref: to add -->
Example simulation code for the upcoming paper on archaeal membranes.
Code allows changing parameters.
Current saved output for:

-  a flexible bolalipid membrane patch $k_\mathrm{bola}=0$ is in `./archaeal_patch`.
    - To reproduce, use the following command: `cd archaeal_patch; ../make_input -aph 1.3 -ufraction 0.5 -kbola 0. -fbi 0. && lmp -in in.lmp -log out.log.cmd`
    - The movie above corresponds to this simulation.
   

## Setup
Install LAMMPS (version: 29 Aug 2024 or newer).
To skip compiling, install directly from conda-forge, we recommend the micromamba package manager.
Exact build used for tests: `lammps 2024.08.29 cpu_py311_h50e90f8_nompi_0  conda-forge`.

Install Python (version : 3.11 or newer) and the numpy package (necessary for topology file and potential parameters generation).

## Running a membrane patch simulation in the NVE ensemble

Create a folder for your data:

`mkdir test`

Create input files (`in.data`, `in.interaction.lmp`,`in.lmp`), e.g.:
`./make_input -ufraction 0.5 -aph 1.25 -kbola 0. -t test/`
Other options can be seen with `./make_input -h`.
All arguments (after default value application) are saved to `args.txt`.

Edit `test/in.lmp` as needed.

Run LAMMPS in the `test` directory:
`(cd test; lmp -in in.lmp)`

For visualizing trajectory files, we recommend Ovito: create a pipeline by loading the `in.data` file containing topology and load the trajectory files `out.0.relax.traj.bin` and `out.1.main.traj.bin`.
