import random
from typing import (
    List,
    Optional,
)

import dpdata
import numpy as np
import scipy.constants as pc
from packaging.version import (
    Version,
)

from dpgen2.constants import (
    lmp_traj_name,
)


def _sample_sphere():
    rng = np.random.default_rng()
    while True:
        vv = np.array([rng.normal(), rng.normal(), rng.normal()])
        vn = np.linalg.norm(vv)
        if vn < 0.2:
            continue
        return vv / vn


def make_lmp_input(
    conf_file: str,
    ensemble: str,
    graphs: List[str],
    nsteps: int,
    dt: float,
    neidelay: Optional[int],
    trj_freq: int,
    mass_map: List[float],
    temp: float,
    tau_t: float = 0.1,
    pres: Optional[float] = None,
    tau_p: float = 0.5,
    use_clusters: bool = False,
    relative_f_epsilon: Optional[float] = None,
    relative_v_epsilon: Optional[float] = None,
    pka_e: Optional[float] = None,
    ele_temp_f: Optional[float] = None,
    ele_temp_a: Optional[float] = None,
    nopbc: bool = False,
    max_seed: int = 1000000,
    deepmd_version="2.0",
    trj_seperate_files=True,
):
    if (ele_temp_f is not None or ele_temp_a is not None) and Version(
        deepmd_version
    ) < Version("1"):
        raise RuntimeError(
            "the electron temperature is only supported by deepmd-kit >= 1.0.0, please upgrade your deepmd-kit"
        )
    if ele_temp_f is not None and ele_temp_a is not None:
        raise RuntimeError(
            "the frame style ele_temp and atom style ele_temp should not be set at the same time"
        )
    if "npt" in ensemble and pres is None:
        raise RuntimeError("the pressre should be provided for npt ensemble")
    ret = "variable        NSTEPS          equal %d\n" % nsteps
    ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq
    ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
    ret += f"variable        TEMP            equal {temp:f}\n"
    if ele_temp_f is not None:
        ret += f"variable        ELE_TEMP        equal {ele_temp_f:f}\n"
    if ele_temp_a is not None:
        ret += f"variable        ELE_TEMP        equal {ele_temp_a:f}\n"
    if pres is not None:
        ret += f"variable        PRES            equal {pres:f}\n"
    ret += f"variable        TAU_T           equal {tau_t:f}\n"
    if pres is not None:
        ret += f"variable        TAU_P           equal {tau_p:f}\n"
    ret += "\n"
    ret += "units           metal\n"
    if nopbc:
        ret += "boundary        f f f\n"
    else:
        ret += "boundary        p p p\n"
    ret += "atom_style      atomic\n"
    ret += "\n"
    ret += "neighbor        1.0 bin\n"
    if neidelay is not None:
        ret += "neigh_modify    delay %d\n" % neidelay
    ret += "\n"
    ret += "box          tilt large\n"
    ret += f'if "${{restart}} > 0" then "read_restart dpgen.restart.*" else "read_data {conf_file}"\n'
    ret += "change_box   all triclinic\n"
    for jj in range(len(mass_map)):
        ret += "mass            %d %f\n" % (jj + 1, mass_map[jj])
    graph_list = ""
    for ii in graphs:
        graph_list += ii + " "
    if Version(deepmd_version) < Version("1"):
        # 0.x
        ret += f"pair_style      deepmd {graph_list} ${{THERMO_FREQ}} model_devi.out\n"
    else:
        # 1.x
        keywords = ""
        if use_clusters:
            keywords += "atomic "
        if relative_f_epsilon is not None:
            keywords += f"relative {relative_f_epsilon} "
        if relative_v_epsilon is not None:
            keywords += f"relative_v {relative_v_epsilon} "
        if ele_temp_f is not None:
            keywords += "fparam ${ELE_TEMP}"
        if ele_temp_a is not None:
            keywords += "aparam ${ELE_TEMP}"
        ret += f"pair_style      deepmd {graph_list} out_freq ${{THERMO_FREQ}} out_file model_devi.out {keywords}\n"
    ret += "pair_coeff      * *\n"
    ret += "\n"
    ret += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"
    ret += "thermo          ${THERMO_FREQ}\n"
    if trj_seperate_files:
        ret += "dump            1 all custom ${DUMP_FREQ} traj/*.lammpstrj id type x y z fx fy fz\n"
    else:
        ret += f"dump            1 all custom ${{DUMP_FREQ}} {lmp_traj_name} id type x y z fx fy fz\n"
    ret += "restart         10000 dpgen.restart\n"
    ret += "\n"
    if pka_e is None:
        ret += 'if "${restart} == 0" then "velocity        all create ${TEMP} %d"' % (
            random.randrange(max_seed - 1) + 1
        )
    else:
        sys = dpdata.System(conf_file, fmt="lammps/lmp")
        sys_data = sys.data
        pka_mass = mass_map[sys_data["atom_types"][0] - 1]
        pka_vn = (
            pka_e
            * pc.electron_volt
            / (0.5 * pka_mass * 1e-3 / pc.Avogadro * (pc.angstrom / pc.pico) ** 2)
        )  # type: ignore
        pka_vn = np.sqrt(pka_vn)
        pka_vec = _sample_sphere()
        pka_vec *= pka_vn
        ret += "group           first id 1\n"
        ret += f'if "${{restart}} == 0" then "velocity        first set {pka_vec[0]:f} {pka_vec[1]:f} {pka_vec[2]:f}"\n'
        ret += "fix           2 all momentum 1 linear 1 1 1\n"
    ret += "\n"
    if ensemble.split("-")[0] == "npt":
        assert pres is not None
        if nopbc:
            raise RuntimeError(f"ensemble {ensemble} is conflicting with nopbc")
    if ensemble == "npt" or ensemble == "npt-i" or ensemble == "npt-iso":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == "npt-a" or ensemble == "npt-aniso":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == "npt-t" or ensemble == "npt-tri":
        ret += "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == "nvt":
        ret += "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
    elif ensemble == "nve":
        ret += "fix             1 all nve\n"
    else:
        raise RuntimeError("unknown emsemble " + ensemble)
    if nopbc:
        ret += "velocity        all zero linear\n"
        ret += "fix             fm all momentum 1 linear 1 1 1\n"
    ret += "\n"
    ret += f"timestep        {dt:f}\n"
    ret += "run             ${NSTEPS} upto\n"
    return ret
