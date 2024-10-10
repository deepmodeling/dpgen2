from collections import (
    defaultdict,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Union,
)

import numpy as np
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter,
)

from dpgen2.utils import (
    set_directory,
)


class RunCalyModelDevi(OP):
    r"""calculate model deviaion of trajectories structures.

    Structure optimization will be executed in `optim_path`. The trajectory
    will be stored in files `op["traj"]` and `op["model_devi"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "type_map": Parameter(List[str]),
                "task_name": Parameter(str),
                "traj_dirs": Artifact(List[Path]),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "traj": Artifact(List[Path]),
                "model_devi": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `type_map`: (`List[str]`) The type map of elements.
            - `task_name`: (`str`) The name of the task.
            - `traj_dirs`: (`Artifact(List[Path])`) The List of paths that contains trajectory files.
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation.

        Returns
        -------
        Any
            Output dict with components:
            - `task_name`: (`str`) The name of task.
            - `traj`: (`Artifact(List[Path])`) The output trajectory.
            - `model_devi`: (`Artifact(List[Path])`) The model deviation. The order of recorded model deviations should be consistent with the order of frames in `traj`.

        """
        from deepmd.infer import (  # type: ignore
            DeepPot,
            calc_model_devi,
        )

        type_map = ip["type_map"]

        models = ip["models"]
        all_models = [model.resolve() for model in models]
        graphs = [DeepPot(model) for model in all_models]

        work_dir = Path(ip["task_name"])

        traj_dirs = ip["traj_dirs"]
        traj_dirs = [traj_dir.resolve() for traj_dir in traj_dirs]

        dump_file_name = "traj.%d.dump"
        model_devi_file_name = "model_devi.%d.out"

        tcount = 0
        with set_directory(work_dir):
            dump_str_dict = defaultdict(list)  # key: natoms, value: dump_strs
            devis_dict = defaultdict(list)  # key: natoms, value: Devis-s
            for traj_dir in traj_dirs:
                for traj_name in traj_dir.rglob("*.traj"):
                    atoms_list = parse_traj(traj_name)
                    if atoms_list is None:
                        continue
                    for atoms in atoms_list:
                        natoms = len(atoms)
                        dump_str = atoms2lmpdump(atoms, tcount, type_map, ignore=True)
                        dump_str_dict[tcount].append(dump_str)

                        pbc = np.all(atoms.get_pbc())
                        coord = atoms.get_positions().reshape(1, -1)
                        cell = atoms.get_cell().array.reshape(1, -1) if pbc else None
                        atype = [type_map.index(atom.symbol) for atom in atoms]  # type: ignore
                        devi = calc_model_devi(coord, cell, atype, graphs)[0]
                        devis_dict[tcount].append(devi)
                    tcount += 1

            traj_file_list = []
            model_devi_file_list = []
            keys = dump_str_dict.keys()
            for key in keys:
                dump_file = Path().joinpath(dump_file_name % key)
                model_devi_file = Path().joinpath(model_devi_file_name % key)

                traj_str = dump_str_dict[key]
                model_devis = devis_dict[key]
                assert len(traj_str) == len(
                    model_devis
                ), "The length of traj_str and model_devis should be same."
                for idx in range(len(model_devis)):
                    traj_str[idx] = traj_str[idx] % idx
                    model_devis[idx][0] = idx

                traj_str = "".join(traj_str)
                dump_file.write_text(traj_str)

                model_devis = np.vstack(model_devis)
                write_model_devi_out(model_devis, model_devi_file)

                traj_file_list.append(dump_file)
                model_devi_file_list.append(model_devi_file)

        for idx in range(len(traj_file_list)):
            traj_file_list[idx] = work_dir / traj_file_list[idx]
            model_devi_file_list[idx] = work_dir / model_devi_file_list[idx]

        ret_dict = {
            "task_name": str(work_dir),
            "traj": traj_file_list,
            "model_devi": model_devi_file_list,
        }

        return OPIO(ret_dict)


def atoms2lmpdump(atoms, struc_idx, type_map, ignore=False):
    """down triangle cell can be obtained from
    cell params: a, b, c, alpha, beta, gamma.
    cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])
    lx, ly, lz = cell[0][0], cell[1][1], cell[2][2]
    xy, xz, yz = cell[1][0], cell[2][0], cell[2][1]
    (lx,ly,lz) = (xhi-xlo,yhi-ylo,zhi-zlo)
    xlo_bound = xlo + MIN(0.0,xy,xz,xy+xz)
    xhi_bound = xhi + MAX(0.0,xy,xz,xy+xz)
    ylo_bound = ylo + MIN(0.0,yz)
    yhi_bound = yhi + MAX(0.0,yz)
    zlo_bound = zlo
    zhi_bound = zhi

    ref: https://docs.lammps.org/Howto_triclinic.html
    """
    from ase import (  # type: ignore
        Atoms,
    )
    from ase.geometry import (  # type: ignore
        cellpar_to_cell,
    )

    dump_str = "ITEM: TIMESTEP\n"
    if not ignore:
        dump_str += f"{struc_idx}\n"
    else:
        dump_str += "%d\n"
    dump_str += "ITEM: NUMBER OF ATOMS\n"
    dump_str += f"{atoms.get_global_number_of_atoms()}\n"

    cellpars = atoms.cell.cellpar()
    new_cell = cellpar_to_cell(cellpars)
    new_atoms = Atoms(
        numbers=atoms.numbers,
        cell=new_cell,
        scaled_positions=atoms.get_scaled_positions(),
    )

    xy, xz, yz = new_cell[1][0], new_cell[2][0], new_cell[2][1]
    xlo, ylo, zlo = 0, 0, 0
    lx, ly, lz = new_cell[0][0], new_cell[1][1], new_cell[2][2]
    xhi, yhi, zhi = lx + xlo, ly + ylo, lz + zlo
    xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
    xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = yhi + max(0.0, yz)
    zlo_bound = zlo
    zhi_bound = zhi

    dump_str += "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
    dump_str += "%20.10f %20.10f %20.10f\n" % (xlo_bound, xhi_bound, xy)
    dump_str += "%20.10f %20.10f %20.10f\n" % (ylo_bound, yhi_bound, xz)
    dump_str += "%20.10f %20.10f %20.10f\n" % (zlo_bound, zhi_bound, yz)
    dump_str += "ITEM: ATOMS id type x y z fx fy fz\n"
    for idx, atom in enumerate(new_atoms):
        type_id = type_map.index(atom.symbol) + 1  # type: ignore
        dump_str += "%5d %5d" % (idx + 1, type_id)
        dump_str += "%20.10f %20.10f %20.10f" % (
            atom.position[0],  # type: ignore
            atom.position[1],  # type: ignore
            atom.position[2],  # type: ignore
        )
        dump_str += "%20.10f %20.10f %20.10f\n" % (0, 0, 0)
    # dump_str = dump_str.strip("\n")
    return dump_str


def parse_traj(traj_file):
    from ase import (  # type: ignore
        Atoms,
    )
    from ase.build import (  # type: ignore
        make_supercell,
    )
    from ase.io import (  # type: ignore
        read,
    )

    safe_dist_dict = {
        "He": 0.0,
        "Li": 1.5,
        "Na": 1.45,
        "K": 2.3,
        "Rb": 2.5,
        "Mg": 1.7,
        "Ca": 2.3,
        "Sr": 2.5,
        "Al": 1.7,
        "Sc": 2.0,
        "Y": 2.1,
        "La": 2.5,
        "Ti": 2.0,
        "Zr": 2.1,
        "Hf": 2.4,
        "Mo": 2.1,
        "W": 2.3,
        "B": 1.1,
        "C": 1.1,
        "Si": 1.6,
        "P": 1.5,
        "As": 2.0,
        "S": 1.5,
        "Se": 2.1,
        "Te": 2.0,
        "Br": 2.3,
        "H": 0.813,
    }

    trajs: List[Atoms] = read(traj_file, index=":", format="traj")  # type: ignore
    dthresh = 0.72
    numb_traj = len(trajs)
    assert numb_traj >= 1, "traj file is broken."

    # 1st Filter, initial configuration
    origin = trajs[0]
    origin = make_supercell(origin, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    dis_mtx = origin.get_all_distances(mic=True)
    row, col = np.diag_indices_from(dis_mtx)
    dis_mtx[row, col] = np.nan
    is_reasonable = np.nanmin(dis_mtx) > dthresh

    selected_traj: Union[List[Atoms], None] = None
    if is_reasonable:
        if len(trajs) >= 20:
            selected_traj = [trajs[iii] for iii in [4, 9, -10, -5, -1]]
        elif 5 <= len(trajs) < 20:
            selected_traj = [
                trajs[np.random.randint(3, len(trajs) - 1)] for _ in range(4)
            ]
            selected_traj.append(trajs[-1])
        elif 3 <= len(trajs) < 5:
            selected_traj = [trajs[round((len(trajs) - 1) / 2)]]
            selected_traj.append(trajs[-1])
        elif len(trajs) == 2:
            selected_traj = [trajs[0], trajs[-1]]
        else:
            selected_traj = [trajs[0]]

        # 2nd filter for selected traj. It filters out all FRAMES that are to close.
        i_keep = []
        for t in selected_traj:
            t2 = make_supercell(t, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

            frame_is_reasonable = True
            dist_dict = t2.get_all_distances(mic=True)
            atype = t2.get_chemical_symbols()
            for a in range(len(atype)):
                for b in range(a + 1, len(atype)):
                    dd = dist_dict[a][b]
                    dr = (
                        (safe_dist_dict[atype[a]] + safe_dist_dict[atype[b]])
                        * 0.529
                        / 1.2
                    )
                    if dd < dr:
                        frame_is_reasonable = False

            if frame_is_reasonable:
                i_keep.append(selected_traj.index(t))
        selected_traj = [selected_traj[iii] for iii in i_keep]
    else:
        selected_traj = None

    return selected_traj


def write_model_devi_out(devi: np.ndarray, fname: Union[str, Path], header: str = ""):
    assert devi.shape[1] == 8
    header = "%s\n%10s" % (header, "step")
    for item in "vf":
        header += "%19s%19s%19s" % (
            f"max_devi_{item}",
            f"min_devi_{item}",
            f"avg_devi_{item}",
        )
        header += "%19s" % "devi_e"
    with open(fname, "ab") as fp:
        np.savetxt(
            fp,
            devi,
            fmt=["%12d"] + ["%19.6e" for _ in range(devi.shape[1] - 1)],
            delimiter="",
            header=header,
        )
    return devi
