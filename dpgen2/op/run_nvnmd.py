import glob
import itertools
import logging
import os
import random
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Union,
)

import numpy as np
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    lmp_input_name,
    lmp_log_name,
    lmp_model_devi_name,
    lmp_traj_name,
    model_name_pattern,
    plm_output_name,
)
from dpgen2.op.run_lmp import (
    RunLmp,
    find_only_one_key,
)
from dpgen2.utils import (
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class RunNvNMD(OP):
    r"""Execute a LAMMPS task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The LAMMPS
    command is exectuted from directory `task_name`. The trajectory
    and the model deviation will be stored in files `op["traj"]` and
    `op["model_devi"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "traj": Artifact(Path),
                "model_devi": Artifact(Path),
                "plm_output": Artifact(Path, optional=True),
                "optional_output": Artifact(Path, optional=True),
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

            - `config`: (`dict`) The config of lmp task. Check `RunLmp.lmp_args` for definitions.
            - `task_name`: (`str`) The name of the task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepLmp`.
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation. The first model with be used to drive molecular dynamics simulation.

        Returns
        -------
        Any
            Output dict with components:
            - `log`: (`Artifact(Path)`) The log file of LAMMPS.
            - `traj`: (`Artifact(Path)`) The output trajectory.
            - `model_devi`: (`Artifact(Path)`) The model deviation. The order of recorded model deviations should be consistent with the order of frames in `traj`.

        Raises
        ------
        TransientError
            On the failure of LAMMPS execution. Handle different failure cases? e.g. loss atoms.
        """
        config = ip["config"] if ip["config"] is not None else {}
        config = RunLmp.normalize_config(config)
        command = config["command"]
        shuffle_models: Optional[bool] = config["shuffle_models"]
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        models = ip["models"]
        # input_files = [lmp_conf_name, lmp_input_name]
        # input_files = [(Path(task_path) / ii).resolve() for ii in input_files]
        input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        model_files = [Path(ii).resolve() / "model.pb" for ii in models]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                # Path(iname).symlink_to(ii)
                try:
                    Path(iname).symlink_to(ii)
                except:
                    logging.warning("failed to link %s, maybe already linked" % iname)
                    pass
            # link models
            model_names = []
            for idx, mm in enumerate(model_files):
                ext = os.path.splitext(mm)[-1]
                if ext == ".pb":
                    mname = model_name_pattern % (idx)
                    # Path(mname).symlink_to(mm)
                    try:
                        Path(mname).symlink_to(mm)
                    except:
                        logging.warning(
                            "failed to link %s, maybe already linked" % mname
                        )
                        pass

                else:
                    raise RuntimeError(
                        "Model file with extension '%s' is not supported" % ext
                    )
                model_names.append(mname)

            if shuffle_models:
                random.shuffle(model_names)

            set_lmp_models(lmp_input_name, model_names)

            # run lmp
            for ii in range(len(model_names)):
                commands = " ".join(
                    [
                        command,
                        "-i",
                        "%d_%s" % (ii, lmp_input_name),
                        "-log",
                        "%d_%s" % (ii, lmp_log_name),
                        "-v",
                        "rerun",
                        "%d" % ii,
                    ]
                )
                ret, out, err = run_command(commands, shell=True)
                if ret != 0:
                    logging.error(
                        "".join(
                            (
                                "lmp failed\n",
                                "command was: ",
                                commands,
                                "out msg: ",
                                out,
                                "\n",
                                "err msg: ",
                                err,
                                "\n",
                            )
                        )
                    )
                    raise TransientError("lmp failed")

            merge_pimd_files()

            traj_files = glob.glob("*_%s" % lmp_traj_name)
            if len(traj_files) > 1:
                calc_model_devi(traj_files, lmp_model_devi_name)

        ret_dict = {
            "log": work_dir / ("%d_%s" % (0, lmp_log_name)),
            "traj": work_dir / ("%d_%s" % (0, lmp_traj_name)),
            "model_devi": self.get_model_devi(work_dir / lmp_model_devi_name),
        }
        plm_output = (
            {"plm_output": work_dir / plm_output_name}
            if (work_dir / plm_output_name).is_file()
            else {}
        )
        ret_dict.update(plm_output)
        return OPIO(ret_dict)

    def get_model_devi(self, model_devi_file):
        return model_devi_file


config_args = RunLmp.lmp_args


def set_lmp_models(lmp_input_name: str, model_names: List[str]):
    with open(lmp_input_name, encoding="utf8") as f:
        lmp_input_lines = f.readlines()

    idx = find_only_one_key(
        lmp_input_lines, ["pair_style", "nvnmd"], raise_not_found=False
    )
    if idx is None:
        return
    new_line_split = lmp_input_lines[idx].split()
    match_idx = find_only_one_key(new_line_split, ["model.pb"], raise_not_found=False)
    if match_idx is None:
        raise RuntimeError("last matching index should not be -1, terribly wrong ")

    for ii, model_name in enumerate(model_names):
        new_line_split[match_idx] = model_name

        lmp_input_lines[idx] = " ".join(new_line_split) + "\n"

        with open("%d_%s" % (ii, lmp_input_name), "w", encoding="utf8") as f:
            f.write("".join(lmp_input_lines))


def merge_pimd_files():
    traj_files = glob.glob("traj.*.dump")
    if len(traj_files) > 0:
        with open(lmp_traj_name, "w") as f:
            for traj_file in sorted(traj_files):
                with open(traj_file, "r") as f2:
                    f.write(f2.read())
    model_devi_files = glob.glob("model_devi.*.out")
    if len(model_devi_files) > 0:
        with open(lmp_model_devi_name, "w") as f:
            for model_devi_file in sorted(model_devi_files):
                with open(model_devi_file, "r") as f2:
                    f.write(f2.read())


def calc_model_devi(
    traj_files,
    fname="model_devi.out",
):
    from ase.io import read  # type: ignore

    trajectories = []
    for f in traj_files:
        traj = read(f, format="lammps-dump-text", index=":", order=True)
        trajectories.append(traj)

    num_frames = len(trajectories[0])
    for traj in trajectories:
        assert (
            len(traj) == num_frames
        ), f"Trajectory length mismatch: expected {num_frames}, got {len(traj)} frames"

    devi = []
    for frame_idx in range(num_frames):
        frames = [traj[frame_idx] for traj in trajectories]

        all_forces = [atoms.get_forces() for atoms in frames]
        all_errors = []

        for atom_idx in range(len(frames[0])):
            forces = [forces_arr[atom_idx] for forces_arr in all_forces]

            for a, b in itertools.combinations(forces, 2):
                error = np.linalg.norm(a - b)
                all_errors.append(error)

        max_error = np.max(all_errors) if all_errors else 0.0
        min_error = np.min(all_errors) if all_errors else 0.0
        avg_error = np.mean(all_errors) if all_errors else 0.0

        # ase verion >= 3.26.0, please update ase using "pip install git+https://gitlab.com/ase/ase.git"
        devi.append(
            [
                trajectories[0][frame_idx].info["timestep"],
                0,
                0,
                0,
                max_error,
                min_error,
                avg_error,
                0,
            ]
        )

    devi = np.array(devi)
    write_model_devi_out(devi, fname=fname)


def write_model_devi_out(devi: np.ndarray, fname: Union[str, Path], header: str = ""):
    assert devi.shape[1] == 8
    header = "%s\n%10s" % (header, "step")
    for item in "vf":
        header += "%19s%19s%19s" % (
            f"max_devi_{item}",
            f"min_devi_{item}",
            f"avg_devi_{item}",
        )
    with open(fname, "ab") as fp:
        np.savetxt(
            fp,
            devi,
            fmt=["%12d"] + ["%19.6e" for _ in range(devi.shape[1] - 1)],
            delimiter="",
            header=header,
        )
    return devi
