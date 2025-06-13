import glob
import itertools
import json
import logging
import os
import random
import re
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from ase.io import (
    read,
)
from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    HDF5Datasets,
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_model_devi_name,
    lmp_traj_name,
    model_name_match_pattern,
    model_name_pattern,
    plm_output_name,
    pytorch_model_name_pattern,
)
from dpgen2.op.run_caly_model_devi import (
    write_model_devi_out,
)
from dpgen2.utils import (
    BinaryFileInput,
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

            - `config`: (`dict`) The config of lmp task. Check `RunNvNMD.lmp_args` for definitions.
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
        config = RunNvNMD.normalize_config(config)
        command = config["command"]
        teacher_model: Optional[BinaryFileInput] = config["teacher_model_path"]
        shuffle_models: Optional[bool] = config["shuffle_models"]
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        models = ip["models"]
        # input_files = [lmp_conf_name, lmp_input_name]
        # input_files = [(Path(task_path) / ii).resolve() for ii in input_files]
        input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        model_files = [Path(ii).resolve() for ii in models]
        work_dir = Path(task_name)

        if teacher_model is not None:
            assert (
                len(model_files) == 1
            ), "One model is enough in knowledge distillation"
            ext = os.path.splitext(teacher_model.file_name)[-1]
            teacher_model_file = "teacher_model" + ext
            teacher_model.save_as_file(teacher_model_file)
            model_files = [Path(teacher_model_file).resolve()] + model_files

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
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
                    try:
                        Path(mname).symlink_to(mm)
                    except:
                        logging.warning(
                            "failed to link %s, maybe already linked" % mname
                        )
                        pass

                elif ext == ".pt":
                    # freeze model
                    mname = pytorch_model_name_pattern % (idx)
                    freeze_model(mm, mname, config.get("model_frozen_head"))
                else:
                    raise RuntimeError(
                        "Model file with extension '%s' is not supported" % ext
                    )
                model_names.append(mname)

            if shuffle_models:
                random.shuffle(model_names)

            set_models(lmp_input_name, model_names)

            # run lmp
            commands = " ; ".join(
                [
                    " ".join(
                        [
                            "cp",
                            str(model_name),
                            "model.pb",
                            "&&",
                            command,
                            "-i",
                            lmp_input_name,
                            "-log",
                            lmp_log_name,
                            "-v",
                            "rerun",
                            "%d" % i,
                            "&&",
                            "cp",
                            lmp_traj_name,
                            lmp_traj_name + ".%d" % i,
                        ]
                    )
                    for i, model_name in enumerate(models)
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

            ele_temp = None
            if config.get("use_ele_temp", 0):
                ele_temp = get_ele_temp(lmp_log_name)
                if ele_temp is not None:
                    data = {
                        "ele_temp": ele_temp,
                    }
                    with open("job.json", "w") as f:
                        json.dump(data, f, indent=4)
            merge_pimd_files()

            if os.path.exists(lmp_traj_name):
                calc_model_devi(
                    [lmp_traj_name + f".{i}" for i in range(len(model_names))]
                )

        ret_dict = {
            "log": work_dir / lmp_log_name,
            "traj": work_dir / lmp_traj_name,
            "model_devi": self.get_model_devi(work_dir / lmp_model_devi_name),
        }
        plm_output = (
            {"plm_output": work_dir / plm_output_name}
            if (work_dir / plm_output_name).is_file()
            else {}
        )
        ret_dict.update(plm_output)
        if ele_temp is not None:
            ret_dict["optional_output"] = work_dir / "job.json"

        return OPIO(ret_dict)

    def get_model_devi(self, model_devi_file):
        return model_devi_file

    @staticmethod
    def lmp_args():
        doc_lmp_cmd = "The command of LAMMPS"
        doc_teacher_model = "The teacher model in `Knowledge Distillation`"
        doc_shuffle_models = "Randomly pick a model from the group of models to drive theexploration MD simulation"
        doc_head = "Select a head from multitask"
        doc_use_ele_temp = "Whether to use electronic temperature, 0 for no, 1 for frame temperature, and 2 for atomic temperature"
        doc_use_hdf5 = "Use HDF5 to store trajs and model_devis"
        return [
            Argument("command", str, optional=True, default="lmp", doc=doc_lmp_cmd),
            Argument(
                "teacher_model_path",
                [BinaryFileInput, str],
                optional=True,
                default=None,
                doc=doc_teacher_model,
            ),
            Argument(
                "shuffle_models",
                bool,
                optional=True,
                default=False,
                doc=doc_shuffle_models,
            ),
            Argument("head", str, optional=True, default=None, doc=doc_head),
            Argument(
                "use_ele_temp", int, optional=True, default=0, doc=doc_use_ele_temp
            ),
            Argument(
                "model_frozen_head", str, optional=True, default=None, doc=doc_head
            ),
            Argument(
                "use_hdf5",
                bool,
                optional=True,
                default=False,
                doc=doc_use_hdf5,
            ),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunNvNMD.lmp_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data


config_args = RunNvNMD.lmp_args


def set_models(lmp_input_name: str, model_names: List[str]):
    with open(lmp_input_name, encoding="utf8") as f:
        lmp_input_lines = f.readlines()

    idx = find_only_one_key(
        lmp_input_lines, ["pair_style", "deepmd"], raise_not_found=False
    )
    if idx is None:
        return
    new_line_split = lmp_input_lines[idx].split()
    match_first = -1
    match_last = -1
    pattern = model_name_match_pattern
    for sidx, ii in enumerate(new_line_split):
        if re.fullmatch(pattern, ii) is not None:
            if match_first == -1:
                match_first = sidx
        else:
            if match_first != -1:
                match_last = sidx
                break
    if match_first == -1:
        raise RuntimeError(
            f"cannot file model pattern {pattern} in line " f" {lmp_input_lines[idx]}"
        )
    if match_last == -1:
        raise RuntimeError(f"last matching index should not be -1, terribly wrong ")
    for ii in range(match_last, len(new_line_split)):
        if re.fullmatch(pattern, new_line_split[ii]) is not None:
            raise RuntimeError(
                f"unexpected matching of model pattern {pattern} "
                f"in line {lmp_input_lines[idx]}"
            )
    new_line_split[match_first:match_last] = model_names
    lmp_input_lines[idx] = " ".join(new_line_split) + "\n"

    with open(lmp_input_name, "w", encoding="utf8") as f:
        f.write("".join(lmp_input_lines))


def find_only_one_key(lmp_lines, key, raise_not_found=True):
    found = []
    for idx in range(len(lmp_lines)):
        words = lmp_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key:
            found.append(idx)
    if len(found) > 1:
        raise RuntimeError("found %d keywords %s" % (len(found), key))
    if len(found) == 0:
        if raise_not_found:
            raise RuntimeError("failed to find keyword %s" % (key))
        else:
            return None
    return found[0]


def get_ele_temp(lmp_log_name):
    with open(lmp_log_name, encoding="utf8") as f:
        lmp_log_lines = f.readlines()

    for line in lmp_log_lines:
        fields = line.split()
        if fields[:2] == ["pair_style", "deepmd"]:
            if "fparam" in fields:
                # for rendering variables
                try:
                    return float(fields[fields.index("fparam") + 1])
                except Exception:
                    pass
            if "aparam" in fields:
                try:
                    return float(fields[fields.index("aparam") + 1])
                except Exception:
                    pass

    return None


def freeze_model(input_model, frozen_model, head=None):
    freeze_args = "-o %s" % frozen_model
    if head is not None:
        freeze_args += " --head %s" % head
    freeze_cmd = "dp --pt freeze -c %s %s" % (input_model, freeze_args)
    ret, out, err = run_command(freeze_cmd, shell=True)
    if ret != 0:
        logging.error(
            "".join(
                (
                    "freeze failed\n",
                    "command was",
                    freeze_cmd,
                    "out msg",
                    out,
                    "\n",
                    "err msg",
                    err,
                    "\n",
                )
            )
        )
        raise TransientError("freeze failed")


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
    trajectories = []
    for f in traj_files:
        traj = read(f, format="lammps-dump-text", index=":", order=True)
        trajectories.append(traj)

    num_frames = len(trajectories[0])
    for traj in trajectories:
        assert len(traj) == num_frames, "Not match"

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
