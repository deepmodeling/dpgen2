import glob
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
    pt2_model_name_pattern,
    pytorch_model_name_pattern,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)

_MODEL_BACKEND_ALIASES = {"pt-expt": "pytorch-exportable"}
_MODEL_BACKEND_FLAGS = {
    "pytorch": "--pt",
    "pytorch-exportable": "--pt-expt",
}


class PrepareDPModels(OP):
    """Freeze DP checkpoints once before exploration tasks are fanned out."""

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({"models": Artifact(List[Path])})

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        config = RunLmp.normalize_config(ip["config"] or {})
        return OPIO({"models": prepare_dp_models(ip["models"], config)})


class RunLmp(OP):
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
                "extra_outputs": Artifact(List[Path]),
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
            ext = os.path.splitext(teacher_model.file_name)[-1]
            teacher_model_file = "teacher_model" + ext
            teacher_model.save_as_file(teacher_model_file)
            model_files = [Path(teacher_model_file).resolve()] + model_files

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)
            # link models
            model_names = []
            for idx, mm in enumerate(model_files):
                ext = os.path.splitext(mm)[-1]
                if ext == ".pb":
                    mname = model_name_pattern % (idx)
                    Path(mname).symlink_to(mm)
                elif ext == ".pth":
                    mname = pytorch_model_name_pattern % (idx)
                    Path(mname).symlink_to(mm)
                elif ext == ".pt2":
                    mname = pt2_model_name_pattern % (idx)
                    Path(mname).symlink_to(mm)
                elif ext == ".pt":
                    # freeze model
                    backend = _model_backend(config)
                    mname = _model_name(idx, config["model_format"])
                    freeze_model(
                        mm,
                        mname,
                        config.get("model_frozen_head"),
                        backend,
                    )
                    if config["dp_compress"]:
                        compressed = _compressed_model_name(idx, config["model_format"])
                        compress_model(mname, compressed, backend)
                        mname = compressed
                else:
                    raise RuntimeError(
                        "Model file with extension '%s' is not supported" % ext
                    )
                model_names.append(mname)

            if shuffle_models:
                random.shuffle(model_names)

            set_models(lmp_input_name, model_names)
            if any(Path(name).suffix == ".pt2" for name in model_names):
                ensure_pt2_atom_map(lmp_input_name)

            # run lmp
            command = " ".join([command, "-i", lmp_input_name, "-log", lmp_log_name])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "lmp failed\n",
                            "command was: ",
                            command,
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

        extra_outputs = []
        for fname in config["extra_output_files"]:
            extra_outputs += list(work_dir.glob(fname))
        ret_dict["extra_outputs"] = extra_outputs  # type: ignore
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
        doc_extra_output_files = "Extra output file names, support wildcards"
        doc_model_devi_backend = (
            "The DeePMD backend used to freeze models for exploration"
        )
        doc_model_format = "The frozen model format. Use 'pt2' for DPA4 and DPA4C"
        doc_dp_compress = "Compress the frozen model before exploration"
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
            Argument(
                "extra_output_files",
                list,
                optional=True,
                default=[],
                doc=doc_extra_output_files,
            ),
            Argument(
                "model_devi_backend",
                str,
                optional=True,
                default="pytorch",
                doc=doc_model_devi_backend,
            ),
            Argument(
                "model_format",
                str,
                optional=True,
                default="pth",
                doc=doc_model_format,
            ),
            Argument(
                "dp_compress",
                bool,
                optional=True,
                default=False,
                doc=doc_dp_compress,
            ),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunLmp.lmp_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data


config_args = RunLmp.lmp_args


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
            f"cannot file model pattern {pattern} in line  {lmp_input_lines[idx]}"
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


def ensure_pt2_atom_map(lmp_input_name: str):
    """Ensure a PT2 LAMMPS input enables the atom map before creating a box.

    Parameters
    ----------
    lmp_input_name : str
        Path to the LAMMPS input file.

    Raises
    ------
    RuntimeError
        If an existing atom-map command follows the first box-creation command
        in a clear-delimited section, or no box-creation command is present.
    """
    with open(lmp_input_name, encoding="utf8") as f:
        lmp_input_lines = f.readlines()

    commands = []
    command_start = 0
    command_parts = []
    for index, line in enumerate(lmp_input_lines):
        code = line.partition("#")[0].rstrip()
        command_parts.append(code[:-1] if code.endswith("&") else code)
        if code.endswith("&"):
            continue
        commands.append((command_start, " ".join(command_parts)))
        command_start = index + 1
        command_parts = []
    if command_parts:
        commands.append((command_start, " ".join(command_parts)))

    sections = [[]]
    for command in commands:
        if re.match(r"^\s*clear(?:\s|$)", command[1]):
            sections.append([])
        else:
            sections[-1].append(command)

    insert_indices = []
    found_box_command = False
    for section in sections:
        boundary_positions = [
            position
            for position, (_, command) in enumerate(section)
            if re.search(r"\b(?:create_box|read_data|read_restart)\b", command)
        ]
        if not boundary_positions:
            continue
        found_box_command = True
        first_boundary = boundary_positions[0]
        map_positions = [
            position
            for position, (_, command) in enumerate(section)
            if re.match(r"^\s*atom_modify\b.*\bmap\s+(?:yes|array|hash)\b", command)
        ]
        if any(position < first_boundary for position in map_positions):
            continue
        if map_positions:
            raise RuntimeError(
                "PT2 LAMMPS inputs require atom_modify map before "
                "create_box, read_data, or read_restart"
            )
        insert_indices.append(section[first_boundary][0])

    if not found_box_command:
        raise RuntimeError(
            "PT2 LAMMPS inputs require create_box, read_data, or read_restart"
        )

    for index in reversed(insert_indices):
        lmp_input_lines.insert(index, "atom_modify        map yes\n")
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


def _model_backend(config):
    backend = _MODEL_BACKEND_ALIASES.get(
        config["model_devi_backend"], config["model_devi_backend"]
    )
    model_format = config["model_format"]
    if backend not in _MODEL_BACKEND_FLAGS:
        raise RuntimeError(f"Unsupported model-deviation backend '{backend}'")
    if model_format not in ["pth", "pt2"]:
        raise RuntimeError(f"Unsupported model format '{model_format}'")
    if model_format == "pth" and backend != "pytorch":
        raise RuntimeError("The pth model format requires the pytorch backend")
    if config["dp_compress"] and not (
        backend == "pytorch-exportable" and model_format == "pt2"
    ):
        raise RuntimeError(
            "Compressed pt2 models require the pytorch-exportable backend"
        )
    return backend


def validate_model_backend(train_backend, config):
    """Validate that a PyTorch checkpoint is frozen by its training backend.

    Parameters
    ----------
    train_backend : str
        DeePMD training backend.
    config : dict
        LAMMPS exploration configuration.

    Raises
    ------
    RuntimeError
        If PyTorch training and deployment backends differ.
    """
    train_backend = _MODEL_BACKEND_ALIASES.get(train_backend, train_backend)
    if train_backend not in _MODEL_BACKEND_FLAGS:
        return
    model_backend = _model_backend(RunLmp.normalize_config(config))
    if model_backend != train_backend:
        raise RuntimeError(
            f"The model-deviation backend '{model_backend}' cannot freeze a "
            f"checkpoint trained by '{train_backend}'; use the same backend "
            "for training and model deployment"
        )


def _model_name(index, model_format):
    if model_format == "pt2":
        return pt2_model_name_pattern % index
    return pytorch_model_name_pattern % index


def _compressed_model_name(index, model_format):
    return "model.%03d.compressed.%s" % (index, model_format)


def prepare_dp_models(models, config):
    """Return frozen models, exporting checkpoints once when needed."""
    prepared = []
    output_dir = Path("prepared_models")
    for idx, model in enumerate(models):
        model = Path(model).resolve()
        ext = model.suffix
        if ext != ".pt":
            if ext not in [".pb", ".pth", ".pt2"]:
                raise RuntimeError(
                    "Model file with extension '%s' is not supported" % ext
                )
            prepared.append(model)
            continue
        backend = _model_backend(config)
        output_dir.mkdir(exist_ok=True)
        frozen_model = output_dir / _model_name(idx, config["model_format"])
        freeze_model(
            model,
            frozen_model,
            config.get("model_frozen_head"),
            backend,
        )
        if config["dp_compress"]:
            compressed_model = output_dir / _compressed_model_name(
                idx, config["model_format"]
            )
            compress_model(frozen_model, compressed_model, backend)
            frozen_model = compressed_model
        prepared.append(frozen_model)
    return prepared


def freeze_model(input_model, frozen_model, head=None, backend="pytorch"):
    backend = _MODEL_BACKEND_ALIASES.get(backend, backend)
    freeze_cmd = [
        "dp",
        _MODEL_BACKEND_FLAGS[backend],
        "freeze",
        "-c",
        str(input_model),
        "-o",
        str(frozen_model),
    ]
    if head is not None:
        freeze_cmd.extend(["--head", str(head)])
    if backend == "pytorch-exportable" and Path(frozen_model).suffix == ".pt2":
        freeze_cmd.extend(["--lower-kind", "graph"])
    ret, out, err = run_command(freeze_cmd)
    if ret != 0:
        logging.error(
            "freeze failed\ncommand was %s\nout msg%s\nerr msg%s\n",
            freeze_cmd,
            out,
            err,
        )
        raise FatalError("freeze failed")


def compress_model(input_model, output_model, backend="pytorch-exportable"):
    backend = _MODEL_BACKEND_ALIASES.get(backend, backend)
    compress_cmd = [
        "dp",
        _MODEL_BACKEND_FLAGS[backend],
        "compress",
        "-i",
        str(input_model),
        "-o",
        str(output_model),
    ]
    ret, out, err = run_command(compress_cmd)
    if ret != 0:
        logging.error(
            "compress failed\ncommand was%s\nout msg%s\nerr msg%s\n",
            compress_cmd,
            out,
            err,
        )
        raise FatalError("compress failed")


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


class RunLmpHDF5(RunLmp):
    @classmethod
    def get_output_sign(cls):
        output_sign = super().get_output_sign()
        output_sign["traj"] = Artifact(HDF5Datasets)
        output_sign["model_devi"] = Artifact(HDF5Datasets)
        return output_sign

    def get_model_devi(self, model_devi_file):
        return np.loadtxt(model_devi_file)
