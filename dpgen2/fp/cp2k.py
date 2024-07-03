import os
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

import dpdata
from dargs import (
    Argument,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)

try:
    from fpop.cp2k import (
        Cp2kInputs,
        PrepCp2k,
        RunCp2k,
    )
except ModuleNotFoundError:
    Cp2kInputs = PrepCp2k = RunCp2k = object

from ..constants import (
    fp_default_out_data_name,
)


class FpOpCp2kInputs(Cp2kInputs):  # type: ignore
    @staticmethod
    def args():
        doc_inp_file = "The path to the user-submitted CP2K input file."
        return [
            Argument("inp_file", str, optional=False, doc=doc_inp_file),
        ]


class PrepFpOpCp2k(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "type_map": List[str],
                "confs": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_names": BigParameter(List[str]),
                "task_paths": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        confs = []
        # remove atom types with 0 atom from type map
        # for all atom types in the type map
        for p in ip["confs"]:
            for f in p.rglob("type.raw"):
                system = f.parent
                s = dpdata.System(system, fmt="deepmd/npy")
                atom_numbs = []
                atom_names = []
                for numb, name in zip(s["atom_numbs"], s["atom_names"]):  # type: ignore https://github.com/microsoft/pyright/issues/5620
                    if numb > 0:
                        atom_numbs.append(numb)
                        atom_names.append(name)
                if atom_names != s["atom_names"]:
                    for i, t in enumerate(s["atom_types"]):  # type: ignore https://github.com/microsoft/pyright/issues/5620
                        s["atom_types"][i] = atom_names.index(s["atom_names"][t])  # type: ignore https://github.com/microsoft/pyright/issues/5620
                    s.data["atom_numbs"] = atom_numbs
                    s.data["atom_names"] = atom_names
                    target = "output/%s" % system
                    s.to("deepmd/npy", target)
                    confs.append(Path(target))
                else:
                    confs.append(system)
        op_in = OPIO(
            {
                "inputs": ip["config"]["inputs"],
                "type_map": ip["type_map"],
                "confs": confs,
                "prep_image_config": ip["config"].get("prep", {}),
            }
        )
        op = PrepCp2k()
        return op.execute(op_in)  # type: ignore in the case of not importing fpop


def get_run_type(lines: List[str]) -> Optional[str]:
    for line in lines:
        if "RUN_TYPE" in line:
            return line.split()[-1]
    return None


class RunFpOpCp2k(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "labeled_data": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        run_config = ip["config"].get("run", {})
        op_in = OPIO(
            {
                "task_name": ip["task_name"],
                "task_path": ip["task_path"],
                "backward_list": [],
                "log_name": "output.log",
                "run_image_config": run_config,
            }
        )
        op = RunCp2k()
        op_out = op.execute(op_in)  # type: ignore in the case of not importing fpop
        workdir = op_out["backward_dir"].parent

        file_path = os.path.join(str(workdir), "output.log")

        # convert the output to deepmd/npy format
        with open(workdir / "input.inp", "r") as f:
            lines = f.readlines()

        # 获取 RUN_TYPE
        run_type = get_run_type(lines)

        if run_type == "ENERGY_FORCE":
            sys = dpdata.LabeledSystem(file_path, fmt="cp2kdata/e_f")
        elif run_type == "MD":
            sys = dpdata.LabeledSystem(
                str(workdir), cp2k_output_name="output.log", fmt="cp2kdata/md"
            )
        else:
            raise ValueError(f"Type of calculation {run_type} not supported")

        # out_name = run_config.get("out", fp_default_out_data_name)
        out_name = fp_default_out_data_name
        sys.to("deepmd/npy", workdir / out_name)

        return OPIO(
            {
                "log": workdir / "output.log",
                "labeled_data": workdir / out_name,
            }
        )

    @staticmethod
    def args():
        doc_cmd = "The command of cp2k"
        doc_out = (
            "The output dir name of labeled data. "
            "In `deepmd/npy` format provided by `dpdata`."
        )
        return [
            Argument("command", str, optional=True, default="cp2k", doc=doc_cmd),
            Argument(
                "out", str, optional=True, default=fp_default_out_data_name, doc=doc_out
            ),
        ]
