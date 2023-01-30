"""Prep and Run Gaussian tasks."""
from dflow.python import TransientError, FatalError
from typing import Tuple, List, Any, Optional
import dpdata
from dargs import (
    dargs,
    Argument,
)

from .prep_fp import PrepFp
from .run_fp import RunFp
from ..utils import BinaryFileInput
from dpgen2.constants import fp_default_out_data_name, fp_default_log_name
from dpgen2.utils.run_command import run_command
from pathlib import Path
import os

import numpy as np

# global static variables
deepmd_input_path = "one_frame_input"

# global static variables
deepmd_temp_path = "one_frame_temp"

# global static variables
deepmd_teacher_model = "teacher_model.pb"


class DeepmdInputs:
    @staticmethod
    def args() -> List[Argument]:
        return []

    def __init__(self, **kwargs: Any):
        self.data = kwargs


class PrepDeepmd(PrepFp):
    def prep_task(
        self,
        conf_frame: dpdata.System,
        inputs,
    ):
        r"""Define how one Deepmd task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs: str or dict
            This parameter is useless in deepmd.
        """
        conf_frame.to("deepmd/npy", deepmd_input_path)


class RunDeepmd(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a Deepmd task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [deepmd_input_path]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a Deepmd task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return []

    def run_task(
        self,
        teacher_model_path: BinaryFileInput,
        out: str,
        log: str,
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        command: str
            The command of running Deepmd task
        out: str
            The name of the output data file.

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """
        log_name = log
        out_name = out

        dp, type_map_teacher = self._get_dp_model(teacher_model_path)

        # Run deepmd
        self._dp_infer(dp, type_map_teacher, out_name)

        run_command(f'echo "job finished!" > {log_name}', shell=True)

        return out_name, log_name

    def _get_dp_model(self, teacher_model_path: BinaryFileInput):
        from deepmd.infer import DeepPot  # type: ignore

        teacher_model_path.save_as_file(deepmd_teacher_model)
        dp = DeepPot(deepmd_teacher_model)

        type_map_teacher = dp.get_type_map()

        os.remove(deepmd_teacher_model)
        return dp, type_map_teacher

    def _prep_input(self, type_map_teacher):
        ss = dpdata.System(deepmd_input_path, fmt="deepmd/npy")
        conf_type_map = ss["atom_names"]

        if not set(conf_type_map).issubset(set(type_map_teacher)):
            err_message = (
                f"the type map of system ({conf_type_map}) is not subset of "
                + f"the type map of the teacher model ({type_map_teacher})."
            )
            raise FatalError("deepmd labeling failed\n", "err msg", err_message, "\n")

        # make sure the order of elements in sys_type_map
        # is the same as that in type_map_teacher
        temp_type_map = [ele for ele in type_map_teacher if ele in set(conf_type_map)]

        ss.apply_type_map(temp_type_map)
        ss.to("deepmd/npy", deepmd_temp_path)
        return conf_type_map, temp_type_map

    def _dp_infer(self, dp, type_map_teacher, out_name):
        conf_type_map, temp_type_map = self._prep_input(type_map_teacher)

        ss = dpdata.System(
            deepmd_temp_path, fmt="deepmd/npy", type_map=type_map_teacher
        )

        coord_npy_path_list = list(Path(deepmd_temp_path).glob("*/coord.npy"))
        assert len(coord_npy_path_list) == 1, coord_npy_path_list
        coord_npy_path = coord_npy_path_list[0]
        energy_npy_path = coord_npy_path.parent / "energy.npy"
        force_npy_path = coord_npy_path.parent / "force.npy"
        virial_npy_path = coord_npy_path.parent / "virial.npy"

        nframe = ss.get_nframes()
        coord = ss["coords"]
        cell = ss["cells"].reshape([nframe, -1])
        atype = ss["atom_types"].tolist()

        energy, force, virial_force = dp.eval(coord, cell, atype)

        with open(energy_npy_path, "wb") as f:
            np.save(f, energy)
        with open(force_npy_path, "wb") as f:
            np.save(f, force)
        with open(virial_npy_path, "wb") as f:
            np.save(f, virial_force)

        ss = dpdata.LabeledSystem(
            deepmd_temp_path, fmt="deepmd/npy", type_map=temp_type_map
        )
        ss.apply_type_map(conf_type_map)
        ss.to("deepmd/npy", out_name)

    @staticmethod
    def args() -> List[dargs.Argument]:
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_deepmd_teacher_model = (
            "The path of teacher model, which can be loaded by deepmd.infer.DeepPot"
        )
        doc_deepmd_log = "The log file name of dp"
        doc_deepmd_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument(
                "teacher_model_path",
                [str, BinaryFileInput],
                optional=False,
                doc=doc_deepmd_teacher_model,
            ),
            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_deepmd_out,
            ),
            Argument(
                "log",
                str,
                optional=True,
                default=fp_default_log_name,
                doc=doc_deepmd_log,
            ),
        ]
