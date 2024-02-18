import logging
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import dpdata
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
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    fp_default_log_name,
    fp_default_out_data_name,
)
from dpgen2.utils.run_command import (
    run_command,
)

from .cp2k_input import (
    Cp2kInputs,
)
from .prep_fp import (
    PrepFp,
)
from .run_fp import (
    RunFp,
)

# global static variables
cp2k_conf_name = "coord_n_cell.inc"
cp2k_input_name = "input.inp"


class PrepCp2k(PrepFp):
    def prep_task(
        self,
        conf_frame: dpdata.System,
        cp2k_inputs: Cp2kInputs,
    ):
        r"""Define how one CP2K task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        cp2k_inputs : Cp2kInputs
            The Cp2kInputs object handels all other input files of the task.
        """
        Path(cp2k_conf_name).write_text(cp2k_inputs.make_cp2k_coord_cell(conf_frame))
        Path(cp2k_input_name).write_text(cp2k_inputs.make_cp2k_input())


class RunCp2k(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a cp2k task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [cp2k_conf_name, cp2k_input_name]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a cp2k task.

        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return []

    def run_task(
        self,
        command: str,
        out: str,
        log: str,
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        command : str
            The command of running cp2k task
        out : str
            The name of the output data file.
        log : str
            The name of the log file

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """

        log_name = log
        out_name = out
        # run cp2k
        command = " ".join([command, ">", log_name])
        ret, out, err = run_command(command, shell=True)
        if ret != 0:
            logging.error(
                "".join(
                    ("cp2k failed\n", "out msg: ", out, "\n", "err msg: ", err, "\n")
                )
            )
            raise TransientError("cp2k failed")
        # convert the output to deepmd/npy format
        sys = dpdata.LabeledSystem(log, fmt="cp2k/output")
        sys.to("deepmd/npy", out_name)
        return out_name, log_name

    @staticmethod
    def args():
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_cp2k_cmd = "The command of CP2K"
        doc_cp2k_log = "The log file name of CP2K"
        doc_cp2k_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument("command", str, optional=True, default="cp2k", doc=doc_cp2k_cmd),
            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_cp2k_out,
            ),
            Argument(
                "log", str, optional=True, default=fp_default_log_name, doc=doc_cp2k_log
            ),
        ]
