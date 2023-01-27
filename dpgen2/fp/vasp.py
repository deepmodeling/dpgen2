from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
    FatalError,
    BigParameter,
)
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    List,
    Set,
    Dict,
    Union,
)
import numpy as np
import dpdata
from dargs import (
    dargs,
    Argument,
    Variant,
    ArgumentEncoder,
)

from .prep_fp import PrepFp
from .run_fp import RunFp
from .vasp_input import VaspInputs, make_kspacing_kpoints
from dpgen2.constants import (
    fp_default_log_name,
    fp_default_out_data_name,
)
from dpgen2.utils.run_command import run_command

# global static variables
vasp_conf_name = "POSCAR"
vasp_input_name = "INCAR"
vasp_pot_name = "POTCAR"
vasp_kp_name = "KPOINTS"


class PrepVasp(PrepFp):
    def prep_task(
        self,
        conf_frame: dpdata.System,
        vasp_inputs: VaspInputs,
    ):
        r"""Define how one Vasp task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs: VaspInputs
            The VaspInputs object handels all other input files of the task.
        """

        conf_frame.to("vasp/poscar", vasp_conf_name)
        Path(vasp_input_name).write_text(vasp_inputs.incar_template)
        # fix the case when some element have 0 atom, e.g. H0O2
        tmp_frame = dpdata.System(vasp_conf_name, fmt="vasp/poscar")
        Path(vasp_pot_name).write_text(vasp_inputs.make_potcar(tmp_frame["atom_names"]))
        Path(vasp_kp_name).write_text(vasp_inputs.make_kpoints(conf_frame["cells"][0]))


class RunVasp(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a vasp task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [vasp_conf_name, vasp_input_name, vasp_pot_name, vasp_kp_name]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a vasp task.

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
        command: str
            The command of running vasp task
        out: str
            The name of the output data file.
        log: str
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
        # run vasp
        command = " ".join([command, ">", log_name])
        ret, out, err = run_command(command, shell=True)
        if ret != 0:
            raise TransientError(
                "vasp failed\n", "out msg", out, "\n", "err msg", err, "\n"
            )
        # convert the output to deepmd/npy format
        sys = dpdata.LabeledSystem("OUTCAR")
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

        doc_vasp_cmd = "The command of VASP"
        doc_vasp_log = "The log file name of VASP"
        doc_vasp_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument("command", str, optional=True, default="vasp", doc=doc_vasp_cmd),
            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_vasp_out,
            ),
            Argument(
                "log", str, optional=True, default=fp_default_log_name, doc=doc_vasp_log
            ),
        ]
