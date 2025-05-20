import json
import logging
import os
import re
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
from dpgen2.utils import (
    run_command,
    setup_ele_temp,
)

from .prep_fp import (
    PrepFp,
)
from .run_fp import (
    RunFp,
)
from .vasp_input import (
    VaspInputs,
    make_kspacing_kpoints,
)

# global static variables
vasp_conf_name = "POSCAR"
vasp_input_name = "INCAR"
vasp_pot_name = "POTCAR"
vasp_kp_name = "KPOINTS"


def clean_lines(string_list, remove_empty_lines=True):
    """[migrated from pymatgen]
    Strips whitespace, carriage returns and empty lines from a list of strings.

    Args:
        string_list: List of strings
        remove_empty_lines: Set to True to skip lines which are empty after
            stripping.

    Returns:
        List of clean strings with no whitespaces.
    """
    for s in string_list:
        clean_s = s
        if "#" in s:
            ind = s.index("#")
            clean_s = s[:ind]
        clean_s = clean_s.strip()
        if (not remove_empty_lines) or clean_s != "":
            yield clean_s


def loads_incar(incar: str):
    lines = list(clean_lines(incar.splitlines()))
    params = {}
    for line in lines:
        for sline in line.split(";"):
            m = re.match(r"(\w+)\s*=\s*(.*)", sline.strip())
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                params[key.upper()] = val
    return params


def dumps_incar(params: dict):
    incar = "\n".join([key + " = " + str(val) for key, val in params.items()]) + "\n"
    return incar


class PrepVasp(PrepFp):
    def set_ele_temp(self, conf_frame, incar):
        use_ele_temp = 0
        ele_temp = None
        if "fparam" in conf_frame.data:
            use_ele_temp = 1
            ele_temp = conf_frame.data["fparam"][0][0]
        if "aparam" in conf_frame.data:
            use_ele_temp = 2
            ele_temp = conf_frame.data["aparam"][0][0][0]
        if ele_temp:
            import scipy.constants as pc

            params = loads_incar(incar)
            params["ISMEAR"] = -1
            params["SIGMA"] = ele_temp * pc.Boltzmann / pc.electron_volt
            incar = dumps_incar(params)
            data = {
                "use_ele_temp": use_ele_temp,
                "ele_temp": ele_temp,
            }
            with open("job.json", "w") as f:
                json.dump(data, f, indent=4)
        return incar

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
        vasp_inputs : VaspInputs
            The VaspInputs object handels all other input files of the task.
        """

        conf_frame.to("vasp/poscar", vasp_conf_name)
        incar = vasp_inputs.incar_template
        incar = self.set_ele_temp(conf_frame, incar)

        Path(vasp_input_name).write_text(incar)
        # fix the case when some element have 0 atom, e.g. H0O2
        tmp_frame = dpdata.System(vasp_conf_name, fmt="vasp/poscar")
        Path(vasp_pot_name).write_text(vasp_inputs.make_potcar(tmp_frame["atom_names"]))
        Path(vasp_kp_name).write_text(vasp_inputs.make_kpoints(conf_frame["cells"][0]))  # type: ignore


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
        return ["job.json"]

    def set_ele_temp(self, system):
        if os.path.exists("job.json"):
            with open("job.json", "r") as f:
                data = json.load(f)
            if "use_ele_temp" in data and "ele_temp" in data:
                if data["use_ele_temp"] == 1:
                    setup_ele_temp(False)
                    system.data["fparam"] = np.tile(data["ele_temp"], [1, 1])
                elif data["use_ele_temp"] == 2:
                    setup_ele_temp(True)
                    system.data["aparam"] = np.tile(
                        data["ele_temp"], [1, system.get_natoms(), 1]
                    )

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
            The command of running vasp task
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
        # run vasp
        command = " ".join([command, ">", log_name])
        ret, out, err = run_command(command, shell=True)
        if ret != 0:
            logging.error(
                "".join(
                    ("vasp failed\n", "out msg: ", out, "\n", "err msg: ", err, "\n")
                )
            )
            raise TransientError("vasp failed")
        # convert the output to deepmd/npy format
        sys = dpdata.LabeledSystem("OUTCAR")
        self.set_ele_temp(sys)
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
