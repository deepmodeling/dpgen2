"""Prep and Run Gaussian tasks."""
import logging
from typing import (
    Any,
    List,
    Tuple,
)

import dpdata
from dargs import (
    Argument,
    dargs,
)
from dflow.python import (
    TransientError,
)

from dpgen2.constants import (
    fp_default_out_data_name,
)
from dpgen2.utils.run_command import (
    run_command,
)

from .prep_fp import (
    PrepFp,
)
from .run_fp import (
    RunFp,
)

# global static variables
gaussian_input_name = "task.gjf"
# this output name is generated by Gaussian
gaussian_output_name = "task.log"


class GaussianInputs:
    @staticmethod
    def args() -> List[Argument]:
        r"""The arguments of the GaussianInputs class."""
        doc_keywords = "Gaussian keywords, e.g. force b3lyp/6-31g**. If a list, run multiple steps."
        doc_multiplicity = (
            "spin multiplicity state. It can be a number. If auto, multiplicity will be detected "
            "automatically, with the following rules:\n\n"
            "fragment_guesses=True multiplicity will +1 for each radical, and +2 for each oxygen molecule\n\n"
            "fragment_guesses=False multiplicity will be 1 or 2, but +2 for each oxygen molecule."
        )
        doc_charge = (
            "molecule charge. Only used when charge is not provided by the system"
        )
        doc_basis_set = "custom basis set"
        doc_keywords_high_multiplicity = (
            "keywords for points with multiple raicals. multiplicity should be auto. "
            "If not set, fallback to normal keywords"
        )
        doc_fragment_guesses = "initial guess generated from fragment guesses. If True, multiplicity should be auto"
        doc_nproc = "Number of CPUs to use"

        return [
            Argument("keywords", [str, list], optional=False, doc=doc_keywords),
            Argument(
                "multiplicity",
                [int, str],
                optional=True,
                default="auto",
                doc=doc_multiplicity,
            ),
            Argument("charge", int, optional=True, default=0, doc=doc_charge),
            Argument("basis_set", str, optional=True, doc=doc_basis_set),
            Argument(
                "keywords_high_multiplicity",
                str,
                optional=True,
                doc=doc_keywords_high_multiplicity,
            ),
            Argument(
                "fragment_guesses",
                bool,
                optional=True,
                default=False,
                doc=doc_fragment_guesses,
            ),
            Argument("nproc", int, optional=True, default=1, doc=doc_nproc),
        ]

    def __init__(self, **kwargs: Any):
        self.data = kwargs


class PrepGaussian(PrepFp):
    def prep_task(
        self,
        conf_frame: dpdata.System,
        inputs: GaussianInputs,
    ):
        r"""Define how one Gaussian task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs : GaussianInputs
            The GaussianInputs object handels all other input files of the task.
        """

        conf_frame.to("gaussian/gjf", gaussian_input_name, **inputs.data)


class RunGaussian(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a Gaussian task.

        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [gaussian_input_name]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a Gaussian task.

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
        post_command: str,
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs

        Parameters
        ----------
        command : str
            The command of running gaussian task
        out : str
            The name of the output data file.

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """
        # run gaussian
        out_name = out
        command = " ".join([command, gaussian_input_name])
        ret, out, err = run_command(command, shell=True)
        if ret != 0:
            logging.error(
                "".join(
                    (
                        "gaussian failed\n",
                        "out msg: ",
                        out,
                        "\n",
                        "err msg: ",
                        err,
                        "\n",
                    )
                )
            )
            raise TransientError("gaussian failed")
        if post_command is not None:
            ret, out, err = run_command(post_command, shell=True)
            if ret != 0:
                logging.error(
                    "".join(
                        (
                            "gaussian postprocessing failed\n",
                            "out msg: ",
                            out,
                            "\n",
                            "err msg: ",
                            err,
                            "\n",
                        )
                    )
                )
                raise TransientError("gaussian postprocessing failed")
        # convert the output to deepmd/npy format
        sys = dpdata.LabeledSystem(gaussian_output_name, fmt="gaussian/log")
        sys.to("deepmd/npy", out_name)
        return out_name, gaussian_output_name

    @staticmethod
    def args() -> List[dargs.Argument]:
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_gaussian_cmd = "The command of Gaussian"
        doc_gaussian_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        doc_post_command = "The command after Gaussian"
        return [
            Argument(
                "command", str, optional=True, default="g16", doc=doc_gaussian_cmd
            ),
            Argument(
                "out",
                str,
                optional=True,
                default=fp_default_out_data_name,
                doc=doc_gaussian_out,
            ),
            Argument(
                "post_command", str, optional=True, default=None, doc=doc_post_command
            ),
        ]
