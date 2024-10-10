import json
import logging
import os
import random
import re
import shutil
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Tuple,
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
    OPIOSign,
    Parameter,
    TransientError,
)

from dpgen2.constants import (
    calypso_log_name,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class CollRunCaly(OP):
    r"""Execute CALYPSO to generate structures in work_dir.

    Changing the work directory into `task_name`. All input files
    have been copied or symbol linked to this directory `task_name` by
    `PrepCalyInput`. The CALYPSO command is exectuted from directory `task_name`.
    The `caly.log` and the `work_dir` will be stored in `op["log"]` and
    `op["work_dir"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),  # for command
                "task_name": BigParameter(str),  # calypso_task.idx
                "cnt_num": Parameter(int),
                "input_file": Artifact(Path),  # input.dat, !!! must be provided
                "step": Artifact(type=Path, optional=True),  # step file
                "results": Artifact(
                    type=Path, optional=True
                ),  # dir named results for evo
                "opt_results_dir": Artifact(
                    type=List[Path], optional=True
                ),  # dir contains POSCAR* CONTCAR* OUTCAR*
                "qhull_input": Artifact(type=Path, optional=True),  # for vsc
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_name": BigParameter(str),  # calypso_task.idx
                "finished": Parameter(str),  # True if cnt_num == maxstep
                "poscar_dir": Artifact(Path),  # dir contains POSCAR* of next step
                "input_file": Artifact(Path),  # input.dat
                "results": Artifact(Path),  # calypso generated results
                "step": Artifact(Path),  # step
                "qhull_input": Artifact(Path),
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

            - `config`: (`dict`) The config of calypso task to obtain the command of calypso.
            - `task_name`: (`str`) The name of the task (calypso_task.{idx}).
            - `input_file`: (`Path`) The input file of the task (input.dat).

            - `step`: (`Path`) The step file from last calypso run
            - `results`: (`Path`) The results dir from last calypso run
            - `opt_results_dir`: (`Path`) The results dir contains POSCAR* CONTCAR* OUTCAR* from last calypso run
            - `qhull_input`: (`Path`) qhull input file `test_qconvex.in`

        Returns
        -------
        Any
            Output dict with components:
            - `poscar_dir`: (`Path`) The dir contains POSCAR*.

            - `task_name`: (`str`) The name of the task (calypso_task.{idx}).
            - `input_file`: (`Path`) The input file of the task (input.dat).
            - `step`: (`Path`) The step file.
            - `results`: (`Path`) The results dir.
            - `qhull_input`: (`Path`) qhull input file.

        Raises
        ------
        TransientError
            On the failure of CALYPSO execution. Resubmit rule should be clear.
        """
        cnt_num = ip["cnt_num"]
        # command
        config = ip["config"] if ip["config"] is not None else {}
        # config = CollRunCaly.normalize_config(config)
        command = config.get("run_calypso_command", "calypso.x")
        # input.dat
        _input_file = ip["input_file"]
        input_file = _input_file.resolve()
        max_step, vsc = get_value_from_inputdat(input_file)
        # work_dir name: calypso_task.idx
        work_dir = Path(ip["task_name"])

        step = ip["step"].resolve() if ip["step"] is not None else ip["step"]
        results = (
            ip["results"].resolve() if ip["results"] is not None else ip["results"]
        )
        opt_results_dir = []
        if ip["opt_results_dir"] is not None:
            for temp in ip["opt_results_dir"]:
                opt_results_dir.append(Path(temp).resolve())

        qhull_input = (
            ip["qhull_input"].resolve()
            if ip["qhull_input"] is not None
            else ip["qhull_input"]
        )

        with set_directory(work_dir):
            # prep files/dirs from last calypso run
            prep_last_calypso_file(step, results, opt_results_dir, qhull_input, vsc)
            # copy input.dat
            Path(input_file.name).symlink_to(input_file)

            finished = "true" if int(cnt_num) == int(max_step) else "false"

            if finished == "false":
                # run calypso
                command = " ".join([command, ">", calypso_log_name])
                ret, out, err = run_command(command, shell=True)
                if ret != 0:
                    logging.error(
                        "".join(
                            (
                                "calypso failed\n",
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
                    raise TransientError("calypso failed")

            poscar_dir = Path("poscar_dir")
            poscar_dir.mkdir(parents=True, exist_ok=True)
            for poscar in Path().glob("POSCAR_*"):
                target = poscar_dir.joinpath(poscar.name)
                shutil.copyfile(poscar, target)

            step = Path("step").read_text().strip()
            # finished = "true" if int(cnt_num) == int(max_step) else "false"

            if not Path("test_qconvex.in").exists():
                Path("test_qconvex.in").write_text("")

        ret_dict = {
            "task_name": str(work_dir),
            "finished": finished,
            "poscar_dir": work_dir.joinpath(poscar_dir),
            "input_file": _input_file,
            "step": work_dir.joinpath("step"),
            "results": work_dir.joinpath("results"),
            "qhull_input": work_dir.joinpath("test_qconvex.in"),
        }
        return OPIO(ret_dict)

    @staticmethod
    def calypso_args():
        doc_calypso_cmd = "The command of calypso (absolute path of calypso.x)."
        return [
            Argument(
                "run_calypso_command",
                str,
                optional=True,
                default="calypso.x",
                doc=doc_calypso_cmd,
            ),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = CollRunCaly.calypso_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data


config_args = CollRunCaly.calypso_args


def prep_last_calypso_file(step, results, opt_results_dir, qhull_input, vsc):
    if step is not None and results is not None and opt_results_dir is not None:
        Path(step.name).symlink_to(step)
        Path(results.name).symlink_to(results)
        assert isinstance(opt_results_dir, list), "opt_results_dir should be a list."
        for opt_results_name in opt_results_dir:
            for file_name in opt_results_name.iterdir():
                Path(file_name.name).symlink_to(file_name)

    if vsc and qhull_input is not None:
        Path(qhull_input.name).symlink_to(qhull_input)


def get_value_from_inputdat(filename):
    max_step = 0
    vsc = False
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "MaxStep" in line:
                max_step = int(line.strip().split("#")[0].split("=")[1])
                continue
            if "VSC" in line:
                vsc_str = line.strip().split("#")[0].split("=")[1].lower().strip()
                if vsc_str.startswith("t"):
                    vsc = True
        return max_step, vsc
