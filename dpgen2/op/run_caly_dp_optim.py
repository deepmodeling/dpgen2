import json
import logging
import pickle
import shutil
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter,
    TransientError,
)

from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_run_opt_file,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class RunCalyDPOptim(OP):
    r"""Perform structure optimization with DP in `ip["work_path"]`.

    The `optim_results_dir` and `traj_results` will be returned as `op["optim_results_dir"]`
    and `op["traj_results"]`.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": Parameter(str),  # calypso_task.idx
                "finished": Parameter(str),
                "cnt_num": Parameter(int),
                "task_dir": Artifact(Path),  # ready to run structure optimization
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "optim_results_dir": Artifact(Path),
                "traj_results": Artifact(Path),
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
            - `task_name` : (`str`)
            - `finished` : (`str`)
            - `cnt_num` : (`int`)
            - `task_dir` : (`Path`)

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_name`: (`str`)
            - `optim_results_dir`: (`List[str]`)
            - `traj_results`: (`Artifact(List[Path])`)
        """
        finished = ip["finished"]
        cnt_num = ip["cnt_num"]

        task_path = ip["task_dir"]
        if task_path is not None:
            input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        else:
            input_files = []

        config = ip["config"] if ip["config"] is not None else {}
        command = config.get(
            "run_opt_command", "python -u calypso_run_opt.py model.ckpt.pt"
        )

        work_dir = Path(ip["task_name"])

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)

            if finished == "false":
                ret, out, err = run_command(command, shell=True)
                if ret != 0:
                    logging.error(
                        "".join(
                            (
                                "opt failed\n",
                                "\ncommand was: ",
                                command,
                                "\nout msg: ",
                                out,
                                "\n",
                                "\nerr msg: ",
                                err,
                                "\n",
                            )
                        )
                    )
                    raise TransientError("opt failed")

                optim_results_dir = Path("optim_results_dir")
                optim_results_dir.mkdir(parents=True, exist_ok=True)
                for poscar in Path().glob("POSCAR_*"):
                    target = optim_results_dir.joinpath(poscar.name)
                    shutil.copyfile(poscar, target)
                for contcar in Path().glob("CONTCAR_*"):
                    target = optim_results_dir.joinpath(contcar.name)
                    shutil.copyfile(contcar, target)
                for outcar in Path().glob("OUTCAR_*"):
                    target = optim_results_dir.joinpath(outcar.name)
                    shutil.copyfile(outcar, target)

                traj_results_dir = Path("traj_results")
                traj_results_dir.mkdir(parents=True, exist_ok=True)
                for traj in Path().glob("*.traj"):
                    target = traj_results_dir.joinpath(str(cnt_num) + "." + traj.name)
                    shutil.copyfile(traj, target)

            else:
                optim_results_dir = Path("optim_results_dir")
                optim_results_dir.mkdir(parents=True, exist_ok=True)
                traj_results_dir = Path("traj_results")
                traj_results_dir.mkdir(parents=True, exist_ok=True)

        return OPIO(
            {
                "task_name": str(work_dir),
                "optim_results_dir": work_dir / optim_results_dir,
                "traj_results": work_dir / traj_results_dir,
            }
        )
