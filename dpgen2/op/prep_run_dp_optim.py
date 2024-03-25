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
    calypso_opt_dir_name,
    model_name_pattern,
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


class PrepRunDPOptim(OP):
    r"""Prepare the working directories and input file for structure optimization with DP.

    `POSCAR_*`, `model.000.pb`, `calypso_run_opt.py` and `calypso_check_opt.py` will be copied
    or symlink to each optimization directory from `ip["work_path"]`, according to the
    popsize `ip["caly_input"]["PopSize"]`.
    The paths of these optimization directory will be returned as `op["optim_paths"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": Parameter(str),  # calypso_task.idx
                "finished": Parameter(str),
                "cnt_num": Parameter(int),
                "poscar_dir": Artifact(
                    Path
                ),  # from run_calypso first, then from collect_run_caly
                "models_dir": Artifact(Path),  #
                "caly_run_opt_file": Artifact(Path),  # from prep_caly_input
                "caly_check_opt_file": Artifact(Path),  # from prep_caly_input
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "optim_results_dir": Artifact(Path),
                "traj_results": Artifact(Path),
                "caly_run_opt_file": Artifact(Path),
                "caly_check_opt_file": Artifact(Path),
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
            - `poscar_dir` : (`Path`)
            - `models_dir` : (`Path`)
            - `caly_run_opt_file` : (`Path`)
            - `caly_check_opt_file` : (`Path`)

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_name`: (`str`)
            - `optim_results_dir`: (`List[str]`)
            - `traj_results`: (`Artifact(List[Path])`)
            - `caly_run_opt_file` : (`Path`)
            - `caly_check_opt_file` : (`Path`)
        """
        finished = ip["finished"]
        cnt_num = ip["cnt_num"]

        work_dir = Path(ip["task_name"])
        poscar_dir = ip["poscar_dir"]
        models_dir = ip["models_dir"]
        _caly_run_opt_file = ip["caly_run_opt_file"]
        _caly_check_opt_file = ip["caly_check_opt_file"]
        caly_run_opt_file = _caly_run_opt_file.resolve()
        caly_check_opt_file = _caly_check_opt_file.resolve()
        poscar_list = [poscar.resolve() for poscar in poscar_dir.rglob("POSCAR_*")]

        model_name = "frozen_model.pb"
        model_list = [model.resolve() for model in models_dir.rglob(model_name)]
        if len(model_list) == 0:
            model_name = "model.ckpt.pt"
            model_list = [model.resolve() for model in models_dir.rglob(model_name)]

        model_list = sorted(model_list, key=lambda x: str(x).split(".")[1])
        model_file = model_list[0]

        config = ip["config"] if ip["config"] is not None else {}
        command = config.get(
            "run_opt_command", f"python -u calypso_run_opt.py {model_name}"
        )

        with set_directory(work_dir):
            for idx, poscar in enumerate(poscar_list):
                Path(poscar.name).symlink_to(poscar)
            Path(model_name).symlink_to(model_file)
            Path(caly_run_opt_file.name).symlink_to(caly_run_opt_file)
            Path(caly_check_opt_file.name).symlink_to(caly_check_opt_file)

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
                "caly_run_opt_file": work_dir / caly_run_opt_file.name,
                "caly_check_opt_file": work_dir / caly_check_opt_file.name,
            }
        )
