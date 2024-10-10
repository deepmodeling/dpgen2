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
    calypso_opt_dir_name,
    calypso_run_opt_file,
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


class PrepCalyModelDevi(OP):
    """Prepare the working directories and input file according to slices information
    for making model deviation.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "config": BigParameter(dict),
                "traj_results": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_name_list": Parameter(List[str]),
                "grouped_traj_list": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        """Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `task_name` : (`str`)
            - `config` : (`BigParameter(dict)`)
            - `traj_results` : (`Path`)

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_name_list`: (`List[str]`)
            - `grouped_traj_list`: (`Artifact(List[Path])`)

        """
        work_dir = Path(ip["task_name"])
        traj_results_dir = [
            Path(dir_name).resolve()
            for dir_name in ip["traj_results"]
            if dir_name is not None
        ]
        trajs = [
            traj.resolve()
            for traj_dir in traj_results_dir
            for traj in Path(traj_dir).rglob("*.traj")
        ]
        expl_config = ip["config"]
        group_size = expl_config.get("model_devi_group_size", len(trajs))

        with set_directory(work_dir):
            grouped_trajs_list = [
                trajs[i : i + group_size] for i in range(0, len(trajs), group_size)
            ]

            traj_cnt = 0
            task_dirs = []
            for idx, grouped_trajs in enumerate(grouped_trajs_list):
                trajs_path = Path(f"trajs_part_{idx}")
                task_dirs.append(work_dir / trajs_path)
                with set_directory(trajs_path):
                    for traj in grouped_trajs:
                        Path(f"{traj_cnt}.{traj.name}").symlink_to(traj)
                        traj_cnt += 1

            task_names = [str(task_dir) for task_dir in task_dirs]

        return OPIO(
            {
                "task_name_list": task_names,
                "grouped_traj_list": task_dirs,
            }
        )
