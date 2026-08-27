import os
from pathlib import (
    Path,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
)

from dpgen2.utils.dflow_types import DflowList


class PrepRelax(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "expl_config": dict,
                "cifs": Artifact(DflowList[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "ntasks": int,
                "task_paths": Artifact(DflowList[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        ncifs = len(ip["cifs"])
        config = ip["expl_config"]
        group_size = config["relax_group_size"]
        ntasks = int(ncifs / group_size)
        task_paths = []
        for i in range(ntasks):
            task_dir = Path(f"task.{i:06d}")
            task_dir.mkdir(exist_ok=True)
            for j in range(group_size * i, min(group_size * (i + 1), ncifs)):
                os.symlink(ip["cifs"][j], task_dir / (f"{j}.cif"))
            task_paths.append(task_dir)
        return OPIO(
            {
                "ntasks": ntasks,
                "task_paths": task_paths,
            }
        )
