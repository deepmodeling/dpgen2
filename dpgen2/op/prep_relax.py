import os
from pathlib import (
    Path,
)
from typing import (
    List,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
)


class PrepRelax(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "expl_config": dict,
                "cifs": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "ntasks": int,
                "task_paths": Artifact(List[Path]),
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
            task_dir = Path("task.%06d" % i)
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
