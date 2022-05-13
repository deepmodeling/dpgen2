from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)

import json, pickle
from typing import Tuple, List
from pathlib import Path
from dpgen2.exploration.task import ExplorationTaskGroup
from dpgen2.constants import (
    lmp_task_pattern,
)
from dpgen2.utils import (
    load_object_from_file,
    dump_object_to_file,
)

class PrepLmp(OP):
    r"""Prepare the working directories for LAMMPS tasks.

    A list of working directories (defined by `ip["task"]`)
    containing all files needed to start LAMMPS tasks will be
    created. The paths of the directories will be returned as
    `op["task_paths"]`. The identities of the tasks are returned as
    `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "lmp_task_grp": Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names": List[str],
            "task_paths": Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `lmp_task_grp` : (`Artifact(Path)`) Can be pickle loaded as a ExplorationTaskGroup. Definitions for LAMMPS tasks
        
        Returns
        -------
        op : dict 
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the LAMMPS simulation. The order fo the Paths should be consistent with `op["task_names"]`
        """

        lmp_task_grp = load_object_from_file(ip['lmp_task_grp'])
        cc = 0
        task_paths = []
        for tt in lmp_task_grp:
            ff = tt.files()
            tname = _mk_task_from_files(cc, ff)
            task_paths.append(tname)
            cc += 1
        task_names = [str(ii) for ii in task_paths]
        return OPIO({
            'task_names' : task_names,
            'task_paths' : task_paths,
        })

PrepExplorationTaskGroup = PrepLmp

def _mk_task_from_files(cc, ff):
    tname = Path(lmp_task_pattern % cc)
    tname.mkdir(exist_ok=True, parents=True)
    for nn in ff.keys():
        (tname/nn).write_text(ff[nn])
    return tname


        
