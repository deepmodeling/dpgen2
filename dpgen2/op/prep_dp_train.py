from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)

import json
from typing import Tuple, List
from pathlib import Path

class PrepDPTrain(OP):
    r"""Prepares the working directories for DP training tasks.

    A list of (`numb_models`) working directories containing all files
    needed to start training tasks will be created. The paths of the
    directories will be returned as `op["task_paths"]`. The identities
    of the tasks are returned as `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "template_script" : dict,
            "numb_models" : int,
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names" : List[str],
            "task_paths" : Artifact(List[Path]),
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

            - `template_script`: (`str`) A template of the training script.
            - `numb_models`: (`int`) Number of DP models to train.
        
        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. The order fo the Paths should be consistent with `op["task_names"]`

        """
        raise NotImplementedError
    
