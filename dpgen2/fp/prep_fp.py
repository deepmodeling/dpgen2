from abc import ABC, abstractmethod
import dpdata
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    BigParameter,
)
import os, json
from typing import (
    Tuple, 
    List, 
    Set, 
    Dict,
    Union,
    Any,
)
from pathlib import Path
from dpgen2.utils import (
    set_directory,
)
from dpgen2.constants import (
    fp_task_pattern,
)

class PrepFp(OP, ABC):
    r"""Prepares the working directories for first-principles (FP) tasks.

    A list of (same length as ip["confs"]) working directories
    containing all files needed to start FP tasks will be
    created. The paths of the directories will be returned as
    `op["task_paths"]`. The identities of the tasks are returned as
    `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "config" : BigParameter(dict),
            "type_map": List[str],
            "confs" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names": List[str],
            "task_paths" : Artifact(List[Path]),
        })

    @abstractmethod
    def prep_task(
            self,
            conf_frame: dpdata.System,
            inputs: Any,
    ):
        r"""Define how one FP task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs: Any
            The class object handels all other input files of the task. 
            For example, pseudopotential file, k-point file and so on.
        """
        pass

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

            - `config` : (`dict`) Should have `config['inputs']`, which defines the input files of the FP task.
            - `confs` : (`Artifact(List[Path])`) Configurations for the FP tasks. Stored in folders as deepmd/npy format. Can be parsed as dpdata.MultiSystems. 
        
        Returns
        -------
        op : dict 
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the FP. The order fo the Paths should be consistent with `op["task_names"]`
        """

        inputs = ip['config']['inputs']
        confs = ip['confs']
        type_map = ip['type_map']

        task_names = []
        task_paths = []
        counter=0
        # loop over list of MultiSystems
        for mm in confs:
            ms = dpdata.MultiSystems(type_map=type_map)
            ms.from_deepmd_npy(mm, labeled=False)
            # loop over Systems in MultiSystems
            for ii in range(len(ms)):
                ss = ms[ii]
                # loop over frames
                for ff in range(ss.get_nframes()):
                    nn, pp = self._exec_one_frame(counter, inputs, ss[ff])
                    task_names.append(nn)
                    task_paths.append(pp)
                    counter += 1
        return OPIO({
            'task_names' : task_names,
            'task_paths' : task_paths,
        })


    def _exec_one_frame(
            self,
            idx,
            inputs,
            conf_frame : dpdata.System,
    ) -> Tuple[str, Path]:
        task_name = fp_task_pattern % idx
        task_path = Path(task_name)
        with set_directory(task_path):
            self.prep_task(conf_frame, inputs)
        return task_name, task_path
