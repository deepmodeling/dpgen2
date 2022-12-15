from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
    FatalError,
    BigParameter,
)
import os, json, dpdata
from pathlib import Path
from typing import (
    Tuple, 
    List, 
    Set,
)
from dpgen2.utils.chdir import set_directory


class RunFp(OP):
    r"""Execute a first-principles (FP) task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The FP
    command is exectuted from directory `task_name`. The
    `op["labeled_data"]` in `"deepmd/npy"` format (HF5 in the future)
    provided by `dpdata` will be created.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "config" : BigParameter(dict),
            "task_name": str,
            "task_path" : Artifact(Path),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log": Artifact(Path),
            "labeled_data" : Artifact(Path),
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
        
            - `config`: (`dict`) The config of FP task. Should have `config['run']`, which defines the runtime configuration of the FP task.
            - `task_name`: (`str`) The name of task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepFp`.

        Returns
        -------
            Output dict with components:
        
            - `log`: (`Artifact(Path)`) The log file of FP.
            - `labeled_data`: (`Artifact(Path)`) The path to the labeled data in `"deepmd/npy"` format provided by `dpdata`.
        
        Exceptions
        ----------
        TransientError
            On the failure of FP execution. 
        FatalError
            When mandatory files are not found.
        """
        config = ip['config']['run'] if ip['config']['run'] is not None else {}
        config = type(self).normalize_config(config, strict=False)
        task_name = ip['task_name']
        task_path = ip['task_path']
        input_files = self.input_files()
        input_files = [(Path(task_path)/ii).resolve() for ii in input_files]
        opt_input_files = self.optional_input_files()
        opt_input_files = [(Path(task_path)/ii).resolve() for ii in opt_input_files]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                if not os.path.isfile(ii):
                    raise FatalError(f"cannot file file {ii}")
                iname = ii.name
                Path(iname).symlink_to(ii)
            for ii in opt_input_files:
                if os.path.isfile(ii):
                    iname = ii.name
                    Path(iname).symlink_to(ii)
            out_name, log_name = self.run_task(**config)

        return OPIO({
            "log": work_dir / log_name,
            "labeled_data": work_dir / out_name,
        })


