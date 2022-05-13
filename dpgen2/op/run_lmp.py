import os, json
from pathlib import Path
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
    FatalError,
)
from typing import (
    Tuple,
    List,
    Set,
)
from dpgen2.utils.run_command import run_command
from dpgen2.utils import (
    set_directory,
)
from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_traj_name,
    lmp_model_devi_name,
    model_name_pattern,
)
from dargs import (
    dargs, 
    Argument, 
    Variant, 
    ArgumentEncoder,
)


class RunLmp(OP):
    r"""Execute a LAMMPS task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The LAMMPS
    command is exectuted from directory `task_name`. The trajectory
    and the model deviation will be stored in files `op["traj"]` and
    `op["model_devi"]`, respectively.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "config" : dict,
            "task_name": str,
            "task_path": Artifact(Path),
            "models" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "log" : Artifact(Path),
            "traj" : Artifact(Path),
            "model_devi": Artifact(Path),
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
        
            - `config`: (`dict`) The config of lmp task. Check `RunLmp.lmp_args` for definitions.
            - `task_name`: (`str`) The name of the task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepLmp`.
            - `models`: (`Artifact(List[Path])`) The frozen model to estimate the model deviation. The first model with be used to drive molecular dynamics simulation.

        Returns
        -------
            Output dict with components:
        
            - `log`: (`Artifact(Path)`) The log file of LAMMPS.
            - `traj`: (`Artifact(Path)`) The output trajectory.
            - `model_devi`: (`Artifact(Path)`) The model deviation. The order of recorded model deviations should be consistent with the order of frames in `traj`.
        
        Exceptions
        ----------
        TransientError
            On the failure of LAMMPS execution. Handle different failure cases? e.g. loss atoms.
        """
        config = ip['config'] if ip['config'] is not None else {}
        config = RunLmp.normalize_config(config)
        command = config['command']
        task_name = ip['task_name']
        task_path = ip['task_path']
        models = ip['models']
        input_files = [lmp_conf_name, lmp_input_name]
        input_files = [(Path(task_path)/ii).resolve() for ii in input_files]
        model_files = [Path(ii).resolve() for ii in models]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)
            # link models
            for idx,mm in enumerate(model_files):
                mname = model_name_pattern % (idx)
                Path(mname).symlink_to(mm)
            # run lmp
            command = ' '.join([command, '-i', lmp_input_name, '-log', lmp_log_name])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                raise TransientError(
                    'lmp failed\n',
                    'out msg', out, '\n',
                    'err msg', err, '\n'
                )
            
        return OPIO({
            "log" : work_dir / lmp_log_name,
            "traj" : work_dir / lmp_traj_name,
            "model_devi" : work_dir / lmp_model_devi_name,
        })        

            
    @staticmethod
    def lmp_args():
        doc_lmp_cmd = "The command of LAMMPS"
        return [
            Argument("command", str, optional=True, default='lmp', doc=doc_lmp_cmd),
        ]

    @staticmethod
    def normalize_config(data = {}):
        ta = RunLmp.lmp_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data

    
