from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
    FatalError,
)
import os, json, dpdata
from pathlib import Path
from typing import (
    Tuple, 
    List, 
    Set,
)
from dpgen2.utils.run_command import run_command
from dpgen2.utils.chdir import set_directory
from dpgen2.constants import(
    vasp_conf_name,
    vasp_input_name,
    vasp_pot_name,
    vasp_kp_name,
    vasp_default_log_name,
    vasp_default_out_data_name,
)
from dargs import (
    dargs, 
    Argument, 
    Variant, 
    ArgumentEncoder,
)

class RunVasp(OP):
    r"""Execute a VASP task.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The VASP
    command is exectuted from directory `task_name`. The
    `op["labeled_data"]` in `"deepmd/npy"` format (HF5 in the future)
    provided by `dpdata` will be created.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "config" : dict,
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
        
            - `config`: (`dict`) The config of vasp task. Check `RunVasp.vasp_args` for definitions.
            - `task_name`: (`str`) The name of task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepVasp`.

        Returns
        -------
            Output dict with components:
        
            - `log`: (`Artifact(Path)`) The log file of VASP.
            - `labeled_data`: (`Artifact(Path)`) The path to the labeled data in `"deepmd/npy"` format provided by `dpdata`.
        
        Exceptions
        ----------
        TransientError
            On the failure of VASP execution. 
        """
        config = ip['config'] if ip['config'] is not None else {}
        config = RunVasp.normalize_config(config)
        command = config['command']
        log_name = config['log']
        out_name = config['out']
        task_name = ip['task_name']
        task_path = ip['task_path']
        input_files = [vasp_conf_name, vasp_input_name, vasp_pot_name, vasp_kp_name]
        input_files = [(Path(task_path)/ii).resolve() for ii in input_files]
        work_dir = Path(task_name)

        with set_directory(work_dir):
            # link input files
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)
            # run vasp
            command = ' '.join([command, '>', log_name])
            ret, out, err = run_command(command, shell=True)
            if ret != 0:
                raise TransientError(
                    'vasp failed\n',
                    'out msg', out, '\n',
                    'err msg', err, '\n'
                )                    
            # convert the output to deepmd/npy format
            sys = dpdata.LabeledSystem('OUTCAR')
            sys.to('deepmd/npy', out_name)

        return OPIO({
            "log": work_dir / log_name,
            "labeled_data": work_dir / out_name,
        })


    @staticmethod
    def vasp_args():
        doc_vasp_cmd = "The command of VASP"
        doc_vasp_log = "The log file name of VASP"
        doc_vasp_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument("command", str, optional=True, default='vasp', doc=doc_vasp_cmd),
            Argument("log", str, optional=True, default=vasp_default_log_name, doc=doc_vasp_log),
            Argument("out", str, optional=True, default=vasp_default_out_data_name, doc=doc_vasp_out),
        ]

    @staticmethod
    def normalize_config(data = {}):
        ta = RunVasp.vasp_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)
        return data

    
