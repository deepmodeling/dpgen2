from pathlib import Path
from typing import (
    Optional,
    Tuple, 
    List, 
    Set, 
    Dict,
    Union,
)
import numpy as np
import dpdata
from dargs import (
    dargs, 
    Argument, 
    Variant, 
    ArgumentEncoder,
)

from .prep_fp import PrepFp
from .run_fp import RunFp
from .vasp_input import VaspInputs, make_kspacing_kpoints


# global static variables
vasp_conf_name = 'POSCAR'
vasp_input_name = 'INCAR'
vasp_pot_name = 'POTCAR'
vasp_kp_name = 'KPOINTS'


class PrepVasp(PrepFp):
    def prep_task(
            conf_frame: dpdata.System,
            vasp_inputs: VaspInputs,
    ):
        conf_frame.to('vasp/poscar', vasp_conf_name)
        Path(vasp_input_name).write_text(
            vasp_inputs.incar_template
        )
        # fix the case when some element have 0 atom, e.g. H0O2
        tmp_frame = dpdata.System(vasp_conf_name, fmt='vasp/poscar')
        Path(vasp_pot_name).write_text(
            vasp_inputs.make_potcar(tmp_frame['atom_names'])
        )
        Path(vasp_kp_name).write_text(
            vasp_inputs.make_kpoints(conf_frame['cells'][0])
        )

        
class RunVasp(RunFp):
    def input_files() -> List[str]:
        return [vasp_conf_name, vasp_input_name, vasp_pot_name, vasp_kp_name]

    def optional_input_files() -> List[str]:
        return []

    def run_task(
            command : str,
            log_name: str,
            out_name: str,
    ) -> Tuple[str, str]:
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
        return out_name, log_name


    @staticmethod
    def args():
        doc_vasp_cmd = "The command of VASP"
        doc_vasp_log = "The log file name of VASP"
        doc_vasp_out = "The output dir name of labeled data. In `deepmd/npy` format provided by `dpdata`."
        return [
            Argument("command", str, optional=True, default='vasp', doc=doc_vasp_cmd),
            Argument("log", str, optional=True, default=vasp_default_log_name, doc=doc_vasp_log),
            Argument("out", str, optional=True, default=vasp_default_out_data_name, doc=doc_vasp_out),
        ]

    @staticmethod
    def normalize_config(data = {}, strict=True):
        ta = RunVasp.vasp_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=strict)
        return data
