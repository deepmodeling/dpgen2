"""Prep and Run Gaussian tasks."""
from dflow.python import (
    TransientError,
)
from typing import (
    Tuple,
    List,
    Any,
)
import dpdata
from dargs import (
    dargs, 
    Argument,
)

from .prep_fp import PrepFp
from .run_fp import RunFp
from dpgen2.constants import (
    fp_default_out_data_name,
)
from dpgen2.utils.run_command import run_command
from pathlib import Path
import os

# global static variables
deepmd_input_path = 'one_frame_input'

# global static variables
log_output_name = 'deepmd.log'

# global static variables
deepmd_teacher_model = 'teacher_model.pb'


class DeepmdInputs:
    @staticmethod
    def args() -> List[Argument]:
        return []

    def __init__(self, **kwargs: Any):
        self.data = kwargs


class PrepDeepmd(PrepFp):
    def prep_task(
            self,
            conf_frame: dpdata.System,
            inputs,
    ):
        r"""Define how one Deepmd task is prepared.

        Parameters
        ----------
        conf_frame : dpdata.System
            One frame of configuration in the dpdata format.
        inputs: str or dict
            This parameter is useless in deepmd.
        """
        conf_frame.to('deepmd/npy', deepmd_input_path)

        
class RunDeepmd(RunFp):
    def input_files(self) -> List[str]:
        r"""The mandatory input files to run a Deepmd task.
        
        Returns
        -------
        files: List[str]
            A list of madatory input files names.

        """
        return [deepmd_input_path]

    def optional_input_files(self) -> List[str]:
        r"""The optional input files to run a Deepmd task.
        
        Returns
        -------
        files: List[str]
            A list of optional input files names.

        """
        return []

    def run_task(
            self,
            teacher_model: str,
            type_map: List[str] = None
    ) -> Tuple[str, str]:
        r"""Defines how one FP task runs
        
        Parameters
        ----------
        command: str
            The command of running Deepmd task
        out: str
            The name of the output data file.

        Returns
        -------
        out_name: str
            The file name of the output data in the dpdata.LabeledSystem format.
        log_name: str
            The file name of the log.
        """
        # run Deepmd
        self._dp_infer(teacher_model, type_map)

        ret, out, err = run_command(f'echo "job finished!" > {log_output_name}', shell=True)
        if ret != 0:
            raise TransientError(
                'vasp failed\n',
                'out msg', out, '\n',
                'err msg', err, '\n'
            )     

        return deepmd_input_path, log_output_name
    
    def _dp_infer(self, teacher_model: str, type_map):
        model_path = Path('teacher_model.pb')
        model_path.write_text(teacher_model)

        from deepmd.infer import DeepPot
        import numpy as np

        if type_map is None:
            assert dp.model_type == "ener", 'type_map should be define or model type should be "ener"'
            type_map = dp.get_type_map()
        elif dp.model_type == "ener":
            assert type_map == dp.get_type_map(), \
                f'type_map({type_map}) and deepmd model type_map{dp.get_type_map()} are not the same!'

        dp = DeepPot(model_path)

        ss = dpdata.System()
        ss = ss.from_deepmd_npy(deepmd_input_path, type_map=type_map)

        coord_npy_path_list = list(Path(deepmd_input_path).glob('*/coord.npy'))
        assert len(coord_npy_path_list) == 1, coord_npy_path_list
        coord_npy_path = coord_npy_path_list[0]
        energy_npy_path = coord_npy_path.parent / 'energy.npy'
        force_npy_path = coord_npy_path.parent / 'force.npy'
        virial_npy_path = coord_npy_path.parent / 'virial.npy'
        
        nframe = ss.get_nframes()
        coord = ss['coords']
        cell = ss['cells'].reshape([nframe, -1])
        atype = ss['atom_types'].tolist()
        
        e, f, v = dp.eval(coord, cell, atype)
        
        with open(energy_npy_path, 'wb') as f:
            np.save(f, e)
        with open(force_npy_path, 'wb') as f:
            np.save(f, f)
        with open(virial_npy_path, 'wb') as f:
            np.save(f, v)
        
        os.remove(str(model_path))


    @staticmethod
    def args() -> List[dargs.Argument]:
        r"""The argument definition of the `run_task` method.

        Returns
        -------
        arguments: List[dargs.Argument]
            List of dargs.Argument defines the arguments of `run_task` method.
        """

        doc_deepmd_teacher_model = "The path of teacher model, which can be loaded by deepmd.infer.DeepPot"
        doc_deepmd_type_map = "The type map of teacher model. It can be set automatically when the type of teacher model is \"ener\", otherwise it should be provided by the user."
        return [
            Argument("teacher_model", str, optional=False, doc=doc_deepmd_teacher_model),
            Argument("type_map", str, optional=True, default=None, doc=doc_deepmd_type_map),
        ]
