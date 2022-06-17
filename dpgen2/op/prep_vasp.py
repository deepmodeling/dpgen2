import dpdata
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact
)
import os, json
from typing import (
    Tuple, 
    List, 
    Set, 
    Dict,
    Union,
)
from pathlib import Path
from dpgen2.fp.vasp import VaspInputs
from dpgen2.utils import (
    set_directory,
    load_object_from_file,
    dump_object_to_file,
)
from dpgen2.constants import (
    vasp_task_pattern,
    vasp_conf_name,
    vasp_input_name,
    vasp_pot_name,
    vasp_kp_name,
)

class PrepVasp(OP):
    r"""Prepares the working directories for VASP tasks.

    A list of (same length as ip["confs"]) working directories
    containing all files needed to start VASP tasks will be
    created. The paths of the directories will be returned as
    `op["task_paths"]`. The identities of the tasks are returned as
    `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "type_map": List[str],
            "inputs": Artifact(Path),
            "confs" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "task_names": List[str],
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

            - `inputs` : (`VaspInputs`) Definitions for the VASP inputs
            - `confs` : (`Artifact(List[Path])`) Configurations for the VASP tasks. Stored in folders as deepmd/npy format. Can be parsed as dpdata.MultiSystems. 
        
        Returns
        -------
        op : dict 
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. Contains all input files needed to start the VASP. The order fo the Paths should be consistent with `op["task_names"]`
        """

        inputs_fname = ip['inputs']        
        inputs = load_object_from_file(inputs_fname)
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
            vasp_inputs : VaspInputs,
            conf_frame : dpdata.System,
    ) -> str:
        task_name = vasp_task_pattern % idx
        task_path = Path(task_name)
        with set_directory(task_path):
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
        return task_name, task_path
