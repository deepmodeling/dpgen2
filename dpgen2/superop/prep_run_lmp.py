from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    argo_range,
    argo_len,
    argo_sequence,
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    Slices,
)
from dpgen2.constants import (
    lmp_index_pattern,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
from dpgen2.utils.step_config import init_executor

import os
from typing import Set, List
from pathlib import Path
from copy import deepcopy

class PrepRunLmp(Steps):
    def __init__(
            self,
            name : str,
            prep_op : OP,
            run_op : OP,
            prep_config : dict = normalize_step_dict({}),
            run_config : dict = normalize_step_dict({}),
            upload_python_package : str = None,
    ):
        self._input_parameters = {
            "block_id" : InputParameter(type=str, value=""),
            "lmp_config" : InputParameter()
        }
        self._input_artifacts = {
            "lmp_task_grp": InputArtifact(),
            "models" : InputArtifact()
        }
        self._output_parameters={
            "task_names": OutputParameter(),
        }
        self._output_artifacts={
            "logs": OutputArtifact(),
            "trajs": OutputArtifact(),
            "model_devis": OutputArtifact(),
        }        

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )
        
        self._keys = ['prep-lmp', 'run-lmp']
        self.step_keys = {}
        ii = 'prep-lmp'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii]
        )
        ii = 'run-lmp'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _prep_run_lmp(
            self, 
            self.step_keys,
            prep_op,
            run_op,
            prep_config = prep_config,
            run_config = run_config,
            upload_python_package = upload_python_package,
        )            
        
    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys


def _prep_run_lmp(
        prep_run_steps,
        step_keys,
        prep_op : OP,
        run_op : OP,
        prep_config : dict = normalize_step_dict({}),
        run_config : dict = normalize_step_dict({}),
        upload_python_package : str = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop('template_config')
    run_template_config = run_config.pop('template_config')
    prep_executor = init_executor(prep_config.pop('executor'))
    run_executor = init_executor(run_config.pop('executor'))

    prep_lmp = Step(
        'prep-lmp',
        template=PythonOPTemplate(
            prep_op,
            output_artifact_archive={
                "task_paths": None
            },
            python_packages = upload_python_package,
            **prep_template_config,
        ),
        parameters={
        },
        artifacts={
            "lmp_task_grp": prep_run_steps.inputs.artifacts['lmp_task_grp'],
        },
        key = step_keys['prep-lmp'],
        executor = prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_lmp)

    run_lmp = Step(
        'run-lmp',
        template=PythonOPTemplate(
            run_op,
            slices = Slices(
                "int('{{item}}')",
                input_parameter = ["task_name"],
                input_artifact = ["task_path"],
                output_artifact = ["log", "traj", "model_devi"],
            ),
            python_packages = upload_python_package,
            **run_template_config,
        ),
        parameters={
            "task_name" : prep_lmp.outputs.parameters["task_names"],
            "config" : prep_run_steps.inputs.parameters["lmp_config"],
        },
        artifacts={
            'task_path' : prep_lmp.outputs.artifacts['task_paths'],
            "models" : prep_run_steps.inputs.artifacts['models'],
        },
        with_sequence=argo_sequence(argo_len(prep_lmp.outputs.parameters["task_names"]), format=lmp_index_pattern),
        # with_param=argo_range(argo_len(prep_lmp.outputs.parameters["task_names"])),
        key = step_keys['run-lmp'],
        executor = run_executor,
        **run_config,
    )
    prep_run_steps.add(run_lmp)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_lmp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_lmp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["trajs"]._from = run_lmp.outputs.artifacts["traj"]
    prep_run_steps.outputs.artifacts["model_devis"]._from = run_lmp.outputs.artifacts["model_devi"]

    return prep_run_steps


