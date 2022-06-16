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
    train_index_pattern,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
from dpgen2.utils.step_config import init_executor

import os
from typing import Set, List
from pathlib import Path
from copy import deepcopy

class PrepRunDPTrain(Steps):
    def __init__(
            self,
            name : str,
            prep_train_op : OP,
            run_train_op : OP,
            prep_config : dict = normalize_step_dict({}),
            run_config : dict = normalize_step_dict({}),
            upload_python_package : str = None,
    ):
        self._input_parameters = {
            "block_id" : InputParameter(type=str, value=""),
            "numb_models": InputParameter(type=int),
            "template_script" : InputParameter(),
            "train_config" : InputParameter(),
        }        
        self._input_artifacts = {
            "init_models" : InputArtifact(optional=True),
            "init_data" : InputArtifact(),
            "iter_data" : InputArtifact(),
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "scripts": OutputArtifact(),
            "models": OutputArtifact(),
            "logs": OutputArtifact(),
            "lcurves": OutputArtifact(),
        }

        super().__init__(        
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts,
            ),
            outputs=Outputs(
                artifacts=self._output_artifacts,
            ),
        )
        
        self._keys = ['prep-train', 'run-train']
        self.step_keys = {}
        ii = 'prep-train'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii]
        )
        ii = 'run-train'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _prep_run_dp_train(
            self, 
            self.step_keys,
            prep_train_op,
            run_train_op,
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
    

def _prep_run_dp_train(
        train_steps,
        step_keys,
        prep_train_op : OP,
        run_train_op : OP,
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

    prep_train = Step(
        'prep-train',
        template=PythonOPTemplate(
            prep_train_op,
            output_artifact_archive={
                "task_paths": None
            },
            python_packages = upload_python_package,
            **prep_template_config,
        ),
        parameters={
            "numb_models": train_steps.inputs.parameters['numb_models'],
            "template_script": train_steps.inputs.parameters['template_script'],
        },
        artifacts={
        },
        key = step_keys['prep-train'],
        executor = prep_executor,
        **prep_config,
    )
    train_steps.add(prep_train)

    run_train = Step(
        'run-train',
        template=PythonOPTemplate(
            run_train_op,
            slices = Slices(
                "int('{{item}}')",
                input_parameter = ["task_name"],
                input_artifact = ["task_path", "init_model"],
                output_artifact = ["model", "lcurve", "log", "script"],
            ),
            python_packages = upload_python_package,
            **run_template_config,
        ),
        parameters={
            "config" : train_steps.inputs.parameters["train_config"],
            "task_name" : prep_train.outputs.parameters["task_names"],
        },
        artifacts={
            'task_path' : prep_train.outputs.artifacts['task_paths'],
            "init_model" : train_steps.inputs.artifacts['init_models'],
            "init_data": train_steps.inputs.artifacts['init_data'],
            "iter_data": train_steps.inputs.artifacts['iter_data'],
        },
        with_sequence=argo_sequence(argo_len(prep_train.outputs.parameters["task_names"]), format=train_index_pattern),
        # with_param=argo_range(train_steps.inputs.parameters["numb_models"]),
        key = step_keys['run-train'],
        executor = run_executor,
        **run_config,
    )
    train_steps.add(run_train)

    train_steps.outputs.artifacts["scripts"]._from = run_train.outputs.artifacts["script"]
    train_steps.outputs.artifacts["models"]._from = run_train.outputs.artifacts["model"]
    train_steps.outputs.artifacts["logs"]._from = run_train.outputs.artifacts["log"]
    train_steps.outputs.artifacts["lcurves"]._from = run_train.outputs.artifacts["lcurve"]

    return train_steps


