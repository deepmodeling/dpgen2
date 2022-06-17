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
    vasp_index_pattern,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
from dpgen2.utils.step_config import init_executor

import os
from typing import Set, List
from pathlib import Path
from copy import deepcopy

class PrepRunFp(Steps):
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
            "fp_config" : InputParameter(),
            "type_map" : InputParameter(),
        }
        self._input_artifacts = {
            "inputs": InputArtifact(),
            "confs" : InputArtifact()
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
            "logs": OutputArtifact(),
            "labeled_data": OutputArtifact(),
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
        
        self._keys = ['prep-fp', 'run-fp']
        self.step_keys = {}
        ii = 'prep-fp'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii]
        )
        ii = 'run-fp'
        self.step_keys[ii] = '--'.join(
            ["%s"%self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _prep_run_fp(
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



def _prep_run_fp(
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

    prep_fp = Step(
        'prep-fp',
        template=PythonOPTemplate(
            prep_op,
            output_artifact_archive={
                "task_paths": None
            },
            python_packages = upload_python_package,
            **prep_template_config,
        ),
        parameters={
            "type_map" : prep_run_steps.inputs.parameters['type_map'],
        },
        artifacts={
            "inputs": prep_run_steps.inputs.artifacts['inputs'],
            "confs" : prep_run_steps.inputs.artifacts['confs'],
        },
        key = step_keys['prep-fp'],
        executor = prep_executor,
        **prep_config,        
    )
    prep_run_steps.add(prep_fp)

    run_fp = Step(
        'run-fp',
        template=PythonOPTemplate(
            run_op,
            slices = Slices(
                "int('{{item}}')",
                input_parameter = ["task_name"],
                input_artifact = ["task_path"],
                output_artifact = ["log", "labeled_data"],
            ),
            python_packages = upload_python_package,
            **run_template_config,
        ),
        parameters={
            "task_name" : prep_fp.outputs.parameters["task_names"],
            "config" : prep_run_steps.inputs.parameters["fp_config"],
        },
        artifacts={
            'task_path' : prep_fp.outputs.artifacts['task_paths'],
        },
        with_sequence=argo_sequence(argo_len(prep_fp.outputs.parameters["task_names"]), format=vasp_index_pattern),
        # with_param=argo_range(argo_len(prep_fp.outputs.parameters["task_names"])),
        key = step_keys['run-fp'],
        executor = run_executor,
        **run_config,
    )
    prep_run_steps.add(run_fp)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_fp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_fp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["labeled_data"]._from = run_fp.outputs.artifacts["labeled_data"]

    return prep_run_steps


