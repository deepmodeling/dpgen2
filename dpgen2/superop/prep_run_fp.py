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
import os
from typing import Set, List
from pathlib import Path


class PrepRunFp(Steps):
    def __init__(
            self,
            name : str,
            prep_op : OP,
            run_op : OP,
            prep_image : str = "dflow:v1.0",
            run_image : str = "dflow:v1.0",
            upload_python_package : str = None,
    ):
        self._input_parameters = {
            "block_id" : InputParameter(type=str, value=""),
            "fp_config" : InputParameter(),
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
            prep_image = prep_image,
            run_image = run_image,
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
        prep_image : str = "dflow:v1.0",
        run_image : str = "dflow:v1.0",
        upload_python_package : str = None,
):
    prep_fp = Step(
        'prep-fp',
        template=PythonOPTemplate(
            prep_op,
            image=prep_image,
            output_artifact_archive={
                "task_paths": None
            },
            python_packages = upload_python_package,
        ),
        parameters={
        },
        artifacts={
            "inputs": prep_run_steps.inputs.artifacts['inputs'],
            "confs" : prep_run_steps.inputs.artifacts['confs'],
        },
        key = step_keys['prep-fp'],
    )
    prep_run_steps.add(prep_fp)

    run_fp = Step(
        'run-fp',
        template=PythonOPTemplate(
            run_op,
            image=run_image,
            slices = Slices(
                "int('{{item}}')",
                input_parameter = ["task_name"],
                input_artifact = ["task_path"],
                output_artifact = ["log", "labeled_data"],
            ),
            python_packages = upload_python_package,
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
    )
    prep_run_steps.add(run_fp)

    prep_run_steps.outputs.parameters["task_names"].value_from_parameter = prep_fp.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_fp.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["labeled_data"]._from = run_fp.outputs.artifacts["labeled_data"]

    return prep_run_steps


