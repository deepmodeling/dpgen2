import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.constants import (
    train_index_pattern,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class Finetune(Steps):
    def __init__(
        self,
        name: str,
        prep_train_op: OP,
        run_train_op: OP,
        modify_train_script_op: OP,
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value="finetune"),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "run_optional_parameter": InputParameter(
                type=dict, value=run_train_op.default_optional_parameter
            ),
        }
        self._input_artifacts = {
            "init_models": InputArtifact(optional=True),
            "init_data": InputArtifact(),
            "iter_data": InputArtifact(),
        }
        self._output_parameters = {
            "template_script": OutputParameter(),
        }
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
                parameters=self._output_parameters,
                artifacts=self._output_artifacts,
            ),
        )

        self._keys = ["prep-train", "run-train", "modify-train-script"]
        self.step_keys = {}
        ii = "prep-train"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-train"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )
        ii = "modify-train-script"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])

        self = _finetune(
            self,
            self.step_keys,
            prep_train_op,
            run_train_op,
            modify_train_script_op,
            prep_config=prep_config,
            run_config=run_config,
            upload_python_packages=upload_python_packages,
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


def _finetune(
    finetune_steps,
    step_keys,
    prep_train_op: OP,
    run_train_op: OP,
    modify_train_script_op: OP,
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))

    prep_train = Step(
        "prep-train",
        template=PythonOPTemplate(
            prep_train_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "numb_models": finetune_steps.inputs.parameters["numb_models"],
            "template_script": finetune_steps.inputs.parameters["template_script"],
        },
        artifacts={},
        key=step_keys["prep-train"],
        executor=prep_executor,
        **prep_config,
    )
    finetune_steps.add(prep_train)

    run_train = Step(
        "run-train",
        template=PythonOPTemplate(
            run_train_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name"],
                input_artifact=["task_path", "init_model"],
                output_artifact=["model", "lcurve", "log", "script"],
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": finetune_steps.inputs.parameters["train_config"],
            "task_name": prep_train.outputs.parameters["task_names"],
            "optional_parameter": finetune_steps.inputs.parameters[
                "run_optional_parameter"
            ],
        },
        artifacts={
            "task_path": prep_train.outputs.artifacts["task_paths"],
            "init_model": finetune_steps.inputs.artifacts["init_models"],
            "init_data": finetune_steps.inputs.artifacts["init_data"],
            "iter_data": finetune_steps.inputs.artifacts["iter_data"],
        },
        with_sequence=argo_sequence(
            argo_len(prep_train.outputs.parameters["task_names"]),
            format=train_index_pattern,
        ),
        # with_param=argo_range(finetune_steps.inputs.parameters["numb_models"]),
        key=step_keys["run-train"],
        executor=run_executor,
        **run_config,
    )
    finetune_steps.add(run_train)

    # print(modify_train_script_op.get_input_sign())
    modify_train_script = Step(
        "modify-train-script",
        template=PythonOPTemplate(
            modify_train_script_op,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "numb_models": finetune_steps.inputs.parameters["numb_models"],
        },
        artifacts={
            "scripts": run_train.outputs.artifacts["script"],
        },
        key=step_keys["modify-train-script"],
        executor=prep_executor,
        **prep_config,
    )
    finetune_steps.add(modify_train_script)

    finetune_steps.outputs.artifacts["scripts"]._from = run_train.outputs.artifacts[
        "script"
    ]
    finetune_steps.outputs.artifacts["models"]._from = run_train.outputs.artifacts[
        "model"
    ]
    finetune_steps.outputs.artifacts["logs"]._from = run_train.outputs.artifacts["log"]
    finetune_steps.outputs.artifacts["lcurves"]._from = run_train.outputs.artifacts[
        "lcurve"
    ]
    finetune_steps.outputs.parameters[
        "template_script"
    ].value_from_parameter = modify_train_script.outputs.parameters["template_script"]

    return finetune_steps
