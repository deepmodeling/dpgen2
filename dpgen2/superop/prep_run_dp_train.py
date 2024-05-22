import json
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
    Type,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    S3Artifact,
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
    BigParameter,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.constants import (
    train_index_pattern,
    train_script_name,
    train_task_pattern,
)
from dpgen2.op import (
    RunDPTrain,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class ModifyTrainScript(OP):
    r"""Modify the training scripts to prepare them for training
    tasks in dpgen step.

    Read the training scripts modified by finetune, and replace
    the original template scripts to be compatible with pre-trained models.
    New templates are returned as `op["template_script"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "numb_models": int,
                "scripts": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "template_script": BigParameter(List[dict]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:

            - `scripts`: (`Artifact(Path)`) Training scripts from finetune.
            - `numb_models`: (`int`) Number of DP models to train.

        Returns
        -------
        op : dict
            Output dict with components:

            - `template_script`: (`List[dict]`) One template from one finetuning task. The length of the list should be the same as `numb_models`.

        """
        scripts = ip["scripts"]
        new_template_script = []
        numb_models = ip["numb_models"]

        for ii in range(numb_models):
            subdir = Path(train_task_pattern % ii)
            train_script = Path(scripts) / subdir / train_script_name
            with open(train_script, "r") as fp:
                train_dict = json.load(fp)
            new_template_script.append(train_dict)

        op = OPIO(
            {
                "template_script": new_template_script,
            }
        )
        return op


class PrepRunDPTrain(Steps):
    def __init__(
        self,
        name: str,
        prep_train_op: Type[OP],
        run_train_op: Type[RunDPTrain],
        modify_train_script_op: Type[ModifyTrainScript] = ModifyTrainScript,
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
        finetune: bool = False,
        valid_data: Optional[S3Artifact] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
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

        self._keys = ["prep-train", "run-train"]
        if finetune:
            self._keys.append("modify-train-script")
        self.step_keys = {}
        ii = "prep-train"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-train"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )
        ii = "modify-train-script"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])

        self = _prep_run_dp_train(
            self,
            self.step_keys,
            prep_train_op,
            run_train_op,
            modify_train_script_op,
            prep_config=prep_config,
            run_config=run_config,
            upload_python_packages=upload_python_packages,
            finetune=finetune,
            valid_data=valid_data,
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
    prep_train_op: Type[OP],
    run_train_op: Type[RunDPTrain],
    modify_train_script_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
    finetune: bool = False,
    valid_data: Optional[S3Artifact] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    prep_train = Step(
        "prep-train",
        template=PythonOPTemplate(
            prep_train_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "numb_models": train_steps.inputs.parameters["numb_models"],
            "template_script": train_steps.inputs.parameters["template_script"],
        },
        artifacts={},
        key=step_keys["prep-train"],
        executor=prep_executor,
        **prep_config,
    )
    train_steps.add(prep_train)

    run_train = Step(
        "run-train",
        template=PythonOPTemplate(
            run_train_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name"],
                input_artifact=["task_path", "init_model"],
                output_artifact=["model", "lcurve", "log", "script"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": train_steps.inputs.parameters["train_config"],
            "task_name": prep_train.outputs.parameters["task_names"],
            "optional_parameter": train_steps.inputs.parameters[
                "run_optional_parameter"
            ],
        },
        artifacts={
            "task_path": prep_train.outputs.artifacts["task_paths"],
            "init_model": train_steps.inputs.artifacts["init_models"],
            "init_data": train_steps.inputs.artifacts["init_data"],
            "iter_data": train_steps.inputs.artifacts["iter_data"],
            "valid_data": valid_data,
        },
        with_sequence=argo_sequence(
            argo_len(prep_train.outputs.parameters["task_names"]),
            format=train_index_pattern,
        ),
        # with_param=argo_range(train_steps.inputs.parameters["numb_models"]),
        key=step_keys["run-train"],
        executor=run_executor,
        **run_config,
    )
    train_steps.add(run_train)

    if finetune:
        modify_train_script = Step(
            "modify-train-script",
            template=PythonOPTemplate(
                modify_train_script_op,
                python_packages=upload_python_packages,
                **prep_template_config,
            ),
            parameters={
                "numb_models": train_steps.inputs.parameters["numb_models"],
            },
            artifacts={
                "scripts": run_train.outputs.artifacts["script"],
            },
            key=step_keys["modify-train-script"],
            executor=prep_executor,
            **prep_config,
        )
        train_steps.add(modify_train_script)
        train_steps.outputs.parameters[
            "template_script"
        ].value_from_parameter = modify_train_script.outputs.parameters[
            "template_script"
        ]
    else:
        train_steps.outputs.parameters[
            "template_script"
        ].value_from_parameter = train_steps.inputs.parameters["template_script"]
    train_steps.outputs.artifacts["scripts"]._from = run_train.outputs.artifacts[
        "script"
    ]
    train_steps.outputs.artifacts["models"]._from = run_train.outputs.artifacts["model"]
    train_steps.outputs.artifacts["logs"]._from = run_train.outputs.artifacts["log"]
    train_steps.outputs.artifacts["lcurves"]._from = run_train.outputs.artifacts[
        "lcurve"
    ]

    return train_steps
