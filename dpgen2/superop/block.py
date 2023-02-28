import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OPTemplate,
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

from dpgen2.op import (
    CollectData,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

block_default_optional_parameter = {
    "data_mixed_type": False,
}


def make_collect_data_optional_parameter(block_optional_parameter):
    return {
        "mixed_type": block_optional_parameter["data_mixed_type"],
    }


def make_run_dp_train_optional_parameter(block_optional_parameter):
    return {
        "mixed_type": block_optional_parameter["data_mixed_type"],
    }


class ConcurrentLearningBlock(Steps):
    def __init__(
        self,
        name: str,
        prep_run_dp_train_op: OPTemplate,
        prep_run_lmp_op: OPTemplate,
        select_confs_op: OP,
        prep_run_fp_op: OPTemplate,
        collect_data_op: OP,
        select_confs_config: dict = normalize_step_dict({}),
        collect_data_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
            "type_map": InputParameter(),
            "numb_models": InputParameter(type=int),
            "template_script": InputParameter(),
            "train_config": InputParameter(),
            "lmp_config": InputParameter(),
            "conf_selector": InputParameter(),
            "fp_config": InputParameter(),
            "lmp_task_grp": InputParameter(),
            "optional_parameter": InputParameter(
                type=dict, value=block_default_optional_parameter
            ),
        }
        self._input_artifacts = {
            "init_models": InputArtifact(optional=True),
            "init_data": InputArtifact(),
            "iter_data": InputArtifact(),
        }
        self._output_parameters = {
            "exploration_report": OutputParameter(),
        }
        self._output_artifacts = {
            "models": OutputArtifact(),
            "iter_data": OutputArtifact(),
            "trajs": OutputArtifact(),
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

        self._my_keys = ["select-confs", "collect-data"]
        self._keys = (
            prep_run_dp_train_op.keys
            + prep_run_lmp_op.keys
            + self._my_keys[:1]
            + prep_run_fp_op.keys
            + self._my_keys[1:2]
        )
        self.step_keys = {}
        for ii in self._my_keys:
            self.step_keys[ii] = "--".join(
                ["%s" % self.inputs.parameters["block_id"], ii]
            )

        self = _block_cl(
            self,
            self.step_keys,
            name,
            prep_run_dp_train_op,
            prep_run_lmp_op,
            select_confs_op,
            prep_run_fp_op,
            collect_data_op,
            select_confs_config=select_confs_config,
            collect_data_config=collect_data_config,
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


def _block_cl(
    block_steps: Steps,
    step_keys: Dict[str, Any],
    name: str,
    prep_run_dp_train_op: OPTemplate,
    prep_run_lmp_op: OPTemplate,
    select_confs_op: OP,
    prep_run_fp_op: OPTemplate,
    collect_data_op: OP,
    select_confs_config: dict = normalize_step_dict({}),
    collect_data_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    select_confs_config = deepcopy(select_confs_config)
    collect_data_config = deepcopy(collect_data_config)
    select_confs_template_config = select_confs_config.pop("template_config")
    collect_data_template_config = collect_data_config.pop("template_config")
    select_confs_executor = init_executor(select_confs_config.pop("executor"))
    collect_data_executor = init_executor(collect_data_config.pop("executor"))
    run_dp_train_optional_parameter = make_run_dp_train_optional_parameter(
        block_steps.inputs.parameters["optional_parameter"]
    )
    collect_data_optional_parameter = make_collect_data_optional_parameter(
        block_steps.inputs.parameters["optional_parameter"]
    )

    prep_run_dp_train = Step(
        name + "-prep-run-dp-train",
        template=prep_run_dp_train_op,
        parameters={
            "block_id": block_steps.inputs.parameters["block_id"],
            "train_config": block_steps.inputs.parameters["train_config"],
            "numb_models": block_steps.inputs.parameters["numb_models"],
            "template_script": block_steps.inputs.parameters["template_script"],
            "run_optional_parameter": run_dp_train_optional_parameter,
        },
        artifacts={
            "init_models": block_steps.inputs.artifacts["init_models"],
            "init_data": block_steps.inputs.artifacts["init_data"],
            "iter_data": block_steps.inputs.artifacts["iter_data"],
        },
        key="--".join(
            ["%s" % block_steps.inputs.parameters["block_id"], "prep-run-train"]
        ),
    )
    block_steps.add(prep_run_dp_train)

    prep_run_lmp = Step(
        name=name + "-prep-run-lmp",
        template=prep_run_lmp_op,
        parameters={
            "block_id": block_steps.inputs.parameters["block_id"],
            "lmp_config": block_steps.inputs.parameters["lmp_config"],
            "lmp_task_grp": block_steps.inputs.parameters["lmp_task_grp"],
        },
        artifacts={
            "models": prep_run_dp_train.outputs.artifacts["models"],
        },
        key="--".join(
            ["%s" % block_steps.inputs.parameters["block_id"], "prep-run-lmp"]
        ),
    )
    block_steps.add(prep_run_lmp)

    select_confs = Step(
        name=name + "-select-confs",
        template=PythonOPTemplate(
            select_confs_op,
            output_artifact_archive={"confs": None},
            python_packages=upload_python_packages,
            **select_confs_template_config,
        ),
        parameters={
            "conf_selector": block_steps.inputs.parameters["conf_selector"],
            "type_map": block_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "trajs": prep_run_lmp.outputs.artifacts["trajs"],
            "model_devis": prep_run_lmp.outputs.artifacts["model_devis"],
        },
        key=step_keys["select-confs"],
        executor=select_confs_executor,
        **select_confs_config,
    )
    block_steps.add(select_confs)

    prep_run_fp = Step(
        name=name + "-prep-run-fp",
        template=prep_run_fp_op,
        parameters={
            "block_id": block_steps.inputs.parameters["block_id"],
            "fp_config": block_steps.inputs.parameters["fp_config"],
            "type_map": block_steps.inputs.parameters["type_map"],
        },
        artifacts={
            "confs": select_confs.outputs.artifacts["confs"],
        },
        key="--".join(
            ["%s" % block_steps.inputs.parameters["block_id"], "prep-run-fp"]
        ),
    )
    block_steps.add(prep_run_fp)

    collect_data = Step(
        name=name + "-collect-data",
        template=PythonOPTemplate(
            collect_data_op,
            output_artifact_archive={"iter_data": None},
            python_packages=upload_python_packages,
            **collect_data_template_config,
        ),
        parameters={
            "name": block_steps.inputs.parameters["block_id"],
            "type_map": block_steps.inputs.parameters["type_map"],
            "optional_parameter": collect_data_optional_parameter,
        },
        artifacts={
            "iter_data": block_steps.inputs.artifacts["iter_data"],
            "labeled_data": prep_run_fp.outputs.artifacts["labeled_data"],
        },
        key=step_keys["collect-data"],
        executor=collect_data_executor,
        **collect_data_config,
    )
    block_steps.add(collect_data)

    block_steps.outputs.parameters[
        "exploration_report"
    ].value_from_parameter = select_confs.outputs.parameters["report"]
    block_steps.outputs.artifacts["models"]._from = prep_run_dp_train.outputs.artifacts[
        "models"
    ]
    block_steps.outputs.artifacts["iter_data"]._from = collect_data.outputs.artifacts[
        "iter_data"
    ]
    block_steps.outputs.artifacts["trajs"]._from = prep_run_lmp.outputs.artifacts[
        "trajs"
    ]

    return block_steps
