import os
from copy import (
    deepcopy,
)
from typing import (
    List,
    Optional,
    Type,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    Outputs,
    Step,
    Steps,
    argo_sequence,
)
from dflow.python import (
    OP,
    PythonOPTemplate,
    Slices,
)

from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class PrepRunDiffCSP(Steps):
    def __init__(
        self,
        name: str,
        diffcsp_gen_op: Type[OP],
        prep_relax_op: Type[OP],
        run_relax_op: Type[OP],
        prep_config: Optional[dict] = None,
        run_config: Optional[dict] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        prep_config = normalize_step_dict({}) if prep_config is None else prep_config
        run_config = normalize_step_dict({}) if run_config is None else run_config
        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "expl_task_grp": InputParameter(),
            "explore_config": InputParameter(),
            "type_map": InputParameter(),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {}
        self._output_artifacts = {
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
        self._keys = ["diffcsp-gen", "prep-relax", "run-relax"]

        self = _prep_run_diffcsp(
            self,
            diffcsp_gen_op,
            prep_relax_op,
            run_relax_op,
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


def _prep_run_diffcsp(
    prep_run_diffcsp_steps: Steps,
    diffcsp_gen_op: Type[OP],
    prep_relax_op: Type[OP],
    run_relax_op: Type[OP],
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
    template_slice_config = run_config.pop("template_slice_config", {})

    block_id = prep_run_diffcsp_steps.inputs.parameters["block_id"]
    expl_task_grp = prep_run_diffcsp_steps.inputs.parameters["expl_task_grp"]
    expl_config = prep_run_diffcsp_steps.inputs.parameters["explore_config"]
    prep_run_diffcsp_steps.inputs.parameters["type_map"]
    models = prep_run_diffcsp_steps.inputs.artifacts["models"]

    diffcsp_gen = Step(
        "diffcsp-gen",
        template=PythonOPTemplate(
            diffcsp_gen_op,
            python_packages=upload_python_packages,
            slices=Slices(
                "int('{{item}}')",
                output_artifact=["cifs"],
                **template_slice_config,
            ),
            **prep_template_config,
        ),
        parameters={
            "task_id": "{{item}}",
            "config": expl_config,
        },
        key="%s--diffcsp-gen-{{item}}" % block_id,
        executor=prep_executor,
        with_sequence=argo_sequence(expl_config["gen_tasks"], format="%06d"),  # type: ignore
    )
    prep_run_diffcsp_steps.add(diffcsp_gen)

    prep_relax = Step(
        "prep-relax",
        template=PythonOPTemplate(
            prep_relax_op,
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "expl_config": expl_config,
        },
        artifacts={
            "cifs": diffcsp_gen.outputs.artifacts["cifs"],
        },
        key="%s--prep-relax" % block_id,
        executor=prep_executor,
    )
    prep_run_diffcsp_steps.add(prep_relax)

    run_relax = Step(
        "run-relax",
        template=PythonOPTemplate(
            run_relax_op,
            python_packages=upload_python_packages,
            slices=Slices(
                "int('{{item}}')",
                input_artifact=["task_path"],
                output_artifact=["trajs", "model_devis"],
                **template_slice_config,
            ),
            **run_template_config,
        ),
        parameters={
            "diffcsp_task_grp": expl_task_grp,
            "expl_config": expl_config,
        },
        artifacts={
            "models": models,
            "task_path": prep_relax.outputs.artifacts["task_paths"],
        },
        key="%s--run-relax-{{item}}" % block_id,
        executor=run_executor,
        with_sequence=argo_sequence(
            prep_relax.outputs.parameters["ntasks"], format="%06d"
        ),
    )
    prep_run_diffcsp_steps.add(run_relax)

    prep_run_diffcsp_steps.outputs.artifacts[
        "trajs"
    ]._from = run_relax.outputs.artifacts["trajs"]
    prep_run_diffcsp_steps.outputs.artifacts[
        "model_devis"
    ]._from = run_relax.outputs.artifacts["model_devis"]
    return prep_run_diffcsp_steps
