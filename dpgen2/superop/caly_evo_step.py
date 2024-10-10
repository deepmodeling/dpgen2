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
    Step,
    Steps,
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    download_artifact,
    if_expression,
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
    calypso_index_pattern,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class CalyEvoStep(Steps):
    def __init__(
        self,
        name: str,
        collect_run_caly: Type[OP],
        prep_dp_optim: Type[OP],
        run_dp_optim: Type[OP],
        expl_mode: str = "default",
        prep_config: dict = normalize_step_dict({}),
        run_config: dict = normalize_step_dict({}),
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        self.expl_mode = expl_mode
        self._input_parameters = {
            "iter_num": InputParameter(type=int, value=0),
            "cnt_num": InputParameter(type=int, value=0),
            "block_id": InputParameter(type=str, value=""),
            "task_name": InputParameter(type=str),
            "expl_config": InputParameter(),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
            "input_file": InputArtifact(),  # input.dat
            "caly_run_opt_file": InputArtifact(),
            "caly_check_opt_file": InputArtifact(),
            # calypso evo needed
            "results": InputArtifact(optional=True),
            "step": InputArtifact(optional=True),
            "opt_results_dir": InputArtifact(optional=True),
            "qhull_input": InputArtifact(optional=True),
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "traj_results": OutputArtifact(),
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

        self.step_keys = {}
        self._keys = []
        self = _caly_evo_step(
            self,
            self.step_keys,
            collect_run_caly,
            prep_dp_optim,
            run_dp_optim,
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


def _caly_evo_step(
    caly_evo_step_steps,
    step_keys,
    collect_run_calypso_op: Type[OP],
    prep_dp_optim_op: Type[OP],
    run_dp_optim_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
):
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor_config = prep_config.pop("executor")
    run_executor_config = run_config.pop("executor")
    template_slice_config = run_config.pop("template_slice_config", {})
    expl_mode = caly_evo_step_steps.expl_mode
    no_slice_run_config = deepcopy(run_config)
    no_slice_run_config.pop("continue_on_num_success", None)
    no_slice_run_config.pop("continue_on_success_ratio", None)

    def wise_executor(expl_mode, origin_executor_config):
        if expl_mode == "default":
            return init_executor(deepcopy(origin_executor_config))
        elif expl_mode == "merge":
            return None
        else:
            raise NotImplementedError(
                f"Unknown expl_mode {expl_mode}, only support `default` and `merge`."
            )

    # collect the last step files and run calypso.x to generate structures
    collect_run_calypso = Step(
        "collect-run-calypso",
        template=PythonOPTemplate(
            collect_run_calypso_op,
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": caly_evo_step_steps.inputs.parameters["expl_config"],
            "task_name": caly_evo_step_steps.inputs.parameters["task_name"],
            "cnt_num": caly_evo_step_steps.inputs.parameters["cnt_num"],
        },
        artifacts={
            "input_file": caly_evo_step_steps.inputs.artifacts["input_file"],
            "step": caly_evo_step_steps.inputs.artifacts["step"],
            "results": caly_evo_step_steps.inputs.artifacts["results"],
            "opt_results_dir": caly_evo_step_steps.inputs.artifacts["opt_results_dir"],
            "qhull_input": caly_evo_step_steps.inputs.artifacts["qhull_input"],
        },
        key="%s--collect-run-calypso-%s-%s"
        % (
            caly_evo_step_steps.inputs.parameters["block_id"],
            caly_evo_step_steps.inputs.parameters["iter_num"],
            caly_evo_step_steps.inputs.parameters["cnt_num"],
        ),
        executor=wise_executor(expl_mode, prep_executor_config),
        **no_slice_run_config,
    )
    caly_evo_step_steps.add(collect_run_calypso)

    # prep_dp_optim
    prep_dp_optim = Step(
        "prep-dp-optim",
        template=PythonOPTemplate(
            prep_dp_optim_op,
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "task_name": caly_evo_step_steps.inputs.parameters["task_name"],
            "finished": collect_run_calypso.outputs.parameters["finished"],
            "template_slice_config": template_slice_config,
        },
        artifacts={
            "poscar_dir": collect_run_calypso.outputs.artifacts["poscar_dir"],
            "models_dir": caly_evo_step_steps.inputs.artifacts["models"],
            "caly_run_opt_file": caly_evo_step_steps.inputs.artifacts[
                "caly_run_opt_file"
            ],
            "caly_check_opt_file": caly_evo_step_steps.inputs.artifacts[
                "caly_check_opt_file"
            ],
        },
        key="%s--prep-dp-optim-%s-%s"
        % (
            caly_evo_step_steps.inputs.parameters["block_id"],
            caly_evo_step_steps.inputs.parameters["iter_num"],
            caly_evo_step_steps.inputs.parameters["cnt_num"],
        ),
        executor=wise_executor(expl_mode, prep_executor_config),
        **no_slice_run_config,
    )
    caly_evo_step_steps.add(prep_dp_optim)

    # run_dp_optim
    run_dp_optim = Step(
        "run-dp-optim",
        template=PythonOPTemplate(
            run_dp_optim_op,
            slices=Slices(
                input_parameter=["task_name"],
                input_artifact=["task_dir"],
                output_artifact=["traj_results", "optim_results_dir"],
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "config": caly_evo_step_steps.inputs.parameters["expl_config"],
            "task_name": prep_dp_optim.outputs.parameters["task_names"],
            "finished": collect_run_calypso.outputs.parameters["finished"],
            "cnt_num": caly_evo_step_steps.inputs.parameters["cnt_num"],
        },
        artifacts={
            "task_dir": prep_dp_optim.outputs.artifacts["task_dirs"],
        },
        key="%s--run-dp-optim-%s-%s-{{item}}"
        % (
            caly_evo_step_steps.inputs.parameters["block_id"],
            caly_evo_step_steps.inputs.parameters["iter_num"],
            caly_evo_step_steps.inputs.parameters["cnt_num"],
        ),
        executor=wise_executor(expl_mode, run_executor_config),
        **run_config,
    )
    caly_evo_step_steps.add(run_dp_optim)

    name = "calypso-block"
    next_step = Step(
        name=name + "-nextstep",
        template=caly_evo_step_steps,
        parameters={
            "iter_num": caly_evo_step_steps.inputs.parameters["iter_num"],
            "cnt_num": caly_evo_step_steps.inputs.parameters["cnt_num"] + 1,
            "block_id": caly_evo_step_steps.inputs.parameters["block_id"],
            "expl_config": caly_evo_step_steps.inputs.parameters["expl_config"],
            "task_name": collect_run_calypso.outputs.parameters["task_name"],
        },
        artifacts={
            "models": caly_evo_step_steps.inputs.artifacts["models"],
            "input_file": collect_run_calypso.outputs.artifacts[
                "input_file"
            ],  # input.dat
            "results": collect_run_calypso.outputs.artifacts["results"],
            "step": collect_run_calypso.outputs.artifacts["step"],
            "qhull_input": collect_run_calypso.outputs.artifacts["qhull_input"],
            "opt_results_dir": run_dp_optim.outputs.artifacts["optim_results_dir"],
            "caly_run_opt_file": prep_dp_optim.outputs.artifacts["caly_run_opt_file"],
            "caly_check_opt_file": prep_dp_optim.outputs.artifacts[
                "caly_check_opt_file"
            ],
        },
        when="%s == false" % (collect_run_calypso.outputs.parameters["finished"]),
    )
    caly_evo_step_steps.add(next_step)

    caly_evo_step_steps.outputs.artifacts[
        "traj_results"
    ].from_expression = if_expression(
        _if=(collect_run_calypso.outputs.parameters["finished"] == "false"),
        _then=(next_step.outputs.artifacts["traj_results"]),
        _else=(run_dp_optim.outputs.artifacts["traj_results"]),
    )

    return caly_evo_step_steps
