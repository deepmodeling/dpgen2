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
    argo_len,
    argo_sequence,
)
from dflow.python import (
    OP,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.constants import (
    ase_index_pattern,
)
from dpgen2.utils.step_config import (
    init_executor,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class PrepRunAse(Steps):
    """Super-OP that chains :class:`~dpgen2.op.PrepAse` and
    :class:`~dpgen2.op.RunAse` into a single dflow ``Steps`` object.

    The interface mirrors :class:`~dpgen2.superop.PrepRunLmp` so that
    :class:`~dpgen2.superop.ConcurrentLearningBlock` can use it as a
    drop-in replacement for the exploration step.

    Parameters
    ----------
    name : str
        Name of the ``Steps`` object (used as the Argo workflow step name).
    prep_op : Type[OP]
        The :class:`~dpgen2.op.PrepAse` class (or a subclass).
    run_op : Type[OP]
        The :class:`~dpgen2.op.RunAse` class (or a subclass).
    prep_config : dict, optional
        Step configuration for the prep step (executor, resources, …).
    run_config : dict, optional
        Step configuration for the run step (executor, resources, …).
    upload_python_packages : List[os.PathLike], optional
        Extra Python packages to upload to the workflow executor.
    """

    def __init__(
        self,
        name: str,
        prep_op: Type[OP],
        run_op: Type[OP],
        prep_config: Optional[dict] = None,
        run_config: Optional[dict] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        prep_config = normalize_step_dict({}) if prep_config is None else prep_config
        run_config = normalize_step_dict({}) if run_config is None else run_config

        self._input_parameters = {
            "block_id": InputParameter(type=str, value=""),
            "explore_config": InputParameter(),
            "expl_task_grp": InputParameter(),
            "type_map": InputParameter(value=[]),
        }
        self._input_artifacts = {
            "models": InputArtifact(),
        }
        self._output_parameters = {
            "task_names": OutputParameter(),
        }
        self._output_artifacts = {
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

        self._keys = ["prep-ase", "run-ase"]
        self.step_keys = {}
        ii = "prep-ase"
        self.step_keys[ii] = "--".join(["%s" % self.inputs.parameters["block_id"], ii])
        ii = "run-ase"
        self.step_keys[ii] = "--".join(
            ["%s" % self.inputs.parameters["block_id"], ii + "-{{item}}"]
        )

        self = _prep_run_ase(
            self,
            self.step_keys,
            prep_op,
            run_op,
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


def _prep_run_ase(
    prep_run_steps: Steps,
    step_keys: dict,
    prep_op: Type[OP],
    run_op: Type[OP],
    prep_config: dict = normalize_step_dict({}),
    run_config: dict = normalize_step_dict({}),
    upload_python_packages: Optional[List[os.PathLike]] = None,
) -> Steps:
    """Wire up the prep and run steps inside *prep_run_steps*.

    Parameters
    ----------
    prep_run_steps : Steps
        The parent ``Steps`` object to add steps to.
    step_keys : dict
        Mapping ``{"prep-ase": key, "run-ase": key}`` for step caching.
    prep_op, run_op : Type[OP]
        OP classes for the prep and run steps.
    prep_config, run_config : dict
        Normalised step configurations.
    upload_python_packages : List[os.PathLike], optional
        Extra packages to upload.

    Returns
    -------
    Steps
        The modified *prep_run_steps* (same object, returned for chaining).
    """
    prep_config = deepcopy(prep_config)
    run_config = deepcopy(run_config)
    prep_template_config = prep_config.pop("template_config")
    run_template_config = run_config.pop("template_config")
    prep_executor = init_executor(prep_config.pop("executor"))
    run_executor = init_executor(run_config.pop("executor"))
    template_slice_config = run_config.pop("template_slice_config", {})

    prep_ase = Step(
        "prep-ase",
        template=PythonOPTemplate(
            prep_op,
            output_artifact_archive={"task_paths": None},
            python_packages=upload_python_packages,
            **prep_template_config,
        ),
        parameters={
            "ase_task_grp": prep_run_steps.inputs.parameters["expl_task_grp"],
        },
        artifacts={},
        key=step_keys["prep-ase"],
        executor=prep_executor,
        **prep_config,
    )
    prep_run_steps.add(prep_ase)

    run_ase = Step(
        "run-ase",
        template=PythonOPTemplate(
            run_op,
            slices=Slices(
                "int('{{item}}')",
                input_parameter=["task_name"],
                input_artifact=["task_path"],
                output_artifact=["log", "traj", "model_devi"],
                **template_slice_config,
            ),
            python_packages=upload_python_packages,
            **run_template_config,
        ),
        parameters={
            "task_name": prep_ase.outputs.parameters["task_names"],
            "config": prep_run_steps.inputs.parameters["explore_config"],
        },
        artifacts={
            "task_path": prep_ase.outputs.artifacts["task_paths"],
            "models": prep_run_steps.inputs.artifacts["models"],
        },
        with_sequence=argo_sequence(
            argo_len(prep_ase.outputs.parameters["task_names"]),
            format=ase_index_pattern,
        ),
        key=step_keys["run-ase"],
        executor=run_executor,
        **run_config,
    )
    prep_run_steps.add(run_ase)

    prep_run_steps.outputs.parameters[
        "task_names"
    ].value_from_parameter = prep_ase.outputs.parameters["task_names"]
    prep_run_steps.outputs.artifacts["logs"]._from = run_ase.outputs.artifacts["log"]
    prep_run_steps.outputs.artifacts["trajs"]._from = run_ase.outputs.artifacts["traj"]
    prep_run_steps.outputs.artifacts["model_devis"]._from = run_ase.outputs.artifacts[
        "model_devi"
    ]

    return prep_run_steps
