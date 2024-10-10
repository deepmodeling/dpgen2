import json
import logging
import pickle
import shutil
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow import (
    Step,
    Workflow,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter,
    Slices,
    TransientError,
)
from dflow.utils import (
    flatten,
)

from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_run_opt_file,
)
from dpgen2.exploration.task import (
    ExplorationTaskGroup,
)
from dpgen2.superop.caly_evo_step import (
    CalyEvoStep,
)
from dpgen2.utils import (
    BinaryFileInput,
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


class CalyEvoStepMerge(OP):
    def __init__(self, mode="debug", *args, **kwargs):
        self.mode = mode
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "iter_num": int,
                "cnt_num": Parameter(int, default=0),
                "block_id": Parameter(str, default=""),
                "task_name": BigParameter(str),
                "expl_config": BigParameter(dict),
                "models": Artifact(Path),
                "input_file": Artifact(Path),
                "caly_run_opt_file": Artifact(Path),
                "caly_check_opt_file": Artifact(Path),
                "results": Artifact(Path, optional=True),
                "step": Artifact(Path, optional=True),
                "opt_results_dir": Artifact(List[Path], optional=True),
                "qhull_input": Artifact(Path, optional=True),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "traj_results": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        from dflow import (
            config,
        )

        config["mode"] = self.mode
        wf = Workflow("caly-evo-workflow")
        steps = CalyEvoStep(*self.args, **self.kwargs)
        step = Step(
            "caly-evo-step",
            template=steps,
            slices=Slices(output_artifact=["traj_results"]),
            parameters={k: ip[k] for k in steps.inputs.parameters},
            artifacts={
                k: upload_artifact(ip[k]) if ip[k] is not None else None
                for k in steps.inputs.artifacts
            },
            with_param=[0],
        )
        wf.add(step)
        wf.submit()
        wf.wait()
        assert wf.query_status() == "Succeeded"
        out = OPIO()
        step = wf.query_step("caly-evo-step")[0]
        for k in step.outputs.parameters:
            out[k] = step.outputs.parameters[k].value
        output_sign = self.get_output_sign()
        for k in step.outputs.artifacts:
            path_list = download_artifact(step.outputs.artifacts[k])
            if output_sign[k].type == List[Path]:
                if not isinstance(path_list, list) or any(
                    [p is not None and not isinstance(p, str) for p in path_list]
                ):
                    path_list = list(flatten(path_list).values())
                out[k] = [Path(p) for p in path_list]
            elif output_sign[k].type == Path:
                assert len(path_list) == 1
                out[k] = Path(path_list[0])
        return out
