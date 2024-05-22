import json
import os
import pickle
import shutil
import time
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
)

import jsonpickle
import numpy as np
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
    argo_range,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
)

from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_index_pattern,
    calypso_input_file,
    calypso_run_opt_file,
)

try:
    from context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from context import (
    default_host,
    default_image,
    skip_ut_with_dflow,
    skip_ut_with_dflow_reason,
    upload_python_packages,
)
from mocked_ops import (
    MockedCollRunCaly,
    MockedRunCalyDPOptim,
    MockedRunCalyModelDevi,
    mocked_numb_models,
)

from dpgen2.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTask,
)
from dpgen2.op.prep_caly_dp_optim import (
    PrepCalyDPOptim,
)
from dpgen2.op.prep_caly_input import (
    PrepCalyInput,
)
from dpgen2.op.prep_caly_model_devi import (
    PrepCalyModelDevi,
)
from dpgen2.op.run_caly_model_devi import (
    RunCalyModelDevi,
)
from dpgen2.superop.caly_evo_step import (
    CalyEvoStep,
)
from dpgen2.superop.prep_run_calypso import (
    PrepRunCaly,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

prep_default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        },
    }
)
run_default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        },
        "template_slice_config": {
            "group_size": 2,
            "pool_size": 1,
            "model_devi_group_size": 30,
        },
    }
)


def make_task_group_list(njobs):
    tgrp = BaseExplorationTaskGroup()
    for ii in range(njobs):
        tt = ExplorationTask()
        tt.add_file(calypso_input_file, "3")
        tt.add_file(calypso_run_opt_file, f"run_{ii}")
        tt.add_file(calypso_check_opt_file, f"check_{ii}")
        tgrp.add_task(tt)
    return tgrp


# @unittest.skip("temporary pass")
@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestPrepRunCaly(unittest.TestCase):
    def setUp(self):
        self.expl_config = {}
        self.work_dir = Path("storge_files")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.nmodels = mocked_numb_models
        self.model_list = []
        for ii in range(self.nmodels):
            model_path = self.work_dir.joinpath(f"task.{ii}")
            model_path.mkdir(parents=True, exist_ok=True)
            model = model_path.joinpath(f"frozen_model.pb")
            model.write_text(f"model {ii}")
            self.model_list.append(model)
        self.models = upload_artifact(self.model_list)

        self.block_id = "id123id"
        self.expl_task_grp = make_task_group_list(njobs=2)
        self.type_map = ["Mg", "Al"]

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)
        for i in Path().glob("prep-run-caly-step*"):
            shutil.rmtree(i, ignore_errors=True)

    def test(self):
        caly_evo_step_op = CalyEvoStep(
            "caly-evo-run",
            MockedCollRunCaly,
            PrepCalyDPOptim,
            MockedRunCalyDPOptim,
            prep_config=prep_default_config,
            run_config=run_default_config,
            upload_python_packages=upload_python_packages,
        )
        prep_run_caly_op = PrepRunCaly(
            "prep-run-calypso",
            PrepCalyInput,
            caly_evo_step_op,
            PrepCalyModelDevi,
            MockedRunCalyModelDevi,
            prep_config=prep_default_config,
            run_config=run_default_config,
            upload_python_packages=upload_python_packages,
        )
        prep_run_caly_step = Step(
            "prep-run-caly-step",
            template=prep_run_caly_op,
            parameters={
                "block_id": self.block_id,
                "expl_task_grp": self.expl_task_grp,
                "explore_config": self.expl_config,
                "type_map": self.type_map,
            },
            artifacts={
                "models": self.models,
            },
        )

        wf = Workflow(name="prep-run-caly-step", host=default_host)
        wf.add(prep_run_caly_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="prep-run-caly-step")[0]
        self.assertEqual(step.phase, "Succeeded")

        # download_artifact(step.outputs.artifacts["model_devis"])
        # download_artifact(step.outputs.artifacts["trajs"])
        # download_artifact(step.outputs.artifacts["logs"])

        # for ii in step.outputs.parameters["task_names"].value:
        #     self.check_run_lmp_output(ii, self.model_list)
