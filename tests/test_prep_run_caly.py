import json
import os
import pickle
import shutil
import time
import unittest
from collections import (
    defaultdict,
)
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
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
from mock import (
    Mock,
    patch,
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
from dpgen2.op.caly_evo_step_merge import (
    CalyEvoStepMerge,
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
    _prep_run_caly,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

prep_default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        },
    }
)


class TestPrepRunCalyConfiguration(unittest.TestCase):
    def test_model_deviation_step_uses_run_config(self):
        """Route every step-level control from the intended phase config."""
        step_config_keys = (
            "continue_on_failed",
            "continue_on_num_success",
            "continue_on_success_ratio",
            "parallelism",
        )

        def make_mapping():
            return defaultdict(Mock)

        prep_config = normalize_step_dict(
            {
                "continue_on_failed": False,
                "continue_on_num_success": 1,
                "continue_on_success_ratio": 0.1,
                "parallelism": 3,
            }
        )
        run_config = normalize_step_dict(
            {
                "continue_on_failed": True,
                "continue_on_num_success": 9,
                "continue_on_success_ratio": 0.9,
                "parallelism": 7,
            }
        )

        for expl_mode in ("default", "merge"):
            with self.subTest(expl_mode=expl_mode):
                step_calls = []

                def make_step(name, *args, **kwargs):
                    step_calls.append((name, kwargs))
                    return SimpleNamespace(
                        outputs=SimpleNamespace(
                            parameters=make_mapping(),
                            artifacts=make_mapping(),
                        )
                    )

                prep_run_steps = SimpleNamespace(
                    inputs=SimpleNamespace(
                        parameters=make_mapping(),
                        artifacts=make_mapping(),
                    ),
                    outputs=SimpleNamespace(artifacts=make_mapping()),
                    add=Mock(),
                )

                with (
                    patch(
                        "dpgen2.superop.prep_run_calypso.Step",
                        side_effect=make_step,
                    ),
                    patch("dpgen2.superop.prep_run_calypso.PythonOPTemplate"),
                    patch("dpgen2.superop.prep_run_calypso.Slices"),
                    patch("dpgen2.superop.prep_run_calypso.argo_range"),
                    patch(
                        "dpgen2.superop.prep_run_calypso.init_executor",
                        side_effect=lambda value: value,
                    ),
                ):
                    _prep_run_caly(
                        prep_run_steps,
                        defaultdict(str),
                        Mock(),
                        Mock(),
                        Mock(),
                        Mock(),
                        expl_mode=expl_mode,
                        prep_config=prep_config,
                        run_config=run_config,
                    )

                calls_by_name = dict(step_calls)
                expected_configs = {
                    "prep-caly-input": prep_config,
                    "caly-evo-step": (
                        prep_config if expl_mode == "default" else run_config
                    ),
                    "run-caly-model-devi": run_config,
                }
                for step_name, expected_config in expected_configs.items():
                    actual_kwargs = calls_by_name[step_name]
                    self.assertEqual(
                        {key: actual_kwargs[key] for key in step_config_keys},
                        {key: expected_config[key] for key in step_config_keys},
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

    def test_caly_evo_step_merge_merge_mode(self):
        run_default_config = normalize_step_dict(
            {
                "template_config": {
                    "image": default_image,
                },
                "template_slice_config": {"group_size": 2, "pool_size": 1},
            }
        )
        explore_config = {"mode": "merge", "model_devi_group_size": 30}
        expl_mode = explore_config.get("mode")
        caly_evo_step_op = CalyEvoStepMerge(
            name="caly-evo-step",
            collect_run_caly=MockedCollRunCaly,
            prep_dp_optim=PrepCalyDPOptim,
            run_dp_optim=MockedRunCalyDPOptim,
            expl_mode=expl_mode,
            prep_config=prep_default_config,
            run_config=run_default_config,
            upload_python_packages=None,
        )
        prep_run_caly_op = PrepRunCaly(
            "prep-run-calypso",
            PrepCalyInput,
            caly_evo_step_op,
            PrepCalyModelDevi,
            MockedRunCalyModelDevi,
            expl_mode=expl_mode,
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

    def test_caly_evo_step_merge_default_mode(self):
        run_default_config = normalize_step_dict(
            {
                "template_config": {
                    "image": default_image,
                },
                "template_slice_config": {"group_size": 2, "pool_size": 1},
            }
        )
        explore_config = {"mode": "default", "model_devi_group_size": 30}
        expl_mode = explore_config.get("mode")
        caly_evo_step_op = CalyEvoStep(
            "caly-evo-run",
            MockedCollRunCaly,
            PrepCalyDPOptim,
            MockedRunCalyDPOptim,
            expl_mode=expl_mode,
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
            expl_mode=expl_mode,
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
