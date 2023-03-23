import json
import os
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
    MockedModifyTrainScript,
    MockedPrepDPTrain,
    MockedRunDPTrain,
    make_mocked_init_data,
    make_mocked_init_models,
    mocked_numb_models,
    mocked_template_script,
)
from test_prep_run_dp_train import (
    TestMockedPrepDPTrain,
    TestMockedRunDPTrain,
    check_run_train_dp_output,
)

from dpgen2.constants import (
    train_task_pattern,
)
from dpgen2.superop.finetune import (
    Finetune,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        }
    }
)


@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestFinetune(unittest.TestCase):
    def setUp(self):
        self.numb_models = mocked_numb_models

        tmp_models = make_mocked_init_models(self.numb_models)
        self.init_models = upload_artifact(tmp_models)
        self.str_init_models = tmp_models

        tmp_init_data = make_mocked_init_data()
        self.init_data = upload_artifact(tmp_init_data)
        self.path_init_data = tmp_init_data

        tmp_iter_data = [Path("iter_data/foo"), Path("iter_data/bar")]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii / "a").write_text("data a")
            (ii / "b").write_text("data b")
        self.iter_data = upload_artifact(tmp_iter_data)
        self.path_iter_data = tmp_iter_data

        self.template_script = mocked_template_script.copy()

        self.task_names = ["task.0000", "task.0001", "task.0002"]
        self.task_paths = [Path(ii) for ii in self.task_names]
        self.train_scripts = [
            Path("task.0000/input.json"),
            Path("task.0001/input.json"),
            Path("task.0002/input.json"),
        ]

    def tearDown(self):
        for ii in ["init_data", "iter_data"] + self.task_names:
            if Path(ii).exists():
                shutil.rmtree(str(ii))
        for ii in self.str_init_models:
            if Path(ii).exists():
                os.remove(ii)

    def test_finetune(self):
        steps = Finetune(
            "finetune-steps",
            MockedPrepDPTrain,
            MockedRunDPTrain,
            MockedModifyTrainScript,
            upload_python_packages=upload_python_packages,
            prep_config=default_config,
            run_config=default_config,
        )
        finetune_step = Step(
            "finetune-step",
            template=steps,
            parameters={
                "numb_models": self.numb_models,
                "template_script": self.template_script,
                "train_config": {},
            },
            artifacts={
                "init_models": self.init_models,
                "init_data": self.init_data,
                "iter_data": self.iter_data,
            },
        )
        wf = Workflow(name="dp-finetune", host=default_host)
        wf.add(finetune_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="finetune-step")[0]
        self.assertEqual(step.phase, "Succeeded")
        
        new_template_script = step.outputs.parameters["template_script"].value
        expected_list = [{"foo": "bar"} for i in range(self.numb_models)]
        assert (new_template_script == expected_list)

        download_artifact(step.outputs.artifacts["scripts"])
        download_artifact(step.outputs.artifacts["models"])
        download_artifact(step.outputs.artifacts["logs"])
        download_artifact(step.outputs.artifacts["lcurves"])

        for ii in range(3):
            check_run_train_dp_output(
                self,
                self.task_names[ii],
                self.train_scripts[ii],
                self.str_init_models[ii],
                self.path_init_data,
                self.path_iter_data,
                only_check_name=True,
            )
