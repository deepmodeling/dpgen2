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
    MockedPrepDPTrain,
    MockedRunDPTrain,
    MockedRunDPTrainNoneInitModel,
    MockedRunNvNMDTrain,
    MockedRunNvNMDTrainNoneInitModel,
    make_mocked_init_data,
    make_mocked_init_models,
    make_mocked_init_nvnmd_models,
    mocked_numb_models,
    mocked_template_script,
)

from dpgen2.constants import (
    train_task_pattern,
)
from dpgen2.superop.prep_run_dp_train import (
    PrepRunDPTrain,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        }
    }
)


def _check_log(
    tcase, fname, path, script, init_model, init_data, iter_data, only_check_name=False
):
    with open(fname) as fp:
        lines_ = fp.read().strip().split("\n")
    if only_check_name:
        lines = []
        for ii in lines_:
            ww = ii.split(" ")
            ww[1] = str(Path(ww[1]).name)
            lines.append(" ".join(ww))
    else:
        lines = lines_
    revised_fname = lambda ff: Path(ff).name if only_check_name else Path(ff)
    tcase.assertEqual(
        lines[0].split(" "),
        ["init_model", str(revised_fname(Path(path) / init_model)), "OK"],
    )
    for ii in range(2):
        tcase.assertEqual(
            lines[1 + ii].split(" "),
            [
                "data",
                str(revised_fname(Path(path) / sorted(list(init_data))[ii])),
                "OK",
            ],
        )
    for ii in range(2):
        tcase.assertEqual(
            lines[3 + ii].split(" "),
            [
                "data",
                str(revised_fname(Path(path) / sorted(list(iter_data))[ii])),
                "OK",
            ],
        )
    tcase.assertEqual(
        lines[5].split(" "), ["script", str(revised_fname(Path(path) / script)), "OK"]
    )


def _check_model(
    tcase,
    fname,
    path,
    model,
):
    with open(fname) as fp:
        flines = fp.read().strip().split("\n")
    with open(Path(path) / model) as fp:
        mlines = fp.read().strip().split("\n")
    tcase.assertEqual(flines[0], "read from init model: ")
    for ii in range(len(mlines)):
        tcase.assertEqual(flines[ii + 1], mlines[ii])


def _check_nvnmd_model_files(tcase, cwd, init_model):
    """Helper to check all nvnmd model files."""
    model_checks = [
        ("nvnmd_models/frozen_model.pb", init_model / "frozen_model.pb"),
        ("nvnmd_models/model.pb", init_model / "frozen_model.pb"),
        ("nvnmd_models/model.ckpt.meta", init_model / "model.ckpt.meta"),
        ("nvnmd_models/model.ckpt.data-00000-of-00001", init_model / "model.ckpt.data"),
        ("nvnmd_models/model.ckpt.index", init_model / "model.ckpt.index"),
    ]
    for output_file, expected_file in model_checks:
        _check_model(tcase, output_file, cwd, expected_file)


def _check_lcurve(
    tcase,
    fname,
    path,
    script,
):
    with open(fname) as fp:
        flines = fp.read().strip().split("\n")
    with open(Path(path) / script) as fp:
        mlines = fp.read().strip().split("\n")
    tcase.assertEqual(flines[0], "read from train_script: ")
    for ii in range(len(mlines)):
        tcase.assertEqual(flines[ii + 1], mlines[ii])


def check_run_train_dp_output(
    tcase,
    work_dir,
    script,
    init_model,
    init_data,
    iter_data,
    only_check_name=False,
):
    cwd = os.getcwd()
    os.chdir(work_dir)
    _check_log(
        tcase,
        "log",
        cwd,
        script,
        init_model,
        init_data,
        iter_data,
        only_check_name=only_check_name,
    )
    _check_model(tcase, "model.pb", cwd, init_model)
    _check_lcurve(tcase, "lcurve.out", cwd, script)
    os.chdir(cwd)


def check_run_train_nvnmd_output(
    tcase,
    work_dir,
    script,
    init_model,
    init_data,
    iter_data,
    only_check_name=False,
):
    cwd = os.getcwd()
    os.chdir(work_dir)
    _check_log(
        tcase,
        "log",
        cwd,
        script,
        init_model,
        init_data,
        iter_data,
        only_check_name=only_check_name,
    )
    _check_nvnmd_model_files(tcase, cwd, init_model)
    _check_lcurve(tcase, "nvnmd_cnn/lcurve.out", cwd, script)
    os.chdir(cwd)


class TestMockedPrepDPTrain(unittest.TestCase):
    def setUp(self):
        self.numb_models = mocked_numb_models
        self.template_script = mocked_template_script.copy()
        self.expected_subdirs = ["task.0000", "task.0001", "task.0002"]
        self.expected_train_scripts = [
            Path("task.0000/input.json"),
            Path("task.0001/input.json"),
            Path("task.0002/input.json"),
        ]

    def tearDown(self):
        for ii in self.expected_subdirs:
            if Path(ii).exists():
                shutil.rmtree(ii)

    def test(self):
        prep = MockedPrepDPTrain()
        ip = OPIO(
            {
                "template_script": self.template_script,
                "numb_models": self.numb_models,
            }
        )
        op = prep.execute(ip)
        # self.assertEqual(self.expected_train_scripts, op["train_scripts"])
        self.assertEqual(self.expected_subdirs, op["task_names"])
        self.assertEqual([Path(ii) for ii in self.expected_subdirs], op["task_paths"])


class TestMockedRunDPTrain(unittest.TestCase):
    def setUp(self):
        self.numb_models = mocked_numb_models

        self.init_models = make_mocked_init_models(self.numb_models)

        tmp_init_data = make_mocked_init_data()
        self.init_data = tmp_init_data

        tmp_iter_data = [Path("iter_data/foo"), Path("iter_data/bar")]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii / "a").write_text("data a")
            (ii / "b").write_text("data b")
        self.iter_data = tmp_iter_data

        self.template_script = mocked_template_script.copy()

        self.task_names = ["task.0000", "task.0001", "task.0002"]
        self.task_paths = [Path(ii) for ii in self.task_names]
        self.train_scripts = [
            Path("task.0000/input.json"),
            Path("task.0001/input.json"),
            Path("task.0002/input.json"),
        ]

        for ii in range(3):
            Path(self.task_names[ii]).mkdir(exist_ok=True, parents=True)
            Path(self.train_scripts[ii]).write_text("{}")

    def tearDown(self):
        for ii in ["init_data", "iter_data"] + self.task_names:
            if Path(ii).exists():
                shutil.rmtree(str(ii))
        for ii in self.init_models:
            if Path(ii).exists():
                os.remove(ii)

    def test(self):
        for ii in range(3):
            run = MockedRunDPTrain()
            ip = OPIO(
                {
                    "config": {},
                    "task_name": self.task_names[ii],
                    "task_path": self.task_paths[ii],
                    "init_model": self.init_models[ii],
                    "init_data": self.init_data,
                    "iter_data": self.iter_data,
                }
            )
            op = run.execute(ip)
            self.assertEqual(op["script"], Path(train_task_pattern % ii) / "input.json")
            self.assertTrue(op["script"].is_file())
            self.assertEqual(op["model"], Path(train_task_pattern % ii) / "model.pb")
            self.assertEqual(op["log"], Path(train_task_pattern % ii) / "log")
            self.assertEqual(op["lcurve"], Path(train_task_pattern % ii) / "lcurve.out")
            check_run_train_dp_output(
                self,
                self.task_names[ii],
                self.train_scripts[ii],
                self.init_models[ii],
                self.init_data,
                self.iter_data,
            )


class TestMockedRunNvNMDTrain(unittest.TestCase):
    def setUp(self):
        self.numb_models = mocked_numb_models

        self.init_models = make_mocked_init_nvnmd_models(self.numb_models)

        tmp_init_data = make_mocked_init_data()
        self.init_data = tmp_init_data

        tmp_iter_data = [Path("iter_data/foo"), Path("iter_data/bar")]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii / "a").write_text("data a")
            (ii / "b").write_text("data b")
        self.iter_data = tmp_iter_data

        self.template_script = mocked_template_script.copy()

        self.task_names = ["task.0000", "task.0001", "task.0002"]
        self.task_paths = [Path(ii) for ii in self.task_names]
        self.train_scripts = [
            Path("task.0000/input.json"),
            Path("task.0001/input.json"),
            Path("task.0002/input.json"),
        ]

        for ii in range(3):
            Path(self.task_names[ii]).mkdir(exist_ok=True, parents=True)
            Path(self.train_scripts[ii]).write_text("{}")

    def tearDown(self):
        for ii in ["init_data", "iter_data"] + self.task_names:
            if Path(ii).exists():
                shutil.rmtree(str(ii))
        for ii in self.init_models:
            if Path(ii).exists():
                shutil.rmtree(ii)

    def test(self):
        for ii in range(3):
            run = MockedRunNvNMDTrain()
            ip = OPIO(
                {
                    "config": {},
                    "task_name": self.task_names[ii],
                    "task_path": self.task_paths[ii],
                    "init_model": self.init_models[ii],
                    "init_data": self.init_data,
                    "iter_data": self.iter_data,
                }
            )
            op = run.execute(ip)
            self.assertEqual(op["script"], Path(train_task_pattern % ii) / "input.json")
            self.assertTrue(op["script"].is_file())
            self.assertEqual(
                op["model"] / "frozen_model.pb",
                Path(train_task_pattern % ii) / "nvnmd_models/frozen_model.pb",
            )
            self.assertEqual(
                op["model"] / "model.pb",
                Path(train_task_pattern % ii) / "nvnmd_models/model.pb",
            )
            self.assertEqual(
                op["model"] / "model.ckpt.meta",
                Path(train_task_pattern % ii) / "nvnmd_models/model.ckpt.meta",
            )
            self.assertEqual(
                op["model"] / "model.ckpt.data-00000-of-00001",
                Path(train_task_pattern % ii)
                / "nvnmd_models/model.ckpt.data-00000-of-00001",
            )
            self.assertEqual(
                op["model"] / "model.ckpt.index",
                Path(train_task_pattern % ii) / "nvnmd_models/model.ckpt.index",
            )
            self.assertEqual(op["log"], Path(train_task_pattern % ii) / "log")
            self.assertEqual(
                op["lcurve"], Path(train_task_pattern % ii) / "nvnmd_cnn/lcurve.out"
            )
            check_run_train_nvnmd_output(
                self,
                self.task_names[ii],
                self.train_scripts[ii],
                self.init_models[ii],
                self.init_data,
                self.iter_data,
            )


@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestTrainDp(unittest.TestCase):
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

    def test_train(self):
        steps = PrepRunDPTrain(
            "train-steps",
            MockedPrepDPTrain,
            MockedRunDPTrain,
            upload_python_packages=upload_python_packages,
            prep_config=default_config,
            run_config=default_config,
        )
        train_step = Step(
            "train-step",
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
        wf = Workflow(name="dp-train", host=default_host)
        wf.add(train_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="train-step")[0]
        self.assertEqual(step.phase, "Succeeded")

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

    def test_train_no_init_model(self):
        steps = PrepRunDPTrain(
            "train-steps",
            MockedPrepDPTrain,
            MockedRunDPTrainNoneInitModel,
            upload_python_packages=upload_python_packages,
            prep_config=default_config,
            run_config=default_config,
        )
        train_step = Step(
            "train-step",
            template=steps,
            parameters={
                "numb_models": self.numb_models,
                "template_script": self.template_script,
                "train_config": {},
            },
            artifacts={
                "init_models": None,
                "init_data": self.init_data,
                "iter_data": self.iter_data,
            },
        )
        wf = Workflow(name="dp-train", host=default_host)
        wf.add(train_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="train-step")[0]
        self.assertEqual(step.phase, "Succeeded")

    def test_finetune(self):
        steps = PrepRunDPTrain(
            "finetune-steps",
            MockedPrepDPTrain,
            MockedRunDPTrain,
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
                "run_optional_parameter": {
                    "data_mixed_type": False,
                    "finetune_mode": "finetune",
                },
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
        assert new_template_script == self.template_script

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


@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestTrainNvNMD(unittest.TestCase):
    def setUp(self):
        self.numb_models = mocked_numb_models

        tmp_models = make_mocked_init_nvnmd_models(self.numb_models)
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
                shutil.rmtree(ii)

    def test_train(self):
        steps = PrepRunDPTrain(
            "train-steps",
            MockedPrepDPTrain,
            MockedRunNvNMDTrain,
            upload_python_packages=upload_python_packages,
            prep_config=default_config,
            run_config=default_config,
        )
        train_step = Step(
            "train-step",
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
        wf = Workflow(name="nvnmd-train", host=default_host)
        wf.add(train_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="train-step")[0]
        self.assertEqual(step.phase, "Succeeded")

        download_artifact(step.outputs.artifacts["scripts"])
        download_artifact(step.outputs.artifacts["models"])
        download_artifact(step.outputs.artifacts["logs"])
        download_artifact(step.outputs.artifacts["lcurves"])

        for ii in range(3):
            check_run_train_nvnmd_output(
                self,
                self.task_names[ii],
                self.train_scripts[ii],
                self.str_init_models[ii],
                self.path_init_data,
                self.path_iter_data,
                only_check_name=True,
            )

    def test_train_no_init_model(self):
        steps = PrepRunDPTrain(
            "train-steps",
            MockedPrepDPTrain,
            MockedRunNvNMDTrainNoneInitModel,
            upload_python_packages=upload_python_packages,
            prep_config=default_config,
            run_config=default_config,
        )
        train_step = Step(
            "train-step",
            template=steps,
            parameters={
                "numb_models": self.numb_models,
                "template_script": self.template_script,
                "train_config": {},
            },
            artifacts={
                "init_models": None,
                "init_data": self.init_data,
                "iter_data": self.iter_data,
            },
        )
        wf = Workflow(name="nvnmd-train", host=default_host)
        wf.add(train_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="train-step")[0]
        self.assertEqual(step.phase, "Succeeded")
