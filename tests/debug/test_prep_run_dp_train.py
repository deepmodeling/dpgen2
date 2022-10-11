import json
import os
import shutil
from tabnanny import check
import time
import unittest
from pathlib import Path
from typing import List, Set

import numpy as np
from dflow import (InputArtifact, InputParameter, Inputs, OutputArtifact,
                   OutputParameter, Outputs, S3Artifact, Step, Steps, Workflow,
                   argo_range, download_artifact, upload_artifact)
from dflow.python import OP, OPIO, Artifact, OPIOSign, PythonOPTemplate

try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.constants import train_task_pattern
from dpgen2.superop.prep_run_dp_train import PrepRunDPTrain
from dpgen2.utils.step_config import normalize as normalize_step_dict

from context import (default_host, default_image, skip_ut_with_dflow,
                     skip_ut_with_dflow_reason, upload_python_package)
from mocked_ops import (MockedPrepDPTrain, MockedRunDPTrain,
                        MockedRunDPTrainNoneInitModel, make_mocked_init_data,
                        make_mocked_init_models, mocked_numb_models,
                        mocked_template_script)

default_config = normalize_step_dict(
    {"template_config": {
        "image": default_image,
    }})


def _check_log(tcase,
               fname,
               path,
               script,
               init_model,
               init_data,
               iter_data,
               only_check_name=False):
    with open(fname) as fp:
        lines_ = fp.read().strip().split('\n')
    if only_check_name:
        lines = []
        for ii in lines_:
            ww = ii.split(' ')
            ww[1] = str(Path(ww[1]).name)
            lines.append(' '.join(ww))
    else:
        lines = lines_

    def revised_fname(ff):
        return Path(ff).name if only_check_name else Path(ff)

    tcase.assertEqual(
        lines[0].split(' '),
        ['init_model',
         str(revised_fname(Path(path) / init_model)), 'OK'])
    for ii in range(2):
        tcase.assertEqual(lines[1 + ii].split(' '), [
            'data',
            str(revised_fname(Path(path) / sorted(list(init_data))[ii])), 'OK'
        ])
    for ii in range(2):
        tcase.assertEqual(lines[3 + ii].split(' '), [
            'data',
            str(revised_fname(Path(path) / sorted(list(iter_data))[ii])), 'OK'
        ])
    tcase.assertEqual(
        lines[5].split(' '),
        ['script', str(revised_fname(Path(path) / script)), 'OK'])


def _check_model(
    tcase,
    fname,
    path,
    model,
):
    with open(fname) as fp:
        flines = fp.read().strip().split('\n')
    with open(Path(path) / model) as fp:
        mlines = fp.read().strip().split('\n')
    tcase.assertEqual(flines[0], "read from init model: ")
    for ii in range(len(mlines)):
        tcase.assertEqual(flines[ii + 1], mlines[ii])


def _check_lcurve(
    tcase,
    fname,
    path,
    script,
):
    with open(fname) as fp:
        flines = fp.read().strip().split('\n')
    with open(Path(path) / script) as fp:
        mlines = fp.read().strip().split('\n')
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
    if config["mode"] == "debug":
        cwd = os.getcwd()
        start = work_dir.find("{{") + 2
        if start != 1:
            end = work_dir.find("}}")
            task_name = work_dir[start:end]
            work_dir = work_dir[:start - 2]
            script = Path(work_dir + 'script') / script
            os.chdir(work_dir + 'log/' + task_name)
        _check_log(tcase,
                   "log",
                   cwd,
                   script,
                   init_model,
                   init_data,
                   iter_data,
                   only_check_name=only_check_name)
        os.chdir(cwd)
        if start != 1:
            os.chdir(work_dir + 'model/' + task_name)
        _check_model(tcase, "model.pb", cwd, init_model)
        os.chdir(cwd)
        if start != 1:
            os.chdir(work_dir + 'lcurve/' + task_name)
        _check_lcurve(tcase, "lcurve.out", cwd, script)
        os.chdir(cwd)

    else:
        cwd = os.getcwd()
        os.chdir(work_dir)
        _check_log(tcase,
                   "log",
                   cwd,
                   script,
                   init_model,
                   init_data,
                   iter_data,
                   only_check_name=only_check_name)
        _check_model(tcase, "model.pb", cwd, init_model)
        _check_lcurve(tcase, "lcurve.out", cwd, script)
        os.chdir(cwd)



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

        tmp_iter_data = [Path('iter_data/foo'), Path('iter_data/bar')]
        for ii in tmp_iter_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii / 'a').write_text('data a')
            (ii / 'b').write_text('data b')
        self.iter_data = upload_artifact(tmp_iter_data)
        self.path_iter_data = tmp_iter_data

        self.template_script = mocked_template_script.copy()

        self.task_names = ['task.0000', 'task.0001', 'task.0002']
        self.task_paths = [Path(ii) for ii in self.task_names]
        self.train_scripts = [
            Path('task.0000/input.json'),
            Path('task.0001/input.json'),
            Path('task.0002/input.json'),
        ]

    def tearDown(self):
        for ii in ['init_data', 'iter_data'] + self.task_names:
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
            upload_python_package=upload_python_package,
            prep_config=default_config,
            run_config=default_config,
        )
        train_step = Step(
            'train-step',
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

        if config["mode"] == "debug":
            step = train_step
        else:
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
            if config["mode"] == "debug":
                check_run_train_dp_output(
                    self,
                    wf.id + '/--run-train-group/{{%s}}' % self.task_names[ii],
                    self.train_scripts[ii],
                    self.str_init_models[ii],
                    self.path_init_data,
                    self.path_iter_data,
                    only_check_name=True,
                )
            else:
                check_run_train_dp_output(self,
                                          self.task_names[ii],
                                          self.train_scripts[ii],
                                          self.str_init_models[ii],
                                          self.path_init_data,
                                          self.path_iter_data,
                                          only_check_name=True)

    def test_train_no_init_model(self):
        steps = PrepRunDPTrain(
            "train-steps",
            MockedPrepDPTrain,
            MockedRunDPTrainNoneInitModel,
            upload_python_package=upload_python_package,
            prep_config=default_config,
            run_config=default_config,
        )
        train_step = Step(
            'train-step',
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
        if config["mode"] == "debug":
            step = train_step
        else:
            while wf.query_status() in ["Pending", "Running"]:
                time.sleep(4)
            self.assertEqual(wf.query_status(), "Succeeded")
            step = wf.query_step(name="train-step")[0]
        self.assertEqual(step.phase, "Succeeded")


if __name__ == "__main__":
    from dflow import config
    config["mode"] = "debug"
    unittest.main()
