import json
import os
import pickle
import shutil
import textwrap
import time
import unittest
from pathlib import Path
from typing import List, Set

import jsonpickle
import numpy as np
from dflow import (InputArtifact, InputParameter, Inputs, OutputArtifact,
                   OutputParameter, Outputs, S3Artifact, Step, Steps, Workflow,
                   argo_range, download_artifact, upload_artifact)
from dflow.python import (OP, OPIO, Artifact, OPIOSign, PythonOPTemplate,
                          upload_packages)

try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dflow.python import FatalError
from dpgen2.constants import (lmp_conf_name, lmp_input_name, lmp_log_name,
                              lmp_traj_name, model_name_pattern,
                              train_log_name, train_script_name,
                              train_task_pattern, vasp_conf_name,
                              vasp_input_name, vasp_pot_name,
                              vasp_task_pattern)
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.scheduler import ExplorationScheduler
from dpgen2.exploration.selector import TrustLevel
from dpgen2.exploration.task import (ExplorationStage, ExplorationTask,
                                     ExplorationTaskGroup)
from dpgen2.flow.dpgen_loop import ConcurrentLearning
from dpgen2.fp.vasp import VaspInputs
from dpgen2.op.prep_lmp import PrepLmp
from dpgen2.superop.block import ConcurrentLearningBlock
from dpgen2.superop.prep_run_dp_train import PrepRunDPTrain
from dpgen2.superop.prep_run_fp import PrepRunFp
from dpgen2.superop.prep_run_lmp import PrepRunLmp
from dpgen2.utils.step_config import normalize as normalize_step_dict

from context import (default_host, default_image, skip_ut_with_dflow,
                     skip_ut_with_dflow_reason, upload_python_package)
from mocked_ops import (
    MockedCollectData, MockedCollectDataFailed, MockedCollectDataRestart,
    MockedConfSelector, MockedConstTrustLevelStageScheduler,
    MockedExplorationReport, MockedExplorationTaskGroup,
    MockedExplorationTaskGroup1, MockedExplorationTaskGroup2,
    MockedPrepDPTrain, MockedPrepVasp, MockedRunDPTrain, MockedRunLmp,
    MockedRunVasp, MockedRunVaspFail1, MockedRunVaspRestart, MockedSelectConfs,
    MockedStage, MockedStage1, MockedStage2, make_mocked_init_data,
    make_mocked_init_models, mocked_incar_template, mocked_numb_models,
    mocked_numb_select, mocked_template_script)

default_config = normalize_step_dict(
    {"template_config": {
        "image": default_image,
    }})


@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestLoop(unittest.TestCase):

    def _setUp_ops(self):
        self.prep_run_dp_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            MockedPrepDPTrain,
            MockedRunDPTrain,
            upload_python_package=upload_python_package,
            prep_config=default_config,
            run_config=default_config,
        )
        self.prep_run_lmp_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            MockedRunLmp,
            upload_python_package=upload_python_package,
            prep_config=default_config,
            run_config=default_config,
        )
        self.prep_run_fp_op = PrepRunFp(
            "prep-run-fp",
            MockedPrepVasp,
            MockedRunVasp,
            upload_python_package=upload_python_package,
            prep_config=default_config,
            run_config=default_config,
        )
        self.block_cl_op = ConcurrentLearningBlock(
            self.name + '-block',
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op,
            MockedCollectData,
            upload_python_package=upload_python_package,
            select_confs_config=default_config,
            collect_data_config=default_config,
        )
        self.dpgen_op = ConcurrentLearning(
            self.name,
            self.block_cl_op,
            upload_python_package=upload_python_package,
            step_config=default_config,
        )

    def _setUp_data(self):
        self.numb_models = mocked_numb_models

        tmp_models = []
        for ii in range(self.numb_models):
            ff = Path(model_name_pattern % ii)
            ff.write_text(f'This is init model {ii}')
            tmp_models.append(ff)
        self.init_models = upload_artifact(tmp_models)
        self.str_init_models = tmp_models

        tmp_init_data = [Path('init_data/foo'), Path('init_data/bar')]
        for ii in tmp_init_data:
            ii.mkdir(exist_ok=True, parents=True)
            (ii / 'a').write_text('data a')
            (ii / 'b').write_text('data b')
        self.init_data = upload_artifact(tmp_init_data)
        self.path_init_data = tmp_init_data

        self.iter_data = upload_artifact([])
        self.path_iter_data = None

        self.template_script = mocked_template_script

        self.type_map = ['H', 'O']

        self.incar = Path('incar')
        self.incar.write_text(mocked_incar_template)
        self.potcar = Path('potcar')
        self.potcar.write_text('bar')
        self.vasp_inputs = VaspInputs(
            0.16,
            True,
            self.incar,
            {'foo': 'potcar'},
        )

        self.scheduler = ExplorationScheduler()
        self.trust_level = TrustLevel(0.1, 0.3)
        trust_level = TrustLevel(0.1, 0.3)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage(),
            trust_level,
            conv_accuracy=0.7,
            max_numb_iter=2,
        )
        self.scheduler.add_stage_scheduler(stage_scheduler)
        trust_level = TrustLevel(0.2, 0.4)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage1(),
            trust_level,
            conv_accuracy=0.7,
            max_numb_iter=2,
        )
        self.scheduler.add_stage_scheduler(stage_scheduler)

    def setUp(self):
        self.name = 'dpgen'
        self._setUp_ops()
        self._setUp_data()

    def tearDown(self):
        for ii in ['init_data', 'iter_data', 'models']:
            ii = Path(ii)
            if ii.is_dir():
                shutil.rmtree(ii)
        for ii in range(self.numb_models):
            name = Path(model_name_pattern % ii)
            if name.is_file():
                os.remove(name)
        for ii in [
                self.incar,
                self.potcar,
        ]:
            if ii.is_file():
                os.remove(ii)

    def test(self):
        self.assertEqual(self.dpgen_op.loop_keys, [
            "loop",
            'block',
            'prep-train',
            'run-train',
            'prep-lmp',
            'run-lmp',
            'select-confs',
            'prep-fp',
            'run-fp',
            'collect-data',
            "scheduler",
            "id",
        ])
        self.assertEqual(self.dpgen_op.init_keys, [
            "scheduler",
            "id",
        ])

        dpgen_step = Step(
            'dpgen-step',
            template=self.dpgen_op,
            parameters={
                "type_map": self.type_map,
                "numb_models": self.numb_models,
                "template_script": self.template_script,
                "train_config": {},
                "lmp_config": {},
                "fp_config": {},
                'fp_inputs': self.vasp_inputs,
                "exploration_scheduler": self.scheduler,
            },
            artifacts={
                "init_models": self.init_models,
                "init_data": self.init_data,
                "iter_data": self.iter_data,
            },
        )

        # wf = Workflow(name="dpgen", host=default_host)
        wf = Workflow(name="dpgen")
        wf.add(dpgen_step)
        wf.submit()
        if config["mode"] == "debug":
            pass
        else:
            while wf.query_status() in ["Pending", "Running"]:
                time.sleep(4)
            self.assertEqual(wf.query_status(), "Succeeded")
        step = dpgen_step
        self.assertEqual(step.phase, "Succeeded")

        scheduler = step.outputs.parameters['exploration_scheduler'].value
        download_artifact(step.outputs.artifacts["iter_data"],
                          path='iter_data')
        download_artifact(step.outputs.artifacts["models"],
                          path=Path('models') / self.name)
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 1)

        if config["mode"] != "debug":
            # # we know number of selected data is 2
            # # by MockedConfSelector
            for ii in range(mocked_numb_select):
                self.assertEqual(
                    (Path(wf.id + 'iter_data') / 'iter-000000' /
                     ('data_' + vasp_task_pattern % ii) /
                     'data').read_text().strip(), '\n'.join([
                         'labeled_data of ' + vasp_task_pattern % ii,
                         f'select conf.{ii}',
                         f'mocked conf {ii}',
                         f'mocked input {ii}',
                     ]).strip())
            for ii in range(mocked_numb_select):
                self.assertEqual(
                    (Path(wf.id + 'iter_data') / 'iter-000001' /
                     ('data_' + vasp_task_pattern % ii) /
                     'data').read_text().strip(), '\n'.join([
                         'labeled_data of ' + vasp_task_pattern % ii,
                         f'select conf.{ii}',
                         f'mocked 1 conf {ii}',
                         f'mocked 1 input {ii}',
                     ]).strip())

            # new model is read from init model
            for ii in range(self.numb_models):
                model = Path('models')/self.name / \
                    (train_task_pattern % ii)/'model.pb'
                flines = model.read_text().strip().split('\n')
                # two iteratins, to lines of reading
                self.assertEqual(flines[0], "read from init model: ")
                self.assertEqual(flines[1], "read from init model: ")
                self.assertEqual(flines[2], f"This is init model {ii}")


if __name__ == "__main__":
    from dflow import config
    config["mode"] = "debug"
    unittest.main()
