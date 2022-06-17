import os, textwrap, pickle
import numpy as np
import unittest

from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    S3Artifact,
    argo_range
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    upload_packages,
)

import time, shutil, json, jsonpickle

from typing import Set, List
from pathlib import Path
try:
    from context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from context import (
    upload_python_package,
    skip_ut_with_dflow,
    skip_ut_with_dflow_reason,
    default_image,
    default_host,
)
from dflow.python import (
    FatalError,
)
from dpgen2.exploration.scheduler import (
    ExplorationScheduler,
)
from dpgen2.op.prep_lmp import PrepLmp
from dpgen2.superop.prep_run_dp_train import PrepRunDPTrain
from dpgen2.superop.prep_run_lmp import PrepRunLmp
from dpgen2.superop.prep_run_fp import PrepRunFp
from dpgen2.superop.block import ConcurrentLearningBlock
from dpgen2.exploration.task import ExplorationTask, ExplorationTaskGroup
from dpgen2.fp.vasp import VaspInputs
from dpgen2.flow.dpgen_loop import ConcurrentLearning
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.task import ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.selector import TrustLevel
from dpgen2.utils import(
    dump_object_to_file,
    load_object_from_file,
)

from dpgen2.constants import (
    train_task_pattern,
    train_script_name,
    train_log_name,
    model_name_pattern,
    lmp_conf_name,
    lmp_input_name,
    lmp_traj_name,
    lmp_log_name,
    vasp_task_pattern,
    vasp_conf_name,
    vasp_input_name,
    vasp_pot_name,
)
from mocked_ops import (
    mocked_template_script,
    mocked_numb_models,
    make_mocked_init_models,
    make_mocked_init_data,
    mocked_incar_template,
    mocked_numb_select,
    MockedPrepDPTrain,
    MockedRunDPTrain,    
    MockedRunLmp,
    MockedPrepVasp,
    MockedRunVasp,
    MockedRunVaspFail1,
    MockedRunVaspRestart,
    MockedSelectConfs,
    MockedConfSelector,
    MockedCollectData,
    MockedCollectDataFailed,
    MockedCollectDataRestart,
    MockedExplorationReport,
    MockedExplorationTaskGroup,
    MockedExplorationTaskGroup1,
    MockedExplorationTaskGroup2,
    MockedStage,
    MockedStage1,
    MockedStage2,
    MockedConstTrustLevelStageScheduler,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
default_config = normalize_step_dict(
    {
        "template_config" : {
            "image" : default_image,
        }
    }
)


@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestLoop(unittest.TestCase):
    def _setUp_ops(self):
        self.prep_run_dp_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            MockedPrepDPTrain,
            MockedRunDPTrain,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        self.prep_run_lmp_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            MockedRunLmp,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        self.prep_run_fp_op = PrepRunFp(
            "prep-run-fp",
            MockedPrepVasp,
            MockedRunVasp,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        self.block_cl_op = ConcurrentLearningBlock(
            self.name+'-block', 
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op,
            MockedCollectData,
            upload_python_package = upload_python_package,
            select_confs_config = default_config,
            collect_data_config = default_config,
        )        
        self.dpgen_op = ConcurrentLearning(
            self.name,
            self.block_cl_op,
            upload_python_package = upload_python_package,
            step_config = default_config,
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
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
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
            0.16, True,
            self.incar,
            {'foo': 'potcar'},
        )
        self.vasp_inputs_fname = Path('vasp_inputs.dat')
        self.vasp_inputs_arti = upload_artifact(
            dump_object_to_file(self.vasp_inputs, self.vasp_inputs_fname))

        self.scheduler = ExplorationScheduler()        
        self.trust_level = TrustLevel(0.1, 0.3)
        trust_level = TrustLevel(0.1, 0.3)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler.add_stage_scheduler(stage_scheduler)
        trust_level = TrustLevel(0.2, 0.4)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage1(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler.add_stage_scheduler(stage_scheduler)        
        self.scheduler_artifact = upload_artifact(
            dump_object_to_file(self.scheduler, 'in_scheduler.dat')
        )
        
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
        for ii in [self.incar, self.potcar, Path('scheduler.dat'), Path('in_scheduler.dat'), self.vasp_inputs_fname]:
            if ii.is_file():
                os.remove(ii)

    def test(self):
        self.assertEqual(
            self.dpgen_op.loop_keys, 
            [
                "loop", 'block',
                'prep-train', 'run-train', 'prep-lmp', 'run-lmp', 'select-confs',
                'prep-fp', 'run-fp', 'collect-data', 
                "scheduler", "id",
            ]
        )
        self.assertEqual(
            self.dpgen_op.init_keys, 
            [
                "scheduler", "id", 
            ]
        )

        dpgen_step = Step(
            'dpgen-step', 
            template = self.dpgen_op,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf = Workflow(name="dpgen", host=default_host)
        wf.add(dpgen_step)
        wf.submit()
        
        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)
        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name='dpgen-step')[0]
        self.assertEqual(step.phase, "Succeeded")        
        
        # scheduler = jsonpickle.decode(step.outputs.parameters['exploration_scheduler'].value)
        scheduler = load_object_from_file(
            download_artifact(step.outputs.artifacts["exploration_scheduler"])[0]
        )
        download_artifact(step.outputs.artifacts["iter_data"], path = 'iter_data')
        download_artifact(step.outputs.artifacts["models"], path = Path('models')/self.name)
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 1)
        
        # # we know number of selected data is 2
        # # by MockedConfSelector
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000000'/
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join(['labeled_data of '+vasp_task_pattern%ii,
                           f'select conf.{ii}',
                           f'mocked conf {ii}',
                           f'mocked input {ii}',
                           ]).strip()
            )
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000001'/\
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join(['labeled_data of '+vasp_task_pattern%ii,
                           f'select conf.{ii}',
                           f'mocked 1 conf {ii}',
                           f'mocked 1 input {ii}',
                           ]).strip()
            )
                           

        # new model is read from init model
        for ii in range(self.numb_models):
            model = Path('models')/self.name/(train_task_pattern%ii)/'model.pb'
            flines = model.read_text().strip().split('\n')            
            # two iteratins, to lines of reading
            self.assertEqual(flines[0], "read from init model: ")            
            self.assertEqual(flines[1], "read from init model: ")            
            self.assertEqual(flines[2], f"This is init model {ii}")



@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestLoopRestart(unittest.TestCase):
    def _setUp_ops(self):
        self.prep_run_dp_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            MockedPrepDPTrain,
            MockedRunDPTrain,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        self.prep_run_lmp_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            MockedRunLmp,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        self.prep_run_fp_op = PrepRunFp(
            "prep-run-fp",
            MockedPrepVasp,
            MockedRunVasp,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        self.prep_run_fp_op_f1 = PrepRunFp(
            "prep-run-fp",
            MockedPrepVasp,
            MockedRunVaspFail1,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        self.prep_run_fp_op_res = PrepRunFp(
            "prep-run-fp",
            MockedPrepVasp,
            MockedRunVaspRestart,
            upload_python_package = upload_python_package,
            prep_config = default_config,
            run_config = default_config,
        )
        # failed collect data
        self.block_cl_op_0 = ConcurrentLearningBlock(
            self.name+'-block', 
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op,
            MockedCollectDataFailed,
            upload_python_package = upload_python_package,
            select_confs_config = default_config,
            collect_data_config = default_config,
        )        
        self.dpgen_op_0 = ConcurrentLearning(
            self.name,
            self.block_cl_op_0,
            upload_python_package = upload_python_package,
            step_config = default_config,
        )
        # restart collect data
        self.block_cl_op_1 = ConcurrentLearningBlock(
            self.name+'-block', 
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op,
            MockedCollectDataRestart,
            upload_python_package = upload_python_package,
            select_confs_config = default_config,
            collect_data_config = default_config,
        )        
        self.dpgen_op_1 = ConcurrentLearning(
            self.name,
            self.block_cl_op_1,
            upload_python_package = upload_python_package,
            step_config = default_config,
        )
        # failed vasp 1
        self.block_cl_op_2 = ConcurrentLearningBlock(
            self.name+'-block', 
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op_f1,
            MockedCollectData,
            upload_python_package = upload_python_package,
            select_confs_config = default_config,
            collect_data_config = default_config,
        )        
        self.dpgen_op_2 = ConcurrentLearning(
            self.name,
            self.block_cl_op_2,
            upload_python_package = upload_python_package,
            step_config = default_config,
        )
        # restart vasp
        self.block_cl_op_3 = ConcurrentLearningBlock(
            self.name+'-block', 
            self.prep_run_dp_train_op,
            self.prep_run_lmp_op,
            MockedSelectConfs,
            self.prep_run_fp_op_res,
            MockedCollectData,
            upload_python_package = upload_python_package,
            select_confs_config = default_config,
            collect_data_config = default_config,
        )        
        self.dpgen_op_3 = ConcurrentLearning(
            self.name,
            self.block_cl_op_3,
            upload_python_package = upload_python_package,
            step_config = default_config,
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
            (ii/'a').write_text('data a')
            (ii/'b').write_text('data b')
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
            0.16, True,
            self.incar,
            {'foo': self.potcar},
        )
        self.vasp_inputs_fname = Path('vasp_inputs.dat')
        self.vasp_inputs_arti = upload_artifact(
            dump_object_to_file(self.vasp_inputs, self.vasp_inputs_fname))

        self.scheduler_0 = ExplorationScheduler()        
        self.trust_level = TrustLevel(0.1, 0.3)
        trust_level = TrustLevel(0.1, 0.3)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler_0.add_stage_scheduler(stage_scheduler)
        trust_level = TrustLevel(0.2, 0.4)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage1(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler_0.add_stage_scheduler(stage_scheduler)
        self.scheduler_0_artifact = upload_artifact(
            dump_object_to_file(self.scheduler_0, 'in_scheduler_0.dat'))

        self.scheduler_1 = ExplorationScheduler()        
        self.trust_level = TrustLevel(0.1, 0.3)
        trust_level = TrustLevel(0.1, 0.3)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler_1.add_stage_scheduler(stage_scheduler)
        trust_level = TrustLevel(0.2, 0.4)
        stage_scheduler = MockedConstTrustLevelStageScheduler(
            MockedStage2(),
            trust_level,
            conv_accuracy = 0.7,
            max_numb_iter = 2,
        )
        self.scheduler_1.add_stage_scheduler(stage_scheduler)        
        self.scheduler_1.add_stage_scheduler(stage_scheduler)
        self.scheduler_1_artifact = upload_artifact(
            dump_object_to_file(self.scheduler_1, 'in_scheduler_1.dat'))
        
        
    def setUp(self):
        self.name = 'dpgen'
        self._setUp_ops()
        self._setUp_data()

    def tearDown(self):
        for ii in ['init_data', 'iter_data', 'models', 'failed_res']:
            ii = Path(ii)
            if ii.is_dir():
                shutil.rmtree(ii)            
        for ii in range(self.numb_models):
            name = Path(model_name_pattern % ii)
            if name.is_file():
                os.remove(name)
        for ii in [self.incar, self.potcar]:
            if ii.is_file():
                os.remove(ii)
        for ii in [Path('scheduler_0.dat'), Path('in_scheduler_0.dat')]:
            if ii.is_file():
                os.remove(ii)
        for ii in [Path('scheduler_1.dat'), Path('in_scheduler_1.dat')]:
            if ii.is_file():
                os.remove(ii)
        for ii in [Path('scheduler.dat'), Path('scheduler_new.dat')]:
            if ii.is_file():
                os.remove(ii)
        for ii in [self.vasp_inputs_fname]:
            if ii.is_file():
                os.remove(ii)

    def get_restart_step(
            self,
            wf,
            steps_exceptions = [],
            phase = 'Succeeded',
    ):
        _steps = wf.query_step(phase=phase)
        steps = []
        for ii in _steps:
            if ii.type != "Steps" or (ii.key is not None and ii.key in steps_exceptions):
                steps.append(ii)
        return steps

    def test_update_artifact(self):
        # failed at collect_data
        # revise the output of prep-run-fp
        # restart the workflow with restarted collect_data and the old scheduler

        dpgen_step_0 = Step(
            'dpgen-step', 
            template = self.dpgen_op_0,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_0 = Workflow(name="dpgen")
        wf_0.add(dpgen_step_0)
        wf_0.submit()
        id_0 = wf_0.id

        # wf_0 = Workflow(id='dpgen-rdhxw')

        while wf_0.query_status() in ["Pending", "Running"]:
            time.sleep(4)
        self.assertEqual(wf_0.query_status(), "Failed")

        # _steps_0 = wf_0.query_step()
        # steps_0 = []
        # for ii in _steps_0:
        #     if ii['phase'] == 'Succeeded':
        #         steps_0.append(ii)

        steps_0 = self.get_restart_step(
            wf_0, steps_exceptions = ['iter-000000--prep-run-fp'])
        
        fpout_idx = None
        for idx, ii in enumerate(steps_0):
            if ii.key is not None and ii.key == 'iter-000000--prep-run-fp':
                fpout_idx = idx
        self.assertTrue(fpout_idx is not None)
        step_fp = steps_0.pop(fpout_idx)

        self.assertEqual(step_fp['phase'], 'Succeeded')
        download_artifact(step_fp.outputs.artifacts['labeled_data'], path='failed_res')
        for modi_file in [
                Path('failed_res')/'task.000000'/'data_task.000000'/'data',
                Path('failed_res')/'task.000001'/'data_task.000001'/'data',
        ]:
            fc = modi_file.read_text()
            fc = 'modified\n' + fc
            modi_file.write_text(fc)
        os.chdir('failed_res')
        new_arti = upload_artifact([
            'task.000000/data_task.000000', 
            'task.000001/data_task.000001',
        ])
        step_fp.modify_output_artifact('labeled_data', new_arti)
        os.chdir('..')
        steps_0.append(step_fp)

        # keys_0 =  []
        # for ii in steps_0:
        #     if ii.key is not None:
        #         keys_0.append(ii.key)
        # keys_0.sort()
        # print(keys_0)

        dpgen_step_1 = Step(
            'dpgen-step', 
            template = self.dpgen_op_1,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_1_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_1 = Workflow(name="dpgen")
        wf_1.add(dpgen_step_1)
        wf_1.submit(reuse_step = steps_0)
        id_1 = wf_1.id
        
        while wf_1.query_status() in ["Pending", "Running"]:
            time.sleep(2)
        self.assertEqual(wf_1.query_status(), "Succeeded")

        step = wf_1.query_step(name='dpgen-step')[0]
        self.assertEqual(step.phase, "Succeeded")
        download_artifact(step.outputs.artifacts["iter_data"], path = 'iter_data')
        download_artifact(step.outputs.artifacts["models"], path = Path('models')/self.name)
        # scheduler = jsonpickle.decode(step.outputs.parameters['exploration_scheduler'].value)
        scheduler = load_object_from_file(
            download_artifact(step.outputs.artifacts["exploration_scheduler"])[0])
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 1)
        
        # # we know number of selected data is 2
        # # by MockedConfSelector
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000000'/
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join([
                    'restart',
                    'modified',
                    'labeled_data of '+vasp_task_pattern%ii,
                    f'select conf.{ii}',
                    f'mocked conf {ii}',
                    f'mocked input {ii}',
                ]).strip()
            )
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000001'/\
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join([
                    'restart',
                    'labeled_data of '+vasp_task_pattern%ii,
                    f'select conf.{ii}',
                    f'mocked 1 conf {ii}',
                    f'mocked 1 input {ii}',
                ]).strip()
            )


    def test_update_artifact_scheduler(self):
        # failed at collect_data
        # revise the output of prep-run-fp
        # revise the scheduler
        # restart the workflow with restarted collect_data and the new scheduler

        dpgen_step_0 = Step(
            'dpgen-step', 
            template = self.dpgen_op_0,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_0 = Workflow(name="dpgen")
        wf_0.add(dpgen_step_0)
        wf_0.submit()
        id_0 = wf_0.id

        # wf_0 = Workflow(id='dpgen-xjvww')

        while wf_0.query_status() in ["Pending", "Running"]:
            time.sleep(4)
        self.assertEqual(wf_0.query_status(), "Failed")

        _steps_0 = wf_0.query_step()
        steps_0 = []
        for ii in _steps_0:
            if ii['phase'] == 'Succeeded':
                steps_0.append(ii)
        
        # update the output artifact of iter-000000--prep-run-fp
        fpout_idx = None
        for idx, ii in enumerate(steps_0):
            if ii.key is not None and ii.key == 'iter-000000--prep-run-fp':
                fpout_idx = idx
        self.assertTrue(fpout_idx is not None)
        step_fp = steps_0.pop(fpout_idx)

        self.assertEqual(step_fp['phase'], 'Succeeded')
        download_artifact(step_fp.outputs.artifacts['labeled_data'], path='failed_res')
        for modi_file in [
                Path('failed_res')/'task.000000'/'data_task.000000'/'data',
                Path('failed_res')/'task.000001'/'data_task.000001'/'data',
        ]:
            fc = modi_file.read_text()
            fc = 'modified\n' + fc
            modi_file.write_text(fc)
        os.chdir('failed_res')
        new_arti = upload_artifact([
            'task.000000/data_task.000000', 
            'task.000001/data_task.000001',
        ])
        step_fp.modify_output_artifact('labeled_data', new_arti)
        os.chdir('..')
        steps_0.append(step_fp)
        
        # update the output artifact of init--scheduler
        scheduler_idx = None
        for idx, ii in enumerate(steps_0):
            if ii.key is not None and ii.key == 'init--scheduler':
                scheduler_idx = idx
        self.assertTrue(scheduler_idx is not None)
        step_scheduler = steps_0.pop(scheduler_idx)
        self.assertEqual(step_scheduler['phase'], 'Succeeded')
        # old_scheduler = jsonpickle.decode(step_scheduler.outputs.parameters['exploration_scheduler'].value)
        old_scheduler = load_object_from_file(            
            download_artifact(step_scheduler.outputs.artifacts['exploration_scheduler'])[0])
        self.assertEqual(old_scheduler.get_stage(), 0)
        # update a stage scheduler
        old_scheduler.stage_schedulers[1] = self.scheduler_1.stage_schedulers[1]        
        # step_scheduler.modify_output_parameter("exploration_scheduler", old_scheduler)
        old_scheduler_artifact = upload_artifact(
            dump_object_to_file(old_scheduler, 'scheduler_new.dat'))
        step_scheduler.modify_output_artifact("exploration_scheduler", old_scheduler_artifact)
        steps_0.append(step_scheduler)

        dpgen_step_1 = Step(
            'dpgen-step', 
            template = self.dpgen_op_1,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_1 = Workflow(name="dpgen")
        wf_1.add(dpgen_step_1)
        wf_1.submit(reuse_step = steps_0)
        id_1 = wf_1.id
        
        while wf_1.query_status() in ["Pending", "Running"]:
            time.sleep(2)
        self.assertEqual(wf_1.query_status(), "Succeeded")

        step = wf_1.query_step(name='dpgen-step')[0]
        self.assertEqual(step.phase, "Succeeded")
        download_artifact(step.outputs.artifacts["iter_data"], path = 'iter_data')
        download_artifact(step.outputs.artifacts["models"], path = Path('models')/self.name)
        # scheduler = jsonpickle.decode(step.outputs.parameters['exploration_scheduler'].value)
        scheduler = load_object_from_file(
            download_artifact(step.outputs.artifacts["exploration_scheduler"])[0])
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 1)
        
        # # we know number of selected data is 2
        # # by MockedConfSelector
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000000'/
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join([
                    'restart',
                    'modified',
                    'labeled_data of '+vasp_task_pattern%ii,
                    f'select conf.{ii}',
                    f'mocked conf {ii}',
                    f'mocked input {ii}',
                ]).strip()
            )
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000001'/\
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join([
                    'restart',
                    'labeled_data of '+vasp_task_pattern%ii,
                    f'select conf.{ii}',
                    f'mocked 2 conf {ii}',
                    f'mocked 2 input {ii}',
                ]).strip()
            )


    def test_update_slice_item_input(self):
        # failed at run-fp-000001
        # revise the item 1 of the output of prep-fp
        # restart the workflow with restarted run_fp, the same scheduler

        dpgen_step_0 = Step(
            'dpgen-step', 
            template = self.dpgen_op_2,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_0 = Workflow(name="dpgen")
        wf_0.add(dpgen_step_0)
        wf_0.submit()
        id_0 = wf_0.id

        # wf_0 = Workflow(id='dpgen-skxp5')

        while wf_0.query_status() in ["Pending", "Running"]:
            time.sleep(2)
        self.assertEqual(wf_0.query_status(), "Failed")

        _steps_0 = wf_0.query_step()
        steps_0 = []
        for ii in _steps_0:
            if ii['phase'] == 'Succeeded':
                steps_0.append(ii)
        
        fpout_idx = None
        for idx, ii in enumerate(steps_0):
            if ii.key is not None and ii.key == 'iter-000000--prep-fp':
                fpout_idx = idx
        self.assertTrue(fpout_idx is not None)
        step_fp = steps_0.pop(fpout_idx)

        self.assertEqual(step_fp['phase'], 'Succeeded')
        download_artifact(step_fp.outputs.artifacts['task_paths'], path='failed_res')
        for modi_file in [
                Path('failed_res')/'task.000001'/'POSCAR',
        ]:
            fc = modi_file.read_text()
            fc = 'modified\n' + fc
            modi_file.write_text(fc)
        os.chdir('failed_res')
        new_arti = upload_artifact([
            'task.000000/', 
            'task.000001/',
        ])
        step_fp.modify_output_artifact('task_paths', new_arti)
        os.chdir('..')
        steps_0.append(step_fp)

        # keys_0 =  []
        # for ii in steps_0:
        #     if ii.key is not None:
        #         keys_0.append(ii.key)
        # keys_0.sort()
        # print(keys_0)

        dpgen_step_1 = Step(
            'dpgen-step', 
            template = self.dpgen_op_3,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_1 = Workflow(name="dpgen")
        wf_1.add(dpgen_step_1)
        wf_1.submit(reuse_step = steps_0)
        id_1 = wf_1.id
        
        while wf_1.query_status() in ["Pending", "Running"]:
            time.sleep(2)
        self.assertEqual(wf_1.query_status(), "Succeeded")

        step = wf_1.query_step(name='dpgen-step')[0]
        self.assertEqual(step.phase, "Succeeded")
        download_artifact(step.outputs.artifacts["iter_data"], path = 'iter_data')
        download_artifact(step.outputs.artifacts["models"], path = Path('models')/self.name)
        # scheduler = jsonpickle.decode(step.outputs.parameters['exploration_scheduler'].value)
        scheduler = load_object_from_file(
            download_artifact(step.outputs.artifacts["exploration_scheduler"])[0])
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 1)
        
        # # we know number of selected data is 2
        # # by MockedConfSelector
        for ii in range(mocked_numb_select):
            if ii == 0:
                self.assertEqual(
                    (Path('iter_data')/'iter-000000'/
                     ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                    '\n'.join([
                        'labeled_data of '+vasp_task_pattern%ii,
                        f'select conf.{ii}',
                        f'mocked conf {ii}',
                        f'mocked input {ii}',
                    ]).strip()
                )
            else :
                self.assertEqual(
                    (Path('iter_data')/'iter-000000'/
                     ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                    '\n'.join([
                        'restarted',
                        'labeled_data of '+vasp_task_pattern%ii,
                        'modified',
                        f'select conf.{ii}',
                        f'mocked conf {ii}',
                        f'mocked input {ii}',
                    ]).strip()
                )
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000001'/\
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join([
                    'restarted',
                    'labeled_data of '+vasp_task_pattern%ii,
                    f'select conf.{ii}',
                    f'mocked 1 conf {ii}',
                    f'mocked 1 input {ii}',
                ]).strip()
            )


    def test_update_slice_item_output(self):        
        # failed at collect_data
        # revise the output of run-fp-000001
        # restart the workflow with restarted collect_data, the same scheduler

        dpgen_step_0 = Step(
            'dpgen-step', 
            template = self.dpgen_op_0,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_0 = Workflow(name="dpgen")
        wf_0.add(dpgen_step_0)
        wf_0.submit()
        id_0 = wf_0.id

        # wf_0 = Workflow(id='dpgen-nmgsw')

        while wf_0.query_status() in ["Pending", "Running"]:
            time.sleep(4)
        self.assertEqual(wf_0.query_status(), "Failed")

        steps_0 = self.get_restart_step(wf_0)
        # for ii in steps_0:
        #     print(ii.key)
        
        fpout_idx = None
        for idx, ii in enumerate(steps_0):
            if ii.key is not None and ii.key == 'iter-000000--run-fp-000001':
                fpout_idx = idx
        self.assertTrue(fpout_idx is not None)
        step_fp = steps_0.pop(fpout_idx)

        self.assertEqual(step_fp['phase'], 'Succeeded')
        download_artifact(step_fp.outputs.artifacts['labeled_data'], path='failed_res')
        for modi_file in [
                Path('failed_res')/'task.000001'/'data_task.000001'/'data',
        ]:
            fc = modi_file.read_text()
            fc = 'modified\n' + fc
            modi_file.write_text(fc)
        os.chdir('failed_res')
        new_arti = upload_artifact([
            None,
            'task.000001/data_task.000001',
        ], archive = None)
        step_fp.modify_output_artifact('labeled_data', new_arti)
        os.chdir('..')
        steps_0.append(step_fp)

        # keys_0 =  []
        # for ii in steps_0:
        #     if ii.key is not None:
        #         keys_0.append(ii.key)
        # keys_0.sort()
        # print(keys_0)

        dpgen_step_1 = Step(
            'dpgen-step', 
            template = self.dpgen_op_1,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_1 = Workflow(name="dpgen")
        wf_1.add(dpgen_step_1)
        wf_1.submit(reuse_step = steps_0)
        id_1 = wf_1.id
        
        while wf_1.query_status() in ["Pending", "Running"]:
            time.sleep(2)
        self.assertEqual(wf_1.query_status(), "Succeeded")

        step = wf_1.query_step(name='dpgen-step')[0]
        self.assertEqual(step.phase, "Succeeded")
        download_artifact(step.outputs.artifacts["iter_data"], path = 'iter_data')
        download_artifact(step.outputs.artifacts["models"], path = Path('models')/self.name)
        # scheduler = jsonpickle.decode(step.outputs.parameters['exploration_scheduler'].value)
        scheduler = load_object_from_file(
            download_artifact(step.outputs.artifacts["exploration_scheduler"])[0])
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 1)
        
        # # we know number of selected data is 2
        # # by MockedConfSelector
        for ii in range(mocked_numb_select):
            if ii == 0:
                self.assertEqual(
                    (Path('iter_data')/'iter-000000'/
                     ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                    '\n'.join([
                        'restart',
                        'labeled_data of '+vasp_task_pattern%ii,
                        f'select conf.{ii}',
                        f'mocked conf {ii}',
                        f'mocked input {ii}',
                    ]).strip()
                )
            elif ii == 1:
                self.assertEqual(
                    (Path('iter_data')/'iter-000000'/
                     ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                    '\n'.join([
                        'restart',
                        'modified',
                        'labeled_data of '+vasp_task_pattern%ii,
                        f'select conf.{ii}',
                        f'mocked conf {ii}',
                        f'mocked input {ii}',
                    ]).strip()
                )
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000001'/\
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join([
                    'restart',
                    'labeled_data of '+vasp_task_pattern%ii,
                    f'select conf.{ii}',
                    f'mocked 1 conf {ii}',
                    f'mocked 1 input {ii}',
                ]).strip()
            )



    def test_update_slice_item_output_1(self):        
        # failed at collect_data
        # revise the output of run-fp-000001
        # restart the workflow with restarted collect_data, the same scheduler

        dpgen_step_0 = Step(
            'dpgen-step', 
            template = self.dpgen_op_0,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_0 = Workflow(name="dpgen")
        wf_0.add(dpgen_step_0)
        wf_0.submit()
        id_0 = wf_0.id

        # wf_0 = Workflow(id='dpgen-nmgsw')

        while wf_0.query_status() in ["Pending", "Running"]:
            time.sleep(4)
        self.assertEqual(wf_0.query_status(), "Failed")

        steps_0 = self.get_restart_step(wf_0)
        # for ii in steps_0:
        #     print(ii.key)
        
        fpout_idx = None
        for idx, ii in enumerate(steps_0):
            if ii.key is not None and ii.key == 'iter-000000--run-fp-000001':
                fpout_idx = idx
        self.assertTrue(fpout_idx is not None)
        step_fp = steps_0.pop(fpout_idx)

        self.assertEqual(step_fp['phase'], 'Succeeded')
        step_fp.download_sliced_output_artifact('labeled_data', path='failed_res')
        for modi_file in [
                Path('failed_res')/'task.000001'/'data_task.000001'/'data',
        ]:
            fc = modi_file.read_text()
            fc = 'modified\n' + fc
            modi_file.write_text(fc)
        os.chdir('failed_res')
        step_fp.upload_and_modify_sliced_output_artifact(
            'labeled_data', 
            ['task.000001/data_task.000001']
        )
        os.chdir('..')
        steps_0.append(step_fp)

        # keys_0 =  []
        # for ii in steps_0:
        #     if ii.key is not None:
        #         keys_0.append(ii.key)
        # keys_0.sort()
        # print(keys_0)

        dpgen_step_1 = Step(
            'dpgen-step', 
            template = self.dpgen_op_1,
            parameters = {
                "type_map" : self.type_map,
                "numb_models" : self.numb_models,
                "template_script" : self.template_script,
                "train_config" : {},
                "lmp_config" : {},
                "fp_config" : {},
            },
            artifacts = {
                "exploration_scheduler" : self.scheduler_0_artifact,
                'fp_inputs' : self.vasp_inputs_arti,
                "init_models" : self.init_models,
                "init_data" : self.init_data,
                "iter_data" : self.iter_data,
            },
        )
        
        wf_1 = Workflow(name="dpgen")
        wf_1.add(dpgen_step_1)
        wf_1.submit(reuse_step = steps_0)
        id_1 = wf_1.id
        
        while wf_1.query_status() in ["Pending", "Running"]:
            time.sleep(2)
        self.assertEqual(wf_1.query_status(), "Succeeded")

        step = wf_1.query_step(name='dpgen-step')[0]
        self.assertEqual(step.phase, "Succeeded")
        download_artifact(step.outputs.artifacts["iter_data"], path = 'iter_data')
        download_artifact(step.outputs.artifacts["models"], path = Path('models')/self.name)
        # scheduler = jsonpickle.decode(step.outputs.parameters['exploration_scheduler'].value)
        scheduler = load_object_from_file(
            download_artifact(step.outputs.artifacts["exploration_scheduler"])[0])
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 1)
        
        # # we know number of selected data is 2
        # # by MockedConfSelector
        for ii in range(mocked_numb_select):
            if ii == 0:
                self.assertEqual(
                    (Path('iter_data')/'iter-000000'/
                     ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                    '\n'.join([
                        'restart',
                        'labeled_data of '+vasp_task_pattern%ii,
                        f'select conf.{ii}',
                        f'mocked conf {ii}',
                        f'mocked input {ii}',
                    ]).strip()
                )
            elif ii == 1:
                self.assertEqual(
                    (Path('iter_data')/'iter-000000'/
                     ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                    '\n'.join([
                        'restart',
                        'modified',
                        'labeled_data of '+vasp_task_pattern%ii,
                        f'select conf.{ii}',
                        f'mocked conf {ii}',
                        f'mocked input {ii}',
                    ]).strip()
                )
        for ii in range(mocked_numb_select):
            self.assertEqual(
                (Path('iter_data')/'iter-000001'/\
                 ('data_'+vasp_task_pattern%ii)/'data').read_text().strip(),
                '\n'.join([
                    'restart',
                    'labeled_data of '+vasp_task_pattern%ii,
                    f'select conf.{ii}',
                    f'mocked 1 conf {ii}',
                    f'mocked 1 input {ii}',
                ]).strip()
            )
                           

