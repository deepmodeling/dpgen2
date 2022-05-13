from op.context import dpgen2
import numpy as np
import unittest, json, shutil
from mock import mock, patch, call
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    TransientError,
)
from pathlib import Path
from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_traj_name,
    lmp_model_devi_name,
    model_name_pattern,
)
from dpgen2.op.run_lmp import RunLmp


class TestRunLmp(unittest.TestCase):
    def setUp(self):
        self.task_path = Path('task/path')
        self.task_path.mkdir(parents=True, exist_ok=True)
        self.model_path = Path('models/path')
        self.model_path.mkdir(parents=True, exist_ok=True)
        (self.task_path/lmp_conf_name).write_text('foo')
        (self.task_path/lmp_input_name).write_text('bar')
        self.task_name = 'task_000'
        self.models = [self.model_path/Path(f'model_{ii}.pb') for ii in range(4)]
        for idx,ii in enumerate(self.models):
            ii.write_text(f'model{idx}')
        
    def tearDown(self):
        if Path('task').is_dir():
            shutil.rmtree('task')
        if Path('models').is_dir():
            shutil.rmtree('models')
        if Path(self.task_name).is_dir():
            shutil.rmtree(self.task_name)

    @patch('dpgen2.op.run_lmp.run_command')
    def test_success(self, mocked_run):
        mocked_run.side_effect = [ (0, 'foo\n', '') ]
        op = RunLmp()
        out = op.execute(
            OPIO({
                'config' : {'command' : 'mylmp'},
                'task_name' : self.task_name,
                'task_path' : self.task_path,
                'models' : self.models,
            }))
        work_dir = Path(self.task_name)
        # check output
        self.assertEqual(out['log'], work_dir/lmp_log_name)
        self.assertEqual(out['traj'], work_dir/lmp_traj_name)
        self.assertEqual(out['model_devi'], work_dir/lmp_model_devi_name)
        # check call
        calls = [
            call(' '.join(['mylmp', '-i', lmp_input_name, '-log', lmp_log_name]), shell=True),
        ]
        mocked_run.assert_has_calls(calls)
        # check input files are correctly linked
        self.assertEqual((work_dir/lmp_conf_name).read_text(), 'foo')
        self.assertEqual((work_dir/lmp_input_name).read_text(), 'bar')
        for ii in range(4):
            self.assertEqual((work_dir/(model_name_pattern%ii)).read_text(), f'model{ii}')

    
    @patch('dpgen2.op.run_lmp.run_command')
    def test_error(self, mocked_run):
        mocked_run.side_effect = [ (1, 'foo\n', '') ]
        op = RunLmp()
        with self.assertRaises(TransientError) as ee:
            out = op.execute(
                OPIO({
                    'config' : {'command' : 'mylmp'},
                    'task_name' : self.task_name,
                    'task_path' : self.task_path,
                    'models' : self.models,
                }))
        # check call
        calls = [
            call(' '.join(['mylmp', '-i', lmp_input_name, '-log', lmp_log_name]), shell=True),
        ]
        mocked_run.assert_has_calls(calls)
                        
        
