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
    vasp_conf_name,
    vasp_input_name,
    vasp_pot_name,
    vasp_kp_name,
    vasp_default_log_name,
    vasp_default_out_data_name,
)
from dpgen2.op.run_vasp import RunVasp


class TestRunVasp(unittest.TestCase):
    def setUp(self):
        self.task_path = Path('task/path')
        self.task_path.mkdir(parents=True, exist_ok=True)
        (self.task_path/vasp_conf_name).write_text('foo')
        (self.task_path/vasp_input_name).write_text('bar')
        (self.task_path/vasp_pot_name).write_text('dee')
        (self.task_path/vasp_kp_name).write_text('por')
        self.task_name = 'task_000'
        
    def tearDown(self):
        if Path('task').is_dir():
            shutil.rmtree('task')
        if Path(self.task_name).is_dir():
            shutil.rmtree(self.task_name)

    @patch('dpgen2.op.run_vasp.run_command')
    def test_success(self, mocked_run):
        mocked_run.side_effect = [ (0, 'foo\n', '') ]
        op = RunVasp()
        def new_to(obj, foo, bar):
            data_path = Path('data')
            data_path.mkdir()
            (data_path/'foo').write_text('bar')
        def new_init(obj, foo):
            pass
        with mock.patch.object(dpgen2.op.run_vasp.dpdata.LabeledSystem, 
                               'to', 
                               new=new_to):
            with mock.patch.object(dpgen2.op.run_vasp.dpdata.LabeledSystem, 
                                   '__init__', 
                                   new=new_init):
                out = op.execute(
                    OPIO({
                        'config' : {'command' : 'myvasp',
                                    'log': 'foo.log', 
                                    'out':'data',
                                    },
                        'task_name' : self.task_name,
                        'task_path' : self.task_path,
                    }))
        work_dir = Path(self.task_name)
        # check output
        self.assertEqual(out['log'], work_dir/'foo.log')
        self.assertEqual(out['labeled_data'], work_dir/'data')
        # check call
        calls = [
            call(' '.join(['myvasp', '>', 'foo.log']), shell=True),
        ]
        mocked_run.assert_has_calls(calls)
        # check input files are correctly linked
        self.assertEqual((work_dir/vasp_conf_name).read_text(), 'foo')
        self.assertEqual((work_dir/vasp_input_name).read_text(), 'bar')
        self.assertEqual((work_dir/vasp_pot_name).read_text(), 'dee')
        self.assertEqual((work_dir/vasp_kp_name).read_text(), 'por')
        # check output
        self.assertEqual((Path(self.task_name)/'data'/'foo').read_text(), 'bar')


    @patch('dpgen2.op.run_vasp.run_command')
    def test_success(self, mocked_run):
        mocked_run.side_effect = [ (0, 'foo\n', '') ]
        op = RunVasp()
        def new_to(obj, foo, bar):
            data_path = Path('data')
            data_path.mkdir()
            (data_path/'foo').write_text('bar')
        def new_init(obj, foo):
            pass
        with mock.patch.object(dpgen2.op.run_vasp.dpdata.LabeledSystem, 
                               'to', 
                               new=new_to):
            with mock.patch.object(dpgen2.op.run_vasp.dpdata.LabeledSystem, 
                                   '__init__', 
                                   new=new_init):
                out = op.execute(
                    OPIO({
                        'config' : {'command' : 'myvasp',
                                    },
                        'task_name' : self.task_name,
                        'task_path' : self.task_path,
                    }))
        work_dir = Path(self.task_name)
        # check output
        self.assertEqual(out['log'], work_dir/vasp_default_log_name)
        self.assertEqual(out['labeled_data'], work_dir/vasp_default_out_data_name)
        # check call
        calls = [
            call(' '.join(['myvasp', '>', vasp_default_log_name]), shell=True),
        ]
        mocked_run.assert_has_calls(calls)
        # check input files are correctly linked
        self.assertEqual((work_dir/vasp_conf_name).read_text(), 'foo')
        self.assertEqual((work_dir/vasp_input_name).read_text(), 'bar')
        self.assertEqual((work_dir/vasp_pot_name).read_text(), 'dee')
        self.assertEqual((work_dir/vasp_kp_name).read_text(), 'por')
        # check output
        self.assertEqual((Path(self.task_name)/'data'/'foo').read_text(), 'bar')
    

    @patch('dpgen2.op.run_vasp.run_command')
    def test_error(self, mocked_run):
        mocked_run.side_effect = [ (1, 'foo\n', '') ]
        op = RunVasp()
        with self.assertRaises(TransientError) as ee:
            out = op.execute(
                OPIO({
                    'config' : {'command' : 'myvasp'},
                    'task_name' : self.task_name,
                    'task_path' : self.task_path,
                }))
        # check call
        calls = [
            call(' '.join(['myvasp', '>', vasp_default_log_name]), shell=True),
        ]
        mocked_run.assert_has_calls(calls)
                        
        
