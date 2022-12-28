from utils.context import dpgen2
import numpy as np
import unittest, json, shutil, os
import random
import tempfile
import dpdata
import dflow
from pathlib import Path
from dpgen2.utils.download_dpgen2_artifacts import (
    download_dpgen2_artifacts,
)
from dpgen2.entrypoint.watch import (
    update_finished_steps,
)

import mock

class MockedArti:
    def __getitem__(
            self,
            key,
    ):
        return 'arti-' + key

class MockedDef:
    artifacts = MockedArti()

class MockedStep:
    inputs = MockedDef()
    outputs = MockedDef()

class Mockedwf:    
    keys = ['iter-0--prep-run-train',]
    def query_step(self, key):
        return [MockedStep()]

    def query_keys_of_steps(self):
        return self.keys


class TestDownloadDpgen2Artifact(unittest.TestCase):
    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_train_download(self, mocked_dl):
        download_dpgen2_artifacts(Mockedwf(), 'iter-000000--prep-run-train', 'foo')
        expected = [
            mock.call("arti-init_models", path=Path("foo/iter-000000/prep-run-train/inputs"), skip_exists=True),
            mock.call("arti-init_data", path=Path("foo/iter-000000/prep-run-train/inputs"), skip_exists=True),
            mock.call("arti-iter_data", path=Path("foo/iter-000000/prep-run-train/inputs"), skip_exists=True),
            mock.call("arti-scripts", path=Path("foo/iter-000000/prep-run-train/outputs"), skip_exists=True),
            mock.call("arti-models", path=Path("foo/iter-000000/prep-run-train/outputs"), skip_exists=True),
            mock.call("arti-logs", path=Path("foo/iter-000000/prep-run-train/outputs"), skip_exists=True),
            mock.call("arti-lcurves", path=Path("foo/iter-000000/prep-run-train/outputs"), skip_exists=True),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)

    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_lmp_download(self, mocked_dl):
        download_dpgen2_artifacts(Mockedwf(), 'iter-000001--prep-run-lmp', None)
        expected = [
            mock.call("arti-logs", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
            mock.call("arti-trajs", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
            mock.call("arti-model_devis", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)

    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_fp_download(self, mocked_dl):
        download_dpgen2_artifacts(Mockedwf(), 'iter-000001--prep-run-fp', None)
        expected = [
            mock.call("arti-confs", path=Path("iter-000001/prep-run-fp/inputs"), skip_exists=True),
            mock.call("arti-logs", path=Path("iter-000001/prep-run-fp/outputs"), skip_exists=True),
            mock.call("arti-labeled_data", path=Path("iter-000001/prep-run-fp/outputs"), skip_exists=True),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)


    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_fp_download_chkpnt(self, mocked_dl):
        if Path('iter-000001').exists():
            shutil.rmtree('iter-000001')
        Path("iter-000001/prep-run-fp/inputs").mkdir(parents=True, exist_ok=True)
        Path("iter-000001/prep-run-fp/outputs").mkdir(parents=True, exist_ok=True)
        download_dpgen2_artifacts(Mockedwf(), 'iter-000001--prep-run-fp', None, chk_pnt=True)
        expected = [
            mock.call("arti-confs", path=Path("iter-000001/prep-run-fp/inputs"), skip_exists=True),
            mock.call("arti-logs", path=Path("iter-000001/prep-run-fp/outputs"), skip_exists=True),
            mock.call("arti-labeled_data", path=Path("iter-000001/prep-run-fp/outputs"), skip_exists=True),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)        
        self.assertTrue(Path("iter-000001/prep-run-fp/inputs/done").is_file())
        self.assertTrue(Path("iter-000001/prep-run-fp/outputs/done").is_file())

        download_dpgen2_artifacts(Mockedwf(), 'iter-000001--prep-run-fp', None, chk_pnt=True)
        expected = [
            mock.call("arti-confs", path=Path("iter-000001/prep-run-fp/inputs"), skip_exists=True),
            mock.call("arti-logs", path=Path("iter-000001/prep-run-fp/outputs"), skip_exists=True),
            mock.call("arti-labeled_data", path=Path("iter-000001/prep-run-fp/outputs"), skip_exists=True),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)        
        if Path('iter-000001').exists():
            shutil.rmtree('iter-000001')

    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_empty_download(self, mocked_dl):
        download_dpgen2_artifacts(Mockedwf(), 'iter-000001--foo', None)
        expected = [
        ]
        self.assertEqual(mocked_dl.call_args_list, expected)

        
    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_update_finished_steps_exist_steps(self, mocked_dl):
        wf = Mockedwf()
        wf.keys = ['iter-000000--prep-run-train', 'iter-000001--prep-run-lmp']
        finished_keys = update_finished_steps(wf, ['iter-000000--prep-run-train'], True)
        self.assertEqual(finished_keys, wf.keys)
        expected = [
            mock.call("arti-logs", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
            mock.call("arti-trajs", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
            mock.call("arti-model_devis", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)

    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_update_finished_steps_none_steps(self, mocked_dl):
        wf = Mockedwf()
        wf.keys = ['iter-000000--prep-run-train', 'iter-000001--prep-run-lmp']
        finished_keys = update_finished_steps(wf, None, True)
        self.assertEqual(finished_keys, wf.keys)
        expected = [
            mock.call("arti-init_models", path=Path("iter-000000/prep-run-train/inputs"), skip_exists=True),
            mock.call("arti-init_data", path=Path("iter-000000/prep-run-train/inputs"), skip_exists=True),
            mock.call("arti-iter_data", path=Path("iter-000000/prep-run-train/inputs"), skip_exists=True),
            mock.call("arti-scripts", path=Path("iter-000000/prep-run-train/outputs"), skip_exists=True),
            mock.call("arti-models", path=Path("iter-000000/prep-run-train/outputs"), skip_exists=True),
            mock.call("arti-logs", path=Path("iter-000000/prep-run-train/outputs"), skip_exists=True),
            mock.call("arti-lcurves", path=Path("iter-000000/prep-run-train/outputs"), skip_exists=True),
            mock.call("arti-logs", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
            mock.call("arti-trajs", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
            mock.call("arti-model_devis", path=Path("iter-000001/prep-run-lmp/outputs"), skip_exists=True),
        ]
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)
        
