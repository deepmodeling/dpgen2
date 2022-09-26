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
    def query_step(self, key):
        return [MockedStep()]

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
        for ii,jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii,jj)


    @mock.patch('dpgen2.utils.download_dpgen2_artifacts.download_artifact')
    def test_empty_download(self, mocked_dl):
        download_dpgen2_artifacts(Mockedwf(), 'iter-000001--foo', None)
        expected = [
        ]
        self.assertEqual(mocked_dl.call_args_list, expected)
