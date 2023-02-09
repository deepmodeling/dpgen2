import json
import os
import random
import shutil
import tempfile
import textwrap
import unittest
from pathlib import (
    Path,
)

import dflow
import dpdata
import mock
import numpy as np
from utils.context import (
    dpgen2,
)

from dpgen2.utils.download_dpgen2_artifacts import (
    download_dpgen2_artifacts_by_def,
    print_op_download_setting,
    DownloadDefinition,
)


class MockedArti:
    def __getitem__(
        self,
        key,
    ):
        return "arti-" + key


class MockedDef:
    artifacts = MockedArti()


class MockedStep:
    inputs = MockedDef()
    outputs = MockedDef()

    def __getitem__(
        self,
        kk,
    ):
        return "Succeeded"

class Mockedwf:
    keys = [
        "iter-000000--prep-run-train",
        "iter-000001--prep-run-train",
        "iter-000000--prep-run-lmp",
    ]

    def query_step_by_key(self, key):
        if (key == sorted(self.keys)):
            return [MockedStep(), MockedStep(), MockedStep()]
        else:
            return [MockedStep() for kk in key]

    def query_keys_of_steps(self):
        return self.keys

class TestDownloadDpgen2Artifact(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree("foo", ignore_errors=True)

    @mock.patch("dpgen2.utils.download_dpgen2_artifacts.download_artifact")
    def test_download(self, mocked_dl):
        with self.assertLogs(level='WARN') as log:
            download_dpgen2_artifacts_by_def(
                Mockedwf(), 
                iterations=[0,1,2],
                step_defs=[
                    "prep-run-train/input/init_models",
                    "prep-run-train/output/logs",
                    "prep-run-lmp/input/foo",
                    "prep-run-lmp/output/trajs",
                ],
                prefix="foo",
                chk_pnt=False,
            )
        self.assertEqual(len(log.output), 1)
        self.assertEqual(len(log.records), 1)
        self.assertIn(
            'cannot find download settings for prep-run-lmp/input/foo',
            log.output[0],
        )
        expected = [
            mock.call(
                "arti-init_models",
                path=Path("foo/iter-000000/prep-run-train/input/init_models"),
                skip_exists=True,
            ),
            mock.call(
                "arti-logs",
                path=Path("foo/iter-000000/prep-run-train/output/logs"),
                skip_exists=True,
            ),
            mock.call(
                "arti-trajs",
                path=Path("foo/iter-000000/prep-run-lmp/output/trajs"),
                skip_exists=True,
            ),
            mock.call(
                "arti-init_models",
                path=Path("foo/iter-000001/prep-run-train/input/init_models"),
                skip_exists=True,
            ),
            mock.call(
                "arti-logs",
                path=Path("foo/iter-000001/prep-run-train/output/logs"),
                skip_exists=True,
            ),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii, jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii, jj)


    @mock.patch("dpgen2.utils.download_dpgen2_artifacts.download_artifact")
    def test_download_with_ckpt(self, mocked_dl):
        with self.assertLogs(level='WARN') as log:
            download_dpgen2_artifacts_by_def(
                Mockedwf(), 
                iterations=[0,1,2],
                step_defs=[
                    "prep-run-train/input/init_models",
                    "prep-run-train/output/logs",
                    "prep-run-lmp/input/foo",
                    "prep-run-lmp/output/trajs",
                ],
                prefix="foo",
                chk_pnt=True,
            )
        self.assertEqual(len(log.output), 1)
        self.assertEqual(len(log.records), 1)
        self.assertIn(
            'cannot find download settings for prep-run-lmp/input/foo',
            log.output[0],
        )
        expected = [
            mock.call(
                "arti-init_models",
                path=Path("foo/iter-000000/prep-run-train/input/init_models"),
                skip_exists=True,
            ),
            mock.call(
                "arti-logs",
                path=Path("foo/iter-000000/prep-run-train/output/logs"),
                skip_exists=True,
            ),
            mock.call(
                "arti-trajs",
                path=Path("foo/iter-000000/prep-run-lmp/output/trajs"),
                skip_exists=True,
            ),
            mock.call(
                "arti-init_models",
                path=Path("foo/iter-000001/prep-run-train/input/init_models"),
                skip_exists=True,
            ),
            mock.call(
                "arti-logs",
                path=Path("foo/iter-000001/prep-run-train/output/logs"),
                skip_exists=True,
            ),
        ]
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii, jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii, jj)
        
        download_dpgen2_artifacts_by_def(
            Mockedwf(), 
            iterations=[0,1],
            step_defs=[
                "prep-run-train/input/init_models",
                "prep-run-train/output/logs",
                "prep-run-lmp/output/trajs",
                "prep-run-lmp/output/model_devis",
            ],
            prefix="foo",
            chk_pnt=True,
        )
        expected = [
            mock.call(
                "arti-model_devis",
                path=Path("foo/iter-000000/prep-run-lmp/output/model_devis"),
                skip_exists=True,
            ),
        ]
        self.assertEqual(len(mocked_dl.call_args_list[5:]), len(expected))
        for ii, jj in zip(mocked_dl.call_args_list[5:], expected):
            self.assertEqual(ii, jj)


    def test_print_op_dld_setting(self):
        setting = {
            "foo" : DownloadDefinition()
            .add_input("i0")
            .add_input("i1")
            .add_output("o0"),
            "bar" : DownloadDefinition()
            .add_output("o0")
            .add_output("o1"),
        }
        ret = print_op_download_setting(setting)

        expected = textwrap.dedent("""step: foo
  input:
    i0 i1
  output:
    o0
step: bar
  output:
    o0 o1
""")
        self.assertEqual(ret.rstrip(), expected.rstrip())
