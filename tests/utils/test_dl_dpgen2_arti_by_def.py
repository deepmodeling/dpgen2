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
    DownloadDefinition,
    _get_all_iterations,
    _get_all_step_defs,
    download_dpgen2_artifacts_by_def,
    print_op_download_setting,
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
        if key == sorted(self.keys):
            return [MockedStep(), MockedStep(), MockedStep()]
        else:
            return [MockedStep() for kk in key]

    def query_keys_of_steps(self):
        return self.keys


class TestDownloadDpgen2Artifact(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree("foo", ignore_errors=True)

    def test_get_all_iterations(self):
        step_keys = [
            "init--scheduler",
            "iter-000000--foo",
            "iter-000000--bar",
            "iter-000002--tar",
        ]
        iterations = _get_all_iterations(step_keys)
        self.assertEqual(iterations, [0, 2])

    def test_get_step_defs(self):
        setting = {
            "foo": DownloadDefinition()
            .add_input("i0")
            .add_input("i1")
            .add_output("o0"),
            "bar": DownloadDefinition().add_output("o0").add_output("o1"),
        }
        expected = [
            "foo/input/i0",
            "foo/input/i1",
            "foo/output/o0",
            "bar/output/o0",
            "bar/output/o1",
        ]
        step_defs = _get_all_step_defs(setting)
        self.assertEqual(step_defs, expected)

    @mock.patch("dpgen2.utils.download_dpgen2_artifacts.download_artifact")
    def test_download(self, mocked_dl):
        with self.assertLogs(level="WARN") as log:
            download_dpgen2_artifacts_by_def(
                Mockedwf(),
                iterations=[0, 1, 2],
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
            "cannot find download settings for prep-run-lmp/input/foo",
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
    def test_download_empty(self, mocked_dl):
        with self.assertLogs(level="WARN") as log:
            download_dpgen2_artifacts_by_def(
                Mockedwf(),
                iterations=[0, 1, 2],
                step_defs=[
                    "foo/input/init_models",
                    "prep-run-train/output/bar",
                ],
                prefix="foo",
                chk_pnt=False,
            )
        self.assertEqual(len(log.output), 2)
        self.assertEqual(len(log.records), 2)
        self.assertIn(
            "cannot find download settings for foo/input/init_models",
            log.output[0],
        )
        self.assertIn(
            "cannot find download settings for prep-run-train/output/bar",
            log.output[1],
        )
        expected = []
        self.assertEqual(len(mocked_dl.call_args_list), len(expected))
        for ii, jj in zip(mocked_dl.call_args_list, expected):
            self.assertEqual(ii, jj)

    @mock.patch("dpgen2.utils.download_dpgen2_artifacts.download_artifact")
    def test_download_with_ckpt(self, mocked_dl):
        with self.assertLogs(level="WARN") as log:
            download_dpgen2_artifacts_by_def(
                Mockedwf(),
                iterations=[0, 1, 2],
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
            "cannot find download settings for prep-run-lmp/input/foo",
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
            iterations=[0, 1],
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
            "foo": DownloadDefinition()
            .add_input("i0")
            .add_input("i1")
            .add_output("o0"),
            "bar": DownloadDefinition().add_output("o0").add_output("o1"),
        }
        ret = print_op_download_setting(setting)

        expected = textwrap.dedent(
            """step: foo
  input:
    i0 i1
  output:
    o0
step: bar
  output:
    o0 o1
"""
        )
        self.assertEqual(ret.rstrip(), expected.rstrip())
