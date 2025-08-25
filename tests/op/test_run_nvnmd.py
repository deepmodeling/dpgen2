import json
import os
import shutil
import unittest
from pathlib import (
    Path,
)

import dpdata
import numpy as np
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    TransientError,
)
from mock import (
    call,
    mock,
    patch,
)

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_model_devi_name,
    lmp_traj_name,
    model_name_pattern,
)
from dpgen2.op.run_lmp import get_ele_temp, set_models
from dpgen2.op.run_nvnmd import (
    RunNvNMD,
    merge_pimd_files,
)
from dpgen2.utils import (
    BinaryFileInput,
)

# isort: on


class TestRunNvNMD(unittest.TestCase):
    def setUp(self):
        self.task_path = Path("task/path")
        self.task_path.mkdir(parents=True, exist_ok=True)
        self.model_path = Path("models/path")
        self.model_path.mkdir(parents=True, exist_ok=True)
        (self.task_path / lmp_conf_name).write_text("foo")
        (self.task_path / lmp_input_name).write_text("bar")
        self.task_name = "task_000"
        self.models = [self.model_path / Path(f"model_{ii}") for ii in range(4)]
        for idx, ii in enumerate(self.models):
            ii.mkdir(parents=True, exist_ok=True)
            model_file = ii / Path("model.pb")
            model_file.write_text(f"model{idx}")

    def tearDown(self):
        if Path("task").is_dir():
            shutil.rmtree("task")
        if Path("models").is_dir():
            shutil.rmtree("models")
        if Path(self.task_name).is_dir():
            shutil.rmtree(self.task_name)

    @patch("dpgen2.op.run_nvnmd.run_command")
    def test_success(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", "")] * 4
        op = RunNvNMD()
        out = op.execute(
            OPIO(
                {
                    "config": {"command": "mylmp"},
                    "task_name": self.task_name,
                    "task_path": self.task_path,
                    "models": self.models,
                }
            )
        )
        work_dir = Path(self.task_name)
        # check output
        self.assertEqual(out["log"], work_dir / ("0_%s" % lmp_log_name))
        self.assertEqual(out["traj"], work_dir / ("0_%s" % lmp_traj_name))
        self.assertEqual(out["model_devi"], work_dir / lmp_model_devi_name)
        # check call
        models = ["models/path/model_%d.pb" % i for i in range(len(self.models))]
        models = ["models/path/model_%d.pb" % i for i in range(len(self.models))]
        calls = [
            call(
                " ".join(
                    [
                        "mylmp",
                        "-i",
                        "%d_%s" % (ii, lmp_input_name),
                        "-log",
                        "%d_%s" % (ii, lmp_log_name),
                        "-v",
                        "rerun",
                        "%d" % ii,
                    ]
                ),
                shell=True,
            )
            for ii in range(len(models))
        ]
        mocked_run.assert_has_calls(calls)
        # check input files are correctly linked
        self.assertEqual((work_dir / lmp_conf_name).read_text(), "foo")
        self.assertEqual((work_dir / lmp_input_name).read_text(), "bar")
        for ii in range(4):
            self.assertEqual(
                (work_dir / (model_name_pattern % ii)).read_text(), f"model{ii}"
            )

    @patch("dpgen2.op.run_nvnmd.run_command")
    def test_error(self, mocked_run):
        mocked_run.side_effect = [(1, "foo\n", "")]
        op = RunNvNMD()
        with self.assertRaises(TransientError) as ee:
            out = op.execute(
                OPIO(
                    {
                        "config": {"command": "mylmp"},
                        "task_name": self.task_name,
                        "task_path": self.task_path,
                        "models": self.models,
                    }
                )
            )
        # check call
        models = ["models/path/model_%d.pb" % i for i in range(len(self.models))]
        calls = [
            call(
                " ".join(
                    [
                        "mylmp",
                        "-i",
                        "%d_%s" % (ii, lmp_input_name),
                        "-log",
                        "%d_%s" % (ii, lmp_log_name),
                        "-v",
                        "rerun",
                        "%d" % ii,
                    ]
                ),
                shell=True,
            )
            for ii in range(1)
        ]
        mocked_run.assert_has_calls(calls)


def swap_element(arg):
    bk = arg.copy()
    arg[1] = bk[0]
    arg[0] = bk[1]


class TestSetModels(unittest.TestCase):
    def setUp(self):
        self.input_name = Path("lmp.input")
        self.model_names = ["model.000.pb", "model.001.pb"]

    def tearDown(self):
        os.remove(self.input_name)

    def test(self):
        lmp_config = "pair_style      nvnmd model.000.pb\n"
        expected_output = "pair_style      nvnmd model.000.pb\n"
        input_name = self.input_name
        input_name.write_text(lmp_config)
        set_models(input_name, self.model_names)
        self.assertEqual(input_name.read_text(), expected_output)

    def test_failed(self):
        lmp_config = "pair_style      deepmd model.000.pb\n"
        input_name = self.input_name
        input_name = Path("lmp.input")
        input_name.write_text(lmp_config)
        with self.assertRaises(RuntimeError) as re:
            set_models(input_name, self.model_names)

    def test_failed_no_matching(self):
        lmp_config = "pair_style      deepmd\n"
        input_name = self.input_name
        input_name = Path("lmp.input")
        input_name.write_text(lmp_config)
        with self.assertRaises(RuntimeError) as re:
            set_models(input_name, self.model_names)
