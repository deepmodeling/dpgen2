import json
import os
import shutil
import unittest
from pathlib import (
    Path,
)

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
from op.context import (
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
from dpgen2.op.run_lmp import (
    RunLmp,
    randomly_shuffle_models,
)
from dpgen2.utils import (
    BinaryFileInput,
)


class TestRunLmp(unittest.TestCase):
    def setUp(self):
        self.task_path = Path("task/path")
        self.task_path.mkdir(parents=True, exist_ok=True)
        self.model_path = Path("models/path")
        self.model_path.mkdir(parents=True, exist_ok=True)
        (self.task_path / lmp_conf_name).write_text("foo")
        (self.task_path / lmp_input_name).write_text("bar")
        self.task_name = "task_000"
        self.models = [self.model_path / Path(f"model_{ii}.pb") for ii in range(4)]
        for idx, ii in enumerate(self.models):
            ii.write_text(f"model{idx}")

    def tearDown(self):
        if Path("task").is_dir():
            shutil.rmtree("task")
        if Path("models").is_dir():
            shutil.rmtree("models")
        if Path(self.task_name).is_dir():
            shutil.rmtree(self.task_name)

    @patch("dpgen2.op.run_lmp.run_command")
    def test_success(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", "")]
        op = RunLmp()
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
        self.assertEqual(out["log"], work_dir / lmp_log_name)
        self.assertEqual(out["traj"], work_dir / lmp_traj_name)
        self.assertEqual(out["model_devi"], work_dir / lmp_model_devi_name)
        # check call
        calls = [
            call(
                " ".join(["mylmp", "-i", lmp_input_name, "-log", lmp_log_name]),
                shell=True,
            ),
        ]
        mocked_run.assert_has_calls(calls)
        # check input files are correctly linked
        self.assertEqual((work_dir / lmp_conf_name).read_text(), "foo")
        self.assertEqual((work_dir / lmp_input_name).read_text(), "bar")
        for ii in range(4):
            self.assertEqual(
                (work_dir / (model_name_pattern % ii)).read_text(), f"model{ii}"
            )

    @patch("dpgen2.op.run_lmp.run_command")
    def test_error(self, mocked_run):
        mocked_run.side_effect = [(1, "foo\n", "")]
        op = RunLmp()
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
        calls = [
            call(
                " ".join(["mylmp", "-i", lmp_input_name, "-log", lmp_log_name]),
                shell=True,
            ),
        ]
        mocked_run.assert_has_calls(calls)


class TestRunLmpDist(unittest.TestCase):
    lmp_config = """variable        NSTEPS          equal 1000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box          tilt large
if "${restart} > 0" then "read_restart dpgen.restart.*" else "read_data conf.lmp"

group target_element_1 type 4
#set group other_element type/subset ${ELEMENT_TYPE_4} ${ELEMENT_NUMB_4} ${OUTER_RANDOM_SEED_4}

change_box   all triclinic
mass            6 26.980000
pair_style      deepmd model.000.pb out_freq 10 out_file model_devi.out
pair_coeff      * * 

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}
#dump            1 all custom ${DUMP_FREQ} traj/*.lammpstrj id type x y z fx fy fz

if "${restart} == 0" then "velocity        all create 2754.34 709383"
fix             1 all npt temp 2754.34 2754.34 ${TAU_T} iso 1.0 1.0 ${TAU_P}
timestep        0.002000
run             3000 upto
"""

    def setUp(self):
        self.task_path = Path("task/path")
        self.task_path.mkdir(parents=True, exist_ok=True)
        self.model_path = Path("models/path")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.teacher_path = Path("models/teacher")
        self.teacher_path.mkdir(parents=True, exist_ok=True)

        (self.task_path / lmp_conf_name).write_text("foo")
        (self.task_path / lmp_input_name).write_text(TestRunLmpDist.lmp_config)

        self.task_name = "task_000"
        self.models = [self.model_path / Path(f"model_{ii}.pb") for ii in range(1)]
        for idx, ii in enumerate(self.models):
            ii.write_text(f"model{idx}")

        (self.teacher_path / "teacher.pb").write_text("teacher model")
        self.teacher_model = BinaryFileInput(self.teacher_path / "teacher.pb", "pb")

        self.maxDiff = None

    def tearDown(self):
        if Path("task").is_dir():
            shutil.rmtree("task")
        if Path("models").is_dir():
            shutil.rmtree("models")
        if Path(self.task_name).is_dir():
            shutil.rmtree(self.task_name)

    @patch("dpgen2.op.run_lmp.run_command")
    def test_success(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", "")]
        op = RunLmp()
        out = op.execute(
            OPIO(
                {
                    "config": {
                        "command": "mylmp",
                        "teacher_model_path": self.teacher_model,
                    },
                    "task_name": self.task_name,
                    "task_path": self.task_path,
                    "models": self.models,
                }
            )
        )
        work_dir = Path(self.task_name)

        # check input files are correctly linked
        self.assertEqual((work_dir / lmp_conf_name).read_text(), "foo")

        lmp_config = TestRunLmpDist.lmp_config.replace(
            "model.000.pb", "model.000.pb model.001.pb"
        )
        self.assertEqual((work_dir / lmp_input_name).read_text(), lmp_config)

        # check if the teacher model is linked to model.000.pb
        ii = 0
        self.assertEqual(
            (work_dir / (model_name_pattern % ii)).read_text(), f"teacher model"
        )

        ii = 1
        self.assertEqual(
            (work_dir / (model_name_pattern % ii)).read_text(), f"model{ii - 1}"
        )

        # The number of models have to be 2 in knowledge distillation
        self.assertEqual(len(list((work_dir.glob("*.pb")))), 2)


def swap_element(arg):
    bk = arg.copy()
    arg[1] = bk[0]
    arg[0] = bk[1]


class TestRandomShuffleModels(unittest.TestCase):
    def setUp(self):
        self.input_name = Path("lmp.input")

    def tearDown(self):
        os.remove(self.input_name)

    @patch("dpgen2.op.run_lmp.random.shuffle")
    def test(self, mock_shuffle):
        mock_shuffle.side_effect = swap_element
        lmp_config = "pair_style      deepmd model.000.pb model.001.pb out_freq 10 out_file model_devi.out"
        expected_output = "pair_style deepmd model.001.pb model.000.pb out_freq 10 out_file model_devi.out"
        input_name = self.input_name
        input_name.write_text(lmp_config)
        randomly_shuffle_models(input_name)
        self.assertEqual(input_name.read_text(), expected_output)

    def test_failed(self):
        lmp_config = "pair_style      deepmd model.000.pb model.001.pb out_freq 10 out_file model_devi.out model.002.pb"
        input_name = self.input_name
        input_name = Path("lmp.input")
        input_name.write_text(lmp_config)
        with self.assertRaises(RuntimeError) as re:
            randomly_shuffle_models(input_name)

    def test_failed_no_matching(self):
        lmp_config = "pair_style      deepmd  out_freq 10 out_file model_devi.out"
        input_name = self.input_name
        input_name = Path("lmp.input")
        input_name.write_text(lmp_config)
        with self.assertRaises(RuntimeError) as re:
            randomly_shuffle_models(input_name)
