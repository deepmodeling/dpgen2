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
from dpgen2.op.run_nvnmd import (
    RunNvNMD,
    get_ele_temp,
    merge_pimd_files,
    set_models,
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

    @patch("dpgen2.op.run_nvnmd.run_command")
    def test_success(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", "")]
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
        self.assertEqual(out["log"], work_dir / lmp_log_name)
        self.assertEqual(out["traj"], work_dir / lmp_traj_name)
        self.assertEqual(out["model_devi"], work_dir / lmp_model_devi_name)
        # check call
        models = ["models/path/model_%d.pb" % i for i in range(len(self.models))]
        calls = [
            call(
                " ; ".join(
                    [
                        " ".join(
                            [
                                "cp",
                                model_name,
                                "model.pb",
                                "&&",
                                "mylmp",
                                "-i",
                                lmp_input_name,
                                "-log",
                                lmp_log_name,
                                "-v",
                                "rerun",
                                "%d" % i,
                                "&&",
                                "cp",
                                lmp_traj_name,
                                lmp_traj_name + ".%d" % i,
                            ]
                        )
                        for i, model_name in enumerate(models)
                    ]
                ),
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
                " ; ".join(
                    [
                        " ".join(
                            [
                                "cp",
                                model_name,
                                "model.pb",
                                "&&",
                                "mylmp",
                                "-i",
                                lmp_input_name,
                                "-log",
                                lmp_log_name,
                                "-v",
                                "rerun",
                                "%d" % i,
                                "&&",
                                "cp",
                                lmp_traj_name,
                                lmp_traj_name + ".%d" % i,
                            ]
                        )
                        for i, model_name in enumerate(models)
                    ]
                ),
                shell=True,
            ),
        ]
        mocked_run.assert_has_calls(calls)


class TestRunNvNMDDist(unittest.TestCase):
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
        (self.task_path / lmp_input_name).write_text(TestRunNvNMDDist.lmp_config)

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

    @patch("dpgen2.op.run_nvnmd.run_command")
    def test_success(self, mocked_run):
        mocked_run.side_effect = [(0, "foo\n", "")]
        op = RunNvNMD()
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

        lmp_config = TestRunNvNMDDist.lmp_config.replace(
            "pair_style      deepmd model.000.pb",
            "pair_style deepmd model.000.pb model.001.pb",
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


class TestSetModels(unittest.TestCase):
    def setUp(self):
        self.input_name = Path("lmp.input")
        self.model_names = ["model.000.pth", "model.001.pb"]

    def tearDown(self):
        os.remove(self.input_name)

    def test(self):
        lmp_config = "pair_style      deepmd model.000.pb model.001.pb out_freq 10 out_file model_devi.out\n"
        expected_output = "pair_style deepmd model.000.pth model.001.pb out_freq 10 out_file model_devi.out\n"
        input_name = self.input_name
        input_name.write_text(lmp_config)
        set_models(input_name, self.model_names)
        self.assertEqual(input_name.read_text(), expected_output)

    def test_failed(self):
        lmp_config = "pair_style      deepmd model.000.pb model.001.pb out_freq 10 out_file model_devi.out model.002.pb\n"
        input_name = self.input_name
        input_name = Path("lmp.input")
        input_name.write_text(lmp_config)
        with self.assertRaises(RuntimeError) as re:
            set_models(input_name, self.model_names)

    def test_failed_no_matching(self):
        lmp_config = "pair_style      deepmd  out_freq 10 out_file model_devi.out\n"
        input_name = self.input_name
        input_name = Path("lmp.input")
        input_name.write_text(lmp_config)
        with self.assertRaises(RuntimeError) as re:
            set_models(input_name, self.model_names)


class TestGetEleTemp(unittest.TestCase):
    def test_get_ele_temp_none(self):
        with open("log", "w") as f:
            f.write(
                "pair_style      deepmd model.000.pb model.001.pb model.002.pb model.003.pb model.004.pb out_freq 10 out_file model_devi.out"
            )
        ele_temp = get_ele_temp("log")
        self.assertIsNone(ele_temp)

    def test_get_ele_temp(self):
        with open("log", "w") as f:
            f.write(
                "pair_style      deepmd model.000.pb model.001.pb model.002.pb model.003.pb model.004.pb out_freq 10 out_file model_devi.out fparam 6.6"
            )
        ele_temp = get_ele_temp("log")
        self.assertEqual(ele_temp, 6.6)

    def tearDown(self):
        if os.path.exists("log"):
            os.remove("log")


class TestMergePIMDFiles(unittest.TestCase):
    def test_merge_pimd_files(self):
        for i in range(1, 3):
            with open("traj.%s.dump" % i, "w") as f:
                f.write(
                    """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0000000000000000e+00 1.2444661140399999e+01 0.0000000000000000e+00
0.0000000000000000e+00 1.2444661140399999e+01 0.0000000000000000e+00
0.0000000000000000e+00 1.2444661140399999e+01 0.0000000000000000e+00
ITEM: ATOMS id type x y z
1 8 7.23489 0.826309 4.61669
2 1 8.04419 0.520382 5.14395
3 1 6.48126 0.446895 4.99766
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0000000000000000e+00 1.2444661140399999e+01 0.0000000000000000e+00
0.0000000000000000e+00 1.2444661140399999e+01 0.0000000000000000e+00
0.0000000000000000e+00 1.2444661140399999e+01 0.0000000000000000e+00
ITEM: ATOMS id type x y z
1 8 7.23103 0.814939 4.59892
2 1 7.96453 0.61699 5.19158
3 1 6.43661 0.370311 5.09854
"""
                )
        for i in range(1, 3):
            with open("model_devi.%s.out" % i, "w") as f:
                f.write(
                    """#       step         max_devi_v         min_devi_v         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f
           0       9.023897e-17       3.548771e-17       5.237314e-17       8.196123e-16       1.225653e-16       3.941002e-16
          10       1.081667e-16       4.141596e-17       7.534462e-17       9.070597e-16       1.067947e-16       4.153524e-16
"""
                )

        merge_pimd_files()
        self.assertTrue(os.path.exists(lmp_traj_name))
        self.assertTrue(os.path.exists(lmp_model_devi_name))
        s = dpdata.System(lmp_traj_name, fmt="lammps/dump")
        assert len(s) == 4
        model_devi = np.loadtxt(lmp_model_devi_name)
        assert model_devi.shape[0] == 4

    def tearDown(self):
        for f in [
            lmp_traj_name,
            "traj.1.dump",
            "traj.2.dump",
            lmp_model_devi_name,
            "model_devi.1.out",
            "model_devi.2.out",
        ]:
            if os.path.exists(f):
                os.remove(f)
