"""Unit tests for PrepAse and RunAse OPs.

These tests mock out all heavy dependencies (ASE, DeePMD) so that the
test suite can run without those packages installed.  The tests verify:

* PrepAse — directory creation, file content, task naming
* RunAse  — symlink setup, _run_ase_md dispatch, output paths,
            TransientError on failure
* _atoms_to_lmpdump — LAMMPS dump format correctness
* _write_model_devi_out — column count and header
"""

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
from dflow.python import (
    OPIO,
    TransientError,
)

# isort: off
from .context import dpgen2  # noqa: F401 — adds repo root to sys.path

from dpgen2.constants import (
    ase_conf_name,
    ase_input_name,
    ase_log_name,
    ase_model_devi_name,
    ase_task_pattern,
    ase_traj_name,
    model_name_pattern,
)
from dpgen2.op.prep_ase import PrepAse, _mk_task_from_files
from dpgen2.op.run_ase import (
    RunAse,
    RunAseHDF5,
    _atoms_to_lmpdump,
    _run_ase_md,
    _write_model_devi_out,
)
from dpgen2.exploration.task import AseTaskGroup

# isort: on

# ---------------------------------------------------------------------------
# Minimal LAMMPS dump content used as a fake initial configuration
# ---------------------------------------------------------------------------
_FAKE_CONF_LMP = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
   0.0000000000   4.0500000000   0.0000000000
   0.0000000000   4.0500000000   0.0000000000
   0.0000000000   4.0500000000   0.0000000000
ITEM: ATOMS id type x y z fx fy fz
    1     1   0.0000000000   0.0000000000   0.0000000000   0.0   0.0   0.0
    2     1   2.0250000000   2.0250000000   2.0250000000   0.0   0.0   0.0
"""

_FAKE_ASE_INPUT = {
    "type_map": ["Al"],
    "mass_map": [26.98],
    "numb_models": 2,
    "ensemble": "nvt",
    "temperature": 300.0,
    "pressure": None,
    "dt": 0.001,
    "nsteps": 10,
    "trj_freq": 5,
    "tau_t": 0.1,
    "tau_p": 0.5,
    "init_velocities": False,
}


# ===========================================================================
# PrepAse tests
# ===========================================================================


class TestPrepAse(unittest.TestCase):
    """Tests for :class:`PrepAse`."""

    def setUp(self):
        self.work_dir = Path("test_prep_ase_work")
        self.work_dir.mkdir(exist_ok=True)
        os.chdir(self.work_dir)

        # Build a minimal AseTaskGroup with 2 tasks
        tg = AseTaskGroup()
        tg.set_md(
            numb_models=2,
            mass_map=[26.98],
            type_map=["Al"],
            temps=[300.0, 600.0],
        )
        tg.set_conf([_FAKE_CONF_LMP])
        tg.make_task()
        self.task_group = tg

    def tearDown(self):
        os.chdir("..")
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_returns_two_task_paths(self):
        op = PrepAse()
        out = op.execute(OPIO({"ase_task_grp": self.task_group}))
        self.assertEqual(len(out["task_paths"]), 2)

    def test_task_names_match_paths(self):
        op = PrepAse()
        out = op.execute(OPIO({"ase_task_grp": self.task_group}))
        for name, path in zip(out["task_names"], out["task_paths"]):
            self.assertEqual(name, str(path))

    def test_task_pattern(self):
        op = PrepAse()
        out = op.execute(OPIO({"ase_task_grp": self.task_group}))
        self.assertEqual(out["task_names"][0], ase_task_pattern % 0)
        self.assertEqual(out["task_names"][1], ase_task_pattern % 1)

    def test_conf_file_written(self):
        op = PrepAse()
        out = op.execute(OPIO({"ase_task_grp": self.task_group}))
        for p in out["task_paths"]:
            self.assertTrue((p / ase_conf_name).is_file())

    def test_input_json_written(self):
        op = PrepAse()
        out = op.execute(OPIO({"ase_task_grp": self.task_group}))
        for p in out["task_paths"]:
            self.assertTrue((p / ase_input_name).is_file())

    def test_input_json_parseable(self):
        op = PrepAse()
        out = op.execute(OPIO({"ase_task_grp": self.task_group}))
        for p in out["task_paths"]:
            data = json.loads((p / ase_input_name).read_text())
            self.assertIn("temperature", data)
            self.assertIn("ensemble", data)

    def test_temperatures_differ(self):
        op = PrepAse()
        out = op.execute(OPIO({"ase_task_grp": self.task_group}))
        temps = []
        for p in out["task_paths"]:
            data = json.loads((p / ase_input_name).read_text())
            temps.append(data["temperature"])
        self.assertEqual(sorted(temps), [300.0, 600.0])

    def test_mk_task_from_files_creates_dir(self):
        files = {"conf.lmp": "foo", "ase_input.json": "{}"}
        tname = _mk_task_from_files(99, files)
        self.assertTrue(tname.is_dir())
        self.assertEqual((tname / "conf.lmp").read_text(), "foo")
        shutil.rmtree(tname)


# ===========================================================================
# RunAse tests
# ===========================================================================


class TestRunAse(unittest.TestCase):
    """Tests for :class:`RunAse` (mocking _run_ase_md)."""

    def setUp(self):
        self.work_dir = Path("test_run_ase_work")
        self.work_dir.mkdir(exist_ok=True)
        os.chdir(self.work_dir)

        # Create a fake task directory
        self.task_path = Path("task_path")
        self.task_path.mkdir()
        (self.task_path / ase_conf_name).write_text(_FAKE_CONF_LMP)
        (self.task_path / ase_input_name).write_text(
            json.dumps(_FAKE_ASE_INPUT)
        )

        # Create fake model files
        self.model_dir = Path("models")
        self.model_dir.mkdir()
        self.models = [self.model_dir / f"model.{i:03d}.pb" for i in range(2)]
        for m in self.models:
            m.write_text("fake_model")

        self.task_name = "task.000000"

    def tearDown(self):
        os.chdir("..")
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _make_fake_outputs(self, work_dir: Path):
        """Create the output files that _run_ase_md would produce."""
        (work_dir / ase_log_name).write_text("ase log")
        (work_dir / ase_traj_name).write_text("ITEM: TIMESTEP\n0\n")
        devi = np.zeros((2, 7))
        devi[:, 0] = [0, 1]
        _write_model_devi_out(devi, str(work_dir / ase_model_devi_name))

    @patch("dpgen2.op.run_ase._run_ase_md")
    def test_output_paths(self, mock_run):
        def side_effect(model_names, config):
            work_dir = Path(".")
            self._make_fake_outputs(work_dir)

        mock_run.side_effect = side_effect

        op = RunAse()
        out = op.execute(
            OPIO(
                {
                    "config": {},
                    "task_name": self.task_name,
                    "task_path": self.task_path,
                    "models": self.models,
                }
            )
        )
        work_dir = Path(self.task_name)
        self.assertEqual(out["log"], work_dir / ase_log_name)
        self.assertEqual(out["traj"], work_dir / ase_traj_name)
        self.assertEqual(out["model_devi"], work_dir / ase_model_devi_name)

    @patch("dpgen2.op.run_ase._run_ase_md")
    def test_input_files_symlinked(self, mock_run):
        def side_effect(model_names, config):
            self._make_fake_outputs(Path("."))

        mock_run.side_effect = side_effect

        op = RunAse()
        op.execute(
            OPIO(
                {
                    "config": {},
                    "task_name": self.task_name,
                    "task_path": self.task_path,
                    "models": self.models,
                }
            )
        )
        work_dir = Path(self.task_name)
        self.assertTrue((work_dir / ase_conf_name).exists())
        self.assertTrue((work_dir / ase_input_name).exists())

    @patch("dpgen2.op.run_ase._run_ase_md")
    def test_model_files_symlinked(self, mock_run):
        def side_effect(model_names, config):
            self._make_fake_outputs(Path("."))

        mock_run.side_effect = side_effect

        op = RunAse()
        op.execute(
            OPIO(
                {
                    "config": {},
                    "task_name": self.task_name,
                    "task_path": self.task_path,
                    "models": self.models,
                }
            )
        )
        work_dir = Path(self.task_name)
        for i in range(2):
            self.assertTrue((work_dir / (model_name_pattern % i)).exists())

    @patch("dpgen2.op.run_ase._run_ase_md")
    def test_run_ase_md_called_with_model_names(self, mock_run):
        def side_effect(model_names, config):
            self._make_fake_outputs(Path("."))

        mock_run.side_effect = side_effect

        op = RunAse()
        op.execute(
            OPIO(
                {
                    "config": {},
                    "task_name": self.task_name,
                    "task_path": self.task_path,
                    "models": self.models,
                }
            )
        )
        self.assertTrue(mock_run.called)
        called_model_names = mock_run.call_args[0][0]
        self.assertEqual(len(called_model_names), 2)
        for name in called_model_names:
            self.assertTrue(name.endswith(".pb"))

    @patch("dpgen2.op.run_ase._run_ase_md")
    def test_transient_error_on_failure(self, mock_run):
        mock_run.side_effect = RuntimeError("ASE exploded")

        op = RunAse()
        with self.assertRaises(TransientError):
            op.execute(
                OPIO(
                    {
                        "config": {},
                        "task_name": self.task_name,
                        "task_path": self.task_path,
                        "models": self.models,
                    }
                )
            )

    def test_normalize_config_empty(self):
        cfg = RunAse.normalize_config({})
        self.assertIn("model_frozen_head", cfg)
        self.assertIsNone(cfg["model_frozen_head"])

    def test_normalize_config_with_head(self):
        cfg = RunAse.normalize_config({"model_frozen_head": "head0"})
        self.assertEqual(cfg["model_frozen_head"], "head0")

    def test_unsupported_model_extension_raises(self):
        bad_model = self.model_dir / "model.000.xyz"
        bad_model.write_text("bad")

        op = RunAse()
        with self.assertRaises(RuntimeError):
            op.execute(
                OPIO(
                    {
                        "config": {},
                        "task_name": self.task_name,
                        "task_path": self.task_path,
                        "models": [bad_model],
                    }
                )
            )


# ===========================================================================
# RunAseHDF5 tests
# ===========================================================================


class TestRunAseHDF5(unittest.TestCase):
    """Tests for :class:`RunAseHDF5` output signature."""

    def test_output_sign_has_hdf5_traj(self):
        from dflow.python import HDF5Datasets

        sign = RunAseHDF5.get_output_sign()
        self.assertEqual(sign["traj"].type, HDF5Datasets)

    def test_output_sign_has_hdf5_model_devi(self):
        from dflow.python import HDF5Datasets

        sign = RunAseHDF5.get_output_sign()
        self.assertEqual(sign["model_devi"].type, HDF5Datasets)


# ===========================================================================
# _atoms_to_lmpdump tests
# ===========================================================================


class TestAtomsToDump(unittest.TestCase):
    """Tests for :func:`_atoms_to_lmpdump`."""

    def _make_atoms(self):
        """Return a minimal 2-atom Al FCC unit cell."""
        try:
            import ase  # type: ignore
            import ase.build  # type: ignore
        except ImportError:
            self.skipTest("ase not installed")
        atoms = ase.build.bulk("Al", "fcc", a=4.05)
        return atoms

    def test_contains_timestep_header(self):
        atoms = self._make_atoms()
        dump = _atoms_to_lmpdump(atoms, 0, ["Al"])
        self.assertIn("ITEM: TIMESTEP", dump)

    def test_frame_index_in_dump(self):
        atoms = self._make_atoms()
        dump = _atoms_to_lmpdump(atoms, 42, ["Al"])
        lines = dump.splitlines()
        ts_idx = lines.index("ITEM: TIMESTEP")
        self.assertEqual(lines[ts_idx + 1], "42")

    def test_number_of_atoms_correct(self):
        atoms = self._make_atoms()
        dump = _atoms_to_lmpdump(atoms, 0, ["Al"])
        lines = dump.splitlines()
        na_idx = lines.index("ITEM: NUMBER OF ATOMS")
        self.assertEqual(int(lines[na_idx + 1]), len(atoms))

    def test_atoms_section_present(self):
        atoms = self._make_atoms()
        dump = _atoms_to_lmpdump(atoms, 0, ["Al"])
        self.assertIn("ITEM: ATOMS id type x y z fx fy fz", dump)

    def test_type_id_is_one_for_first_element(self):
        atoms = self._make_atoms()
        dump = _atoms_to_lmpdump(atoms, 0, ["Al"])
        lines = dump.splitlines()
        atom_start = lines.index("ITEM: ATOMS id type x y z fx fy fz") + 1
        first_atom = lines[atom_start].split()
        self.assertEqual(int(first_atom[1]), 1)  # type index 1-based


# ===========================================================================
# _write_model_devi_out tests
# ===========================================================================


class TestWriteModelDeviOut(unittest.TestCase):
    """Tests for :func:`_write_model_devi_out`."""

    def setUp(self):
        self.work_dir = Path("test_write_devi_work")
        self.work_dir.mkdir(exist_ok=True)
        os.chdir(self.work_dir)

    def tearDown(self):
        os.chdir("..")
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_writes_file(self):
        devi = np.zeros((3, 7))
        devi[:, 0] = [0, 1, 2]
        _write_model_devi_out(devi, "model_devi.out")
        self.assertTrue(Path("model_devi.out").is_file())

    def test_seven_columns(self):
        devi = np.zeros((3, 7))
        devi[:, 0] = [0, 1, 2]
        _write_model_devi_out(devi, "model_devi.out")
        data = np.loadtxt("model_devi.out", comments="#")
        self.assertEqual(data.shape[1], 7)

    def test_step_column_correct(self):
        devi = np.zeros((3, 7))
        devi[:, 0] = [0, 5, 10]
        _write_model_devi_out(devi, "model_devi.out")
        data = np.loadtxt("model_devi.out", comments="#")
        np.testing.assert_array_equal(data[:, 0], [0, 5, 10])

    def test_wrong_column_count_raises(self):
        devi = np.zeros((3, 8))  # 8 columns — wrong
        with self.assertRaises(AssertionError):
            _write_model_devi_out(devi, "model_devi.out")

    def test_header_written(self):
        devi = np.zeros((1, 7))
        _write_model_devi_out(devi, "model_devi.out")
        content = Path("model_devi.out").read_text()
        self.assertIn("max_devi_f", content)
        self.assertIn("max_devi_v", content)


if __name__ == "__main__":
    unittest.main()
