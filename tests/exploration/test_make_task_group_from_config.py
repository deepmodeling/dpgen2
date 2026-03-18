import itertools
import json
import os
import textwrap
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
)

import numpy as np

try:
    from exploration.context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.constants import (
    ase_conf_name,
    ase_input_name,
)
from dpgen2.exploration.task import (
    AseTaskGroup,
    CalyTaskGroup,
    LmpTemplateTaskGroup,
    NPTTaskGroup,
    make_ase_task_group_from_config,
    make_calypso_task_group_from_config,
    make_lmp_task_group_from_config,
)
from dpgen2.exploration.task.calypso import (
    make_calypso_input,
)


class TestMakeLmpTaskGroupFromConfig(unittest.TestCase):
    def setUp(self):
        self.config_npt = {
            "type": "lmp-md",
            "Ts": [100],
        }
        self.config_template = {
            "type": "lmp-template",
            "lmp_template_fname": "foo",
        }
        from .test_lmp_templ_task_group import (
            in_lmp_template,
        )

        Path(self.config_template["lmp_template_fname"]).write_text(in_lmp_template)
        self.mass_map = [1.0, 2.0]
        self.numb_models = 4

    def tearDown(self):
        os.remove(self.config_template["lmp_template_fname"])

    def test_npt(self):
        tgroup = make_lmp_task_group_from_config(
            self.numb_models, self.mass_map, self.config_npt
        )
        self.assertTrue(isinstance(tgroup, NPTTaskGroup))

    def test_template(self):
        tgroup = make_lmp_task_group_from_config(
            self.numb_models, self.mass_map, self.config_template
        )
        self.assertTrue(isinstance(tgroup, LmpTemplateTaskGroup))


class TestMakeCalyTaskGroupFromConfig(unittest.TestCase):
    def setUp(self):
        self.config = {
            "name_of_atoms": ["Li", "La"],
            "numb_of_atoms": [10, 10],
            "numb_of_species": 2,
            "atomic_number": [3, 4],
            "distance_of_ions": [[1.0, 1.0], [1.0, 1.0]],
        }
        self.config_err = {
            "name_of_atoms": ["Li", "La"],
            "numb_of_atoms": [10, 10],
            "numb_of_species": 4,
            "atomic_number": [3, 4],
            "distance_of_ions": [[1.0, 1.0], [1.0, 1.0]],
        }
        self.ref_input = """NumberOfSpecies = 2
NameOfAtoms = Li La
AtomicNumber = 3 4
NumberOfAtoms = 10 10
PopSize = 30
MaxStep = 5
SystemName = CALYPSO
NumberOfFormula = 1 1
Volume = 0
Ialgo = 2
PsoRatio = 0.6
ICode = 15
NumberOfLbest = 4
NumberOfLocalOptim = 4
Command = sh submit.sh
MaxTime = 9000
GenType = 1
PickUp = False
PickStep = 1
Parallel = F
Split = T
SpeSpaceGroup = 2 230
VSC = F
MaxNumAtom = 100
@DistanceOfIon
1.0 1.0
1.0 1.0
@End
@CtrlRange
1 10
@End
"""

    def tearDown(self):
        # os.remove(self.config_template["lmp_template_fname"])
        pass

    def test_make_caly_input(self):
        input_file_str, run_opt_str, check_opt_str = make_calypso_input(**self.config)
        self.assertEqual(input_file_str, self.ref_input)
        self.assertRaises(AssertionError, make_calypso_input, **self.config_err)

    def test_caly_task_group(self):
        tgroup = make_calypso_task_group_from_config(self.config)
        self.assertTrue(isinstance(tgroup, CalyTaskGroup))


# A minimal LAMMPS dump string used as a fake initial configuration.
_fake_conf_lmp = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 4.0
0.0 4.0
0.0 4.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 2 2.0 2.0 2.0
"""


class TestMakeAseTaskGroupFromConfig(unittest.TestCase):
    """Tests for :func:`make_ase_task_group_from_config` and :class:`AseTaskGroup`."""

    def setUp(self):
        self.numb_models = 4
        self.mass_map = [27.0, 24.0]
        self.type_map = ["Al", "Mg"]
        self.config_nvt = {
            "type": "ase-md",
            "conf_idx": [0],
            "temps": [300, 600],
            "ens": "nvt",
            "dt": 0.001,
            "nsteps": 100,
            "trj_freq": 10,
        }
        self.config_npt = {
            "type": "ase-md",
            "conf_idx": [0],
            "temps": [300],
            "press": [1.0, 10.0],
            "ens": "npt",
            "dt": 0.001,
            "nsteps": 100,
            "trj_freq": 10,
        }

    # ------------------------------------------------------------------
    # make_ase_task_group_from_config
    # ------------------------------------------------------------------

    def test_returns_ase_task_group(self):
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        self.assertIsInstance(tgroup, AseTaskGroup)

    def test_npt_returns_ase_task_group(self):
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_npt
        )
        self.assertIsInstance(tgroup, AseTaskGroup)

    # ------------------------------------------------------------------
    # AseTaskGroup.make_task — task count
    # ------------------------------------------------------------------

    def test_task_count_nvt(self):
        """NVT: n_conf × n_temps × 1 (no pressure)."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        confs = [_fake_conf_lmp, _fake_conf_lmp]
        tgroup.set_conf(confs)
        tgroup.make_task()
        # 2 confs × 2 temps × 1 press(None) = 4
        self.assertEqual(len(tgroup), 2 * 2 * 1)

    def test_task_count_npt(self):
        """NPT: n_conf × n_temps × n_press."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_npt
        )
        confs = [_fake_conf_lmp]
        tgroup.set_conf(confs)
        tgroup.make_task()
        # 1 conf × 1 temp × 2 press = 2
        self.assertEqual(len(tgroup), 1 * 1 * 2)

    # ------------------------------------------------------------------
    # AseTaskGroup.make_task — file contents
    # ------------------------------------------------------------------

    def test_task_files_present(self):
        """Each task must contain conf.lmp and ase_input.json."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        for task in tgroup:
            files = task.files()
            self.assertIn(ase_conf_name, files)
            self.assertIn(ase_input_name, files)

    def test_conf_content(self):
        """conf.lmp must equal the input configuration string."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        for task in tgroup:
            self.assertEqual(task.files()[ase_conf_name], _fake_conf_lmp)

    def test_ase_input_json_keys(self):
        """ase_input.json must contain all required keys."""
        required_keys = {
            "type_map",
            "mass_map",
            "numb_models",
            "ensemble",
            "temperature",
            "pressure",
            "dt",
            "nsteps",
            "trj_freq",
            "tau_t",
            "tau_p",
            "init_velocities",
        }
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        for task in tgroup:
            data = json.loads(task.files()[ase_input_name])
            self.assertTrue(required_keys.issubset(data.keys()))

    def test_temperatures_assigned(self):
        """Each task's ase_input.json must have the correct temperature."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        temps_found = sorted(
            {json.loads(t.files()[ase_input_name])["temperature"] for t in tgroup}
        )
        self.assertEqual(temps_found, sorted(self.config_nvt["temps"]))

    def test_pressures_assigned_npt(self):
        """NPT tasks must have the correct pressure values."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_npt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        press_found = sorted(
            {json.loads(t.files()[ase_input_name])["pressure"] for t in tgroup}
        )
        self.assertEqual(press_found, sorted(self.config_npt["press"]))

    def test_nvt_pressure_is_none(self):
        """NVT tasks must have pressure=null in ase_input.json."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        for task in tgroup:
            data = json.loads(task.files()[ase_input_name])
            self.assertIsNone(data["pressure"])

    def test_type_map_in_input(self):
        """ase_input.json must carry the correct type_map."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        for task in tgroup:
            data = json.loads(task.files()[ase_input_name])
            self.assertEqual(data["type_map"], self.type_map)

    def test_numb_models_in_input(self):
        """ase_input.json must carry the correct numb_models."""
        tgroup = make_ase_task_group_from_config(
            self.numb_models, self.mass_map, self.type_map, self.config_nvt
        )
        tgroup.set_conf([_fake_conf_lmp])
        tgroup.make_task()
        for task in tgroup:
            data = json.loads(task.files()[ase_input_name])
            self.assertEqual(data["numb_models"], self.numb_models)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_make_task_without_set_conf_raises(self):
        """make_task() must raise RuntimeError if set_conf was not called."""
        tgroup = AseTaskGroup()
        tgroup.set_md(
            numb_models=self.numb_models,
            mass_map=self.mass_map,
            type_map=self.type_map,
            temps=[300],
        )
        with self.assertRaises(RuntimeError):
            tgroup.make_task()

    def test_make_task_without_set_md_raises(self):
        """make_task() must raise RuntimeError if set_md was not called."""
        tgroup = AseTaskGroup()
        tgroup.set_conf([_fake_conf_lmp])
        with self.assertRaises(RuntimeError):
            tgroup.make_task()
