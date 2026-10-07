import copy
import itertools
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
from dargs.dargs import (
    ArgumentTypeError,
)

try:
    from exploration.context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.exploration.task import (
    CalyTaskGroup,
    LmpTemplateTaskGroup,
    NPTTaskGroup,
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
        self.assertNotIn("conf_idx", self.config_npt)
        self.assertEqual(self.config_npt, {"type": "lmp-md", "Ts": [100]})

    def test_npt_accepts_explicit_conf_idx_without_mutating_caller(self):
        """Accept the explicit normalization key without changing the caller dict."""
        config = {
            "type": "lmp-md",
            "Ts": [100],
            "conf_idx": [2],
        }

        tgroup = make_lmp_task_group_from_config(
            self.numb_models, self.mass_map, config
        )

        self.assertTrue(isinstance(tgroup, NPTTaskGroup))
        self.assertEqual(config, {"type": "lmp-md", "Ts": [100], "conf_idx": [2]})
        # This helper does not select configurations; set_conf is called later.
        self.assertFalse(tgroup.conf_set)

    def test_npt_accepts_sys_idx_alias_without_mutating_caller(self):
        config = dict(self.config_npt, sys_idx=[2])
        before = copy.deepcopy(config)
        tgroup = make_lmp_task_group_from_config(
            self.numb_models, self.mass_map, config
        )
        self.assertIsInstance(tgroup, NPTTaskGroup)
        self.assertEqual(config, before)
        self.assertFalse(tgroup.conf_set)

    def test_invalid_conf_idx_does_not_mutate_caller(self):
        config = dict(self.config_npt, conf_idx="invalid")
        before = copy.deepcopy(config)
        with self.assertRaises(ArgumentTypeError):
            make_lmp_task_group_from_config(self.numb_models, self.mass_map, config)
        self.assertEqual(config, before)

    def test_template(self):
        tgroup = make_lmp_task_group_from_config(
            self.numb_models, self.mass_map, self.config_template
        )
        self.assertTrue(isinstance(tgroup, LmpTemplateTaskGroup))
        self.assertNotIn("conf_idx", self.config_template)
        self.assertFalse(tgroup.strict_revisions)

    def test_template_strict_revisions(self):
        strict_config = {
            **self.config_template,
            "strict_revisions": True,
        }
        tgroup = make_lmp_task_group_from_config(
            self.numb_models, self.mass_map, strict_config
        )
        self.assertTrue(tgroup.strict_revisions)


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

    def test_caly_does_not_mutate_caller(self):
        config = dict(self.config, type="calypso")
        before = copy.deepcopy(config)
        tgroup = make_calypso_task_group_from_config(config)
        self.assertIsInstance(tgroup, CalyTaskGroup)
        self.assertEqual(config, before)
        tgroup.name_of_atoms[0] = "Na"
        self.assertEqual(config, before)
