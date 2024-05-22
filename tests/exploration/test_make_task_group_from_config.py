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
