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
    LmpTemplateTaskGroup,
    NPTTaskGroup,
    make_task_group_from_config,
)


class TestMakeTaskGroupFromConfig(unittest.TestCase):
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
        tgroup = make_task_group_from_config(
            self.numb_models, self.mass_map, self.config_npt
        )
        self.assertTrue(isinstance(tgroup, NPTTaskGroup))

    def test_template(self):
        tgroup = make_task_group_from_config(
            self.numb_models, self.mass_map, self.config_template
        )
        self.assertTrue(isinstance(tgroup, LmpTemplateTaskGroup))
