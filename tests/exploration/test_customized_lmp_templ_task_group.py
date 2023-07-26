import itertools
import os
import shutil
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
from unittest.mock import (
    Mock,
    patch,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    plm_input_name,
)
from dpgen2.exploration.task import (
    CustomizedLmpTemplateTaskGroup,
    ExplorationStage,
)

from .test_lmp_templ_task_group import (
    expected_lmp_plm_template,
    expected_lmp_template,
    in_lmp_plm_template,
    in_lmp_template,
    in_plm_template,
)

in_extra_py_file = textwrap.dedent(
    """from pathlib import Path
import os,shutil
for ii in range(2):
    task_name = Path(f"task_{ii}")
    task_name.mkdir(exist_ok=True)
    shutil.copy("foo.lmp", task_name/"bar.lmp")
    shutil.copy("lmp.template", task_name)
    shutil.copy("plm.template", task_name)
    os.chdir(task_name)
    ss = Path("lmp.template").read_text()
    ss = ss.replace("0.500000", f"{ii}")
    Path("lmp.template").write_text(ss)
    os.chdir("..")
"""
)

in_extra_lmp_py_file = textwrap.dedent(
    """from pathlib import Path
import os,shutil
for ii in range(2):
    task_name = Path(f"task_{ii}")
    task_name.mkdir(exist_ok=True)
    shutil.copy("foo.lmp", task_name/"bar.lmp")
    shutil.copy("lmp.template", task_name)
    os.chdir(task_name)
    ss = Path("lmp.template").read_text()
    ss = ss.replace("0.500000", f"{ii}")
    Path("lmp.template").write_text(ss)
    os.chdir("..")
"""
)


class TestLmpTemplateTaskGroup(unittest.TestCase):
    def setUp(self):
        self.lmp_template_fname = Path("lmp.template")
        self.lmp_template_fname.write_text(in_lmp_plm_template)
        self.plm_template_fname = Path("plm.template")
        self.plm_template_fname.write_text(in_plm_template)
        self.numb_models = 4
        self.confs = ["foo", "bar"]
        self.lmp_rev_mat = {
            "V_NSTEPS": [1000],
            "V_TEMP": [50, 100],
            "V_DIST0": [3, 4],
        }
        self.rev_empty = {}
        self.traj_freq = 20
        self.py_script = Path("modify.py")
        self.py_script.write_text(in_extra_py_file)
        self.shell_cmd = ["python3 modify.py"]

    def tearDown(self):
        os.remove(self.lmp_template_fname)
        os.remove(self.plm_template_fname)
        os.remove(self.py_script)
        # generated by CustomizedLmpTemplateTaskGroup
        # shutil.rmtree("task_0")
        # shutil.rmtree("task_1")

    def test_lmp(self):
        task_group = CustomizedLmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            custom_shell_commands=self.shell_cmd,
            revisions=self.lmp_rev_mat,
            traj_freq=self.traj_freq,
            input_lmp_conf_name="foo.lmp",
            input_lmp_tmpl_name=self.lmp_template_fname,
            input_plm_tmpl_name=self.plm_template_fname,
            input_extra_files=[self.py_script],
            output_dir_pattern="task_*",
            output_lmp_conf_name="bar.lmp",
            output_lmp_tmpl_name="lmp.template",
            output_plm_tmpl_name="plm.template",
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs)
            * len(self.lmp_rev_mat["V_NSTEPS"])
            * len(self.lmp_rev_mat["V_TEMP"])
            * len(self.lmp_rev_mat["V_DIST0"])
            * 2,
        )
        idx = 0
        for cc, dd, ii, jj, kk in itertools.product(
            range(len(self.confs)),
            range(2),
            range(len(self.lmp_rev_mat["V_NSTEPS"])),
            range(len(self.lmp_rev_mat["V_TEMP"])),
            range(len(self.lmp_rev_mat["V_DIST0"])),
        ):
            eel = expected_lmp_plm_template.split("\n")
            eel[0] = eel[0].replace("V_NSTEPS", str(self.lmp_rev_mat["V_NSTEPS"][ii]))
            eel[3] = eel[3].replace("V_TEMP", str(self.lmp_rev_mat["V_TEMP"][jj]))
            # replaced by the shell script.
            eel[6] = eel[6].replace("0.500000", str(dd))
            eep = in_plm_template.split("\n")
            eep[0] = eep[0].replace("V_TEMP", str(self.lmp_rev_mat["V_TEMP"][jj]))
            eep[3] = eep[3].replace("V_DIST0", str(self.lmp_rev_mat["V_DIST0"][kk]))
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                eel,
            )
            self.assertEqual(
                task_group[idx].files()[plm_input_name].split("\n"),
                eep,
            )
            idx += 1

    def test_no_match(self):
        task_group = CustomizedLmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            custom_shell_commands=self.shell_cmd,
            revisions=self.lmp_rev_mat,
            traj_freq=self.traj_freq,
            input_lmp_conf_name="foo.lmp",
            input_lmp_tmpl_name=self.lmp_template_fname,
            input_plm_tmpl_name=self.plm_template_fname,
            input_extra_files=[self.py_script],
            output_dir_pattern="aaa_*",
            output_lmp_conf_name="bar.lmp",
            output_lmp_tmpl_name="lmp.template",
            output_plm_tmpl_name="plm.template",
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(ngroup, 0)


class TestLmpTemplateTaskGroupLmp(unittest.TestCase):
    def setUp(self):
        self.lmp_template_fname = Path("lmp.template")
        self.lmp_template_fname.write_text(in_lmp_template)
        self.numb_models = 4
        self.confs = ["foo", "bar"]
        self.lmp_rev_mat = {
            "V_NSTEPS": [1000],
            "V_TEMP": [50, 100],
        }
        self.rev_empty = {}
        self.traj_freq = 20
        self.py_script = Path("modify.py")
        self.py_script.write_text(in_extra_lmp_py_file)
        self.shell_cmd = ["python3 modify.py"]

    def tearDown(self):
        os.remove(self.lmp_template_fname)
        os.remove(self.py_script)

    def test_lmp(self):
        task_group = CustomizedLmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            custom_shell_commands=self.shell_cmd,
            revisions=self.lmp_rev_mat,
            traj_freq=self.traj_freq,
            input_lmp_conf_name="foo.lmp",
            input_lmp_tmpl_name=self.lmp_template_fname,
            input_plm_tmpl_name=None,
            input_extra_files=[self.py_script],
            output_dir_pattern="task_*",
            output_lmp_conf_name="bar.lmp",
            output_lmp_tmpl_name="lmp.template",
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs)
            * len(self.lmp_rev_mat["V_NSTEPS"])
            * len(self.lmp_rev_mat["V_TEMP"])
            * 2,
        )
        idx = 0
        for cc, dd, ii, jj in itertools.product(
            range(len(self.confs)),
            range(2),
            range(len(self.lmp_rev_mat["V_NSTEPS"])),
            range(len(self.lmp_rev_mat["V_TEMP"])),
        ):
            eel = expected_lmp_template.split("\n")
            eel[0] = eel[0].replace("V_NSTEPS", str(self.lmp_rev_mat["V_NSTEPS"][ii]))
            eel[3] = eel[3].replace("V_TEMP", str(self.lmp_rev_mat["V_TEMP"][jj]))
            # replaced by the shell script.
            eel[6] = eel[6].replace("0.500000", str(dd))
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                eel,
            )
            idx += 1
