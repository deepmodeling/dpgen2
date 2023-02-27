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

import dpdata
import numpy as np

from dpgen2.entrypoint.submit import (
    copy_scheduler_plans,
    expand_idx,
    print_list_steps,
    submit_concurrent_learning,
    update_reuse_step_scheduler,
)
from dpgen2.exploration.render import (
    TrajRenderLammps,
)
from dpgen2.exploration.report import (
    ExplorationReport,
    ExplorationReportTrustLevelsRandom,
)
from dpgen2.exploration.scheduler import (
    ConvergenceCheckStageScheduler,
    ExplorationScheduler,
)
from dpgen2.exploration.selector import (
    ConfSelectorFrames,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTaskGroup,
)

# isort: off
from .context import (
    dpgen2,
)
from mocked_ops import (
    MockedExplorationReport,
    MockedExplorationTaskGroup,
    MockedExplorationTaskGroup1,
    MockedStage,
    MockedStage1,
)

# isort: on

ifc0 = """Al1 
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Al 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc0 = "\n1 atoms\n2 atom types\n   0.0000000000    2.0000000000 xlo xhi\n   0.0000000000    2.0000000000 ylo yhi\n   0.0000000000    2.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      1    0.0000000000    0.0000000000    0.0000000000\n"

ifc1 = """Mg1 
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Mg 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc1 = "\n1 atoms\n2 atom types\n   0.0000000000    3.0000000000 xlo xhi\n   0.0000000000    3.0000000000 ylo yhi\n   0.0000000000    3.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n"

ifc2 = """Mg1 
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Mg 
1 
cartesian
   0.0000000000    0.0000000000    0.0000000000
"""
ofc2 = "\n1 atoms\n2 atom types\n   0.0000000000    4.0000000000 xlo xhi\n   0.0000000000    4.0000000000 ylo yhi\n   0.0000000000    4.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n"


class MockedScheduler:
    def __init__(self, value=0):
        self.value = value


class MockedStep:
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self.key = f"iter-{self.scheduler.value}--scheduler"

    def modify_output_parameter(self, key, scheduler):
        assert key == "exploration_scheduler"
        self.scheduler = scheduler


class TestSubmit(unittest.TestCase):
    def test_expand_idx(self):
        ilist = ["1", "3-5", "10-20:2"]
        olist = expand_idx(ilist)
        expected_olist = [1, 3, 4, 10, 12, 14, 16, 18]
        self.assertEqual(olist, expected_olist)

    def test_print_list_steps(self):
        ilist = ["foo", "bar"]
        ostr = print_list_steps(ilist)
        expected_ostr = "       0    foo\n       1    bar"
        self.assertEqual(ostr, expected_ostr)

    def test_update_reuse_step_scheduler(self):
        reuse_steps = [
            MockedStep(MockedScheduler(0)),
            MockedStep(MockedScheduler(1)),
            MockedStep(MockedScheduler(2)),
            MockedStep(MockedScheduler(3)),
        ]

        reuse_steps = update_reuse_step_scheduler(
            reuse_steps,
            MockedScheduler(4),
        )
        self.assertEqual(len(reuse_steps), 4)
        self.assertEqual(reuse_steps[0].scheduler.value, 0)
        self.assertEqual(reuse_steps[1].scheduler.value, 1)
        self.assertEqual(reuse_steps[2].scheduler.value, 2)
        self.assertEqual(reuse_steps[3].scheduler.value, 4)

    def test_copy_scheduler(self):
        scheduler = ExplorationScheduler()
        scheduler_new = ExplorationScheduler()
        report = ExplorationReportTrustLevelsRandom(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler_new.add_stage_scheduler(stage_scheduler)

        report = ExplorationReportTrustLevelsRandom(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=3,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=3,
        )
        scheduler_new.add_stage_scheduler(stage_scheduler)

        foo_report = MockedExplorationReport()
        foo_report.accurate = 0.5
        foo_report.failed = 0.5
        bar_report = MockedExplorationReport()
        bar_report.accurate = 1.0
        bar_report.failed = 0.0

        conv, ltg, sel = scheduler.plan_next_iteration()
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 0)
        self.assertEqual(scheduler.get_iteration(), 0)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.2)
        self.assertEqual(sel.report.level_f_hi, 0.4)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 1)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())
        self.assertFalse(scheduler.complete())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report)
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.2)
        self.assertEqual(sel.report.level_f_hi, 0.4)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 2)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())

        scheduler_new = copy_scheduler_plans(scheduler_new, scheduler)

        self.assertEqual(scheduler.get_stage(), scheduler_new.get_stage())
        self.assertEqual(scheduler.get_iteration(), scheduler_new.get_iteration())
        self.assertEqual(scheduler.complete(), scheduler_new.complete())
        self.assertEqual(
            scheduler.print_convergence(), scheduler_new.print_convergence()
        )

    def test_copy_scheduler_complete(self):
        scheduler = ExplorationScheduler()
        scheduler_new = ExplorationScheduler()
        report = ExplorationReportTrustLevelsRandom(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=1,
            fatal_at_max=False,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler_new.add_stage_scheduler(stage_scheduler)

        report = ExplorationReportTrustLevelsRandom(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=3,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=3,
        )
        scheduler_new.add_stage_scheduler(stage_scheduler)

        foo_report = MockedExplorationReport()
        foo_report.accurate = 0.5
        foo_report.failed = 0.5
        bar_report = MockedExplorationReport()
        bar_report.accurate = 1.0
        bar_report.failed = 0.0

        conv, ltg, sel = scheduler.plan_next_iteration()
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 0)
        self.assertEqual(scheduler.get_iteration(), 0)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.2)
        self.assertEqual(sel.report.level_f_hi, 0.4)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 1)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())
        self.assertFalse(scheduler.complete())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report)
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.2)
        self.assertEqual(sel.report.level_f_hi, 0.4)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 2)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())

        scheduler_new = copy_scheduler_plans(scheduler_new, scheduler)

        self.assertEqual(scheduler.get_iteration(), scheduler_new.get_iteration())
        self.assertEqual(scheduler.get_stage(), scheduler_new.get_stage())
        self.assertEqual(scheduler.complete(), scheduler_new.complete())
        # 1st stage of scheduler_new is forced complete.
        self.assertEqual(
            scheduler.print_convergence().replace(
                "reached max numb iterations YES",
                "reached max numb iterations NO ",
            ),
            scheduler_new.print_convergence(),
        )


class TestSubmitCmdStd(unittest.TestCase):
    def setUp(self):
        from dflow import (
            config,
        )

        config["mode"] = "debug"
        self.touched_files = [
            "foo",
            "init",
            "bar",
            "tar",
            "INCAR",
            "POTCAR.Al",
            "POTCAR.Mg",
        ]
        for ii in self.touched_files:
            Path(ii).touch()

    def tearDown(self):
        from dflow import (
            config,
        )

        config["mode"] = None
        for ii in self.touched_files:
            os.remove(ii)

    def test(self):
        wf_config = json.loads(input_std)
        submit_concurrent_learning(wf_config, no_submission=True)


class TestSubmitCmdDist(unittest.TestCase):
    def setUp(self):
        from dflow import (
            config,
        )

        config["mode"] = "debug"
        self.touched_files = [
            "foo",
            "init",
            "teacher_model.pb",
            "student_model.pb",
        ]
        for ii in self.touched_files:
            Path(ii).touch()
        Path("POSCAR").write_text(ifc0)

    def tearDown(self):
        from dflow import (
            config,
        )

        config["mode"] = None
        for ii in self.touched_files + ["POSCAR"]:
            os.remove(ii)

    def test(self):
        wf_config = json.loads(input_dist)
        submit_concurrent_learning(wf_config, no_submission=True)


input_std = textwrap.dedent(
    """
{
    "default_step_config" : {
	"template_config" : {
	    "image" : "dflow:1.1.4",
	    "_comment" : "all"
	},
	"_comment" : "all"
    },

    "step_configs":{
	"run_train_config" : {
	    "template_config" : {
		"image" : "deepmd-kit:wanghan",
		"_comment" : "all"
	    },
	    "executor" : {
                "type": "dispatcher",
                "username": "foo"
	    },
	    "_comment" : "all"
	},
	"run_explore_config" : {
	    "template_config" : {
		"image" : "deepmd-kit:wanghan",
		"_comment" : "all"
	    },
	    "executor" : {
                "type": "dispatcher",
                "username": "foo"
	    },
	    "_comment" : "all"
	},
	"run_fp_config" : {
	    "template_config" : {
		"image" : "vasp:wanghan",
		"_comment" : "all"
	    },
	    "executor" : {
                "type": "dispatcher",
                "username": "foo"
	    },
	    "_comment" : "all"
	},
	"_comment" : "all"
    },

    "inputs": {
	"type_map":		["Al", "Mg"],
	"mass_map":		[27, 24],
	"init_data_prefix":	null,
	"init_data_sys":	[
	    "init"
	],
	"_comment" : "all"
    },
    "train":{
	"type" :	"dp",
	"numb_models" : 2,
	"config" : {},
	"template_script" : "foo",
"init_models_paths" : ["bar", "tar"],
	"_comment" : "all"
    },

    "explore" : {
	"type" : "lmp",
	"config" : {
	    "command": "lmp -var restart 0"
	},
	"convergence": {
	    "type" :	"fixed-levels",
	    "conv_accuracy" :	0.9,
	    "level_f_lo":	0.05,
	    "level_f_hi":	0.50,
	    "_comment" : "all"
	},
	"max_numb_iter" :	5,
	"fatal_at_max" :	false,
	"output_nopbc":		false,
	"configuration_prefix": null,
	"configurations":	[
	    {
		"type": "alloy",
		"lattice" : ["fcc", 4.57],
		"replicate" : [2, 2, 2],
		"numb_confs" : 30,
		"concentration" : [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
	    }
	],
	"_comment" : "Stage is of type List[List[dict]]. ",
	"_comment" : "The outer list gives stages, the inner list gives the task groups of the stage, and dict describes the task group.",
	"stages":	[
	    [
		{
		    "type" : "lmp-md",
		    "ensemble": "nvt", "nsteps":  50, "press": [1e0], "temps": [50], "trj_freq": 10,
		    "conf_idx": [0], "n_sample" : 3
		}
	    ]
	],
	"_comment" : "all"
    },
    "fp" : {
	"type" :	"vasp",
	"task_max":	2,
	"inputs_config" : {
	    "pp_files":	{"Al" : "POTCAR.Al", "Mg" : "POTCAR.Mg"},
	    "incar":    "INCAR",
	    "kspacing":	0.32,
	    "kgamma":	true
	},
	"run_config" : {
	    "command": "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
	},
	"_comment" : "all"
    }
}
"""
)

input_dist = textwrap.dedent(
    """
{
    "default_step_config": {
        "template_config": {
            "image": "",
            "_comment": "all"
        },
        "executor": {
            "type": "dispatcher",
            "image_pull_policy": "IfNotPresent",
            "machine_dict": {
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type": "c2_m4_cpu"
                    }
                }
            }
        },
        "_comment": "all"
    },
    "step_configs": {
        "run_train_config": {
            "template_config": {
                "image": "",
                "_comment": "all"
            },
            "executor": {
                "type": "dispatcher",
                "image_pull_policy": "IfNotPresent",
                "machine_dict": {
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": "c8_m31_1 * NVIDIA T4"
                        }
                    }
                }
            },
            "_comment": "all"
        },
        "run_explore_config": {
            "template_config": {
                "image": "",
                "_comment": "all"
            },
            "executor": {
                "type": "dispatcher",
                "image_pull_policy": "IfNotPresent",
                "machine_dict": {
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": "c8_m31_1 * NVIDIA T4"
                        }
                    }
                }
            },
            "_comment": "all"
        },
        "run_fp_config": {
            "template_config": {
                "image": "",
                "_comment": "all"
            },
            "executor": {
                "type": "dispatcher",
                "image_pull_policy": "IfNotPresent",
                "machine_dict": {
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": "c8_m32_cpu"
                        }
                    }
                }
            },
            "_comment": "all"
        },
        "_comment": "all"
    },
    "upload_python_packages": [
    ],
    "inputs": {
        "type_map": [
            "H",
            "C",
            "Al"
        ],
        "mass_map": [
            4,
            12
        ],
        "init_data_prefix": null,
        "init_data_sys": [
            "init"
        ],
        "_comment": "all"
    },
    "train": {
        "student_model_path": "student_model.pb",
        "type": "dp-dist",
        "config": {
            "init_model_policy": "yes",
            "init_model_old_ratio": 0.5,
            "init_model_numb_steps": 200000,
            "init_model_start_lr": 1e-4,
            "init_model_start_pref_e": 0.25,
            "init_model_start_pref_f": 100,
            "_comment": "all"
        },
        "template_script": "foo",
        "_comment": "all"
    },
    "fp" : {
        "type" :	"deepmd",
        "task_max":	2,
        "run_config" : {
            "teacher_model_path": "teacher_model.pb",
            "type_map": ["H", "C"]
        },
        "inputs_config" : {},
        "_comment" : "all"
    },
    "explore": {
        "type": "lmp",
        "config": {
            "teacher_model_path": "teacher_model.pb",
            "command": "lmp -var restart 0"
        },
        "convergence": {
            "type" :	"fixed-levels",
            "conv_accuracy" :	0.9,
            "level_f_lo":	0.05,
            "level_f_hi":	0.50,
            "_comment" : "all"
        },
        "max_numb_iter": 2,
        "fatal_at_max": false,
        "output_nopbc": false,
        "configuration_prefix": null,
        "configurations": [
            {
                "type": "file",
                "files": [
                    "POSCAR"
                ],
                "fmt": "vasp/poscar"
            }
        ],
        "stages": [
            [
            {
                "type" : "lmp-md",
                "ensemble": "nvt", "nsteps":  50, "press": [1e0], "temps": [50], "trj_freq": 10,
                "conf_idx": [0], "n_sample" : 3
            }
            ]
        ],
        "_comment": "all"
    }
}
"""
)
