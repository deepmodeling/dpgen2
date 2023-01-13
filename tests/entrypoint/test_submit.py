from .context import dpgen2
import numpy as np
import unittest, json, shutil, os
import random
import tempfile
import dpdata
from pathlib import Path
from dpgen2.entrypoint.submit import (
    expand_idx,
    print_list_steps,
    update_reuse_step_scheduler,
    copy_scheduler_plans,
)
from dpgen2.exploration.scheduler import (
    ConvergenceCheckStageScheduler,
    ExplorationScheduler,
)
from dpgen2.exploration.report import ExplorationReport, ExplorationReportTrustLevels
from dpgen2.exploration.task import ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.selector import TrustLevel, ConfSelectorFrames
from dpgen2.exploration.render import TrajRenderLammps
from mocked_ops import (
    MockedExplorationReport,
    MockedExplorationTaskGroup,
    MockedExplorationTaskGroup1,
    MockedStage,
    MockedStage1,
)


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
ofc0 = '\n1 atoms\n2 atom types\n   0.0000000000    2.0000000000 xlo xhi\n   0.0000000000    2.0000000000 ylo yhi\n   0.0000000000    2.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      1    0.0000000000    0.0000000000    0.0000000000\n'

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
ofc1 = '\n1 atoms\n2 atom types\n   0.0000000000    3.0000000000 xlo xhi\n   0.0000000000    3.0000000000 ylo yhi\n   0.0000000000    3.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n'

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
ofc2 = '\n1 atoms\n2 atom types\n   0.0000000000    4.0000000000 xlo xhi\n   0.0000000000    4.0000000000 ylo yhi\n   0.0000000000    4.0000000000 zlo zhi\n   0.0000000000    0.0000000000    0.0000000000 xy xz yz\n\nAtoms # atomic\n\n     1      2    0.0000000000    0.0000000000    0.0000000000\n'


class MockedScheduler():
    def __init__(self, value=0):
        self.value = value

class MockedStep():
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self.key = f"iter-{self.scheduler.value}--scheduler"

    def modify_output_parameter(self, key, scheduler):
        assert key == "exploration_scheduler"
        self.scheduler = scheduler


class TestSubmit(unittest.TestCase):
    def test_expand_idx(self):
        ilist = ['1', '3-5', '10-20:2']
        olist = expand_idx(ilist)
        expected_olist = [1, 3, 4, 10, 12, 14, 16, 18]
        self.assertEqual(olist, expected_olist)


    def test_print_list_steps(self):
        ilist = ['foo', 'bar']
        ostr = print_list_steps(ilist)
        expected_ostr = '       0    foo\n       1    bar'
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
        trust_level = TrustLevel(0.1, 0.3)
        report = ExplorationReportTrustLevels(trust_level, 0.9)
        traj_render = TrajRenderLammps()        
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter = 2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter = 2,
        )
        scheduler_new.add_stage_scheduler(stage_scheduler)

        trust_level = TrustLevel(0.2, 0.4)
        report = ExplorationReportTrustLevels(trust_level, 0.9)
        traj_render = TrajRenderLammps()        
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter = 3,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter = 3,
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
        self.assertEqual(sel.report.trust_level.level_f_lo, 0.1)
        self.assertEqual(sel.report.trust_level.level_f_hi, 0.3)
        self.assertTrue(sel.report.trust_level.level_v_lo is None)
        self.assertTrue(sel.report.trust_level.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 0)
        self.assertEqual(scheduler.get_iteration(), 0)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])        
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.trust_level.level_f_lo, 0.2)
        self.assertEqual(sel.report.trust_level.level_f_hi, 0.4)
        self.assertTrue(sel.report.trust_level.level_v_lo is None)
        self.assertTrue(sel.report.trust_level.level_v_hi is None)
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
        self.assertEqual(sel.report.trust_level.level_f_lo, 0.2)
        self.assertEqual(sel.report.trust_level.level_f_hi, 0.4)
        self.assertTrue(sel.report.trust_level.level_v_lo is None)
        self.assertTrue(sel.report.trust_level.level_v_hi is None)
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
        self.assertEqual(scheduler.print_convergence(), scheduler_new.print_convergence())


    def test_copy_scheduler_complete(self):
        scheduler = ExplorationScheduler()
        scheduler_new = ExplorationScheduler()
        trust_level = TrustLevel(0.1, 0.3)
        report = ExplorationReportTrustLevels(trust_level, 0.9)
        traj_render = TrajRenderLammps()        
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter = 1,
            fatal_at_max = False,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter = 2,
        )
        scheduler_new.add_stage_scheduler(stage_scheduler)

        trust_level = TrustLevel(0.2, 0.4)
        report = ExplorationReportTrustLevels(trust_level, 0.9)
        traj_render = TrajRenderLammps()        
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter = 3,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter = 3,
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
        self.assertEqual(sel.report.trust_level.level_f_lo, 0.1)
        self.assertEqual(sel.report.trust_level.level_f_hi, 0.3)
        self.assertTrue(sel.report.trust_level.level_v_lo is None)
        self.assertTrue(sel.report.trust_level.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 0)
        self.assertEqual(scheduler.get_iteration(), 0)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])        
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.trust_level.level_f_lo, 0.2)
        self.assertEqual(sel.report.trust_level.level_f_hi, 0.4)
        self.assertTrue(sel.report.trust_level.level_v_lo is None)
        self.assertTrue(sel.report.trust_level.level_v_hi is None)
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
        self.assertEqual(sel.report.trust_level.level_f_lo, 0.2)
        self.assertEqual(sel.report.trust_level.level_f_hi, 0.4)
        self.assertTrue(sel.report.trust_level.level_v_lo is None)
        self.assertTrue(sel.report.trust_level.level_v_hi is None)
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
            'reached max numb iterations YES',
            'reached max numb iterations NO ',),
            scheduler_new.print_convergence())
