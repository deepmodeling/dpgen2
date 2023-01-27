import os, textwrap
import numpy as np
import unittest

from typing import Set, List
from pathlib import Path

try:
    from exploration.context import dpgen2
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dflow.python import (
    FatalError,
)
from dpgen2.exploration.scheduler import (
    ConvergenceCheckStageScheduler,
    ExplorationScheduler,
)
from dpgen2.exploration.report import ExplorationReport, ExplorationReportTrustLevels
from dpgen2.exploration.task import ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.selector import ConfSelectorFrames
from dpgen2.exploration.render import TrajRenderLammps
from mocked_ops import (
    MockedExplorationReport,
    MockedExplorationTaskGroup,
    MockedExplorationTaskGroup1,
    MockedStage,
    MockedStage1,
)


class TestConvergenceCheckStageScheduler(unittest.TestCase):
    def test_success(self):
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        self.selector = ConfSelectorFrames(traj_render, report)
        self.scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            self.selector,
        )
        foo_report = MockedExplorationReport()
        foo_report.accurate = 1.0
        foo_report.failed = 0.0

        conv, ltg, sel = self.scheduler.plan_next_iteration()
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)

        conv, ltg, sel = self.scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, True)
        self.assertTrue(ltg is None)
        self.assertTrue(sel is None)

    def test_step1(self):
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        self.selector = ConfSelectorFrames(traj_render, report)
        self.scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            self.selector,
        )

        foo_report = MockedExplorationReport()
        foo_report.accurate = 0.5
        foo_report.failed = 0.5
        bar_report = MockedExplorationReport()
        bar_report.accurate = 1.0
        bar_report.failed = 0.0

        conv, ltg, sel = self.scheduler.plan_next_iteration()
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)

        conv, ltg, sel = self.scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)

        conv, ltg, sel = self.scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(conv, True)
        self.assertTrue(ltg is None)
        self.assertTrue(sel is None)
        # self.assertTrue(isinstance(sel, ConfSelectorFrames))
        # self.assertEqual(sel.report.level_f_lo, 0.1)
        # self.assertEqual(sel.report.level_f_hi, 0.3)
        # self.assertTrue(sel.report.level_v_lo is None)
        # self.assertTrue(sel.report.level_v_hi is None)

    def test_max_numb_iter(self):
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        self.selector = ConfSelectorFrames(traj_render, report)
        self.scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            self.selector,
            max_numb_iter=2,
        )

        foo_report = MockedExplorationReport()
        foo_report.accurate = 0.5
        foo_report.failed = 0.5
        bar_report = MockedExplorationReport()
        bar_report.accurate = 1.0
        bar_report.failed = 0.0

        conv, ltg, sel = self.scheduler.plan_next_iteration()
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)

        conv, ltg, sel = self.scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)

        with self.assertRaisesRegex(FatalError, "reached maximal number of iterations"):
            conv, ltg, sel = self.scheduler.plan_next_iteration(foo_report, [])


class TestExplorationScheduler(unittest.TestCase):
    def test_success(self):
        scheduler = ExplorationScheduler()
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

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
        self.assertFalse(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())
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
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
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
        self.assertFalse(scheduler.complete())
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(conv, True)
        self.assertTrue(ltg is None)
        self.assertTrue(sel is None)
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 2)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertTrue(scheduler.stage_schedulers[1].converged())
        self.assertTrue(scheduler.stage_schedulers[1].complete())
        self.assertTrue(scheduler.complete())

    def test_print_scheduler(self):
        scheduler = ExplorationScheduler()
        report_0 = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render_0 = TrajRenderLammps()
        selector_0 = ConfSelectorFrames(traj_render_0, report_0)
        stage_scheduler_0 = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector_0,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler_0)
        report_1 = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render_1 = TrajRenderLammps()
        selector_1 = ConfSelectorFrames(traj_render_1, report_1)
        stage_scheduler_1 = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector_1,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler_1)

        tar_report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        tar_report.record([np.array([0.08, 0.05])])
        foo_report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        foo_report.record([np.array([0.3, 0.9])])
        bar_report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        bar_report.record([np.array([0.1, 0.1])])

        expected_output = [
            "#   stage  id_stg.    iter.      accu.      cand.      fail.   lvl_f_lo   lvl_f_hi    cvged",
            "# Stage    0  --------------------",
            "        0        0        0     1.0000     0.0000     0.0000     0.1000     0.3000     True",
            "# Stage    0  converged YES  reached max numb iterations NO ",
            "# Stage    1  --------------------",
            "        1        0        1     0.0000     0.5000     0.5000     0.2000     0.4000    False",
            "        1        1        2     1.0000     0.0000     0.0000     0.2000     0.4000     True",
            "# Stage    1  converged YES  reached max numb iterations NO ",
            "# All stages converged",
        ]
        self.assertEqual(scheduler.print_convergence(), "No finished iteration found\n")
        conv, ltg, sel = scheduler.plan_next_iteration()
        self.assertEqual(scheduler.print_convergence(), "No finished iteration found\n")
        conv, ltg, sel = scheduler.plan_next_iteration(tar_report, [])
        self.assertEqual(
            scheduler.print_convergence(), "\n".join(expected_output[:3]) + "\n"
        )
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(
            scheduler.print_convergence(), "\n".join(expected_output[:6]) + "\n"
        )
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(
            scheduler.print_convergence(), "\n".join(expected_output) + "\n"
        )

    def test_print_scheduler_last_iteration(self):
        scheduler = ExplorationScheduler()
        report_0 = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render_0 = TrajRenderLammps()
        selector_0 = ConfSelectorFrames(traj_render_0, report_0)
        stage_scheduler_0 = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector_0,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler_0)
        report_1 = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render_1 = TrajRenderLammps()
        selector_1 = ConfSelectorFrames(traj_render_1, report_1)
        stage_scheduler_1 = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector_1,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler_1)

        tar_report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        tar_report.record([np.array([0.08, 0.05])])
        foo_report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        foo_report.record([np.array([0.3, 0.9])])
        bar_report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        bar_report.record([np.array([0.1, 0.1])])

        expected_output = [
            "#   stage  id_stg.    iter.      accu.      cand.      fail.   lvl_f_lo   lvl_f_hi    cvged",
            "        0        0        0     1.0000     0.0000     0.0000     0.1000     0.3000     True",
            "        1        0        1     0.0000     0.5000     0.5000     0.2000     0.4000    False",
            "        1        1        2     1.0000     0.0000     0.0000     0.2000     0.4000     True",
            "# All stages converged",
        ]
        self.assertEqual(
            scheduler.print_last_iteration(), "No finished iteration found\n"
        )
        conv, ltg, sel = scheduler.plan_next_iteration()
        self.assertEqual(
            scheduler.print_last_iteration(), "No finished iteration found\n"
        )
        conv, ltg, sel = scheduler.plan_next_iteration(tar_report, [])
        self.assertEqual(
            scheduler.print_last_iteration(print_header=True),
            "\n".join(expected_output[:2]) + "\n",
        )
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(
            scheduler.print_last_iteration(), "\n".join(expected_output[2:3]) + "\n"
        )
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(
            scheduler.print_last_iteration(), "\n".join(expected_output[3:]) + "\n"
        )

    def test_success_and_ratios(self):
        scheduler = ExplorationScheduler()
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=4,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=4,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

        foo_report = MockedExplorationReport()
        foo_report.accurate = 0.5
        foo_report.failed = 0.2
        foo_report.candidate = 0.3
        bar_report = MockedExplorationReport()
        bar_report.accurate = 1.0
        bar_report.failed = 0.0
        bar_report.candidate = 0.0

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
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.2)
        self.assertEqual(sel.report.level_f_hi, 0.4)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 2)
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.2)
        self.assertEqual(sel.report.level_f_hi, 0.4)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 3)
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(conv, True)
        self.assertTrue(ltg is None)
        self.assertTrue(sel is None)
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 3)

        expected_stage_idx = np.array([0, 1, 1, 1], dtype=int)
        expected_idx_in_stage = np.array([0, 0, 1, 2], dtype=int)
        expected_iter_idx = np.array([0, 1, 2, 3], dtype=int)
        expected_accu = np.array([1.0, 0.5, 0.5, 1.0], dtype=float)
        expected_cand = np.array([0.0, 0.3, 0.3, 0.0], dtype=float)
        expected_fail = np.array([0.0, 0.2, 0.2, 0.0], dtype=float)

        stage_idx, idx_in_stage, iter_idx = scheduler.get_stage_of_iterations()
        np.testing.assert_array_equal(stage_idx, expected_stage_idx)
        np.testing.assert_array_equal(idx_in_stage, expected_idx_in_stage)
        np.testing.assert_array_equal(iter_idx, expected_iter_idx)

        accu, cand, fail = scheduler.get_convergence_ratio()
        np.testing.assert_array_almost_equal(accu, expected_accu)
        np.testing.assert_array_almost_equal(cand, expected_cand)
        np.testing.assert_array_almost_equal(fail, expected_fail)

    def test_success_two_stages(self):
        scheduler = ExplorationScheduler()
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

        report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=3,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

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
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup1))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.2)
        self.assertEqual(sel.report.level_f_hi, 0.4)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 3)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())
        self.assertFalse(scheduler.complete())
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(conv, True)
        self.assertTrue(ltg is None)
        self.assertTrue(sel is None)
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 3)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertTrue(scheduler.stage_schedulers[1].converged())
        self.assertTrue(scheduler.stage_schedulers[1].complete())
        self.assertTrue(scheduler.complete())

    def test_continue_adding_success(self):
        scheduler = ExplorationScheduler()
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

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
        self.assertEqual(len(scheduler.stage_schedulers), 1)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(conv, True)
        self.assertTrue(ltg is None)
        self.assertTrue(sel is None)
        self.assertEqual(scheduler.get_stage(), 1)
        self.assertEqual(scheduler.get_iteration(), 0)
        self.assertEqual(len(scheduler.stage_schedulers), 1)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertTrue(scheduler.complete())

        report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())
        self.assertFalse(scheduler.complete())
        conv, ltg, sel = scheduler.plan_next_iteration()
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
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
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
        self.assertFalse(scheduler.complete())
        conv, ltg, sel = scheduler.plan_next_iteration(bar_report, [])
        self.assertEqual(conv, True)
        self.assertTrue(ltg is None)
        self.assertTrue(sel is None)
        self.assertEqual(scheduler.get_stage(), 2)
        self.assertEqual(scheduler.get_iteration(), 2)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertTrue(scheduler.stage_schedulers[0].converged())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertTrue(scheduler.stage_schedulers[1].converged())
        self.assertTrue(scheduler.stage_schedulers[1].complete())
        self.assertTrue(scheduler.complete())

    def test_failed_stage0(self):
        scheduler = ExplorationScheduler()
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

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
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 0)
        self.assertEqual(scheduler.get_iteration(), 1)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.complete())
        with self.assertRaisesRegex(
            FatalError, "stage 0: reached maximal number of iterations"
        ):
            conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])

    def test_failed_stage0_not_fatal(self):
        scheduler = ExplorationScheduler()
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
            fatal_at_max=False,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=2,
            fatal_at_max=False,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

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
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
        self.assertEqual(conv, False)
        self.assertTrue(isinstance(ltg, MockedExplorationTaskGroup))
        self.assertTrue(isinstance(sel, ConfSelectorFrames))
        self.assertEqual(sel.report.level_f_lo, 0.1)
        self.assertEqual(sel.report.level_f_hi, 0.3)
        self.assertTrue(sel.report.level_v_lo is None)
        self.assertTrue(sel.report.level_v_hi is None)
        self.assertEqual(scheduler.get_stage(), 0)
        self.assertEqual(scheduler.get_iteration(), 1)
        self.assertEqual(len(scheduler.stage_schedulers), 2)
        self.assertFalse(scheduler.stage_schedulers[0].converged())
        self.assertFalse(scheduler.stage_schedulers[0].reached_max_iteration())
        self.assertFalse(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
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
        self.assertTrue(scheduler.stage_schedulers[0].reached_max_iteration())
        self.assertTrue(scheduler.stage_schedulers[0].complete())
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        self.assertFalse(scheduler.stage_schedulers[1].complete())
        self.assertFalse(scheduler.complete())

    def test_failed_stage1(self):
        scheduler = ExplorationScheduler()
        report = ExplorationReportTrustLevels(0.1, 0.3, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)
        report = ExplorationReportTrustLevels(0.2, 0.4, conv_accuracy=0.9)
        traj_render = TrajRenderLammps()
        selector = ConfSelectorFrames(traj_render, report)
        stage_scheduler = ConvergenceCheckStageScheduler(
            MockedStage1(),
            selector,
            max_numb_iter=2,
        )
        scheduler.add_stage_scheduler(stage_scheduler)

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
        self.assertFalse(scheduler.stage_schedulers[1].converged())
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
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
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
        self.assertFalse(scheduler.stage_schedulers[1].converged())
        with self.assertRaisesRegex(
            FatalError, "stage 1: reached maximal number of iterations"
        ):
            conv, ltg, sel = scheduler.plan_next_iteration(foo_report, [])
