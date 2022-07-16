from typing import (
    List,
    Tuple,
)
from dflow.python import (
    FatalError,
)
from pathlib import Path
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.task import ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.selector import ConfSelector
from . import StageScheduler

class ConvergenceCheckStageScheduler(StageScheduler):    
    def __init__(
            self,
            stage : ExplorationStage,
            selector : ConfSelector,
            conv_accuracy : float = 0.9,
            max_numb_iter : int = None,
            fatal_at_max : bool = True,
    ):
        self.stage = stage
        self.selector = selector
        self.conv_accuracy = conv_accuracy
        self.max_numb_iter = max_numb_iter
        self.fatal_at_max = fatal_at_max
        self.nxt_iter = 0

    def plan_next_iteration(
            self,
            hist_reports : List[ExplorationReport] = [],
            report : ExplorationReport = None,
            trajs : List[Path] = None,
    ) -> Tuple[bool, ExplorationTaskGroup, ConfSelector] :
        if report is None:
            converged = False
            lmp_task_grp = self.stage.make_task()
            ret_selector = self.selector
        else :
            converged = report.accurate_ratio() >= self.conv_accuracy
            if not converged:
                # check if we have any candidate to improve the quality of the model
                if report.candidate_ratio() == 0.0:
                    raise FatalError(
                        'The iteration is not converted, but we find that '
                        'it does not selected any candidate configuration. '
                        'This means the quality of the model would not be '
                        'improved and the iteraction would not end. '
                        'Please try to increase the higher trust levels. '
                    )
                # if not converged, check max iter
                if self.max_numb_iter is not None and self.nxt_iter == self.max_numb_iter:
                    if self.fatal_at_max:
                        raise FatalError('reached maximal number of iterations')
                    else:
                        converged = True
            # make lmp tasks
            if converged:
                # if converged, no more lmp task
                lmp_task_grp = None
                ret_selector = None
            else :                        
                lmp_task_grp = self.stage.make_task()
                ret_selector = self.selector
        self.nxt_iter += 1
        return converged, lmp_task_grp, ret_selector

