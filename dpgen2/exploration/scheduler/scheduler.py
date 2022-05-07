from typing import (
    List,
    Tuple,
)
from dflow.python import (
    FatalError,
)
from pathlib import Path
from . import (
    StageScheduler,
)
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.task import ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.selector import ConfSelector, TrustLevel


class ExplorationScheduler():
    """
    The exploration scheduler.

    """

    def __init__(
            self,
    ):
        self.stage_schedulers = []
        self.stage_reports = [[]]
        self.cur_stage = 0
        self.iteration = -1
        
    def add_stage_scheduler(
            self,
            stage_scheduler : StageScheduler,
    ):
        """
        Add stage scheduler. 

        All added schedulers can be treated as a `list` (order matters). Only one stage is converged, the iteration goes to the next iteration.

        Parameters
        ----------
        stage_scheduler: StageScheduler
            The added stage scheduler
        
        """
        self.stage_schedulers.append(stage_scheduler)
        return self

    def get_stage(self):
        """
        Get the index of current stage. 

        Stage index increases when the previous stage converges. Usually called after `self.plan_next_iteration`.

        """
        return self.cur_stage

    def get_iteration(self):
        """
        Get the index of the current iteration.

        Iteration index increase when `self.plan_next_iteration` returns valid `lmp_task_grp` and `conf_selector` for the next iteration.

        """
        return self.iteration

    def plan_next_iteration(
            self,
            report : ExplorationReport = None,
            trajs : List[Path] = None,
    ) -> Tuple[bool, ExplorationTaskGroup, ConfSelector] :
        """
        Make the plan for the next DPGEN iteration.

        Parameters
        ----------
        report : ExplorationReport
            The exploration report of this iteration.
        confs: List[Path]
            A list of configurations generated during the exploration. May be used to generate new configurations for the next iteration. 

        Returns
        -------
        converged: bool
            If DPGEN converges.
        task: ExplorationTaskGroup
            A `ExplorationTaskGroup` defining the exploration of the next iteration. Should be `None` if converged.
        conf_selector: ConfSelector
            The configuration selector for the next iteration. Should be `None` if converged.

        """

        try:
            converged, lmp_task_grp, conf_selector = \
                self.stage_schedulers[self.cur_stage].plan_next_iteration(
                    self.stage_reports[self.cur_stage],
                    report,
                    trajs,
                )
            self.stage_reports[self.cur_stage].append(report)
        except FatalError as e:
            raise FatalError(f'stage {self.cur_stage}: ' + str(e))
        
        if converged:
            self.cur_stage += 1
            self.stage_reports.append([])
            if self.cur_stage < len(self.stage_schedulers):
                # goes to next stage
                return self.plan_next_iteration()
            else:
                # all stages converged
                return True, None, None,
        else :
            self.iteration += 1
            return converged, lmp_task_grp, conf_selector

