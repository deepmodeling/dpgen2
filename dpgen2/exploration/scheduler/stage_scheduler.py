from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Tuple,
)
from pathlib import Path
from dpgen2.exploration.report import ExplorationReport
from dpgen2.exploration.task import ExplorationTaskGroup, ExplorationStage
from dpgen2.exploration.selector import ConfSelector, TrustLevel

class StageScheduler(ABC):
    """
    The scheduler for an exploration stage.
    """

    @abstractmethod
    def plan_next_iteration(
            self,
            hist_reports : List[ExplorationReport],
            report : ExplorationReport,
            trajs : List[Path],
    ) -> Tuple[bool, ExplorationTaskGroup, ConfSelector] :
        """
        Make the plan for the next iteration of the stage.

        It checks the report of the current and all historical iterations of the stage, and tells if the iterations are converged. If not converged, it will plan the next ieration for the stage. 

        Parameters
        ----------
        hist_reports: List[ExplorationReport]
            The historical exploration report of the stage. If this is the first iteration of the stage, this list is empty.
        report : ExplorationReport
            The exploration report of this iteration.
        confs: List[Path]
            A list of configurations generated during the exploration. May be used to generate new configurations for the next iteration. 

        Returns
        -------
        converged: bool
            If the stage converged.
        task: ExplorationTaskGroup
            A `ExplorationTaskGroup` defining the exploration of the next iteration. Should be `None` if the stage is converged.
        conf_selector: ConfSelector
            The configuration selector for the next iteration. Should be `None` if the stage is converged.

        """
        pass
