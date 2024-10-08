from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
    Union,
)

from dflow.python.opio import (
    HDF5Dataset,
)

from dpgen2.exploration.report import (
    ExplorationReport,
)
from dpgen2.exploration.selector import (
    ConfSelector,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTaskGroup,
)


class StageScheduler(ABC):
    """
    The scheduler for an exploration stage.
    """

    @abstractmethod
    def converged(self) -> bool:
        """
        Tell if the stage is converged

        Returns
        -------
        converged  bool
            the convergence
        """
        pass

    @abstractmethod
    def complete(self) -> bool:
        """
        Tell if the stage is complete

        Returns
        -------
        converged  bool
            if the stage is complete
        """
        pass

    @abstractmethod
    def force_complete(self):
        """
        For complete the stage

        """
        pass

    @abstractmethod
    def next_iteration(self) -> int:
        """
        Return the index of the next iteration

        Returns
        -------
        index  int
            the index of the next iteration
        """
        pass

    @abstractmethod
    def get_reports(self) -> List[ExplorationReport]:
        """
        Return all exploration reports

        Returns
        -------
        reports  List[ExplorationReport]
            the reports
        """
        pass

    @abstractmethod
    def plan_next_iteration(
        self,
        report: ExplorationReport,
        trajs: Union[List[Path], List[HDF5Dataset]],
    ) -> Tuple[bool, ExplorationTaskGroup, ConfSelector]:
        """
        Make the plan for the next iteration of the stage.

        It checks the report of the current and all historical iterations of the stage, and tells if the iterations are converged. If not converged, it will plan the next ieration for the stage.

        Parameters
        ----------
        report : ExplorationReport
            The exploration report of this iteration.
        trajs : Union[List[Path], List[HDF5Dataset]]
            A list of configurations generated during the exploration. May be used to generate new configurations for the next iteration.

        Returns
        -------
        stg_complete: bool
            If the stage completed. Two cases may happen:
            1. converged.
            2. when not fatal_at_max, not converged but reached max number of iterations.
        task: ExplorationTaskGroup
            A `ExplorationTaskGroup` defining the exploration of the next iteration. Should be `None` if the stage is converged.
        conf_selector: ConfSelector
            The configuration selector for the next iteration. Should be `None` if the stage is converged.

        """
        pass
