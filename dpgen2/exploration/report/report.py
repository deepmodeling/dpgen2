from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np

from ..deviation import (
    DeviManager,
)


class ExplorationReport(ABC):
    @abstractmethod
    def clear(self):
        r"""Clear the report"""
        pass

    @abstractmethod
    def record(
        self,
        model_devi: DeviManager,
    ):
        r"""Record the model deviations of the trajectories

        Parameters
        ----------
        model_devi : DeviManager
            The class which is responsible for model deviation management.
            Model deviations is stored as a List[Optional[np.ndarray]],
            where np.array is a one-dimensional array.
            List[np.ndarray][ii][jj] is the force model deviation of
            the jj-th frame of the ii-th trajectory.
            Model deviations can be List[None], where len(List[None]) is
            the number of trajectory files.
        """
        pass

    @abstractmethod
    def converged(
        self,
        reports,
    ) -> bool:
        r"""Check if the exploration is converged.

        Parameters
        ----------
        reports
            Historical reports

        Returns
        -------
        converged  bool
            If the exploration is converged.
        """
        pass

    def no_candidate(self) -> bool:
        r"""If no candidate configuration is found"""
        return all([len(ii) == 0 for ii in self.get_candidate_ids()])

    @abstractmethod
    def get_candidate_ids(
        self,
        max_nframes: Optional[int] = None,
    ) -> List[List[int]]:
        r"""Get indexes of candidate configurations

        Parameters
        ----------
        max_nframes
            The maximal number of frames of candidates.

        Returns
        -------
        idx:    List[List[int]]
            The frame indices of candidate configurations.
            idx[ii][jj] is the frame index of the jj-th candidate of the
            ii-th trajectory.
        """
        pass

    @abstractmethod
    def print_header(self) -> str:
        r"""Print the header of report"""
        pass

    @abstractmethod
    def print(
        self,
        stage_idx: int,
        idx_in_stage: int,
        iter_idx: int,
    ) -> str:
        r"""Print the report"""
        pass
