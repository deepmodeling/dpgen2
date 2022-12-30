import numpy as np
from abc import ABC, abstractmethod
from typing import (
    Tuple, List, Optional,
)

class ExplorationReport(ABC):
    @abstractmethod
    def clear(self):
        r"""Clear the report"""
        pass

    @abstractmethod
    def record(
            self,
            md_f : List[np.ndarray],
            md_v : Optional[List[np.ndarray]] = None,
    ):
        r"""Record the model deviations of the trajectories

        Parameters
        ----------
        mdf : List[np.ndarray]
                The force model deviations. mdf[ii][jj] is the force model deviation
                of the jj-th frame of the ii-th trajectory.
        mdv : Optional[List[np.ndarray]]
                The virial model deviations. mdv[ii][jj] is the virial model deviation
                of the jj-th frame of the ii-th trajectory.
        """
        pass

    @abstractmethod
    def converged(self) -> bool :
        r"""If the exploration is converged"""
        pass

    def no_candidate(self) -> bool: 
        r"""If no candidate configuration is found"""
        return all([ len(ii) == 0 for ii in self.get_candidate_ids()])

    @abstractmethod
    def get_candidate_ids(
            self, 
            max_nframes : Optional[int] = None,
    ) -> List[List[int]]:
        r"""Get indexes of candidate configurations

        Parameters
        ----------
        max_nframes    int
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
            stage_idx : int,
            idx_in_stage : int,
            iter_idx : int,
    ) -> str:
        r"""Print the report"""
        pass
