from __future__ import (
    annotations,
)

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
)

import dpdata
import numpy as np


class ConfFilter(ABC):
    @abstractmethod
    def check(
        self,
        frame: dpdata.System,
    ) -> bool:
        """Check if the configuration is valid.

        Parameters
        ----------
        frame : dpdata.System
            A dpdata.System containing a single frame

        Returns
        -------
        valid : bool
            `True` if the configuration is a valid configuration, else `False`.

        """
        pass

    def batched_check(
        self,
        frames: List[dpdata.System],
    ) -> List[bool]:
        """Check if a list of configurations are valid.

        Parameters
        ----------
        frames : List[dpdata.System]
            A list of dpdata.System each containing a single frame

        Returns
        -------
        valid : List[bool]
            `True` if the configuration is a valid configuration, else `False`.

        """
        return list(map(self.check, frames))


class ConfFilters:
    def __init__(
        self,
    ):
        self._filters = []

    def add(
        self,
        conf_filter: ConfFilter,
    ) -> ConfFilters:
        self._filters.append(conf_filter)
        return self

    def check(
        self,
        ms: dpdata.MultiSystems,
    ) -> dpdata.MultiSystems:
        selected_idx = []
        for i in range(len(ms)):
            for j in range(ms[i].get_nframes()):
                selected_idx.append((i, j))
        for ff in self._filters:
            res = ff.batched_check([ms[i][j] for i, j in selected_idx])
            selected_idx = [idx for i, idx in enumerate(selected_idx) if res[i]]
        selected_idx_list = [[] for _ in range(len(ms))]
        for i, j in selected_idx:
            selected_idx_list[i].append(j)
        ms2 = dpdata.MultiSystems(type_map=ms.atom_names)
        for i in range(len(ms)):
            if len(selected_idx_list[i]) > 0:
                ms2.append(ms[i].sub_system(selected_idx_list[i]))
        return ms2
