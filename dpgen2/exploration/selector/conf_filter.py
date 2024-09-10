from __future__ import (
    annotations,
)

from abc import (
    ABC,
    abstractmethod,
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
        conf: dpdata.System,
    ) -> dpdata.System:
        natoms = sum(conf["atom_numbs"])  # type: ignore
        selected_idx = np.arange(conf.get_nframes())
        for ff in self._filters:
            fsel = np.where([ff.check(conf[ii]) for ii in range(conf.get_nframes())])[0]
            selected_idx = np.intersect1d(selected_idx, fsel)
        return conf.sub_system(selected_idx)
