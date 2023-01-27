from __future__ import annotations
from abc import ABC, abstractmethod
import dpdata
import numpy as np


class ConfFilter(ABC):
    @abstractmethod
    def check(
        self,
        coords: np.ndarray,
        cell: np.ndarray,
        atom_types: np.ndarray,
        nopbc: bool,
    ) -> bool:
        """Check if the configuration is valid.

        Parameters
        ----------
        coords : numpy.array
                The coordinates, numpy array of shape natoms x 3
        cell : numpy.array
                The cell tensor. numpy array of shape 3 x 3
        atom_types : numpy.array
                The atom types. numpy array of shape natoms
        nopbc : bool
                If no periodic boundary condition.

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
    ) -> bool:
        natoms = sum(conf["atom_numbs"])
        selected_idx = np.arange(conf.get_nframes())
        for ff in self._filters:
            fsel = np.where(
                [
                    ff.check(
                        conf["coords"][ii],
                        conf["cells"][ii],
                        conf["atom_types"],
                        conf.nopbc,
                    )
                    for ii in range(conf.get_nframes())
                ]
            )[0]
            selected_idx = np.intersect1d(selected_idx, fsel)
        return conf.sub_system(selected_idx)
