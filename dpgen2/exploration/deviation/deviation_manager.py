from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
    Optional,
)

import numpy as np


class DeviManager(ABC):
    r"""A class for model deviation management."""
    MAX_DEVI_V = "max_devi_v"
    MIN_DEVI_V = "min_devi_v"
    AVG_DEVI_V = "avg_devi_v"
    MAX_DEVI_F = "max_devi_f"
    MIN_DEVI_F = "min_devi_f"
    AVG_DEVI_F = "avg_devi_f"

    def __init__(self) -> None:
        super().__init__()
        self.ntraj = 0

    def _check_name(self, name: str):
        assert name in (
            DeviManager.MAX_DEVI_V,
            DeviManager.MIN_DEVI_V,
            DeviManager.AVG_DEVI_V,
            DeviManager.MAX_DEVI_F,
            DeviManager.MIN_DEVI_F,
            DeviManager.AVG_DEVI_F,
        ), f"Error: unknown deviation name {name}"

    def add(self, name: str, deviation: np.ndarray) -> None:
        r"""Add a model deviation into this manager.

        Parameters
        ----------
        name : str
            The name of the deviation. The name is restricted to
            (DeviManager.MAX_DEVI_V, DeviManager.MIN_DEVI_V,
            DeviManager.AVG_DEVI_V, DeviManager.MAX_DEVI_F,
            DeviManager.MIN_DEVI_F, DeviManager.AVG_DEVI_F)
        deviation : np.ndarray
            The model deviation is a one-dimensional array extracted
            from a trajectory file.
        """
        self._check_name(name)
        return self._add(name, deviation)

    @abstractmethod
    def _add(self, name: str, deviation: np.ndarray) -> None:
        pass

    def get(self, name: str) -> List[Optional[np.ndarray]]:
        r"""Gat a model deviation from this manager.

        Parameters
        ----------
        name : str
            The name of the deviation. The name is restricted to
            (DeviManager.MAX_DEVI_V, DeviManager.MIN_DEVI_V,
             DeviManager.AVG_DEVI_V, DeviManager.MAX_DEVI_F,
             DeviManager.MIN_DEVI_F, DeviManager.AVG_DEVI_F)
        """
        self._check_name(name)
        self._check_data()
        return self._get(name)

    @abstractmethod
    def _get(self, name: str) -> List[Optional[np.ndarray]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        r"""Clear all data in this manager."""
        pass

    @abstractmethod
    def _check_data(self) -> None:
        r"""Check if data is valid"""
        pass
