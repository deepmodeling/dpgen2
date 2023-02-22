from collections import (
    defaultdict,
)
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from .deviation_manager import (
    DeviManager,
)


class DeviManagerStd(DeviManager):
    r"""The class which is responsible for model deviation management.

    This is the standard implementation of DeviManager. Each deviation
    (e.g. max_devi_f, max_devi_v in file `model_devi.out`) is stored
    as a List[Optional[np.ndarray]], where np.array is a one-dimensional
    array.
    A List[np.ndarray][ii][jj] is the force model deviation of the jj-th
    frame of the ii-th trajectory.
    The model deviation can be List[None], where len(List[None]) is
    the number of trajectory files.

    """

    def __init__(self):
        super().__init__()
        self._data = defaultdict(list)

    def _add(self, name: str, deviation: np.ndarray) -> None:
        self._data[name].append(deviation)
        self.ntraj = max(self.ntraj, len(self._data[name]))

    def _get(self, name: str) -> List[Optional[np.ndarray]]:
        if self.ntraj == 0:
            return []
        elif len(self._data[name]) == 0:
            return [None for _ in range(self.ntraj)]
        else:
            return self._data[name]

    def clear(self) -> None:
        self.__init__()
        return None
