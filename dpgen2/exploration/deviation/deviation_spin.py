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


class DeviManagerSpin(DeviManager):
    r"""The class which is responsible for DeepSPIN model deviation management.

    This is the implementation of DeviManager for DeepSPIN model. Each deviation
    (e.g. max_devi_af, max_devi_mf in file `model_devi.out`) is stored
    as a List[Optional[np.ndarray]], where np.array is a one-dimensional
    array.
    A List[np.ndarray][ii][jj] is the force model deviation of the jj-th
    frame of the ii-th trajectory.
    The model deviation can be List[None], where len(List[None]) is
    the number of trajectory files.

    """

    MAX_DEVI_AF = "max_devi_af"
    MIN_DEVI_AF = "min_devi_af"
    AVG_DEVI_AF = "avg_devi_af"
    MAX_DEVI_MF = "max_devi_mf"
    MIN_DEVI_MF = "min_devi_mf"
    AVG_DEVI_MF = "avg_devi_mf"

    def __init__(self):
        super().__init__()
        self._data = defaultdict(list)

    def _check_name(self, name: str):
        assert name in (
            DeviManager.MAX_DEVI_V,
            DeviManager.MIN_DEVI_V,
            DeviManager.AVG_DEVI_V,
            self.MAX_DEVI_AF,
            self.MIN_DEVI_AF,
            self.AVG_DEVI_AF,
            self.MAX_DEVI_MF,
            self.MIN_DEVI_MF,
            self.AVG_DEVI_MF,
        ), f"Error: unknown deviation name {name}"

    def _add(self, name: str, deviation: np.ndarray) -> None:
        assert isinstance(
            deviation, np.ndarray
        ), f"Error: deviation(type: {type(deviation)}) is not a np.ndarray"
        assert len(deviation.shape) == 1, (
            f"Error: deviation(shape: {deviation.shape}) is not a "
            + f"one-dimensional array"
        )
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

    def _check_data(self) -> None:
        r"""Check if data is valid"""
        model_devi_names = (
            DeviManager.MAX_DEVI_V,
            DeviManager.MIN_DEVI_V,
            DeviManager.AVG_DEVI_V,
            self.MAX_DEVI_AF,
            self.MIN_DEVI_AF,
            self.AVG_DEVI_AF,
            self.MAX_DEVI_MF,
            self.MIN_DEVI_MF,
            self.AVG_DEVI_MF,
        )
        # check the length of model deviations
        frames = {}
        for name in model_devi_names:
            if len(self._data[name]) > 0:
                assert len(self._data[name]) == self.ntraj, (
                    f"Error: the number of model deviation {name} "
                    + f"({len(self._data[name])}) and trajectory files ({self.ntraj}) "
                    + f"are not equal."
                )
            for idx, ndarray in enumerate(self._data[name]):
                assert isinstance(ndarray, np.ndarray), (
                    f"Error: model deviation in {name} is not ndarray, "
                    + f"index: {idx}, type: {type(ndarray)}"
                )

            frames[name] = [arr.shape[0] for arr in self._data[name]]
            if len(frames[name]) == 0:
                frames.pop(name)

        # check if "max_devi_af" and "max_devi_mf" exist
        assert (
            len(self._data[self.MAX_DEVI_AF]) == self.ntraj
        ), f"Error: cannot find model deviation {self.MAX_DEVI_AF}"
        assert (
            len(self._data[self.MAX_DEVI_MF]) == self.ntraj
        ), f"Error: cannot find model deviation {self.MAX_DEVI_MF}"

        # check if the length of the arrays corresponding to the same
        # trajectory has the same number of frames
        non_empty_deviations = list(frames.keys())
        for name in non_empty_deviations[1:]:
            assert frames[name] == frames[non_empty_deviations[0]], (
                f"Error: the number of frames in {name} is different "
                + f"with that in {non_empty_deviations[0]}.\n"
                + f"{name}: {frames[name]}\n"
                + f"{non_empty_deviations[0]}: {frames[non_empty_deviations[0]]}\n"
            )
