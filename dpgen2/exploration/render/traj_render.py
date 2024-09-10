from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)

import dpdata
import numpy as np
from dflow.python.opio import (
    HDF5Dataset,
)

from ..deviation import (
    DeviManager,
)

if TYPE_CHECKING:
    from dpgen2.exploration.selector import (
        ConfFilters,
    )


class TrajRender(ABC):
    @abstractmethod
    def get_model_devi(
        self,
        files: Union[List[Path], List[HDF5Dataset]],
    ) -> DeviManager:
        r"""Get model deviations from recording files.

        Parameters
        ----------
        files : List[Path]
            The paths to the model deviation recording files

        Returns
        -------
        DeviManager: The class which is responsible for model deviation management.
        """
        pass

    @abstractmethod
    def get_confs(
        self,
        traj: Union[List[Path], List[HDF5Dataset]],
        id_selected: List[List[int]],
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> dpdata.MultiSystems:
        r"""Get configurations from trajectory by selection.

        Parameters
        ----------
        traj : List[Path]
            Trajectory files
        id_selected : List[List[int]]
            The selected frames. id_selected[ii][jj] is the jj-th selected frame
            from the ii-th trajectory. id_selected[ii] may be an empty list.
        type_map : List[str]
            The type map.
        conf_filters : ConfFilters
            Configuration filters
        optional_outputs : List[Path]
            Optional outputs of the exploration

        Returns
        -------
        ms:     dpdata.MultiSystems
            The configurations in dpdata.MultiSystems format
        """
        pass
