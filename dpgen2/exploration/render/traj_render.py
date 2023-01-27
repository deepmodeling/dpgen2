import dpdata
import numpy as np
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dpgen2.exploration.selector import ConfFilters


class TrajRender(ABC):
    @abstractmethod
    def get_model_devi(
        self,
        files: List[Path],
    ) -> Tuple[List[np.ndarray], Union[List[np.ndarray], None]]:
        r"""Get model deviations from recording files.

        Parameters
        ----------
        files:  List[Path]
                The paths to the model deviation recording files

        Returns
        -------
        model_devis: Tuple[List[np.array], Union[List[np.array],None]]
                A tuple. model_devis[0] is the force model deviations,
                model_devis[1] is the virial model deviations.
                The model_devis[1] can be None.
                If not None, model_devis[i] is List[np.array], where np.array is a
                one-dimensional array.
                The first dimension of model_devis[i] is the trajectory
                (same size as len(files)), while the second dimension is the frame.
        """
        pass

    @abstractmethod
    def get_confs(
        self,
        traj: List[Path],
        id_selected: List[List[int]],
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
    ) -> dpdata.MultiSystems:
        r"""Get configurations from trajectory by selection.

        Parameters
        ----------
        traj:   List[Path]
                Trajectory files
        id_selected: List[List[int]]
                The selected frames. id_selected[ii][jj] is the jj-th selected frame
                from the ii-th trajectory. id_selected[ii] may be an empty list.
        type_map: List[str]
                The type map.

        Returns
        -------
        ms:     dpdata.MultiSystems
                The configurations in dpdata.MultiSystems format
        """
        pass
