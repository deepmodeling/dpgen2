from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)


import dpdata
from dflow.python.opio import (
    HDF5Dataset,
)

from dpgen2.exploration.report import (
    ExplorationReport,
)


class ConfSelector(ABC):
    """Select configurations from trajectory and model deviation files."""

    @abstractmethod
    def select(
        self,
        trajs: Union[List[Path], List[HDF5Dataset]],
        model_devis: Union[List[Path], List[HDF5Dataset]],
        type_map: Optional[List[str]] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> Tuple[List[Path], ExplorationReport]:
        pass
