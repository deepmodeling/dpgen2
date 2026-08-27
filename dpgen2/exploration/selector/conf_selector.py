from abc import (
    ABC,
    abstractmethod,
)
from pathlib import (
    Path,
)
from typing import (
    Optional,
    Union,
)

import dpdata
from dflow.python.opio import (
    HDF5Dataset,
)

from dpgen2.exploration.report import (
    ExplorationReport,
)

from . import (
    ConfFilters,
)


class ConfSelector(ABC):
    """Select configurations from trajectory and model deviation files."""

    @abstractmethod
    def select(
        self,
        trajs: Union[list[Path], list[HDF5Dataset]],
        model_devis: Union[list[Path], list[HDF5Dataset]],
        type_map: Optional[list[str]] = None,
        optional_outputs: Optional[list[Path]] = None,
    ) -> tuple[list[Path], ExplorationReport]:
        pass
