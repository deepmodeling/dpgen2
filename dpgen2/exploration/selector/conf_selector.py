import dpdata
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Set
from pathlib import Path
from . import (
    ConfFilters,
)
from dpgen2.exploration.report import ExplorationReport


class ConfSelector(ABC):
    """Select configurations from trajectory and model deviation files."""

    @abstractmethod
    def select(
        self,
        trajs: List[Path],
        model_devis: List[Path],
        type_map: Optional[List[str]] = None,
    ) -> Tuple[List[Path], ExplorationReport]:
        pass
