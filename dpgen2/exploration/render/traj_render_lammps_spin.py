from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
)

import dpdata
import numpy as np

from ..deviation import (
    DeviManager,
    DeviManagerSpin,
)
from .traj_render import (
    TrajRender,
)

if TYPE_CHECKING:
    from dpgen2.exploration.selector import (
        ConfFilters,
    )


class TrajRenderLammpsSpin(TrajRender):
    def __init__(
        self,
        nopbc: bool = False,
    ):
        self.nopbc = nopbc

    def get_model_devi(
        self,
        files: List[Path],
    ) -> DeviManagerSpin:
        ntraj = len(files)

        model_devi = DeviManagerSpin()
        for ii in range(ntraj):
            self._load_one_model_devi(files[ii], model_devi)

        return model_devi

    def _load_one_model_devi(self, fname, model_devi):
        dd = np.loadtxt(fname)
        model_devi.add(DeviManagerSpin.MAX_DEVI_AF, dd[:, 1])
        model_devi.add(DeviManagerSpin.MIN_DEVI_AF, dd[:, 2])
        model_devi.add(DeviManagerSpin.AVG_DEVI_AF, dd[:, 3])
        model_devi.add(DeviManagerSpin.MAX_DEVI_MF, dd[:, 4])
        model_devi.add(DeviManagerSpin.MIN_DEVI_MF, dd[:, 5])
        model_devi.add(DeviManagerSpin.AVG_DEVI_MF, dd[:, 6])

    def get_confs(
        self,
        trajs: List[Path],
        id_selected: List[List[int]],
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
    ) -> dpdata.MultiSystems:
        del conf_filters  # by far does not support conf filters
        ntraj = len(trajs)
        traj_fmt = "lammps/spin/dump"
        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in range(ntraj):
            if len(id_selected[ii]) > 0:
                ss = dpdata.System(trajs[ii], fmt=traj_fmt, type_map=type_map)
                ss.nopbc = self.nopbc
                ss = ss.sub_system(id_selected[ii])
                ms.append(ss)
        return ms
