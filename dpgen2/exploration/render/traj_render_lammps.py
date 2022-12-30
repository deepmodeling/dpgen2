import numpy as np
import dpdata
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)
from .traj_render import TrajRender
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dpgen2.exploration.selector import ConfFilters


class TrajRenderLammps(TrajRender):
    def __init__(
            self,
            nopbc : bool=False,
    ):
        self.nopbc = nopbc

    def get_model_devi(
            self,
            files : List[Path],
    ) -> Tuple[List[np.ndarray], Union[List[np.ndarray], None]]:
        nframes = len(files)
        mdfs = []
        mdvs = []
        for ii in range(nframes):
            mdf, mdv = self._load_one_model_devi(files[ii])
            mdfs.append(mdf)
            mdvs.append(mdv)
        return mdfs, mdvs

    def _load_one_model_devi(self, fname):
        dd = np.loadtxt(fname)
        return dd[:,4], dd[:,1]
        
    def get_confs(
            self,
            trajs: List[Path],
            id_selected: List[List[int]],
            type_map: Optional[List[str]] = None,
            conf_filters: Optional["ConfFilters"] = None,
    ) -> dpdata.MultiSystems :
        del conf_filters        # by far does not support conf filters
        ntraj = len(trajs)
        traj_fmt = 'lammps/dump'
        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in range(ntraj):
            if len(id_selected[ii]) > 0:
                ss = dpdata.System(trajs[ii], fmt=traj_fmt, type_map=type_map)
                ss.nopbc = self.nopbc
                ss = ss.sub_system(id_selected[ii])        
                ms.append(ss)
        return ms
