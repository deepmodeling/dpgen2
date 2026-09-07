import json
from io import (
    StringIO,
)
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
from dflow.python.opio import (
    HDF5Dataset,
)

from dpgen2.utils import (
    setup_ele_temp,
)

from ..deviation import (
    DeviManager,
    DeviManagerStd,
)
from .traj_render import (
    TrajRender,
)

if TYPE_CHECKING:
    from dpgen2.exploration.selector import (
        ConfFilters,
    )


class TrajRenderLammps(TrajRender):
    def __init__(
        self,
        nopbc: bool = False,
        use_ele_temp: int = 0,
    ):
        self.nopbc = nopbc
        self.use_ele_temp = use_ele_temp

    def get_model_devi(
        self,
        files: Union[List[Path], List[HDF5Dataset]],
    ) -> DeviManager:
        ntraj = len(files)

        model_devi = DeviManagerStd()
        for ii in range(ntraj):
            self._load_one_model_devi(files[ii], model_devi)

        return model_devi

    def _load_one_model_devi(self, fname, model_devi):
        if isinstance(fname, HDF5Dataset):
            dd = fname.get_data()
        else:
            dd = np.loadtxt(fname)
        if (
            len(np.shape(dd)) == 1  # type: ignore
        ):  # In case model-devi.out is 1-dimensional
            dd = dd.reshape((1, len(dd)))  # type: ignore

        model_devi.add(DeviManager.MAX_DEVI_V, dd[:, 1])  # type: ignore
        model_devi.add(DeviManager.MIN_DEVI_V, dd[:, 2])  # type: ignore
        model_devi.add(DeviManager.AVG_DEVI_V, dd[:, 3])  # type: ignore
        model_devi.add(DeviManager.MAX_DEVI_F, dd[:, 4])  # type: ignore
        model_devi.add(DeviManager.MIN_DEVI_F, dd[:, 5])  # type: ignore
        model_devi.add(DeviManager.AVG_DEVI_F, dd[:, 6])  # type: ignore

    def get_ele_temp(self, optional_outputs):
        ele_temp = []
        for ii in range(len(optional_outputs)):
            with open(optional_outputs[ii], "r") as f:
                data = json.load(f)
            if self.use_ele_temp:
                ele_temp.append(data["ele_temp"])
        if self.use_ele_temp:
            if self.use_ele_temp == 1:
                setup_ele_temp(False)
            elif self.use_ele_temp == 2:
                setup_ele_temp(True)
            else:
                raise ValueError(
                    "Invalid value for 'use_ele_temp': %s", self.use_ele_temp
                )
        return ele_temp

    def set_ele_temp(self, system, ele_temp):
        if self.use_ele_temp == 1 and ele_temp:
            system.data["fparam"] = np.tile(ele_temp, [len(system), 1])
        elif self.use_ele_temp == 2 and ele_temp:
            system.data["aparam"] = np.tile(
                ele_temp, [len(system), system.get_natoms(), 1]
            )

    def get_confs(
        self,
        trajs: Union[List[Path], List[HDF5Dataset]],
        id_selected: List[List[int]],
        type_map: Optional[List[str]] = None,
        conf_filters: Optional["ConfFilters"] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> dpdata.MultiSystems:
        from ase.io import (  # type: ignore
            read,
        )

        ntraj = len(trajs)
        ele_temp = None
        if optional_outputs:
            assert ntraj == len(optional_outputs)
            ele_temp = self.get_ele_temp(optional_outputs)

        traj_fmt = "lammps/dump"
        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in range(ntraj):
            if len(id_selected[ii]) > 0:
                if isinstance(trajs[ii], HDF5Dataset):
                    traj = StringIO(trajs[ii].get_data())  # type: ignore
                else:
                    traj = trajs[ii]
                # ss = dpdata.System(traj, fmt=traj_fmt, type_map=type_map)
                ss = read(
                    str(traj), format="lammps-dump-text", index=":", specorder=type_map
                )
                for jj in id_selected[ii]:
                    s = dpdata.System(ss[jj], fmt="ase/structure", type_map=type_map)
                    s.nopbc = self.nopbc
                    if ele_temp:
                        self.set_ele_temp(s, ele_temp[ii])
                    ms.append(s)
        if conf_filters is not None:
            ms = conf_filters.check(ms)
        return ms
