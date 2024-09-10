from pathlib import (
    Path,
)
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
)

import dpdata
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    Parameter,
)

from dpgen2.utils import (
    setup_ele_temp,
)


class CollectData(OP):
    """Collect labeled data and add to the iteration dataset.

    After running FP tasks, the labeled data are scattered in task
    directories.  This OP collect the labeled data in one data
    directory and add it to the iteration data. The data generated by
    this iteration will be place in `ip["name"]` subdirectory of the
    iteration data directory.

    """

    default_optional_parameter: ClassVar[Dict[str, Any]] = {
        "mixed_type": False,
    }

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "name": str,
                "type_map": List[str],
                "optional_parameter": Parameter(
                    dict,
                    default=CollectData.default_optional_parameter,
                ),
                "labeled_data": Artifact(List[Path]),
                "iter_data": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "iter_data": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP. This OP collect data scattered in directories given by `ip['labeled_data']`
        in to one `dpdata.Multisystems` and store it in a directory named `name`. This directory is appended
        to the list `iter_data`.

        Parameters
        ----------
        ip : dict
            Input dict with components:

            - `name`: (`str`) The name of this iteration. The data generated by this iteration will be place in a sub-directory of `name`.
            - `labeled_data`: (`Artifact(List[Path])`) The paths of labeled data generated by FP tasks of the current iteration.
            - `iter_data`: (`Artifact(List[Path])`) The data paths previous iterations.

        Returns
        -------
        Any
            Output dict with components:
            - `iter_data`: (`Artifact(List[Path])`) The data paths of previous and the current iteration data.

        """
        name = ip["name"]
        type_map = ip["type_map"]
        mixed_type = ip["optional_parameter"]["mixed_type"]
        labeled_data = ip["labeled_data"]
        iter_data = ip["iter_data"]

        ms = dpdata.MultiSystems(type_map=type_map)
        for ii in labeled_data:
            if len(list(ii.rglob("fparam.npy"))) > 0:
                setup_ele_temp(False)
            if len(list(ii.rglob("aparam.npy"))) > 0:
                setup_ele_temp(True)
            ss = dpdata.LabeledSystem(ii, fmt="deepmd/npy")
            ms.append(ss)

        # NOTICE:
        # if ms.get_nframes() == 0, ms.to_deepmd_npy would not make the dir Path(name)
        Path(name).mkdir()
        if mixed_type:
            ms.to_deepmd_npy_mixed(name)  # type: ignore
        else:
            ms.to_deepmd_npy(name)  # type: ignore
        iter_data.append(Path(name))

        return OPIO(
            {
                "iter_data": iter_data,
            }
        )
