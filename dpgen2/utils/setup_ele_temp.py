import dpdata
import numpy as np
from dpdata.data_type import (
    Axis,
    DataType,
)


def setup_ele_temp(atomic: bool):
    """Set electronic temperature as required input data.

    Parameters
    ----------
    atomic : bool
        Whether to use atomic temperature or frame temperature
    """
    if atomic:
        ele_temp_data_type = DataType(
            "aparam",
            np.ndarray,
            shape=(Axis.NFRAMES, Axis.NATOMS, 1),
            required=False,
        )
    else:
        ele_temp_data_type = DataType(
            "fparam",
            np.ndarray,
            shape=(Axis.NFRAMES, 1),
            required=False,
        )

    dpdata.System.register_data_type(ele_temp_data_type)
    dpdata.LabeledSystem.register_data_type(ele_temp_data_type)
