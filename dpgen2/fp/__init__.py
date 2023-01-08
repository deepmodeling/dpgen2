from .vasp import (
    VaspInputs,
    PrepVasp,
    RunVasp,
)
from .gaussian import (
    GaussianInputs,
    PrepGaussian,
    RunGaussian,
)
from .binary_file_input import BinaryFileInput

fp_styles = {
    "vasp" :  {
        "inputs" : VaspInputs,
        "prep" : PrepVasp,
        "run" : RunVasp,
    },
    "gaussian" : {
        "inputs" : GaussianInputs,
        "prep" : PrepGaussian,
        "run" : RunGaussian,
    },
}
