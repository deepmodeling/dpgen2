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

fp_styles = {
    "vasp": {
        "inputs": VaspInputs,
        "prep": PrepVasp,
        "run": RunVasp,
    },
    "gaussian": {
        "inputs": GaussianInputs,
        "prep": PrepGaussian,
        "run": RunGaussian,
    },
}
