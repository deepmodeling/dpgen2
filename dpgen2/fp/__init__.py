from .abacus import (
    FpOpAbacusInputs,
    PrepFpOpAbacus,
    RunFpOpAbacus,
)
from .deepmd import (
    DeepmdInputs,
    PrepDeepmd,
    RunDeepmd,
)
from .gaussian import (
    GaussianInputs,
    PrepGaussian,
    RunGaussian,
)
from .vasp import (
    PrepVasp,
    RunVasp,
    VaspInputs,
)

from .cp2k import (
    PrepCp2k,
    RunCp2k,
    Cp2kInputs,
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
    "deepmd": {
        "inputs": DeepmdInputs,
        "prep": PrepDeepmd,
        "run": RunDeepmd,
    },
    "fpop_abacus": {
        "inputs": FpOpAbacusInputs,
        "prep": PrepFpOpAbacus,
        "run": RunFpOpAbacus,
    },
    "cp2k": {
        "inputs": Cp2kInputs,
        "prep": PrepCp2k,
        "run": RunCp2k,
    },
}
