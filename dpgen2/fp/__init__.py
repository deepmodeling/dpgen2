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

from .deltaspin import (
    PrepDeltaSpin,
    RunDeltaSpin,
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
    "deltaspin": {
        "inputs": VaspInputs,
        "prep": PrepDeltaSpin,
        "run": RunDeltaSpin,
    },
}
