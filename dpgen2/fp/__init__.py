from .abacus import (
    FpOpAbacusInputs,
    PrepFpOpAbacus,
    RunFpOpAbacus,
)
from .cp2k import (
    FpOpCp2kInputs,
    PrepFpOpCp2k,
    RunFpOpCp2k,
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
    "fpop_cp2k": {
        "inputs": FpOpCp2kInputs,
        "prep": PrepFpOpCp2k,
        "run": RunFpOpCp2k,
    },
    "deltaspin": {
        "inputs": VaspInputs,
        "prep": PrepDeltaSpin,
        "run": RunDeltaSpin,
    },
}
