from .vasp import (
    VaspInputs,
    PrepVasp,
    RunVasp,
)

fp_styles = {
    "vasp" :  {
        "inputs" : VaspInputs,
        "prep" : PrepVasp,
        "run" : RunVasp,
    }
}
