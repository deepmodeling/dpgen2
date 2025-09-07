from .alloy_conf import (
    AlloyConfGenerator,
)
from .conf_generator import (
    ConfGenerator as ConfGenerator,
)
from .file_conf import (
    FileConfGenerator,
)

conf_styles = {
    "alloy": AlloyConfGenerator,
    "file": FileConfGenerator,
}
