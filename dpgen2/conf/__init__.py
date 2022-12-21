from .alloy_conf import (
    AlloyConfGenerator,
)
from .file_conf import (
    FileConfGenerator,
)

conf_styles = {
    "alloy" : AlloyConfGenerator,
    "file" : FileConfGenerator,
}
