import os
from typing import (
    List,
    Tuple,
    Union,
)

from dflow.config import (
    config,
)
from dflow.utils import run_command as dflow_run_command


def run_command(
    cmd: Union[str, List[str]],
    shell: bool = False,
) -> Tuple[int, str, str]:
    interactive = False if config["mode"] == "debug" else True
    return dflow_run_command(
        cmd, raise_error=False, try_bash=shell, interactive=interactive
    )
