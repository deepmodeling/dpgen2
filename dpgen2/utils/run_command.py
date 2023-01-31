from dflow.utils import run_command as dflow_run_command
from typing import Tuple, Union, List
import os


def run_command(
    cmd: Union[str, List[str]],
    shell: bool = False,
) -> Tuple[int, str, str]:
    interactive = False if os.getenv("DFLOW_DEBUG") else True
    return dflow_run_command(
        cmd, raise_error=False, try_bash=shell, interactive=interactive
    )
