from dflow.utils import run_command as dflow_run_command
from typing import Tuple

def run_command(
        cmd : str,
        shell: bool = False,
) -> Tuple[int, str, str]:
    return dflow_run_command(
        cmd, 
        raise_error=False,
        try_bash=shell,
    )
