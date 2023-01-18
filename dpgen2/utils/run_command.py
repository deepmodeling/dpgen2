from dflow.utils import run_command as dflow_run_command
from typing import Tuple, Union, List

def run_command(
        cmd : Union[str, List[str]],
        shell: bool = False,
) -> Tuple[int, str, str]:
    return dflow_run_command(
        cmd, 
        raise_error=False,
        try_bash=shell,
    )
