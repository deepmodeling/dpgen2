import os
import subprocess
import sys
import threading
from typing import (
    List,
    Tuple,
    Union,
)

from dflow.config import (
    config,
)
from dflow.utils import run_command as dflow_run_command


def run_command_streaming(
    cmd: Union[str, List[str]],
    shell: bool = False,
    log_file=None,
) -> Tuple[int, str, str]:
    """Run command with streaming output to both terminal and log file."""
    if isinstance(cmd, str):
        cmd = cmd if shell else cmd.split()

    # Open log file if specified
    log_fp = open(log_file, "w") if log_file else None

    try:
        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=shell,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        # Store output
        stdout_buffer = []
        stderr_buffer = []

        def stream_output(pipe, buffer, is_stderr=False):
            for line in iter(pipe.readline, ""):
                buffer.append(line)
                # Print to terminal
                if is_stderr:
                    print(line, end="", file=sys.stderr)
                else:
                    print(line, end="")
                # Write to log file
                if log_fp:
                    log_fp.write(line)
                    log_fp.flush()
            pipe.close()

        # Start threads for streaming
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, stdout_buffer, False))
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, stderr_buffer, True))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for process to complete
        return_code = process.wait()

        # Wait for threads to finish
        stdout_thread.join()
        stderr_thread.join()

        return return_code, "".join(stdout_buffer), "".join(stderr_buffer)

    finally:
        if log_fp:
            log_fp.close()


def run_command(
    cmd: Union[str, List[str]],
    shell: bool = False,
) -> Tuple[int, str, str]:
    interactive = False if config["mode"] == "debug" else True
    return dflow_run_command(
        cmd, raise_error=False, try_bash=shell, interactive=interactive
    )
