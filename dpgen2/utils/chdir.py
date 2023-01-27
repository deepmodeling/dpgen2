import os
from functools import wraps
from typing import Callable
from contextlib import contextmanager
from pathlib import Path
from dflow.python import (
    OPIO,
)


@contextmanager
def set_directory(path: Path):
    """Sets the current working path within the context.

    Parameters
    ----------
    path : Path
        The path to the cwd

    Yields
    ------
    None

    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    cwd = Path().absolute()
    path.mkdir(exist_ok=True, parents=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def chdir(path_key: str):
    """Returns a decorator that can change the current working path.

    Parameters
    ----------
    path_key : str
        key to OPIO

    Examples
    --------
    >>> class SomeOP(OP):
    ...     @chdir("path")
    ...     def execute(self, ip: OPIO):
    ...         do_something()
    """

    def decorator(func: Callable):
        """Change the current working path."""

        @wraps(func)
        def wrapper(self, ip: OPIO):
            with set_directory(Path(ip[path_key])):
                return func(self, ip)

        return wrapper

    return decorator
