import os
from collections.abc import (
    Sequence,
)
from typing import (
    Dict,
    List,
    Tuple,
)


class ExplorationTask:
    """Define the files needed by an exploration task.

    Examples
    --------
    >>> # this example dumps all files needed by the task.
    >>> files = exploration_task.files()
    ... for file_name, file_content in files.items():
    ...     with open(file_name, 'w') as fp:
    ...         fp.write(file_content)

    """

    def __init__(
        self,
    ):
        self._files = {}

    def add_file(
        self,
        fname: str,
        fcont: str,
    ):
        """Add file to the task

        Parameters
        ----------
        fname : str
            The name of the file
        fcont : str
            The content of the file.

        """
        self._files[fname] = fcont
        return self

    def files(self) -> Dict:
        """Get all files for the task.

        Returns
        -------
        files : dict
            The dict storing all files for the task. The file name is a key of the dict, and the file content is the corresponding value.
        """
        return self._files
