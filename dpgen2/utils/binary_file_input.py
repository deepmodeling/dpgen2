"""Binary file inputs"""
from dflow.python import (
    TransientError,
)
from typing import Tuple, List, Any, Union, Optional
from dargs import (
    dargs,
    Argument,
)
import os
from pathlib import Path
import warnings


class BinaryFileInput:
    def __init__(self, path: Union[str, Path], ext: Optional[str] = None) -> None:
        path = str(path)
        assert os.path.exists(path), f"No such file: {str(path)}"
        if ext and not ext.startswith("."):
            ext = "." + ext
        self.ext = ext

        if self.ext:
            assert (
                os.path.splitext(path)[-1] == self.ext
            ), f'File extension mismatch, require "{ext}", current "{os.path.splitext(path)[-1]}", file path: {str(path)}'

        self.file_name = os.path.basename(path)
        with open(path, "rb") as f:
            self._data = f.read()

    def save_as_file(self, path: Union[str, Path]) -> None:
        if self.ext and os.path.splitext(path)[-1] != self.ext:
            warnings.warn(
                f'warning: file extension mismatch! Extension of input file is "{self.ext}",'
                + f"current extension is \"{str(path).split('.')[-1]}\""
            )

        with open(path, "wb") as file:
            file.write(self._data)
