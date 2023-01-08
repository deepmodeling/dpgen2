"""Binary file inputs"""
from dflow.python import (
    TransientError,
)
from typing import (
    Tuple,
    List,
    Any,
    Union,
    Optional
)
import dpdata
from dargs import (
    dargs, 
    Argument,
)
import base64
import os
from pathlib import Path

class BinaryFileInput:
    def __init__(self, path: Union[str, Path], suffix: str = None) -> None:
        path = str(path)
        assert os.path.exists(path), f"No such file: {path}"
        if suffix:
            assert path.endswith(suffix), \
                f"File suffix mismatch, require \"{suffix}\", current \"{path.split('.')[-1]}\", file path: {path}"

        self.suffix = suffix
        self.file_name = os.path.basename(path)
        with open(path, 'rb') as f:
            data = f.read()
            self._base64_data = base64.b64encode(data)
    
    def save_as_file(self, path: Union[str, Path]) -> None:
        if self.suffix and str(path).split('.')[-1] != self.suffix:
            print(f"warning: file suffix mismatch! Suffix of input file is \"{self.suffix}\", current suffix is \"{str(path).split('.')[-1]}\"")

        with open(path, 'wb') as file:
            data = base64.b64decode(self._base64_data)
            file.write(data)
