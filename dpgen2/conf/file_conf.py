import os
import dpdata
import glob
from pathlib import Path
from .conf_generator import ConfGenerator
from typing import (
    Optional, Union, List, Tuple
)
from dargs import (
    Argument,
    Variant,
)


class FileConfGenerator(ConfGenerator):
    def __init__(
            self,
            files : Union[str,List[str]],
            fmt : str = 'auto',
            prefix : Optional[str] = None,
    ):
        if not isinstance(files, list):
            assert(isinstance(files, str))
            files = [files]
        if prefix is not None:
            pfiles = [Path(prefix) / Path(ii) for ii in files]
        else:
            pfiles = [Path(ii) for ii in files]
        self.files = []
        for ii in pfiles:
            ff = glob.glob(str(ii.absolute()))
            ff.sort()
            self.files += ff
        self.fmt = fmt


    def generate(
            self,
            type_map,
    ) -> dpdata.MultiSystems:
        ms = dpdata.MultiSystems(type_map=type_map)
        for ff in self.files:
            ms.append(dpdata.System(ff, fmt=self.fmt))
        return ms


    @staticmethod
    def args() -> List[Argument]:
        doc_files = "The paths to the configuration files. widecards are supported."
        doc_prefix = "The prefix of file paths."
        doc_fmt = "The format (dpdata accepted formats) of the files."

        return [
            Argument("files", [str, list], optional=False, doc=doc_files),
            Argument("prefix", str, optional=True, default=None, doc=doc_prefix),
            Argument("fmt", str, optional=True, default='auto', doc=doc_fmt),
        ]

