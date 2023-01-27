import os
import dpdata
import glob
from pathlib import Path
from .conf_generator import ConfGenerator
from typing import Optional, Union, List, Tuple
from dargs import (
    Argument,
    Variant,
)


class FileConfGenerator(ConfGenerator):
    def __init__(
        self,
        files: Union[str, List[str]],
        fmt: str = "auto",
        prefix: Optional[str] = None,
        remove_pbc: Optional[bool] = False,
    ):
        if not isinstance(files, list):
            assert isinstance(files, str)
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
        self.remove_pbc = remove_pbc

    def generate(
        self,
        type_map,
    ) -> dpdata.MultiSystems:
        ms = dpdata.MultiSystems(type_map=type_map)
        for ff in self.files:
            ss = dpdata.System(ff, fmt=self.fmt, type_map=type_map)
            if self.remove_pbc:
                ss.remove_pbc()
            ms.append(ss)
        return ms

    @staticmethod
    def args() -> List[Argument]:
        doc_files = "The paths to the configuration files. widecards are supported."
        doc_prefix = "The prefix of file paths."
        doc_fmt = "The format (dpdata accepted formats) of the files."
        doc_remove_pbc = "The remove the pbc of the data. shift the coords to the center of box so it can be used with lammps."

        return [
            Argument("files", [str, list], optional=False, doc=doc_files),
            Argument("prefix", str, optional=True, default=None, doc=doc_prefix),
            Argument("fmt", str, optional=True, default="auto", doc=doc_fmt),
            Argument(
                "remove_pbc", bool, optional=True, default=False, doc=doc_remove_pbc
            ),
        ]
