import glob
import logging
import os
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)

import dpdata
from dargs import (
    Argument,
    Variant,
)

from .conf_generator import (
    ConfGenerator,
)


class FileConfGenerator(ConfGenerator):
    def __init__(
        self,
        files: Union[str, List[str]],
        fmt: str = "auto",
        prefix: Optional[str] = None,
        remove_pbc: Optional[bool] = False,
        remove_spins: bool = False,
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
        self.remove_spins = remove_spins

    def generate(
        self,
        type_map,
    ) -> dpdata.MultiSystems:
        if self.fmt in ["deepmd/npy/mixed"]:
            ms = self.generate_mixed(type_map)
        else:
            ms = self.generate_std(type_map)
        if self.remove_spins:
            for system in ms:
                if system.data.pop("spins", None) is not None:
                    logging.warning(
                        "Removing spin metadata from file configuration %s "
                        "because remove_spins is enabled",
                        system.formula,
                    )
        return ms

    def generate_std(
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

    def generate_mixed(
        self,
        type_map,
    ) -> dpdata.MultiSystems:
        if len(self.files) > 1:
            raise ValueError(
                'the file format "deepmd/npy/mixed" is specified, '
                "but more than one file is given, which is invalide "
                "please provide one path that can be interpreted as "
                "the dpdata.MultiSystems. "
            )
        assert "deepmd/npy/mixed" == self.fmt
        ms = dpdata.MultiSystems(type_map=type_map)
        ms.from_deepmd_npy_mixed(self.files[0], fmt="deepmd/npy/mixed", labeled=False)  # type: ignore
        return ms

    @staticmethod
    def doc() -> str:
        return "Generate alloys from user provided file(s). The file(s) are assume to be load by `dpdata`."

    @staticmethod
    def args() -> List[Argument]:
        doc_files = "The paths to the configuration files. widecards are supported."
        doc_prefix = "The prefix of file paths."
        doc_fmt = "The format (dpdata accepted formats) of the files."
        doc_remove_pbc = "The remove the pbc of the data. shift the coords to the center of box so it can be used with lammps."

        doc_remove_spins = (
            "Remove spin metadata from loaded configurations. Enable this for "
            "ABACUS STRU files containing magnetic moments when using LAMMPS "
            "atom_style atomic. Leave disabled to preserve intentional spins, "
            "for example with a custom atom_style spin template. This applies "
            "to both generate() and all exported formats; source files are unchanged."
        )

        return [
            Argument("files", [str, list], optional=False, doc=doc_files),
            Argument("prefix", str, optional=True, default=None, doc=doc_prefix),
            Argument("fmt", str, optional=True, default="auto", doc=doc_fmt),
            Argument(
                "remove_pbc", bool, optional=True, default=False, doc=doc_remove_pbc
            ),
            Argument(
                "remove_spins", bool, optional=True, default=False, doc=doc_remove_spins
            ),
        ]
