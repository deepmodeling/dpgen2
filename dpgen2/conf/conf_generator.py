import dpdata
import dargs
import tempfile
from pathlib import Path
from typing import (
    List, Dict,
)
from abc import (
    ABC, abstractmethod,
)

class ConfGenerator(ABC):
    @abstractmethod
    def generate(self, type_map):
        r"""Method of generating configurations.

        Parameters
        ----------
        type_map: List[str]
                The type map.

        Returns
        -------
        confs:  dpdata.MultiSystems
                The returned configurations in `dpdata.MultiSystems` format
        
        """
        pass


    def get_file_content(
            self, 
            type_map,
            fmt='lammps/lmp',
    ):
        r"""Get the file content of configurations

        Parameters
        ----------
        type_map: List[str]
                The type map.

        Returns
        -------
        conf_list: List[str]
                A list of file content of configurations.

        """
        ret = []
        ms = self.generate(type_map)
        for ii in range(len(ms)):
            ss = ms[ii]
            for jj in range(ss.get_nframes()):
                tf = Path(tempfile.NamedTemporaryFile().name)
                ss[jj].to(fmt, tf)
                ret.append(tf.read_text())
                tf.unlink()
        return ret


    @staticmethod
    @abstractmethod
    def args():
        pass


    @classmethod
    def normalize_config(
            cls, 
            data: Dict={}, 
            strict: bool=True) -> Dict:
        r"""Normalized the argument.

        Parameters
        ----------
        data: Dict
            The input dict of arguments.
        strict: bool
            Strictly check the arguments.
        
        Returns
        -------
        data: Dict
            The normalized arguments.
        
        """
        ta = cls.args()
        base = dargs.Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=strict)
        return data
