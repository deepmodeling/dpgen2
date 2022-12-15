import numpy as np
import dpdata
from pathlib import Path
from typing import (
    Optional,
    Tuple, 
    List, 
    Set, 
    Dict,
    Union,
)
from dargs import (
    dargs, 
    Argument, 
    Variant, 
    ArgumentEncoder,
)

class VaspInputs():
    def __init__(
            self,
            kspacing : Union[float, List[float]],
            incar_template_name : str,
            potcar_names : Dict[str, str],
            kgamma : bool = True,
    ):
        """
        Parameters
        ----------
        kspacing : Union[float, List[float]]
                The kspacing. If it is a number, then three directions use the same
                ksapcing, otherwise it is a list of three numbers, specifying the
                kspacing used in the x, y and z dimension.
        incar_template_name: str
                A template INCAR file. 
        potcar_names : Dict[str,str]
                The potcar files for the elements. For example
                { 
                   "H" : "/path/to/POTCAR_H",
                   "O" : "/path/to/POTCAR_O",
                }
        kgamma : bool
                K-mesh includes the gamma point
        """
        self.kspacing = kspacing
        self.kgamma = kgamma
        self.incar_from_file(incar_template_name)
        self.potcars_from_file(potcar_names)

    @property
    def incar_template(self):
        return self._incar_template

    @property
    def potcars(self):
        return self._potcars

    def incar_from_file(
            self,
            fname : str,
    ):
        self._incar_template = Path(fname).read_text()

    def potcars_from_file(
            self,
            dict_fnames : Dict[str,str],
    ):
        self._potcars = {}
        for kk,vv in dict_fnames.items():
            self._potcars[kk] = Path(vv).read_text()            

    def make_potcar(
            self, 
            atom_names,
    ) -> str:        
        potcar_contents = []
        for nn in atom_names:
            potcar_contents.append(self._potcars[nn])
        return "".join(potcar_contents)            

    def make_kpoints(
            self,
            box : np.ndarray,
    ) -> str:
        return make_kspacing_kpoints(box, self.kspacing, self.kgamma)

    @staticmethod
    def args():
        doc_pp_files = 'The pseudopotential files set by a dict, e.g. {"Al" : "path/to/the/al/pp/file", "Mg" : "path/to/the/mg/pp/file"}'
        doc_incar = "The path to the template incar file"
        doc_kspacing = "The spacing of k-point sampling. `ksapcing` will overwrite the incar template"
        doc_kgamma = "If the k-mesh includes the gamma point. `kgamma` will overwrite the incar template"
        return [
            Argument("pp_files", dict, optional=False, doc=doc_pp_files),
            Argument("incar", str, optional=False, doc=doc_pp_files),
            Argument("kspacing", float, optional=False, doc=doc_kspacing),
            Argument("kgamma", bool, optional=True, default=True, doc=doc_kgamma),
        ]

    @staticmethod
    def normalize_config(data = {}, strict=True):
        ta = VaspInputs.args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=strict)
        return data



def make_kspacing_kpoints(box, kspacing, kgamma) :
    if type(kspacing) is not list:
        kspacing = [kspacing, kspacing, kspacing]
    box = np.array(box)
    rbox = _reciprocal_box(box)
    kpoints = [max(1,(np.ceil(2 * np.pi * np.linalg.norm(ii) / ks).astype(int))) for ii,ks in zip(rbox,kspacing)]
    ret = _make_vasp_kpoints(kpoints, kgamma)
    return ret


def _make_vasp_kp_gamma(kpoints):
    ret = ""
    ret += "Automatic mesh\n"
    ret += "0\n"
    ret += "Gamma\n"
    ret += "%d %d %d\n" % (kpoints[0], kpoints[1], kpoints[2])
    ret += "0  0  0\n"
    return ret

def _make_vasp_kp_mp(kpoints):
    ret = ""
    ret += "K-Points\n"
    ret += "0\n"
    ret += "Monkhorst Pack\n"
    ret += "%d %d %d\n" % (kpoints[0], kpoints[1], kpoints[2])
    ret += "0  0  0\n"
    return ret

def _make_vasp_kpoints (kpoints, kgamma = False) :
    if kgamma :
        ret = _make_vasp_kp_gamma(kpoints)
    else :
        ret = _make_vasp_kp_mp(kpoints)
    return ret
    
def _reciprocal_box(box) :
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    return rbox

