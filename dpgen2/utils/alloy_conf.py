import random
import dpdata
import tempfile
import numpy as np
from pathlib import Path
from typing import (
    Union, List, Tuple
)
from dargs import (
    Argument,
    Variant,
)
from .unit_cells import generate_unit_cell

class AlloyConf():
    """
    Parameters
    ----------
    lattice     Union[dpdata.System, Tuple[str,float]]
                Lattice of the alloy confs. can be 
                `dpdata.System`: lattice in `dpdata.System`
                `Tuple[str, float]`: pair of lattice type and lattice constant.
                lattice type can be "bcc", "fcc", "hcp", "sc" or "diamond"
    replicate   Union[List[int], Tuple[int], int]
                replicate of the lattice
    type_map    List[str]
                The type map
    """

    def __init__(
            self,
            lattice : Union[dpdata.System, Tuple[str,float]],
            type_map : List[str],
            replicate : Union[List[int], Tuple[int], int] = None,
    )->None:
        # init sys
        if type(lattice) != dpdata.System: 
            sys = generate_unit_cell(lattice[0], lattice[1])
        else:
            sys = lattice
        # replicate
        if type(replicate) == int:
            replicate = [replicate] * 3
        if replicate is not None:
            sys = sys.replicate(replicate)            
        # set atom types
        self.ntypes = len(type_map)
        self.natoms = sum(sys['atom_numbs'])
        sys.data['atom_names'] = type_map
        sys.data['atom_numbs'] = [0] * self.ntypes
        sys.data['atom_numbs'][0] = self.natoms
        sys.data['atom_types'] = np.array([0] * self.natoms, dtype=int)
        self.type_population = [ii for ii in range(self.ntypes)]
        # record sys
        self.sys = sys

    def generate_file_content(
            self,
            numb_confs,
            concentration: Union[List[List[float]], List[float], None] = None,
            cell_pert_frac: float = 0.0,
            atom_pert_dist: float = 0.0,
            fmt : str = 'lammps/lmp'
    ) -> List[str]:
        """
        Parameters
        ----------
        numb_confs      int
                        Number of configurations to generate
        concentration   List[List[float]] or List[float] or None
                        If `List[float]`, the concentrations of each element. The length of
                        the list should be the same as the `type_map`.
                        If `List[List[float]]`, a list of concentrations (`List[float]`) is
                        randomly picked from the List.
                        If `None`, the elements are assumed to be of equal concentration.
        cell_pert_frac  float
                        fraction of cell perturbation
        atom_pert_dist  float
                        the atom perturbation distance (unit angstrom).
        fmt             str
                        the format of the returned conf strings. 
                        Should be one of the formats supported by `dpdata`

        Returns
        -------
        conf_list       List[str]
                        A list of file content of configurations.
        """
        ret = []
        for ii in range(numb_confs):
            ss = self._generate_one_sys(
                concentration, cell_pert_frac, atom_pert_dist)
            tf = Path(tempfile.NamedTemporaryFile().name)
            ss.to(fmt, tf)
            ret.append(tf.read_text())
            tf.unlink()
        return ret


    def generate_systems(
            self,
            numb_confs,
            concentration: Union[List[List[float]], List[float], None] = None,
            cell_pert_frac: float = 0.0,
            atom_pert_dist: float = 0.0,
    ) -> List[str]:
        """
        Parameters
        ----------
        numb_confs      int
                        Number of configurations to generate
        concentration   List[List[float]] or List[float] or None
                        If `List[float]`, the concentrations of each element. The length of
                        the list should be the same as the `type_map`.
                        If `List[List[float]]`, a list of concentrations (`List[float]`) is
                        randomly picked from the List.
                        If `None`, the elements are assumed to be of equal concentration.
        cell_pert_frac  float
                        fraction of cell perturbation
        atom_pert_dist  float
                        the atom perturbation distance (unit angstrom).

        Returns
        -------
        conf_list       List[dpdata.System]
                        A list of generated confs in `dpdata.System`.
        """
        ret = [self._generate_one_sys(
                concentration, cell_pert_frac, atom_pert_dist)
               for ii in range(numb_confs)]
        return ret


    def _generate_one_sys(
            self,
            concentration: Union[List[List[float]], List[float], None] = None,
            cell_pert_frac: float = 0.0,
            atom_pert_dist: float = 0.0,
    ) -> str:        
        if concentration is None:
            cc = [1./float(self.ntypes) for ii in range(self.ntypes)]
        elif type(concentration) is list and type(concentration[0]) is list:
            cc = random.choice(concentration)
        elif type(concentration) is list and \
             (type(concentration[0]) is float or type(concentration[0]) is int):
            cc = concentration
        else :
            raise RuntimeError('unsupported concentration type')
        ret_sys = self.sys.perturb(1, cell_pert_frac, atom_pert_dist)[0]
        ret_sys.data['atom_types'] = np.array(random.choices(
            self.type_population, 
            weights=cc,
            k=self.natoms,
        ), dtype=int)
        ret_sys.data['atom_numbs'] = list(np.bincount(
            ret_sys.data['atom_types'], 
            minlength=self.ntypes,
        ))
        return ret_sys


def generate_alloy_conf_args():
    doc_lattice = 'The lattice. Should be a list providing [ "lattice_type", lattice_const ], or a list providing [ "/path/to/dpdata/system", "fmt" ]. The two styles are distinguished by the type of the second element.'
    doc_replicate = 'The number of replicates in each direction'
    doc_type_map = 'The type map of the system'
    doc_numb_confs = 'The number of configurations to generate'
    doc_concentration = 'The concentration of each element. If None all elements have the same concentration'
    doc_cell_pert_frac = 'The faction of cell perturbation'
    doc_atom_pert_dist = 'The distance of atomic position perturbation'
    doc_fmt = 'The format of file content'

    return [
        Argument("lattice", [list,tuple], doc=doc_lattice),
        Argument("type_map", list, doc=doc_type_map),
        Argument("replicate", list, optional=True, default=None, doc=doc_replicate),
        Argument("numb_confs", int, optional=True, default=1, doc=doc_numb_confs),
        Argument("concentration", list, optional=True, default=None, doc=doc_concentration),
        Argument("cell_pert_frac", float, optional=True, default=0.0, doc=doc_cell_pert_frac),
        Argument("atom_pert_dist", float, optional=True, default=0.0, doc=doc_atom_pert_dist),
        Argument("fmt", str, optional=True, default="lammps/lmp", doc=doc_fmt),        
    ]


def normalize(data):
    sca = generate_alloy_conf_args()
    base = Argument("base", dict, sca)
    data = base.normalize_value(data, trim_pattern="_*")
    base.check_value(data, strict=True)
    return data    


def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    sca = generate_alloy_conf_args()
    base = Argument("conf_config", dict, sca,
                    doc='Generate file content of alloy configurations')
    ptr = []
    ptr.append(base.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split('\n'):
        if 'argument path' in ii:
            key_words.append(ii.split(':')[1].replace('`','').strip())
    return "\n\n".join(ptr)


def generate_alloy_conf_file_content(
        lattice : Union[dpdata.System, Tuple[str,float]],
        type_map : List[str],
        numb_confs,
        replicate : Union[List[int], Tuple[int], int] = None,
        concentration: Union[List[List[float]], List[float], None] = None,
        cell_pert_frac: float = 0.0,
        atom_pert_dist: float = 0.0,
        fmt : str = 'lammps/lmp'
):
    ac = AlloyConf(lattice, type_map, replicate=replicate)
    return ac.generate_file_content(
        numb_confs, 
        concentration=concentration,
        fmt=fmt,
        cell_pert_frac=cell_pert_frac,
        atom_pert_dist=atom_pert_dist,
    )

