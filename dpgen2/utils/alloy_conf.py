import random
import dpdata
import tempfile
import numpy as np
from pathlib import Path
from typing import (
    Union, List, Tuple
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
            replicate : Union[List[int], Tuple[int], int],
            type_map : List[str],
    )->None:
        # init sys
        if type(lattice) != dpdata.System: 
            sys = generate_unit_cell(lattice[0], lattice[1])
        else:
            sys = lattice
        # replicate
        if type(replicate) == int:
            replicate = [replicate] * 3
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
