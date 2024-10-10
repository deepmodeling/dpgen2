import copy
import logging
import random
from typing import (
    List,
)

import numpy as np

from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_input_file,
    calypso_run_opt_file,
)

from .calypso import (
    make_calypso_input,
)
from .task import (
    ExplorationTask,
)
from .task_group import (
    ExplorationTaskGroup,
)

atomic_symbols = (
    'X',  # placeholder
    'H', 'He',
    'Li', 'Be', 'B',  'C',  'N', 'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S',  'Cl', 'Ar',
    'K',  'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
    'Cs', 'Ba',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra',
    'Ac','Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
)  # fmt: skip
atomic_number_map = {key: value for value, key in enumerate(atomic_symbols)}
# Covalent radii from:
#
#  Covalent radii revisited,
#  Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés,
#  Jorge Echeverría, Eduard Cremades, Flavia Barragán and Santiago Alvarez,
#  Dalton Trans., 2008, 2832-2838 DOI:10.1039/B801115J
UNKN = 0.2
covalent_radii = [
    # X, placeholder
    UNKN,
    # H    He
    0.31, 0.28,
    # Li    Be     B     C     N     O     F    Ne
    1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    # Na    Mg    Al    Si     P     S    Cl    Ar
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
    #  K    Ca    Sc    Ti     V    Cr    Mn    Fe    Co    Ni    Cu    Zn    Ga    Ge    As    Se    Br    Kr
    2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
    # Rb    Sr     Y    Zr    Nb    Mo    Tc    Ru    Rh    Pd    Au    Cd    In    Sn    Sb    Te     I    Xe
    2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
    # Cs    Ba
    2.44, 2.15,
    # La    Ce    Pr    Nd    Pm    Sm    Eu    Gd    Tb    Dy    Ho    Er    Tm    Yb    Lu
    2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,
    # Hf    Ta     W    Re    Os    Ir    Pt    Au    Hg    Tl    Pb    Bi    Po    At    Rn
    1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
    # Fr    Ra
    2.60, 2.21,
    # Ac    Th    Pa     U    Np    Pu    Am    Cm    Bk    Cf    Es    Fm    Md    No    Lr
    2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN,
    # Rf    Db    Sg    Bh    Hs    Mt    Ds    Rg    Cn    Nh    Fl    Mc    Lv    Ts    Og
    UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN,
]  # fmt: skip


class CalyTaskGroup(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()

    def set_params(
        self,
        numb_of_species,
        name_of_atoms,
        numb_of_atoms,
        distance_of_ions=None,
        atomic_number=None,
        pop_size: int = 30,
        max_step: int = 5,
        system_name: str = "CALYPSO",
        numb_of_formula: List[int] = [1, 1],
        pressure: float = 0.001,
        fmax: float = 0.01,
        volume: float = 0,
        ialgo: int = 2,
        pso_ratio: float = 0.6,
        icode: int = 15,
        numb_of_lbest: int = 4,
        numb_of_local_optim: int = 4,
        command: str = "sh submit.sh",
        max_time: int = 9000,
        gen_type: int = 1,
        pick_up: bool = False,
        pick_step: int = 1,
        parallel: bool = False,
        split: bool = True,
        spec_space_group: List[int] = [2, 230],
        vsc: bool = True,
        ctrl_range: List[List[int]] = [[1, 10]],
        max_numb_atoms: int = 100,
        opt_step: int = 1000,
    ):
        """
        Set calypso parameters
        """
        self.numb_of_species = numb_of_species
        self.numb_of_atoms = numb_of_atoms

        if isinstance(name_of_atoms, list) and all(
            [isinstance(i, list) for i in name_of_atoms]
        ):
            overlap = set(name_of_atoms[0])
            for temp in name_of_atoms[1:]:
                overlap = overlap & set(temp)

            if any(map(lambda s: (set(s) - overlap) == 0, name_of_atoms)):
                raise ValueError(
                    f"Any sub-list should not equal with intersection, e.g. [[A,B,C], [B,C], [C]] is not allowed."
                )

            while True:
                choice = []
                for _atoms in name_of_atoms:
                    value = random.choice(_atoms)
                    logging.info(
                        f"randomly choose {value} from {_atoms}, already choose: {choice}"
                    )
                    if value in choice:
                        break
                    choice.append(value)
                else:
                    break
            self.name_of_atoms = choice
            logging.info(f"The final choice is {self.name_of_atoms}")
            self.atomic_number = [atomic_symbols.index(i) for i in self.name_of_atoms]
        else:
            self.name_of_atoms = name_of_atoms
            self.atomic_number = atomic_number

        if isinstance(distance_of_ions, dict):
            updated_table = copy.deepcopy(covalent_radii)
            for key, value in distance_of_ions.items():
                updated_table[atomic_number_map[key]] = value

            temp_distance_mtx = np.zeros((numb_of_species, numb_of_species))
            for i in range(numb_of_species):
                for j in range(numb_of_species):
                    temp_distance_mtx[i][j] = round(
                        updated_table[atomic_number_map[self.name_of_atoms[i]]] * 0.7
                        + updated_table[atomic_number_map[self.name_of_atoms[j]]] * 0.7,
                        2,
                    )
            self.distance_of_ions = temp_distance_mtx
        else:
            self.distance_of_ions = distance_of_ions

        self.pop_size = pop_size
        self.max_step = max_step
        self.system_name = system_name
        self.numb_of_formula = numb_of_formula
        self.pressure = pressure
        self.fmax = fmax
        self.volume = volume
        self.ialgo = ialgo
        self.pso_ratio = pso_ratio
        self.icode = icode
        self.numb_of_lbest = numb_of_lbest
        self.numb_of_local_optim = numb_of_local_optim
        self.command = command
        self.max_time = max_time
        self.gen_type = gen_type
        self.pick_up = pick_up
        self.pick_step = pick_step
        self.parallel = parallel
        self.split = split
        self.spec_space_group = spec_space_group
        self.vsc = vsc
        self.ctrl_range = ctrl_range
        self.max_numb_atoms = max_numb_atoms
        self.opt_step = opt_step

        self.caly_set = True

    def make_task(self) -> ExplorationTaskGroup:
        """
        Make the CALYPSO task group.

        Returns
        -------
        task_grp: ExplorationTaskGroup
            Return one calypso task group.
        """
        if not self.caly_set:
            raise RuntimeError("calypso settings are not set")
        # clear all existing tasks
        self.clear()
        self.add_task(self._make_caly_task())
        return self

    def _make_caly_task(self) -> ExplorationTask:
        input_file_str, run_opt_str, check_opt_str = make_calypso_input(
            self.numb_of_species,
            self.name_of_atoms,
            self.atomic_number,
            self.numb_of_atoms,
            self.distance_of_ions,
            self.pop_size,
            self.max_step,
            self.system_name,
            self.numb_of_formula,
            self.pressure,
            self.fmax,
            self.volume,
            self.ialgo,
            self.pso_ratio,
            self.icode,
            self.numb_of_lbest,
            self.numb_of_local_optim,
            self.command,
            self.max_time,
            self.gen_type,
            self.pick_up,
            self.pick_step,
            self.parallel,
            self.split,
            self.spec_space_group,
            self.vsc,
            self.ctrl_range,
            self.max_numb_atoms,
            opt_step=self.opt_step,
        )
        task = ExplorationTask()
        task.add_file(calypso_input_file, input_file_str)
        task.add_file(calypso_run_opt_file, run_opt_str)
        task.add_file(calypso_check_opt_file, check_opt_str)
        return task
