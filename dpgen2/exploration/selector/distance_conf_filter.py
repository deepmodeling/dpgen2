import os
import shutil

import dpdata
import numpy as np

from . import (
    ConfFilter,
)

safe_dist_dict = {
    "H": 1.2255,
    "He": 0.936,
    "Li": 1.8,
    "Be": 1.56,
    "B": 1.32,
    "C": 1.32,
    "N": 1.32,
    "O": 1.32,
    "F": 1.26,
    "Ne": 1.92,
    "Na": 1.595,
    "Mg": 1.87,
    "Al": 1.87,
    "Si": 1.76,
    "P": 1.65,
    "S": 1.65,
    "Cl": 1.65,
    "Ar": 2.09,
    "K": 2.3,
    "Ca": 2.3,
    "Sc": 2.0,
    "Ti": 2.0,
    "V": 2.0,
    "Cr": 1.9,
    "Mn": 1.95,
    "Fe": 1.9,
    "Co": 1.9,
    "Ni": 1.9,
    "Cu": 1.9,
    "Zn": 1.9,
    "Ga": 2.0,
    "Ge": 2.0,
    "As": 2.0,
    "Se": 2.1,
    "Br": 2.1,
    "Kr": 2.3,
    "Rb": 2.5,
    "Sr": 2.5,
    "Y": 2.1,
    "Zr": 2.1,
    "Nb": 2.1,
    "Mo": 2.1,
    "Tc": 2.1,
    "Ru": 2.1,
    "Rh": 2.1,
    "Pd": 2.1,
    "Ag": 2.1,
    "Cd": 2.1,
    "In": 2.0,
    "Sn": 2.0,
    "Sb": 2.0,
    "Te": 2.0,
    "I": 2.0,
    "Xe": 2.0,
    "Cs": 2.5,
    "Ba": 2.8,
    "La": 2.5,
    "Ce": 2.55,
    "Pr": 2.7,
    "Nd": 2.8,
    "Pm": 2.8,
    "Sm": 2.8,
    "Eu": 2.8,
    "Gd": 2.8,
    "Tb": 2.8,
    "Dy": 2.8,
    "Ho": 2.8,
    "Er": 2.6,
    "Tm": 2.8,
    "Yb": 2.8,
    "Lu": 2.8,
    "Hf": 2.4,
    "Ta": 2.5,
    "W": 2.3,
    "Re": 2.3,
    "Os": 2.3,
    "Ir": 2.3,
    "Pt": 2.3,
    "Au": 2.3,
    "Hg": 2.3,
    "Tl": 2.3,
    "Pb": 2.3,
    "Bi": 2.3,
    "Po": 2.3,
    "At": 2.3,
    "Rn": 2.3,
    "Fr": 2.9,
    "Ra": 2.9,
    "Ac": 2.9,
    "Th": 2.8,
    "Pa": 2.8,
    "U": 2.8,
    "Np": 2.8,
    "Pu": 2.8,
    "Am": 2.8,
    "Cm": 2.8,
    "Cf": 2.3,
}
for k in safe_dist_dict:
    safe_dist_dict[k] *= 0.441


def check_multiples(a, b, c, multiple):
    values = [a, b, c]

    for i in range(len(values)):
        for j in range(len(values)):
            if i != j:
                if values[i] > multiple * values[j]:
                    print(
                        f"Value {values[i]} is {multiple} times greater than {values[j]}"
                    )
                    return True
    return False


class DistanceConfFilter(ConfFilter):
    def __init__(self):
        pass

    def check(
        self,
        coords: np.ndarray,
        cell: np.ndarray,
        atom_types: np.ndarray,
        nopbc: bool,
    ):
        from ase import (
            Atoms,
        )
        from ase.build import (
            make_supercell,
        )

        atom_names = list(safe_dist_dict)
        structure = Atoms(
            positions=coords,
            numbers=[atom_names.index(n) + 1 for n in atom_types],
            cell=cell,
            pbc=(not nopbc),
        )

        cell = structure.get_cell()

        if (
            cell[1][0] > 1.732 * cell[1][1]
            or cell[2][0] > 1.732 * cell[2][2]
            or cell[2][1] > 1.732 * cell[2][2]
        ):
            print("Inclined box")
            return False

        a = cell[0][0]
        b = cell[1][1]
        c = cell[2][2]

        if check_multiples(a, b, c, 5):
            print("One side is 5 larger than another")
            return False

        P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        extended_structure = make_supercell(structure, P)

        coords = extended_structure.positions
        symbols = extended_structure.get_chemical_symbols()

        num_atoms = len(coords)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = extended_structure.get_distance(i, j, mic=True)
                type_i = symbols[i]
                type_j = symbols[j]
                dr = safe_dist_dict[type_i] + safe_dist_dict[type_j]

                if dist < dr:
                    print(
                        f"Dangerous close for {type_i} - {type_j}, {dist:.5f} less than {dr:.5f}"
                    )
                    return False

        print("Valid structure")
        return True