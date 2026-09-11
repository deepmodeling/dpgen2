#!/usr/bin/env python3
"""Regenerate randomized VASP k-point test data on explicit invocation."""

import os

import dpdata
import numpy as np
from ase.geometry import (
    cellpar_to_cell,
)


def make_one(out_dir):
    """Generate one randomized POSCAR fixture in *out_dir*."""
    # [0.5, 1)
    [aa, bb, cc] = np.random.random(3) * 0.5 + 0.5
    # [1, 179)
    [alpha, beta, gamma] = np.random.random(3) * (178 / 180) + 1
    cell = cellpar_to_cell([aa, bb, cc, alpha, beta, gamma])
    system = dpdata.System("POSCAR")
    system["cells"][0] = cell
    os.makedirs(out_dir, exist_ok=True)
    system.to_vasp_poscar(os.path.join(out_dir, "POSCAR"))


def main(ntest=30):
    """Regenerate all randomized fixture directories."""
    for index in range(ntest):
        make_one("test.%03d" % index)


if __name__ == "__main__":
    main()
