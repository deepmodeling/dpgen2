#!/usr/bin/env python3
"""Regenerate randomized VASP k-point test data on explicit invocation."""

from pathlib import (
    Path,
)

import dpdata
import numpy as np
from ase.geometry import (
    cellpar_to_cell,
)

from dpgen2.fp.vasp import (
    make_kspacing_kpoints,
)

FIXTURE_DIR = Path(__file__).resolve().parent


def make_one(out_dir):
    """Generate a consistent randomized POSCAR and kp.ref pair in *out_dir*."""
    # [0.5, 1)
    [aa, bb, cc] = np.random.random(3) * 0.5 + 0.5
    # Preserve the original near-collinear sampling: [1, 1 + 178/180) degrees.
    [alpha, beta, gamma] = np.random.random(3) * (178 / 180) + 1
    cell = cellpar_to_cell([aa, bb, cc, alpha, beta, gamma])
    system = dpdata.System(FIXTURE_DIR / "POSCAR")
    system["cells"][0] = cell
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poscar = out_dir / "POSCAR"
    system.to_vasp_poscar(poscar)
    # Use the serialized cell, exactly as test_vasp reads the fixture.
    cell = dpdata.System(poscar)["cells"][0]
    kpoints = make_kspacing_kpoints(cell, 0.16, False).splitlines()[3]
    (out_dir / "kp.ref").write_text(kpoints + "\n")


def main(ntest=30):
    """Regenerate all randomized fixture directories."""
    for index in range(ntest):
        make_one(FIXTURE_DIR / ("test.%03d" % index))


if __name__ == "__main__":
    main()
