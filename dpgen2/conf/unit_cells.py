import tempfile
from pathlib import (
    Path,
)

import dpdata
import numpy as np


def generate_unit_cell(
    crystal: str,
    latt: float = 1.0,
) -> dpdata.System:
    if crystal == "bcc":
        stru = BCC()
    elif crystal == "fcc":
        stru = FCC()
    elif crystal == "hcp":
        stru = HCP()
        latt = latt * np.sqrt(2)
    elif crystal == "sc":
        stru = SC()
    elif crystal == "diamond":
        stru = DIAMOND()
    else:
        raise RuntimeError("unknown latt")

    tf = tempfile.NamedTemporaryFile()
    Path(tf.name).write_text(stru.poscar_unit(latt))
    return dpdata.System(tf.name, fmt="vasp/poscar")


class BCC:
    def numb_atoms(self):
        return 2

    def gen_box(self):
        return np.eye(3)

    def poscar_unit(self, latt):
        box = self.gen_box()
        ret = ""
        ret += f"BCC : a = {latt:f} \n"
        ret += f"{latt:.16f}\n"
        ret += f"{box[0][0]:.16f} {box[0][1]:.16f} {box[0][2]:.16f}\n"
        ret += f"{box[1][0]:.16f} {box[1][1]:.16f} {box[1][2]:.16f}\n"
        ret += f"{box[2][0]:.16f} {box[2][1]:.16f} {box[2][2]:.16f}\n"
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += f"{0.0:.16f} {0.0:.16f} {0.0:.16f}\n"
        ret += f"{0.5:.16f} {0.5:.16f} {0.5:.16f}\n"
        return ret


class FCC:
    def numb_atoms(self):
        return 4

    def gen_box(self):
        return np.eye(3)

    def poscar_unit(self, latt):
        box = self.gen_box()
        ret = ""
        ret += f"FCC : a = {latt:f} \n"
        ret += f"{latt:.16f}\n"
        ret += f"{box[0][0]:.16f} {box[0][1]:.16f} {box[0][2]:.16f}\n"
        ret += f"{box[1][0]:.16f} {box[1][1]:.16f} {box[1][2]:.16f}\n"
        ret += f"{box[2][0]:.16f} {box[2][1]:.16f} {box[2][2]:.16f}\n"
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += f"{0.0:.16f} {0.0:.16f} {0.0:.16f}\n"
        ret += f"{0.5:.16f} {0.5:.16f} {0.0:.16f}\n"
        ret += f"{0.5:.16f} {0.0:.16f} {0.5:.16f}\n"
        ret += f"{0.0:.16f} {0.5:.16f} {0.5:.16f}\n"
        return ret


class HCP:
    def numb_atoms(self):
        return 2

    def gen_box(self):
        box = np.array(
            [[1, 0, 0], [0.5, 0.5 * np.sqrt(3), 0], [0, 0, 2.0 * np.sqrt(2.0 / 3.0)]]
        )
        return box

    def poscar_unit(self, latt):
        box = self.gen_box()
        ret = ""
        ret += f"HCP : a = {latt:f} / sqrt(2)\n"
        ret += "%.16f\n" % (latt / np.sqrt(2))
        ret += f"{box[0][0]:.16f} {box[0][1]:.16f} {box[0][2]:.16f}\n"
        ret += f"{box[1][0]:.16f} {box[1][1]:.16f} {box[1][2]:.16f}\n"
        ret += f"{box[2][0]:.16f} {box[2][1]:.16f} {box[2][2]:.16f}\n"
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += f"{0:.16f} {0:.16f} {0:.16f}\n"
        ret += f"{1.0 / 3:.16f} {1.0 / 3:.16f} {1.0 / 2:.16f}\n"
        return ret


class SC:
    def numb_atoms(self):
        return 1

    def gen_box(self):
        return np.eye(3)

    def poscar_unit(self, latt):
        box = self.gen_box()
        ret = ""
        ret += f"SC : a = {latt:f} \n"
        ret += f"{latt:.16f}\n"
        ret += f"{box[0][0]:.16f} {box[0][1]:.16f} {box[0][2]:.16f}\n"
        ret += f"{box[1][0]:.16f} {box[1][1]:.16f} {box[1][2]:.16f}\n"
        ret += f"{box[2][0]:.16f} {box[2][1]:.16f} {box[2][2]:.16f}\n"
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += f"{0.0:.16f} {0.0:.16f} {0.0:.16f}\n"
        return ret


class DIAMOND:
    def numb_atoms(self):
        return 2

    def gen_box(self):
        box = [
            [0.000000, 1.000000, 1.000000],
            [1.000000, 0.000000, 1.000000],
            [1.000000, 1.000000, 0.000000],
        ]
        return np.array(box)

    def poscar_unit(self, latt):
        box = self.gen_box()
        ret = ""
        ret += "DIAMOND\n"
        ret += f"{latt:.16f}\n"
        ret += f"{box[0][0]:.16f} {box[0][1]:.16f} {box[0][2]:.16f}\n"
        ret += f"{box[1][0]:.16f} {box[1][1]:.16f} {box[1][2]:.16f}\n"
        ret += f"{box[2][0]:.16f} {box[2][1]:.16f} {box[2][2]:.16f}\n"
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += (
            f"{0.12500000000000:.16f} {0.12500000000000:.16f} {0.12500000000000:.16f}\n"
        )
        ret += (
            f"{0.87500000000000:.16f} {0.87500000000000:.16f} {0.87500000000000:.16f}\n"
        )
        return ret
