import dpdata
import numpy as np
from pathlib import Path
import tempfile

def generate_unit_cell(
        crystal : str,
        latt : float = 1.0,
)->dpdata.System:
    if crystal == 'bcc':
        stru = BCC()
    elif crystal == 'fcc':
        stru = FCC()
    elif crystal == 'hcp':
        stru = HCP()
        latt = latt * np.sqrt(2)
    elif crystal == 'sc' : 
        stru = SC()
    elif crystal == 'diamond':
        stru = DIAMOND()
    else:
        raise RuntimeError('unknown latt')
    
    tf = tempfile.NamedTemporaryFile()
    Path(tf.name).write_text(stru.poscar_unit(latt))
    return dpdata.System(tf.name, fmt='vasp/poscar')    


class BCC():
    def numb_atoms (self) :
        return 2

    def gen_box (self) :    
        return np.eye(3)

    def poscar_unit (self, latt) :
        box = self.gen_box()
        ret  = ""
        ret += "BCC : a = %f \n" % latt
        ret += "%.16f\n" % (latt)
        ret += "%.16f %.16f %.16f\n" % (box[0][0], box[0][1], box[0][2])
        ret += "%.16f %.16f %.16f\n" % (box[1][0], box[1][1], box[1][2])
        ret += "%.16f %.16f %.16f\n" % (box[2][0], box[2][1], box[2][2])
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += "%.16f %.16f %.16f\n" % (0.0, 0.0, 0.0)
        ret += "%.16f %.16f %.16f\n" % (0.5, 0.5, 0.5)
        return ret
    
class FCC():
    def numb_atoms (self) :
        return 4

    def gen_box (self) :
        return np.eye(3)

    def poscar_unit (self, latt) :
        box = self.gen_box()
        ret  = ""
        ret += "FCC : a = %f \n" % latt
        ret += "%.16f\n" % (latt)
        ret += "%.16f %.16f %.16f\n" % (box[0][0], box[0][1], box[0][2])
        ret += "%.16f %.16f %.16f\n" % (box[1][0], box[1][1], box[1][2])
        ret += "%.16f %.16f %.16f\n" % (box[2][0], box[2][1], box[2][2])
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += "%.16f %.16f %.16f\n" % (0.0, 0.0, 0.0)
        ret += "%.16f %.16f %.16f\n" % (0.5, 0.5, 0.0)
        ret += "%.16f %.16f %.16f\n" % (0.5, 0.0, 0.5)
        ret += "%.16f %.16f %.16f\n" % (0.0, 0.5, 0.5)
        return ret
    

class HCP():
    def numb_atoms (self) :
        return 2

    def gen_box (self) :
        box = np.array ([[  1, 0, 0], 
                        [0.5, 0.5 * np.sqrt(3), 0],
                        [0, 0, 2. * np.sqrt(2./3.)]])
        return box

    def poscar_unit (self, latt) :
        box = self.gen_box()
        ret  = ""
        ret += "HCP : a = %f / sqrt(2)\n" % latt
        ret += "%.16f\n" % (latt / np.sqrt(2))
        ret += "%.16f %.16f %.16f\n" % (box[0][0], box[0][1], box[0][2])
        ret += "%.16f %.16f %.16f\n" % (box[1][0], box[1][1], box[1][2])
        ret += "%.16f %.16f %.16f\n" % (box[2][0], box[2][1], box[2][2])
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += "%.16f %.16f %.16f\n" % (0, 0, 0)
        ret += "%.16f %.16f %.16f\n" % (1./3, 1./3, 1./2)
        return ret

class SC():
    def numb_atoms (self) :
        return 1

    def gen_box (self) :
        return np.eye(3)

    def poscar_unit (self, latt) :
        box = self.gen_box()
        ret  = ""
        ret += "SC : a = %f \n" % latt
        ret += "%.16f\n" % (latt)
        ret += "%.16f %.16f %.16f\n" % (box[0][0], box[0][1], box[0][2])
        ret += "%.16f %.16f %.16f\n" % (box[1][0], box[1][1], box[1][2])
        ret += "%.16f %.16f %.16f\n" % (box[2][0], box[2][1], box[2][2])
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += "%.16f %.16f %.16f\n" % (0.0, 0.0, 0.0)
        return ret


class DIAMOND():
    def numb_atoms (self) :
        return 2

    def gen_box (self) :
        box = [[0.000000, 1.000000, 1.000000],
               [1.000000, 0.000000, 1.000000],
               [1.000000, 1.000000, 0.000000]
        ]
        return np.array(box)

    def poscar_unit (self, latt) :
        box = self.gen_box()
        ret  = ""
        ret += "DIAMOND\n"
        ret += "%.16f\n" % (latt)
        ret += "%.16f %.16f %.16f\n" % (box[0][0], box[0][1], box[0][2])
        ret += "%.16f %.16f %.16f\n" % (box[1][0], box[1][1], box[1][2])
        ret += "%.16f %.16f %.16f\n" % (box[2][0], box[2][1], box[2][2])
        ret += "Type\n"
        ret += "%d\n" % self.numb_atoms()
        ret += "Direct\n"
        ret += "%.16f %.16f %.16f\n" % (0.12500000000000,   0.12500000000000,   0.12500000000000)
        ret += "%.16f %.16f %.16f\n" % (0.87500000000000,   0.87500000000000,   0.87500000000000)
        return ret
