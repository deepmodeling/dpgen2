import random
from typing import (
    List,
    Optional,
)

import dpdata
import numpy as np
import scipy.constants as pc
from packaging.version import (
    Version,
)

from dpgen2.constants import (
    lmp_traj_name,
)

calypso_run_opt_str = """#!/usr/bin/env python3

import os
import sys
import time
import glob
import numpy as np

from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter

from deepmd.calculator import DP

# structure optimization with DP model and ASE
# PSTRESS and fmax should exist in input.dat

def Get_Element_Num(elements):
    # Using the Atoms.symples to Know Element&Num
    element = []
    ele = {}
    element.append(elements[0])
    for x in elements:
        if x not in element :
            element.append(x)
    for x in element:
        ele[x] = elements.count(x)
    return element, ele

def Write_Contcar(contcar, element, ele, lat, pos):
    # Write CONTCAR
    f = open(contcar,'w')
    f.write('ASE-DP-OPT\\n')
    f.write('1.0\\n')
    for i in range(3):
        f.write('%15.10f %15.10f %15.10f\\n' % tuple(lat[i]))
    for x in element:
        f.write(x + '  ')
    f.write('\\n')
    for x in element:
        f.write(str(ele[x]) + '  ')
    f.write('\\n')
    f.write('Direct\\n')
    na = sum(ele.values())
    dpos = np.dot(pos,np.linalg.inv(lat))
    for i in range(na):
        f.write('%15.10f %15.10f %15.10f\\n' % tuple(dpos[i]))

def Write_Outcar(outcar, element, ele, volume, lat, pos, ene, force, stress, pstress):
    # Write OUTCAR
    f = open(outcar,'w')
    for x in element:
        f.write('VRHFIN =' + str(x) + '\\n')
    f.write('ions per type =')
    for x in element:
        f.write('%5d' % ele[x])
    f.write('\\nDirection     XX             YY             ZZ             XY             YZ             ZX\\n')
    f.write('in kB')
    f.write('%15.6f' % stress[0])
    f.write('%15.6f' % stress[1])
    f.write('%15.6f' % stress[2])
    f.write('%15.6f' % stress[3])
    f.write('%15.6f' % stress[4])
    f.write('%15.6f' % stress[5])
    f.write('\\n')
    ext_pressure = np.sum(stress[0] + stress[1] + stress[2])/3.0 - pstress
    f.write('external pressure = %20.6f kB    Pullay stress = %20.6f  kB\\n'% (ext_pressure, pstress))
    f.write('volume of cell : %20.6f\\n' % volume)
    f.write('direct lattice vectors\\n')
    for i in range(3):
        f.write('%10.6f %10.6f %10.6f\\n' % tuple(lat[i]))
    f.write('POSITION                                       TOTAL-FORCE(eV/Angst)\\n')
    f.write('-------------------------------------------------------------------\\n')
    na = sum(ele.values())
    for i in range(na):
        f.write('%15.6f %15.6f %15.6f' % tuple(pos[i]))
        f.write('%15.6f %15.6f %15.6f\\n' % tuple(force[i]))
    f.write('-------------------------------------------------------------------\\n')
    f.write('energy  without entropy= %20.6f %20.6f\\n' % (ene, ene/na))
    enthalpy = ene + pstress * volume / 1602.17733
    f.write('enthalpy is  TOTEN    = %20.6f %20.6f\\n' % (enthalpy, enthalpy/na))

def run_opt(fmax, stress, opt_step):
    # Using the ASE&DP to Optimize Configures

    calc = DP(model=sys.argv[1])    # init the model before iteration

    start = time.time()
    # pstress kbar
    pstress = stress
    # kBar to eV/A^3
    # 1 eV/A^3 = 160.21766028 GPa
    # 1 / 160.21766028 ~ 0.006242
    aim_stress = 1.0 * pstress* 0.01 * 0.6242 / 10.0

    poscar_list = sorted(glob.glob("POSCAR_*"), key=lambda x: x.strip("POSCAR_"))
    for poscar in poscar_list:
        to_be_opti = read(poscar)
        to_be_opti.calc = calc
        ucf = UnitCellFilter(to_be_opti, scalar_pressure=aim_stress)
        opt = LBFGS(ucf,trajectory=poscar.strip("POSCAR_") + '.traj')
        opt.run(fmax=fmax,steps=opt_step)
        atoms_lat = to_be_opti.cell
        atoms_pos = to_be_opti.positions
        atoms_force = to_be_opti.get_forces()
        atoms_stress = to_be_opti.get_stress()
        # eV/A^3 to GPa
        atoms_stress = atoms_stress/(0.01*0.6242)
        atoms_symbols = to_be_opti.get_chemical_symbols()
        atoms_ene = to_be_opti.get_potential_energy()
        atoms_vol = to_be_opti.get_volume()
        element, ele = Get_Element_Num(atoms_symbols)
        outcar = poscar.replace("POSCAR", "OUTCAR")
        contcar = poscar.replace("POSCAR", "CONTCAR")

        Write_Contcar(contcar, element, ele, atoms_lat, atoms_pos)
        Write_Outcar(outcar, element, ele, atoms_vol, atoms_lat, atoms_pos, atoms_ene, atoms_force, atoms_stress * -10.0, pstress)
"""

calypso_run_opt_str_end = """
if __name__ == '__main__':
    run_opt(fmax=%.3f, stress=%.3f, opt_step=%.3f)
"""

calypso_check_opt_str = """#!/usr/bin/env python3

import os
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter

# check if structure optimization worked well
# if not, this script will generate a fake outcar

def Get_Element_Num(elements):
    # Using the Atoms.symples to Know Element&Num
    element = []
    ele = {}
    element.append(elements[0])
    for x in elements:
        if x not in element :
            element.append(x)
    for x in element:
        ele[x] = elements.count(x)
    return element, ele

def Write_Contcar(element, ele, lat, pos):
    # Write CONTCAR
    f = open('CONTCAR','w')
    f.write('ASE-DP-FAILED\\n')
    f.write('1.0\\n')
    for i in range(3):
        f.write('%15.10f %15.10f %15.10f\\n' % tuple(lat[i]))
    for x in element:
        f.write(x + '  ')
    f.write('\\n')
    for x in element:
        f.write(str(ele[x]) + '  ')
    f.write('\\n')
    f.write('Direct\\n')
    na = sum(ele.values())
    dpos = np.dot(pos,np.linalg.inv(lat))
    for i in range(na):
        f.write('%15.10f %15.10f %15.10f\\n' % tuple(dpos[i]))

def Write_Outcar(element, ele, volume, lat, pos, ene, force, stress,pstress):
    # Write OUTCAR
    f = open('OUTCAR','w')
    for x in element:
        f.write('VRHFIN =' + str(x) + '\\n')
    f.write('ions per type =')
    for x in element:
        f.write('%5d' % ele[x])
    f.write('\\nDirection     XX             YY             ZZ             XY             YZ             ZX\\n')
    f.write('in kB')
    f.write('%15.6f' % stress[0])
    f.write('%15.6f' % stress[1])
    f.write('%15.6f' % stress[2])
    f.write('%15.6f' % stress[3])
    f.write('%15.6f' % stress[4])
    f.write('%15.6f' % stress[5])
    f.write('\\n')
    ext_pressure = np.sum(stress[0] + stress[1] + stress[2])/3.0 - pstress
    f.write('external pressure = %20.6f kB    Pullay stress = %20.6f  kB\\n'% (ext_pressure, pstress))
    f.write('volume of cell : %20.6f\\n' % volume)
    f.write('direct lattice vectors\\n')
    for i in range(3):
        f.write('%10.6f %10.6f %10.6f\\n' % tuple(lat[i]))
    f.write('POSITION                                       TOTAL-FORCE(eV/Angst)\\n')
    f.write('-------------------------------------------------------------------\\n')
    na = sum(ele.values())
    for i in range(na):
        f.write('%15.6f %15.6f %15.6f' % tuple(pos[i]))
        f.write('%15.6f %15.6f %15.6f\\n' % tuple(force[i]))
    f.write('-------------------------------------------------------------------\\n')
    f.write('energy  without entropy= %20.6f %20.6f\\n' % (ene, ene))
    enthalpy = ene + pstress * volume / 1602.17733
    f.write('enthalpy is  TOTEN    = %20.6f %20.6f\\n' % (enthalpy, enthalpy))

def check():
    to_be_opti = read('POSCAR')
    traj = TrajectoryWriter('traj.traj', 'w', to_be_opti)
    traj.write()
    traj.close()
    atoms_symbols_f = to_be_opti.get_chemical_symbols()
    element_f, ele_f = Get_Element_Num(atoms_symbols_f)
    atoms_vol_f = to_be_opti.get_volume()
    atoms_stress_f = np.array([0, 0, 0, 0, 0, 0])
    atoms_lat_f = to_be_opti.cell
    atoms_pos_f = to_be_opti.positions
    atoms_force_f = np.zeros((atoms_pos_f.shape[0], 3))
    atoms_ene_f =  610612509
    Write_Contcar(element_f, ele_f, atoms_lat_f, atoms_pos_f)
    Write_Outcar(element_f, ele_f, atoms_vol_f, atoms_lat_f, atoms_pos_f, atoms_ene_f, atoms_force_f, atoms_stress_f * -10.0, 0)

if __name__ == "__main__":
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd,'OUTCAR')):
        check()"""


def make_calypso_input(
    numb_of_species: int,
    name_of_atoms: List[str],
    atomic_number,
    numb_of_atoms: List[int],
    distance_of_ions,
    pop_size: int = 30,
    max_step: int = 5,
    system_name: str = "CALYPSO",
    numb_of_formula: List[int] = [1, 1],
    pressure: float = 0.001,  # KBar
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
    vsc: bool = False,
    ctrl_range: List[List[int]] = [[1, 10]],
    max_numb_atoms: int = 100,
    **kwargs,
):
    distance_of_ions = np.array(distance_of_ions)
    assert (
        numb_of_species
        == len(name_of_atoms)
        == len(atomic_number)
        == len(numb_of_atoms)
    ), f"{numb_of_species:}, {name_of_atoms:} {atomic_number:} {numb_of_atoms:}"
    assert distance_of_ions.shape == (
        numb_of_species,
        numb_of_species,
    ), f"{distance_of_ions.shape} {numb_of_species:}"

    necessary_keys = {
        "NumberOfSpecies": numb_of_species,
        "NameOfAtoms": " ".join(list(map(str, name_of_atoms))),
        "AtomicNumber": " ".join(list(map(str, atomic_number))),
        "NumberOfAtoms": " ".join(list(map(str, numb_of_atoms))),
        "PopSize": pop_size,
        "MaxStep": max_step,
        "DistanceOfIon": "\n".join(
            [" ".join(list(map(str, i))) for i in distance_of_ions]
        ),
        # @DistanceOfIon
        # @end
    }

    default_key_value = {
        "SystemName": system_name,
        "NumberOfFormula": " ".join(list(map(str, numb_of_formula))),
        "Volume": volume,
        "Ialgo": ialgo,
        "PsoRatio": pso_ratio,
        "ICode": icode,
        "NumberOfLbest": numb_of_lbest,
        "NumberOfLocalOptim": numb_of_local_optim,
        "Command": command,
        "MaxTime": max_time,
        "GenType": gen_type,
        "PickUp": pick_up,
        "PickStep": pick_step,
        "Parallel": "T" if parallel else "F",
        "Split": "T" if split else "F",
        "SpeSpaceGroup": " ".join(list(map(str, spec_space_group))),
        "VSC": "T" if vsc else "F",
        "MaxNumAtom": max_numb_atoms,
        "CtrlRange": "\n".join([" ".join(list(map(str, i))) for i in ctrl_range]),
        # @CtrlRange
        # @end
    }
    distance_of_ions_str = necessary_keys.pop("DistanceOfIon")
    vsc_ctrl_range = default_key_value.pop("CtrlRange")

    file_str = ""
    for key, value in necessary_keys.items():
        file_str += f"{key} = {str(value)}\n"
    for key, value in default_key_value.items():
        file_str += f"{key} = {str(value)}\n"
    file_str += "@DistanceOfIon\n"
    file_str += distance_of_ions_str + "\n"
    file_str += "@End\n"
    file_str += "@CtrlRange\n"
    file_str += vsc_ctrl_range + "\n"
    file_str += "@End\n"

    opt_step = kwargs.get("opt_step", 1000)
    run_opt_str = calypso_run_opt_str + calypso_run_opt_str_end % (
        fmax,
        pressure,
        opt_step,
    )
    check_opt_str = calypso_check_opt_str

    return file_str, run_opt_str, check_opt_str
