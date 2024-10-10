import json
import pickle
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    Parameter,
)

from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_input_file,
    calypso_run_opt_file,
    calypso_task_pattern,
    model_name_pattern,
)
from dpgen2.exploration.task import (
    BaseExplorationTaskGroup,
    ExplorationTaskGroup,
)
from dpgen2.utils import (
    set_directory,
)

vsc_keys = {
    "VSC": "F",
    "MaxNumAtom": 100,
    "CtrlRange": "",
    # @CtrlRange
    # @end
}

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
'''
structure optimization with DP model and ASE
PSTRESS and fmax should exist in input.dat
'''

def Get_Element_Num(elements):
    '''Using the Atoms.symples to Know Element&Num'''
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
    '''Write CONTCAR'''
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
    '''Write OUTCAR'''
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

def run_opt(fmax, stress):
    '''Using the ASE&DP to Optimize Configures'''

    calc = DP(model=sys.argv[1])    # init the model before iteration

    Opt_Step = 1000
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
        opt.run(fmax=fmax,steps=Opt_Step)
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
        run_opt(fmax=%.3f, stress=%.3f)
"""

calypso_check_opt_str = """#!/usr/bin/env python3

import os
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter

'''
check if structure optimization worked well
if not, this script will generate a fake outcar
'''

def Get_Element_Num(elements):
    '''Using the Atoms.symples to Know Element&Num'''
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
    '''Write CONTCAR'''
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
    '''Write OUTCAR'''
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

    try:
        trajs = Trajectory("traj.traj")
    except:
        pass

    numb_traj = len(trajs)
    assert numb_traj >= 1, "traj file is broken."
    origin = trajs[0]
    dis_mtx = origin.get_all_distances(mic=True)
    row, col = np.diag_indices_from(dis_mtx)
    dis_mtx[row, col] = np.nan
    is_reasonable = np.nanmin(dis_mtx) > 0.6

    if is_reasonable:
        if len(trajs) >= 20 :
           selected_traj = [trajs[iii] for iii in [4, 9, -10, -5, -1]]
        elif 5 <= len(trajs) < 20:
           selected_traj = [trajs[np.random.randint(4, len(trajs) - 1)] for _ in range(4)]
           selected_traj.append(trajs[-1])
        elif 3 <= len(trajs) < 5:
           selected_traj = [trajs[round((len(trajs) - 1) / 2)]]
           selected_traj.append(trajs[-1])
        elif len(trajs) == 2:
           selected_traj = [trajs[0], trajs[-1]]
        else:  # len(trajs) == 1
           selected_traj = [trajs[0]]

        for idx, traj in enumerate(selected_traj):
            write(f"{idx}.poscar", traj)

if __name__ == "__main__":
    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd,'OUTCAR')):
        check()"""


class PrepCalyInput(OP):
    r"""Prepare the working directories and input file for generating structures.

    A calypso input file will be generated according to the given parameters
    (defined by `ip["caly_inputs"]`). The artifact will be return
    (ip[`input_files`]). The name of directory is `ip["task_names"]`.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "caly_task_grp": BigParameter(
                    BaseExplorationTaskGroup
                ),  # calypso input params
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "ntasks": Parameter(int),
                "task_names": BigParameter(List[str]),  # task dir names
                "input_dat_files": Artifact(List[Path]),  # `input.dat`s
                "caly_run_opt_files": Artifact(List[Path]),
                "caly_check_opt_files": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
            - `caly_task_grp` : (`BigParameter()`) Definitions for CALYPSO input file.

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of CALYPSO tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `input_dat_files`: (`Artifact(List[Path])`) The parepared working paths of the task containing input files (`input.dat` and `calypso_run_opt.py`) needed to generate structures by CALYPSO and make structure optimization with DP model.
            - `caly_run_opt_files`: (`Artifact(List[Path])`)
            - `caly_check_opt_files`: (`Artifact(List[Path])`)
        """

        cc = 0
        task_paths = []
        input_dat_files = []
        caly_run_opt_files = []
        caly_check_opt_files = []
        caly_task_grp = ip["caly_task_grp"]
        for tt in caly_task_grp:
            ff = tt.files()
            tname = _mk_task_from_files(cc, ff)
            task_paths.append(tname)
            input_dat_files.append(tname / calypso_input_file)
            caly_run_opt_files.append(tname / calypso_run_opt_file)
            caly_check_opt_files.append(tname / calypso_check_opt_file)
            cc += 1
        task_names = [str(ii) for ii in task_paths]

        return OPIO(
            {
                "ntasks": len(task_names),
                "task_names": task_names,
                "input_dat_files": input_dat_files,
                "caly_run_opt_files": caly_run_opt_files,
                "caly_check_opt_files": caly_check_opt_files,
            }
        )


def _mk_task_from_files(cc, ff):
    tname = Path(calypso_task_pattern % cc)
    tname.mkdir(exist_ok=True, parents=True)
    for file_name, file_content in ff.items():
        (tname / file_name).write_text(file_content)
    return tname
