import dpdata
import numpy as np

def fake_system(
        nframes,
        natoms,
        atom_name = 'foo',
):
    ss = dpdata.LabeledSystem()
    ss.data['atom_names'] = [atom_name]
    ss.data['atom_numbs'] = [natoms]
    ss.data['atom_types'] = [0 for ii in range(natoms)]
    #ss.data['cells'] = np.zeros([nframes, 3, 3])
    ss.data['cells'] = np.tile(np.eye(3), [nframes, 1, 1])
    ss.data['coords'] = np.zeros([nframes, natoms, 3])
    ss.data['forces'] = np.zeros([nframes, natoms, 3])
    ss.data['energies'] = np.zeros([nframes])
    return ss

def fake_multi_sys(
        nframs, 
        natoms, 
        natom_name = 'foo',
):
    nsys = len(nframs)
    ms = dpdata.MultiSystems()
    for ii in range(nsys):
        ss = fake_system(nframs[ii], natoms[ii], natom_name)
        ms.append(ss)
    return ms


