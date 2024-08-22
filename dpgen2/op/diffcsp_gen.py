import os
import subprocess
from pathlib import (
    Path,
)
from typing import (
    List,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
)


def convert_pt_to_cif(input_file, output_dir):
    import numpy as np
    import torch  # type: ignore
    from pymatgen.core.lattice import (  # type: ignore
        Lattice,
    )
    from pymatgen.core.structure import (  # type: ignore
        Structure,
    )

    data = torch.load(input_file, map_location=torch.device("cpu"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lengths = data["lengths"]
    angles = data["angles"]
    num_atoms = data["num_atoms"]
    frac_coors = data["frac_coords"]
    atom_types = data["atom_types"]

    lengths_list = lengths.numpy()
    angles_list = angles.numpy()
    num_atoms_list = num_atoms
    frac_coors_list = frac_coors.numpy()
    atom_types_list = atom_types
    if len(atom_types_list.shape) > 1:
        atom_types_list = np.argmax(atom_types_list, axis=-1) + 1

    now_atom = 0
    for i in range(len(num_atoms_list)):
        length = lengths_list[i]
        angle = angles_list[i]
        atom_num = num_atoms_list[i]

        atom_type = atom_types_list[now_atom : now_atom + atom_num]
        frac_coord = frac_coors_list[now_atom : now_atom + atom_num]
        lattice = Lattice.from_parameters(*(length.tolist() + angle.tolist()))
        structure = Structure(
            lattice, atom_type, frac_coord, coords_are_cartesian=False
        )

        filename = "%s.cif" % i
        file_path = os.path.join(output_dir, filename)
        structure.to(filename=file_path)
        now_atom += atom_num


class DiffCSPGen(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": dict,
                "task_id": str,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "cifs": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        cmd = ip["config"]["gen_command"]
        args = cmd.split()
        try:
            i = args.index("--model_path")
        except ValueError:
            raise RuntimeError("Path of DiffCSP model not provided.")
        model_path = args[i + 1]
        subprocess.run(cmd, shell=True, check=True)
        result_file = os.path.join(model_path, "eval_gen.pt")
        task_dir = "diffcsp.%s" % ip["task_id"]
        convert_pt_to_cif(result_file, task_dir)
        return OPIO(
            {
                "cifs": list(Path(task_dir).glob("*.cif")),
            }
        )
