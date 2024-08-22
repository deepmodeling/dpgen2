import os
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
    BigParameter,
    OPIOSign,
)

from dpgen2.exploration.task import (
    DiffCSPTaskGroup,
)

from .run_caly_model_devi import (
    atoms2lmpdump,
)


class RunRelax(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "diffcsp_task_grp": BigParameter(DiffCSPTaskGroup),
                "task_path": Artifact(Path),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "trajs": Artifact(List[Path]),
                "model_devis": Artifact(List[Path]),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        import pickle
        import traceback

        import ase  # type: ignore
        import numpy as np
        from ase.calculators.singlepoint import (  # type: ignore
            SinglePointCalculator,
        )
        from deepmd.infer import (  # type: ignore
            DeepPot,
        )
        from deepmd.infer.model_devi import (  # type: ignore
            calc_model_devi_f,
            calc_model_devi_v,
        )
        from lam_optimize.main import (  # type: ignore
            relax_run,
        )
        from lam_optimize.relaxer import (  # type: ignore
            Relaxer,
        )

        task_group = ip["diffcsp_task_grp"]
        task = next(iter(task_group))  # Only support single task
        models = ip["models"]
        relaxer = Relaxer(models[0])
        type_map = relaxer.calculator.dp.get_type_map()
        fmax = task.fmax
        steps = task.steps
        timeout = task.timeout
        os.makedirs("relax_trajs", exist_ok=True)
        relax_run(
            ip["task_path"],
            relaxer,
            fmax=fmax,
            steps=steps,
            traj_file="relax_trajs",
            timeout=timeout,
            check_convergence=False,
            check_duplicate=False,
        )

        trajs = []
        model_devis = []
        graphs = [None] + [DeepPot(model) for model in models[1:]]
        trj_freq = task.trj_freq
        for fname in os.listdir("relax_trajs"):
            with open(os.path.join("relax_trajs", fname), "rb") as f:
                try:
                    data = pickle.load(f)
                except Exception:
                    traceback.print_exc()
                    continue
            nsteps = len(data["energy"])
            if nsteps <= 0:
                continue

            step_list = []
            forces_list = [[] for _ in range(len(models))]
            virial_list = [[] for _ in range(len(models))]
            dump_str = ""
            coords_list = []
            cell_list = []
            for i in range(0, nsteps, trj_freq):
                atoms = ase.Atoms(
                    numbers=data["atomic_number"],
                    positions=data["atom_positions"][i],
                    pbc=True,
                    cell=data["cell"][i],
                )
                calc = SinglePointCalculator(
                    atoms,
                    energy=data["energy"][i],
                    forces=data["forces"][i],
                    stress=data["stresses"][i],
                )
                atoms.calc = calc
                dump_str += atoms2lmpdump(atoms, i, type_map)
                coords_list.append(data["atom_positions"][i])
                cell_list.append(data["cell"][i])
                step_list.append(i)
                # Use results of model 0 directly
                forces_list[0].append(data["forces"][i])
                virial_list[0].append(
                    -atoms.get_volume()
                    * atoms.get_stress(False).reshape(9)
                    / len(atoms)
                )
            for j in range(1, len(models)):
                dp = graphs[j]
                atype = [
                    dp.get_type_map().index(ase.Atom(i).symbol)  # type: ignore
                    for i in data["atomic_number"]
                ]
                _, forces, virial = dp.eval(  # type: ignore
                    np.array(coords_list), np.array(cell_list), atype
                )
                forces_list[j] = forces
                virial_list[j] = virial / len(atype)
            traj_file = ip["task_path"] / ("traj.%s.dump" % fname)
            traj_file.write_text(dump_str)
            trajs.append(traj_file)
            devi = [np.array(step_list)]
            devi += list(calc_model_devi_v(np.array(virial_list)))
            devi += list(calc_model_devi_f(np.array(forces_list)))
            devi = np.vstack(devi).T
            header = "%10s%19s%19s%19s%19s%19s%19s" % (
                "step",
                "max_devi_v",
                "min_devi_v",
                "avg_devi_v",
                "max_devi_f",
                "min_devi_f",
                "avg_devi_f",
            )
            model_devi_file = ip["task_path"] / ("model_devi.%s.out" % fname)
            np.savetxt(
                model_devi_file,
                devi,
                fmt=["%12d"] + ["%19.6e"] * 6,
                delimiter="",
                header=header,
            )
            model_devis.append(model_devi_file)
        return OPIO(
            {
                "trajs": trajs,
                "model_devis": model_devis,
            }
        )
