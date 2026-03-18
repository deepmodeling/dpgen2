import json
import logging
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

import numpy as np
from dargs import (
    Argument,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    HDF5Datasets,
    OPIOSign,
    TransientError,
)

from dpgen2.constants import (
    ase_conf_name,
    ase_input_name,
    ase_log_name,
    ase_model_devi_name,
    ase_traj_name,
    model_name_pattern,
    pytorch_model_name_pattern,
)
from dpgen2.utils import (
    set_directory,
)

from .run_lmp import (
    freeze_model,
)


class RunAse(OP):
    r"""Execute an ASE MD task driven by a DeePMD potential.

    A working directory named ``task_name`` is created.  All input
    files are copied or symlinked to that directory.  The ASE MD
    simulation is executed in-process (no subprocess).  The trajectory
    is written in LAMMPS dump format (``traj.dump``) so that
    :class:`~dpgen2.exploration.render.TrajRenderLammps` can be reused
    without modification.  Model deviations are written to
    ``model_devi.out`` in the same 7-column format used by
    :func:`~dpgen2.op.run_caly_model_devi.write_model_devi_out` (but
    with 7 columns: step + 3 virial + 3 force deviations, **no**
    energy column).

    Risk mitigations implemented
    ----------------------------
    * **Timestep units** — ``ase_input.json`` stores ``dt`` in
      picoseconds (LAMMPS convention).  Internally converted via
      ``dt_ps * 1000 * ase.units.fs`` before passing to ASE.
    * **Langevin API** — uses keyword ``temperature_K`` (ASE ≥ 3.22).
    * **NPT ensemble** — ``"npt"`` uses ``NPTBerendsen``; ``"npt-mtk"``
      uses ``ase.md.npt.NPT`` (Martyna-Tobias-Klein).
    * **model_devi.out columns** — 7 columns (step, max/min/avg_devi_v,
      max/min/avg_devi_f) matching :class:`TrajRenderLammps` expectations.
    * **Type map** — ``ase.io.read`` with ``species_order=type_map``
      ensures atom types are mapped correctly from the LAMMPS dump file.
    * **Velocity initialisation** — ``MaxwellBoltzmannDistribution``
      called when ``init_velocities=True``.
    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": BigParameter(dict),
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
                "models": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "log": Artifact(Path),
                "traj": Artifact(Path),
                "model_devi": Artifact(Path),
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

            - ``config`` : (``dict``) RunAse configuration.  See
              :meth:`ase_args` for available keys.
            - ``task_name`` : (``str``) The name of the task.
            - ``task_path`` : (``Artifact(Path)``) The path that
              contains all input files prepared by :class:`PrepAse`.
            - ``models`` : (``Artifact(List[Path])``) The frozen
              DeePMD models.  The **first** model drives the MD; all
              models are used for model-deviation calculation.

        Returns
        -------
        op : dict
            Output dict with components:

            - ``log`` : (``Artifact(Path)``) The ASE log file.
            - ``traj`` : (``Artifact(Path)``) The output trajectory in
              LAMMPS dump format.
            - ``model_devi`` : (``Artifact(Path)``) The model deviation
              file (7 columns).

        Raises
        ------
        TransientError
            On any failure during the ASE MD run.
        """
        config = ip["config"] if ip["config"] is not None else {}
        config = RunAse.normalize_config(config)
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        models = ip["models"]

        input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        model_files = [Path(ii).resolve() for ii in models]
        work_dir = Path(task_name)

        # Optionally freeze pytorch models
        frozen_model_files = []
        for idx, mm in enumerate(model_files):
            ext = mm.suffix
            if ext == ".pb":
                frozen_model_files.append(mm)
            elif ext in (".pt", ".pth"):
                mname = Path(pytorch_model_name_pattern % idx)
                freeze_model(mm, mname, config.get("model_frozen_head"))
                frozen_model_files.append(mname.resolve())
            else:
                raise RuntimeError(
                    f"Model file with extension '{ext}' is not supported"
                )

        with set_directory(work_dir):
            # Symlink input files
            for ii in input_files:
                Path(ii.name).symlink_to(ii)
            # Symlink / copy model files
            linked_models = []
            for idx, mm in enumerate(frozen_model_files):
                ext = mm.suffix
                if ext == ".pb":
                    mname = model_name_pattern % idx
                elif ext in (".pt", ".pth"):
                    mname = pytorch_model_name_pattern % idx
                else:
                    mname = mm.name
                Path(mname).symlink_to(mm)
                linked_models.append(mname)

            try:
                _run_ase_md(linked_models, config)
            except Exception as e:
                logging.error(f"ASE MD failed: {e}", exc_info=True)
                raise TransientError(f"ASE MD failed: {e}") from e

        return OPIO(
            {
                "log": work_dir / ase_log_name,
                "traj": work_dir / ase_traj_name,
                "model_devi": self.get_model_devi(work_dir / ase_model_devi_name),
            }
        )

    def get_model_devi(self, model_devi_file):
        return model_devi_file

    @staticmethod
    def ase_args():
        doc_head = "Select a head from multitask"
        return [
            Argument(
                "model_frozen_head",
                str,
                optional=True,
                default=None,
                doc=doc_head,
            ),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunAse.ase_args()
        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=False)
        return data


class RunAseHDF5(RunAse):
    """Variant of :class:`RunAse` that stores outputs as HDF5 datasets."""

    @classmethod
    def get_output_sign(cls):
        output_sign = super().get_output_sign()
        output_sign["traj"] = Artifact(HDF5Datasets)
        output_sign["model_devi"] = Artifact(HDF5Datasets)
        return output_sign

    def get_model_devi(self, model_devi_file):
        # For HDF5 mode the file path is returned as-is; dflow handles
        # the conversion when the artifact is collected.
        return model_devi_file


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_ase_md(model_names: List[str], config: dict) -> None:
    """Run an ASE MD simulation in the current working directory.

    Reads ``ase_input.json`` and ``conf.lmp``, sets up the ASE
    calculator and integrator, runs the MD, and writes ``traj.dump``
    and ``model_devi.out``.

    Parameters
    ----------
    model_names : List[str]
        Names of the (already symlinked) model files in the current
        working directory.  The first model drives the MD; all models
        are used for model-deviation calculation.
    config : dict
        Normalised RunAse configuration dict.
    """
    import ase.io  # type: ignore
    import ase.units  # type: ignore
    from deepmd.calculator import DP  # type: ignore
    from deepmd.infer import DeepPot  # type: ignore
    from deepmd.infer.model_devi import (  # type: ignore
        calc_model_devi_f,
        calc_model_devi_v,
    )

    # ------------------------------------------------------------------ #
    # 1. Read MD settings
    # ------------------------------------------------------------------ #
    with open(ase_input_name) as f:
        md_config = json.load(f)

    type_map: List[str] = md_config["type_map"]
    mass_map: List[float] = md_config["mass_map"]
    numb_models: int = md_config["numb_models"]
    ensemble: str = md_config["ensemble"]
    temperature: float = md_config["temperature"]
    pressure: Optional[float] = md_config.get("pressure")
    # dt stored in ps (LAMMPS convention) → convert to ASE time units
    dt_ps: float = md_config["dt"]
    dt_ase = dt_ps * 1000.0 * ase.units.fs  # Risk 3 mitigation
    nsteps: int = md_config["nsteps"]
    trj_freq: int = md_config["trj_freq"]
    tau_t_ps: float = md_config.get("tau_t", 0.1)
    tau_p_ps: float = md_config.get("tau_p", 0.5)
    init_velocities: bool = md_config.get("init_velocities", True)

    # Convert time constants from ps to ASE units
    tau_t_ase = tau_t_ps * 1000.0 * ase.units.fs
    tau_p_ase = tau_p_ps * 1000.0 * ase.units.fs

    # ------------------------------------------------------------------ #
    # 2. Read initial configuration
    # ------------------------------------------------------------------ #
    # Risk 6 / Risk 7 mitigation: read lammps-dump-text with species_order
    atoms = ase.io.read(
        ase_conf_name,
        format="lammps-dump-text",
        index=0,
        species_order=type_map,
    )

    # Set masses explicitly (ASE may not know them from the dump file)
    for atom in atoms:
        idx = type_map.index(atom.symbol)
        atom.mass = mass_map[idx]

    # ------------------------------------------------------------------ #
    # 3. Set up DeePMD calculator (first model drives MD)
    # ------------------------------------------------------------------ #
    # Risk 1 mitigation: use deepmd.calculator.DP for ASE integration
    calc = DP(model=model_names[0])
    atoms.calc = calc

    # ------------------------------------------------------------------ #
    # 4. Optionally initialise velocities (Risk 12 mitigation)
    # ------------------------------------------------------------------ #
    if init_velocities:
        from ase.md.velocitydistribution import (  # type: ignore
            MaxwellBoltzmannDistribution,
        )

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # ------------------------------------------------------------------ #
    # 5. Set up integrator
    # ------------------------------------------------------------------ #
    ensemble_lower = ensemble.lower()
    if ensemble_lower == "nvt":
        from ase.md.langevin import (  # type: ignore
            Langevin,
        )

        # Risk 4 mitigation: use temperature_K keyword (ASE ≥ 3.22)
        dyn = Langevin(
            atoms,
            timestep=dt_ase,
            temperature_K=temperature,
            friction=1.0 / tau_t_ase,
            logfile=ase_log_name,
        )
    elif ensemble_lower == "nve":
        from ase.md.verlet import (  # type: ignore
            VelocityVerlet,
        )

        dyn = VelocityVerlet(atoms, timestep=dt_ase, logfile=ase_log_name)
    elif ensemble_lower == "npt":
        # Risk 2 mitigation: NPTBerendsen for "npt"
        from ase.md.nptberendsen import (  # type: ignore
            NPTBerendsen,
        )

        if pressure is None:
            raise ValueError("pressure must be set for NPT ensemble")
        # NPTBerendsen expects pressure in ASE units (eV/Å³)
        # Input pressure is in bar → convert
        pressure_ase = pressure * ase.units.bar
        dyn = NPTBerendsen(
            atoms,
            timestep=dt_ase,
            temperature_K=temperature,
            pressure_au=pressure_ase,
            taut=tau_t_ase,
            taup=tau_p_ase,
            logfile=ase_log_name,
        )
    elif ensemble_lower == "npt-mtk":
        # Risk 2 mitigation: ase.md.npt.NPT for "npt-mtk"
        from ase.md.npt import (  # type: ignore
            NPT,
        )

        if pressure is None:
            raise ValueError("pressure must be set for npt-mtk ensemble")
        pressure_ase = pressure * ase.units.bar
        dyn = NPT(
            atoms,
            timestep=dt_ase,
            temperature_K=temperature,
            externalstress=pressure_ase,
            ttime=tau_t_ase,
            pfactor=tau_p_ase**2 * atoms.get_volume() * pressure_ase,
            logfile=ase_log_name,
        )
    else:
        raise ValueError(
            f"Unknown ensemble '{ensemble}'. "
            "Supported: 'nvt', 'nve', 'npt', 'npt-mtk'."
        )

    # ------------------------------------------------------------------ #
    # 6. Load all models for model-deviation calculation (Risk 8)
    # ------------------------------------------------------------------ #
    # Risk 1 mitigation: use DeepPot for multi-model evaluation
    dp_models = [DeepPot(m) for m in model_names]

    # ------------------------------------------------------------------ #
    # 7. Run MD with trajectory / model-devi callbacks
    # ------------------------------------------------------------------ #
    traj_frames: List[str] = []
    devi_rows: List[np.ndarray] = []
    step_counter = [0]  # mutable container for closure

    def _collect_frame():
        step = step_counter[0]
        step_counter[0] += 1
        if step % trj_freq != 0:
            return

        frame_idx = step // trj_freq

        # Write LAMMPS dump frame
        traj_frames.append(_atoms_to_lmpdump(atoms, frame_idx, type_map))

        # Compute model deviation across all models
        coord = atoms.get_positions().reshape(1, -1)
        cell = atoms.get_cell().array.reshape(1, -1)
        atype = [type_map.index(atom.symbol) for atom in atoms]

        forces_list = []
        virial_list = []
        for dp in dp_models:
            dp_type_map = dp.get_type_map()
            # Remap atype to this model's type ordering
            atype_dp = [dp_type_map.index(type_map[t]) for t in atype]
            _, forces, virial = dp.eval(coord, cell, atype_dp)
            forces_list.append(forces[0])  # shape (natoms, 3)
            # virial from DeepPot is (1, 9) in eV; normalise per atom
            virial_list.append(virial[0].reshape(9) / len(atype))

        forces_arr = np.array(forces_list)  # (nmodels, natoms, 3)
        virial_arr = np.array(virial_list)  # (nmodels, 9)

        # Risk 9 mitigation: 7-column format (no energy column)
        # calc_model_devi_v / _f expect shape (nmodels, nframes, ...)
        devi_v = calc_model_devi_v(virial_arr[np.newaxis].transpose(1, 0, 2))
        devi_f = calc_model_devi_f(forces_arr[np.newaxis].transpose(1, 0, 2, 3))
        # devi_v / devi_f each have shape (nframes, 3) → take frame 0
        row = np.array(
            [frame_idx]
            + list(devi_v[0])  # max_devi_v, min_devi_v, avg_devi_v
            + list(devi_f[0])  # max_devi_f, min_devi_f, avg_devi_f
        )
        devi_rows.append(row)

    dyn.attach(_collect_frame, interval=1)
    dyn.run(nsteps)

    # ------------------------------------------------------------------ #
    # 8. Write outputs
    # ------------------------------------------------------------------ #
    with open(ase_traj_name, "w") as f:
        f.write("".join(traj_frames))

    if devi_rows:
        devi_arr = np.vstack(devi_rows)
        _write_model_devi_out(devi_arr, ase_model_devi_name)
    else:
        # No frames collected — write empty file so downstream doesn't fail
        Path(ase_model_devi_name).touch()


def _atoms_to_lmpdump(atoms, frame_idx: int, type_map: List[str]) -> str:
    """Convert an ASE Atoms object to a LAMMPS dump string.

    Produces the same format as
    :func:`~dpgen2.op.run_caly_model_devi.atoms2lmpdump` so that
    :class:`~dpgen2.exploration.render.TrajRenderLammps` can parse it.
    """
    import ase as _ase  # type: ignore
    from ase.geometry import (  # type: ignore
        cellpar_to_cell,
    )

    cellpars = atoms.cell.cellpar()
    new_cell = cellpar_to_cell(cellpars)
    new_atoms = _ase.Atoms(
        numbers=atoms.numbers,
        cell=new_cell,
        scaled_positions=atoms.get_scaled_positions(),
    )

    xy = new_cell[1][0]
    xz = new_cell[2][0]
    yz = new_cell[2][1]
    lx, ly, lz = new_cell[0][0], new_cell[1][1], new_cell[2][2]
    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi, yhi, zhi = lx, ly, lz
    xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
    xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = yhi + max(0.0, yz)

    lines = [
        "ITEM: TIMESTEP",
        str(frame_idx),
        "ITEM: NUMBER OF ATOMS",
        str(len(new_atoms)),
        "ITEM: BOX BOUNDS xy xz yz pp pp pp",
        "%20.10f %20.10f %20.10f" % (xlo_bound, xhi_bound, xy),
        "%20.10f %20.10f %20.10f" % (ylo_bound, yhi_bound, xz),
        "%20.10f %20.10f %20.10f" % (zlo, zhi, yz),
        "ITEM: ATOMS id type x y z fx fy fz",
    ]
    for idx, atom in enumerate(new_atoms):
        type_id = type_map.index(atom.symbol) + 1
        lines.append(
            "%5d %5d%20.10f %20.10f %20.10f%20.10f %20.10f %20.10f"
            % (
                idx + 1,
                type_id,
                atom.position[0],
                atom.position[1],
                atom.position[2],
                0.0,
                0.0,
                0.0,
            )
        )
    return "\n".join(lines) + "\n"


def _write_model_devi_out(
    devi: np.ndarray,
    fname: str,
    header: str = "",
) -> None:
    """Write model-deviation data to *fname* in 7-column format.

    Columns: step, max_devi_v, min_devi_v, avg_devi_v,
             max_devi_f, min_devi_f, avg_devi_f.

    This is intentionally **different** from
    :func:`~dpgen2.op.run_caly_model_devi.write_model_devi_out` which
    writes 8 columns (adds an energy column).  The 7-column format is
    what :class:`~dpgen2.exploration.render.TrajRenderLammps` expects
    (same as LAMMPS ``model_devi.out``).
    """
    assert (
        devi.shape[1] == 7
    ), f"Expected 7 columns in model_devi array, got {devi.shape[1]}"
    col_header = "%10s%19s%19s%19s%19s%19s%19s" % (
        "step",
        "max_devi_v",
        "min_devi_v",
        "avg_devi_v",
        "max_devi_f",
        "min_devi_f",
        "avg_devi_f",
    )
    if header:
        col_header = header + "\n" + col_header
    np.savetxt(
        fname,
        devi,
        fmt=["%12d"] + ["%19.6e"] * 6,
        delimiter="",
        header=col_header,
        comments="#",
    )
