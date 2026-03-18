import itertools
import json
from typing import (
    List,
    Optional,
)

from dpgen2.constants import (
    ase_conf_name,
    ase_input_name,
    model_name_pattern,
)

from .conf_sampling_task_group import (
    ConfSamplingTaskGroup,
)
from .task import (
    ExplorationTask,
)


class AseTaskGroup(ConfSamplingTaskGroup):
    """A group of ASE MD exploration tasks.

    Each task runs an ASE molecular dynamics simulation driven by a
    DeePMD potential.  The initial configuration is stored as a LAMMPS
    dump file (``conf.lmp``) and the MD settings are stored as a JSON
    file (``ase_input.json``).  The output trajectory is written in
    LAMMPS dump format so that :class:`TrajRenderLammps` can be reused
    without modification.

    Parameters
    ----------
    None — call :meth:`set_md` and :meth:`set_conf` before
    :meth:`make_task`.

    Examples
    --------
    >>> tgroup = AseTaskGroup()
    >>> tgroup.set_md(
    ...     numb_models=4,
    ...     mass_map=[27.0, 24.0],
    ...     type_map=["Al", "Mg"],
    ...     temps=[300, 600],
    ...     press=[1.0],
    ...     ens="nvt",
    ...     dt=0.001,
    ...     nsteps=1000,
    ...     trj_freq=10,
    ... )
    >>> tgroup.set_conf(conf_list, n_sample=3)
    >>> tgroup.make_task()
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.md_set = False

    def set_md(
        self,
        numb_models: int,
        mass_map: List[float],
        type_map: List[str],
        temps: List[float],
        press: Optional[List[float]] = None,
        ens: str = "nvt",
        dt: float = 0.001,
        nsteps: int = 1000,
        trj_freq: int = 10,
        tau_t: float = 0.1,
        tau_p: float = 0.5,
        init_velocities: bool = True,
    ):
        """Set MD parameters.

        Parameters
        ----------
        numb_models : int
            Number of DeePMD models used for model deviation.
        mass_map : List[float]
            Atomic masses in amu, ordered by ``type_map``.
        type_map : List[str]
            Element symbols ordered by type index (e.g. ``["Al", "Mg"]``).
        temps : List[float]
            List of temperatures in Kelvin.
        press : List[float], optional
            List of pressures in bar.  Required for NPT ensembles.
            Pass ``None`` or omit for NVT/NVE.
        ens : str
            Ensemble type.  Supported values:

            * ``"nvt"``     — Langevin thermostat (NVT)
            * ``"nve"``     — Velocity Verlet (NVE, microcanonical)
            * ``"npt"``     — Berendsen NPT (``NPTBerendsen``)
            * ``"npt-mtk"`` — Martyna-Tobias-Klein NPT (``ase.md.npt.NPT``)
        dt : float
            Timestep in **picoseconds** (consistent with LAMMPS convention).
            Internally converted to ASE units via ``dt * 1000 * ase.units.fs``.
        nsteps : int
            Total number of MD steps.
        trj_freq : int
            Frequency (in steps) at which frames are written to the
            trajectory and model deviations are evaluated.
        tau_t : float
            Thermostat time constant in picoseconds.
        tau_p : float
            Barostat time constant in picoseconds.  Only used for NPT.
        init_velocities : bool
            If ``True`` (default), initialise atomic velocities from a
            Maxwell-Boltzmann distribution at the target temperature
            before starting MD.
        """
        self.numb_models = numb_models
        self.graphs = [model_name_pattern % ii for ii in range(numb_models)]
        self.mass_map = mass_map
        self.type_map = type_map
        self.temps = temps
        self.press = press if press is not None else [None]
        self.ens = ens
        self.dt = dt
        self.nsteps = nsteps
        self.trj_freq = trj_freq
        self.tau_t = tau_t
        self.tau_p = tau_p
        self.init_velocities = init_velocities
        self.md_set = True

    def make_task(
        self,
    ) -> "AseTaskGroup":
        """Make the ASE MD task group.

        Returns
        -------
        AseTaskGroup
            The task group with one :class:`ExplorationTask` per
            ``(conf, temperature, pressure)`` combination.  The number
            of tasks equals ``n_sample * len(temps) * len(press)``.

        Raises
        ------
        RuntimeError
            If :meth:`set_conf` or :meth:`set_md` has not been called.
        """
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.md_set:
            raise RuntimeError("MD settings are not set")
        # clear all existing tasks
        self.clear()
        confs = self._sample_confs()
        for cc, tt, pp in itertools.product(confs, self.temps, self.press):
            self.add_task(self._make_ase_task(cc, tt, pp))
        return self

    def _make_ase_task(
        self,
        conf: str,
        tt: float,
        pp: Optional[float],
    ) -> ExplorationTask:
        """Build one :class:`ExplorationTask` for a single (conf, T, P) point.

        Parameters
        ----------
        conf : str
            Content of the LAMMPS dump configuration file.
        tt : float
            Temperature in Kelvin.
        pp : float or None
            Pressure in bar.  ``None`` for NVT/NVE ensembles.

        Returns
        -------
        ExplorationTask
        """
        ase_input = {
            "type_map": self.type_map,
            "mass_map": self.mass_map,
            "numb_models": self.numb_models,
            "ensemble": self.ens,
            "temperature": tt,
            "pressure": pp,
            "dt": self.dt,
            "nsteps": self.nsteps,
            "trj_freq": self.trj_freq,
            "tau_t": self.tau_t,
            "tau_p": self.tau_p,
            "init_velocities": self.init_velocities,
        }
        task = ExplorationTask()
        task.add_file(ase_conf_name, conf)
        task.add_file(ase_input_name, json.dumps(ase_input, indent=2))
        return task
