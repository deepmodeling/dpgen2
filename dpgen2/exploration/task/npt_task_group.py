import itertools, random
from typing import (
    List,
)
from . import (
    ExplorationTask,
    ExplorationTaskGroup,
)
from .lmp import make_lmp_input
from dpgen2.constants import (
    lmp_conf_name, 
    lmp_input_name,
    model_name_pattern,
)

class NPTTaskGroup(ExplorationTaskGroup):
    def __init__(
            self, 
    ):
        super().__init__()
        self.conf_set = False
        self.md_set = False

    def set_conf(
            self,
            conf_list : List[str],
            n_sample : int = None,
            random_sample : bool = False,
    ):
        """
        Set the configurations of exploration

        Parameters
        ----------
        conf_list       str
                        A list of file contents
        n_sample        int
                        Number of samples drawn from the conf list each time 
                        `make_task` is called. If set to `None`, 
                        `n_sample` is set to length of the conf_list.
        random_sample   bool
                        If true the confs are randomly sampled, otherwise are
                        consecutively sampled from the conf_list
        """
        self.conf_list = conf_list
        if n_sample is None:
            self.n_sample = len(self.conf_list)
        else:
            self.n_sample = n_sample
        self.random_sample = random_sample
        self.conf_queue = []
        self.conf_set = True

    def set_md(
            self,
            numb_models,
            mass_map,
            temps : List[float],
            press : List[float] = None,
            ens : str = 'npt',
            dt : float = 0.001,
            nsteps : int = 1000,
            trj_freq : int = 10,
            tau_t : float = 0.1,
            tau_p : float = 0.5,
            pka_e : float = None,
            neidelay : int = None,
            no_pbc : bool = False,
            use_clusters : bool = False,
            relative_f_epsilon : float = None,
            relative_v_epsilon : float = None,
            ele_temp_f : float = None,
            ele_temp_a : float = None,
    ):
        """
        Set MD parameters
        """
        self.graphs = [model_name_pattern % ii for ii in range(numb_models)]
        self.mass_map = mass_map
        self.temps = temps
        self.press = press if press is not None else [None]
        self.ens = ens
        self.dt = dt
        self.nsteps = nsteps
        self.trj_freq = trj_freq
        self.tau_t = tau_t
        self.tau_p = tau_p
        self.pka_e = pka_e
        self.neidelay = neidelay
        self.no_pbc = no_pbc
        self.use_clusters = use_clusters
        self.relative_f_epsilon = relative_f_epsilon
        self.relative_v_epsilon = relative_v_epsilon
        self.ele_temp_f = ele_temp_f
        self.ele_temp_a = ele_temp_a
        self.md_set = True

    def make_task(
            self,
    )->ExplorationTaskGroup:
        """
        Make the LAMMPS task group.
        
        Returns
        -------
        task_grp: ExplorationTaskGroup
            The returned lammps task group. The number of tasks is nconf*nT*nP.
            nconf is set by `n_sample` parameter of `set_conf`. 
            nT and nP are lengths of the `temps` and `press` parameters of `set_md`.

        """
        if not self.conf_set:
            raise RuntimeError('confs are not set')
        if not self.md_set:
            raise RuntimeError('MD settings are not set')
        # clear all existing tasks
        self.clear()
        confs = self._sample_confs()
        for cc,tt,pp in itertools.product(confs, self.temps, self.press):
            self.add_task(self._make_lmp_task(cc, tt, pp))
        return self

    def _sample_confs(
            self,
    ):
        confs = []
        for ii in range(self.n_sample):
            if len(self.conf_queue) == 0:
                add_list = self.conf_list.copy()
                if self.random_sample:
                    random.shuffle(add_list)
                self.conf_queue += add_list
            confs.append(self.conf_queue.pop(0))
        return confs
                
    def _make_lmp_task(
            self,
            conf : str,
            tt : float,
            pp : float,
    ) -> ExplorationTask:
        task = ExplorationTask()
        task\
            .add_file(
                lmp_conf_name, 
                conf,
            )\
            .add_file(
                lmp_input_name,
                make_lmp_input(
                    lmp_conf_name,
                    self.ens,
                    self.graphs,
                    self.nsteps,
                    self.dt,
                    self.neidelay,
                    self.trj_freq,
                    self.mass_map,
                    tt,
                    self.tau_t,
                    pp,
                    self.tau_p,
                    self.use_clusters,
                    self.relative_f_epsilon,
                    self.relative_v_epsilon,
                    self.pka_e,
                    self.ele_temp_f,
                    self.ele_temp_a,
                    self.no_pbc,
                    trj_seperate_files = False,
                )
            )
        return task

