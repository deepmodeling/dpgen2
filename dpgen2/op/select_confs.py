from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    BigParameter,
)
import os, json
from typing import Tuple, List, Set
from pathlib import Path
from dpgen2.exploration.selector import ConfSelector
from dpgen2.exploration.report import ExplorationReport

class SelectConfs(OP):
    """Select configurations from exploration trajectories for labeling.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "conf_selector": ConfSelector,
            "traj_fmt": str,
            "type_map": List[str],

            "trajs": Artifact(List[Path]),
            "model_devis": Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "report" : BigParameter(ExplorationReport),

            "confs" : Artifact(List[Path]),
        })

    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:
        
            - `conf_selector`: (`ConfSelector`) Configuration selector.
            - `traj_fmt`: (`str`) The format of trajectory.
            - `type_map`: (`List[str]`) The type map.
            - `trajs`: (`Artifact(List[Path])`) The trajectories generated in the exploration.
            - `model_devis`: (`Artifact(List[Path])`) The file storing the model deviation of the trajectory. The order of model deviation storage is consistent with that of the trajectories. The order of frames of one model deviation storage is also consistent with tat of the corresponding trajectory.

        Returns
        -------
            Output dict with components:
        
            - `report`: (`ExplorationReport`) The report on the exploration.
            - `conf`: (`Artifact(List[Path])`) The selected configurations.
        
        """

        conf_selector = ip['conf_selector']
        traj_fmt = ip['traj_fmt']
        type_map = ip['type_map']

        trajs = ip['trajs']
        model_devis = ip['model_devis']

        confs, report = conf_selector.select(
            trajs, 
            model_devis, 
            traj_fmt = traj_fmt,
            type_map = type_map,
        )

        return OPIO({
            "report" : report,
            "confs" : confs,
        })
