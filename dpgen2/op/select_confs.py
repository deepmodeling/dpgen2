import json
import os
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
    Tuple,
)

from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    OPIOSign,
)

from dpgen2.exploration.report import (
    ExplorationReport,
)
from dpgen2.exploration.selector import (
    ConfSelector,
)


class SelectConfs(OP):
    """Select configurations from exploration trajectories for labeling."""

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "conf_selector": ConfSelector,
                "type_map": List[str],
                "trajs": Artifact(List[Path]),
                "model_devis": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "report": BigParameter(ExplorationReport),
                "confs": Artifact(List[Path]),
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

            - `conf_selector`: (`ConfSelector`) Configuration selector.
            - `type_map`: (`List[str]`) The type map.
            - `trajs`: (`Artifact(List[Path])`) The trajectories generated in the exploration.
            - `model_devis`: (`Artifact(List[Path])`) The file storing the model deviation of the trajectory. The order of model deviation storage is consistent with that of the trajectories. The order of frames of one model deviation storage is also consistent with tat of the corresponding trajectory.

        Returns
        -------
        Any
            Output dict with components:
            - `report`: (`ExplorationReport`) The report on the exploration.
            - `conf`: (`Artifact(List[Path])`) The selected configurations.

        """

        conf_selector = ip["conf_selector"]
        type_map = ip["type_map"]

        trajs = ip["trajs"]
        model_devis = ip["model_devis"]
        trajs, model_devis = SelectConfs.validate_trajs(trajs, model_devis)

        confs, report = conf_selector.select(
            trajs,
            model_devis,
            type_map=type_map,
        )

        return OPIO(
            {
                "report": report,
                "confs": confs,
            }
        )

    @staticmethod
    def validate_trajs(
        trajs,
        model_devis,
    ):
        ntrajs = len(trajs)
        if ntrajs != len(model_devis):
            raise FatalError(
                "length of trajs list is not equal to the " "model_devis list"
            )
        rett = []
        retm = []
        for tt, mm in zip(trajs, model_devis):
            if (tt is None and mm is not None) or (tt is not None and mm is None):
                raise FatalError("trajs frame is {tt} while model_devis frame is {mm}")
            elif tt is not None and mm is not None:
                rett.append(tt)
                retm.append(mm)
        return rett, retm
