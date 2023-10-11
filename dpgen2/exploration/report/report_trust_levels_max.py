import random
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
from dargs import (
    Argument,
)
from dflow.python import (
    FatalError,
)

from ..deviation import (
    DeviManager,
)
from . import (
    ExplorationReport,
)
from .report_trust_levels_base import (
    ExplorationReportTrustLevels,
)


class ExplorationReportTrustLevelsMax(ExplorationReportTrustLevels):
    def converged(
        self,
        reports: Optional[List[ExplorationReport]] = None,
    ) -> bool:
        r"""Check if the exploration is converged.

        Parameters
        ----------
        reports
            Historical reports

        Returns
        -------
        converged  bool
            If the exploration is converged.
        """
        return self.accurate_ratio() >= self.conv_accuracy

    def get_candidate_ids(
        self,
        max_nframes: Optional[int] = None,
    ) -> List[List[int]]:
        ntraj = len(self.traj_nframes)
        id_cand = self._get_candidates(max_nframes)
        id_cand_list = [[] for ii in range(ntraj)]
        for ii in id_cand:
            id_cand_list[ii[0]].append(ii[1])
        return id_cand_list

    def _get_candidates(
        self,
        max_nframes: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get candidates. If number of candidates is larger than `max_nframes`,
        then select `max_nframes` frames with the largest `max_devi_f` from
        the candidates.

        Parameters
        ----------
        max_nframes
            The maximal number of frames of candidates.

        Returns
        -------
        cand_frames   List[Tuple[int,int]]
            Candidate frames. A list of tuples: [(traj_idx, frame_idx), ...]
        """
        self.traj_cand_picked = []
        for tidx, tt in enumerate(self.traj_cand):
            for ff in tt:
                self.traj_cand_picked.append((tidx, ff))
        if max_nframes is not None and max_nframes < len(self.traj_cand_picked):
            # select by maximum
            max_devi_f = self.model_devi.get(DeviManager.MAX_DEVI_F)  # type: ignore
            ret = sorted(
                self.traj_cand_picked,
                key=lambda x: max_devi_f[x[0]][x[1]],
                reverse=True,
            )
            ret = ret[:max_nframes]
        else:
            ret = self.traj_cand_picked
        return ret

    @staticmethod
    def doc() -> str:
        def make_class_doc_link(key):
            from dpgen2.entrypoint.args import (
                make_link,
            )

            return make_link(
                key, f"explore[lmp]/convergence[fixed-levels-max-select]/{key}"
            )

        level_f_hi_link = make_class_doc_link("level_f_hi")
        level_v_hi_link = make_class_doc_link("level_v_hi")
        level_f_lo_link = make_class_doc_link("level_f_lo")
        level_v_lo_link = make_class_doc_link("level_v_lo")
        conv_accuracy_link = make_class_doc_link("conv_accuracy")
        return f"The configurations with force model deviation between {level_f_lo_link}, {level_f_hi_link} or virial model deviation between {level_v_lo_link} and {level_v_hi_link} are treated as candidates (The virial model deviation check is optional). The configurations with maximal model deviation in the candidates are sent for FP calculations. If the ratio of accurate (below {level_f_lo_link} and {level_v_lo_link}) is higher then {conv_accuracy_link}, the stage is treated as converged."
