import random
from abc import (
    abstractmethod,
)
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
    DeviManagerSpin,
)
from . import (
    ExplorationReport,
)


class ExplorationReportTrustLevelsSpin(ExplorationReport):
    def __init__(
        self,
        level_af_lo,
        level_af_hi,
        level_mf_lo,
        level_mf_hi,
        conv_accuracy = 0.9,
    ):
        self.level_af_lo = level_af_lo
        self.level_af_hi = level_af_hi
        self.level_mf_lo = level_mf_lo
        self.level_mf_hi = level_mf_hi
        self.conv_accuracy = conv_accuracy
        self.clear()
        self.model_devi = None

        print_tuple = (
            "stage",
            "id_stg.",
            "iter.",
            "accu.",
            "cand.",
            "fail.",
            "lvl_af_lo",
            "lvl_af_hi",
            "lvl_mf_lo",
            "lvl_mf_hi",
            "cvged",
        )
        spaces = [8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 8]
        self.fmt_str = " ".join([f"%{ii}s" for ii in spaces])
        self.fmt_flt = "%.4f"
        self.header_str = "#" + self.fmt_str % print_tuple

    @staticmethod
    def args() -> List[Argument]:
        doc_level_af_lo = "The lower trust level of atomic force model deviation"
        doc_level_af_hi = "The higher trust level of atomic force model deviation"
        doc_level_mf_lo = "The lower trust level of magnetic force model deviation"
        doc_level_mf_hi = "The higher trust level of magnetic force model deviation"
        doc_conv_accuracy = "If the ratio of accurate frames is larger than this value, the stage is converged"
        return [
            Argument("level_af_lo", float, optional=False, doc=doc_level_af_lo),
            Argument("level_af_hi", float, optional=False, doc=doc_level_af_hi),
            Argument("level_mf_lo", float, optional=False, doc=doc_level_mf_lo),
            Argument("level_mf_hi", float, optional=False, doc=doc_level_mf_hi),
            Argument("conv_accuracy", float, optional=True, default=0.9, doc=doc_conv_accuracy),
        ]

    def clear(self):
        self.traj_nframes = []
        self.traj_cand = []
        self.traj_accu = []
        self.traj_fail = []
        self.traj_cand_picked = []
        self.model_devi = None

    def record(
        self,
        model_devi: DeviManagerSpin,
    ):
        ntraj = model_devi.ntraj
        md_af = model_devi.get(DeviManagerSpin.MAX_DEVI_AF)
        md_mf = model_devi.get(DeviManagerSpin.MAX_DEVI_MF)

        for ii in range(ntraj):
            id_af_cand, id_af_accu, id_af_fail = self._get_indexes(
                md_af[ii], self.level_af_lo, self.level_af_hi
            )
            id_mf_cand, id_mf_accu, id_mf_fail = self._get_indexes(
                md_mf[ii], self.level_mf_lo, self.level_mf_hi
            )
            nframes, set_accu, set_cand, set_fail = self._record_one_traj(
                id_af_accu,
                id_af_cand,
                id_af_fail,
                id_mf_accu,
                id_mf_cand,
                id_mf_fail,
            )
            # record
            self.traj_nframes.append(nframes)
            self.traj_cand.append(set_cand)
            self.traj_accu.append(set_accu)
            self.traj_fail.append(set_fail)
        assert len(self.traj_nframes) == ntraj
        assert len(self.traj_cand) == ntraj
        assert len(self.traj_accu) == ntraj
        assert len(self.traj_fail) == ntraj
        self.model_devi = model_devi

    def _get_indexes(
        self,
        md,
        level_lo,
        level_hi,
    ):
        if (md is not None) and (level_hi is not None) and (level_lo is not None):
            id_cand = np.where(np.logical_and(md >= level_lo, md < level_hi))[0]
            id_accu = np.where(md < level_lo)[0]
            id_fail = np.where(md >= level_hi)[0]
        else:
            id_cand = id_accu = id_fail = None
        return id_cand, id_accu, id_fail

    def _record_one_traj(
        self,
        id_af_accu,
        id_af_cand,
        id_af_fail,
        id_mf_accu,
        id_mf_cand,
        id_mf_fail,
    ):
        """
        Record one trajctory. inputs are the indexes of candidate, accurate and failed frames.

        """
        # check consistency
        nframes = np.size(np.concatenate((id_af_cand, id_af_accu, id_af_fail)))
        if nframes != np.size(np.concatenate((id_mf_cand, id_mf_accu, id_mf_fail))):
            raise FatalError("the number of frames by atomic force and magnetic force is not consistent")
        # nframes to sets
        set_af_accu = set(id_af_accu)
        set_af_cand = set(id_af_cand)
        set_af_fail = set(id_af_fail)
        set_mf_accu = set(id_mf_accu)
        set_mf_cand = set(id_mf_cand)
        set_mf_fail = set(id_mf_fail)
        # accu, cand, fail
        set_accu = set_af_accu & set_mf_accu
        set_cand = (
            (set_af_cand & set_mf_accu)
            | (set_af_cand & set_mf_cand)
            | (set_af_accu & set_mf_cand)
        )
        set_fail = set_af_fail | set_mf_fail
        # check size
        assert nframes == len(set_accu | set_cand | set_fail)
        assert 0 == len(set_accu & set_cand)
        assert 0 == len(set_accu & set_fail)
        assert 0 == len(set_cand & set_fail)
        return nframes, set_accu, set_cand, set_fail

    def converged(
        self,
        reports: Optional[List[ExplorationReport]] = None,
    ) -> bool:
        return self.accurate_ratio() >= self.conv_accuracy

    def failed_ratio(
        self,
        tag = None,
    ):
        traj_nf = [len(ii) for ii in self.traj_fail]
        return float(sum(traj_nf)) / float(sum(self.traj_nframes))

    def accurate_ratio(
        self,
        tag = None,
    ):
        traj_nf = [len(ii) for ii in self.traj_accu]
        return float(sum(traj_nf)) / float(sum(self.traj_nframes))

    def candidate_ratio(
        self,
        tag = None,
    ):
        traj_nf = [len(ii) for ii in self.traj_cand]
        return float(sum(traj_nf)) / float(sum(self.traj_nframes))

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
        then select `max_nframes` frames with the largest `max_devi_mf` 
        from the candidates.

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
            # select by max magnetic force
            max_devi_af = self.model_devi.get(DeviManagerSpin.MAX_DEVI_AF)
            max_devi_mf = self.model_devi.get(DeviManagerSpin.MAX_DEVI_MF)
            ret = sorted(
                self.traj_cand_picked,
                key=lambda x: max_devi_mf[x[0]][x[1]],
                reverse=True,
            )
            ret = ret[:max_nframes]
        else:
            ret = self.traj_cand_picked
        return ret

    def print_header(self) -> str:
        r"""Print the header of report"""
        return self.header_str

    def print(
        self,
        stage_idx: int,
        idx_in_stage: int,
        iter_idx: int,
    ) -> str:
        r"""Print the report"""
        fmt_str = self.fmt_str
        fmt_flt = self.fmt_flt
        print_tuple = (
            str(stage_idx),
            str(idx_in_stage),
            str(iter_idx),
            fmt_flt % (self.accurate_ratio()),
            fmt_flt % (self.candidate_ratio()),
            fmt_flt % (self.failed_ratio()),
            fmt_flt % (self.level_af_lo),
            fmt_flt % (self.level_af_hi),
            fmt_flt % (self.level_mf_lo),
            fmt_flt % (self.level_mf_hi),
            str(self.converged()),
        )
        ret = " " + fmt_str % print_tuple
        return ret
    
    @staticmethod
    def doc() -> str:
        def make_class_doc_link(key):
            from dpgen2.entrypoint.args import (
                make_link,
            )

            return make_link(
                key, f"explore[lmp]/convergence[fixed-levels-max-select-spin]/{key}"
            )

        level_af_hi_link = make_class_doc_link("level_af_hi")
        level_af_lo_link = make_class_doc_link("level_af_lo")
        level_mf_hi_link = make_class_doc_link("level_mf_hi")
        level_mf_lo_link = make_class_doc_link("level_mf_lo")
        conv_accuracy_link = make_class_doc_link("conv_accuracy")
        return f"The configurations with atomic force model deviation between {level_af_lo_link}, {level_af_hi_link} or magnetic force model deviation between {level_mf_lo_link} and {level_mf_hi_link} are treated as candidates. The configurations with maximal magnetic force model deviation in the candidates are sent for FP calculations. If the ratio of accurate (below {level_af_lo_link} and {level_mf_lo_link}) is higher then {conv_accuracy_link}, the stage is treated as converged."
