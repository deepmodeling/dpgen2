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
    DeviManager,
)
from . import (
    ExplorationReport,
)


class ExplorationReportTrustLevels(ExplorationReport):
    def __init__(
        self,
        level_f_lo,
        level_f_hi,
        level_v_lo=None,
        level_v_hi=None,
        level_mf_lo=None,
        level_mf_hi=None,
        conv_accuracy=0.9,
    ):
        self.level_f_lo = level_f_lo
        self.level_f_hi = level_f_hi
        self.level_v_lo = level_v_lo
        self.level_v_hi = level_v_hi
        self.level_mf_lo = level_mf_lo
        self.level_mf_hi = level_mf_hi
        self.conv_accuracy = conv_accuracy
        self.clear()
        self.v_level = (self.level_v_lo is not None) and (self.level_v_hi is not None)
        self.mf_level = (self.level_mf_lo is not None) and (
            self.level_mf_hi is not None
        )
        self.model_devi = None

        print_tuple = (
            "stage",
            "id_stg.",
            "iter.",
            "accu.",
            "cand.",
            "fail.",
            "lvl_f_lo",
            "lvl_f_hi",
        )
        spaces = [8, 8, 8, 10, 10, 10, 10, 10]
        if self.v_level:
            print_tuple += (
                "v_lo",
                "v_hi",
            )
            spaces += [10, 10]
        if self.mf_level:
            print_tuple += (
                "mf_lo",
                "mf_hi",
            )
            spaces += [10, 10]
        print_tuple += ("cvged",)
        spaces += [8]
        self.fmt_str = " ".join([f"%{ii}s" for ii in spaces])
        self.fmt_flt = "%.4f"
        self.header_str = "#" + self.fmt_str % print_tuple
        self._no_candidate = False
        self._failed_ratio = None
        self._accurate_ratio = None
        self._candidate_ratio = None

    @staticmethod
    def args() -> List[Argument]:
        doc_level_f_lo = "The lower trust level of force model deviation"
        doc_level_f_hi = "The higher trust level of force model deviation"
        doc_level_v_lo = "The lower trust level of virial model deviation"
        doc_level_v_hi = "The higher trust level of virial model deviation"
        doc_level_mf_lo = "The lower trust level of magnetic force model deviation"
        doc_level_mf_hi = "The higher trust level of magnetic force model deviation"
        doc_conv_accuracy = "If the ratio of accurate frames is larger than this value, the stage is converged"
        return [
            Argument("level_f_lo", float, optional=False, doc=doc_level_f_lo),
            Argument("level_f_hi", float, optional=False, doc=doc_level_f_hi),
            Argument(
                "level_v_lo", float, optional=True, default=None, doc=doc_level_v_lo
            ),
            Argument(
                "level_v_hi", float, optional=True, default=None, doc=doc_level_v_hi
            ),
            Argument(
                "level_mf_lo",
                float,
                optional=True,
                default=None,
                doc=doc_level_mf_lo,
            ),
            Argument(
                "level_mf_hi",
                float,
                optional=True,
                default=None,
                doc=doc_level_mf_hi,
            ),
            Argument(
                "conv_accuracy",
                float,
                optional=True,
                default=0.9,
                doc=doc_conv_accuracy,
            ),
        ]

    def clear(
        self,
    ):
        self.traj_nframes = []
        self.traj_cand = []
        self.traj_accu = []
        self.traj_fail = []
        self.traj_cand_picked = []
        self.model_devi = None

    def record(
        self,
        model_devi: DeviManager,
    ):
        ntraj = model_devi.ntraj
        md_f = model_devi.get(DeviManager.MAX_DEVI_F)
        md_v = model_devi.get(DeviManager.MAX_DEVI_V)
        md_mf = model_devi.get(DeviManager.MAX_DEVI_MF)

        for ii in range(ntraj):
            id_f_cand, id_f_accu, id_f_fail = self._get_indexes(
                md_f[ii], self.level_f_lo, self.level_f_hi
            )
            id_v_cand, id_v_accu, id_v_fail = self._get_indexes(
                md_v[ii], self.level_v_lo, self.level_v_hi
            )
            id_mf_cand, id_mf_accu, id_mf_fail = self._get_indexes(
                md_mf[ii], self.level_mf_lo, self.level_mf_hi
            )
            nframes, set_accu, set_cand, set_fail = self._record_one_traj(
                id_f_accu,
                id_f_cand,
                id_f_fail,
                id_v_accu,
                id_v_cand,
                id_v_fail,
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
        self._no_candidate = sum([len(ii) for ii in self.traj_cand]) == 0
        self._failed_ratio = float(sum([len(ii) for ii in self.traj_fail])) / float(
            sum(self.traj_nframes)
        )
        self._accurate_ratio = float(sum([len(ii) for ii in self.traj_accu])) / float(
            sum(self.traj_nframes)
        )
        self._candidate_ratio = float(sum([len(ii) for ii in self.traj_cand])) / float(
            sum(self.traj_nframes)
        )

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
        id_f_accu,
        id_f_cand,
        id_f_fail,
        id_v_accu,
        id_v_cand,
        id_v_fail,
        id_mf_accu,
        id_mf_cand,
        id_mf_fail,
    ):
        """
        Record one trajctory. inputs are the indexes of candidate, accurate and failed frames.

        """
        # check consistency
        novirial = id_v_cand is None
        if novirial:
            assert id_v_accu is None
            assert id_v_fail is None
        nomagforce = id_mf_cand is None
        if nomagforce:
            assert id_mf_accu is None
            assert id_mf_fail is None
        nframes = np.size(np.concatenate((id_f_cand, id_f_accu, id_f_fail)))
        if (not novirial) and nframes != np.size(
            np.concatenate((id_v_cand, id_v_accu, id_v_fail))
        ):
            raise FatalError("number of frames by virial ")
        if (not nomagforce) and nframes != np.size(
            np.concatenate((id_mf_cand, id_mf_accu, id_mf_fail))
        ):
            raise FatalError("number of frames by magnetic force ")
        # nframes
        # to sets
        set_full = set([ii for ii in range(nframes)])
        set_f_accu = set(id_f_accu)
        set_f_cand = set(id_f_cand)
        set_f_fail = set(id_f_fail)
        set_v_accu = set_full if novirial else set(id_v_accu)
        set_v_cand = set([]) if novirial else set(id_v_cand)
        set_v_fail = set([]) if novirial else set(id_v_fail)
        set_mf_accu = set_full if nomagforce else set(id_mf_accu)
        set_mf_cand = set([]) if nomagforce else set(id_mf_cand)
        set_mf_fail = set([]) if nomagforce else set(id_mf_fail)

        # check consistency
        assert set_full == set_f_accu | set_f_cand | set_f_fail
        for accu, cand, fail in [
            [set_f_accu, set_f_cand, set_f_fail],
            [set_v_accu, set_v_cand, set_v_fail],
            [set_mf_accu, set_mf_cand, set_mf_fail],
        ]:
            assert 0 == len(accu & cand)
            assert 0 == len(accu & fail)
            assert 0 == len(cand & fail)

        # accu, cand, fail
        set_accu = set_f_accu & set_v_accu & set_mf_accu
        set_fail = set_f_fail | set_v_fail | set_mf_fail
        set_cand = set_full - set_accu - set_fail
        # check size
        assert nframes == len(set_accu | set_cand | set_fail)
        assert 0 == len(set_accu & set_cand)
        assert 0 == len(set_accu & set_fail)
        assert 0 == len(set_cand & set_fail)
        return nframes, set_accu, set_cand, set_fail

    @abstractmethod
    def converged(
        self,
        reports: Optional[List[ExplorationReport]] = None,
    ) -> bool:
        pass

    def failed_ratio(
        self,
        tag=None,
    ):
        return self._failed_ratio

    def accurate_ratio(
        self,
        tag=None,
    ):
        return self._accurate_ratio

    def candidate_ratio(
        self,
        tag=None,
    ):
        return self._candidate_ratio

    def no_candidate(self) -> bool:
        return self._no_candidate

    @abstractmethod
    def get_candidate_ids(
        self,
        max_nframes: Optional[int] = None,
    ) -> List[List[int]]:
        pass

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
            fmt_flt % (self.level_f_lo),
            fmt_flt % (self.level_f_hi),
        )
        if self.v_level:
            print_tuple += (
                fmt_flt % (self.level_v_lo),
                fmt_flt % (self.level_v_hi),
            )
        if self.mf_level:
            print_tuple += (
                fmt_flt % (self.level_mf_lo),
                fmt_flt % (self.level_mf_hi),
            )
        print_tuple += (str(self.converged()),)
        ret = " " + fmt_str % print_tuple
        return ret
