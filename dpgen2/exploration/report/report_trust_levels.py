import numpy as np
import random
from . import ExplorationReport
from typing import (
    List,
    Optional,
    Tuple,
)
from dflow.python import FatalError
from dargs import Argument


class ExplorationReportTrustLevels(ExplorationReport):
    def __init__(
        self,
        level_f_lo,
        level_f_hi,
        level_v_lo=None,
        level_v_hi=None,
        conv_accuracy=0.9,
    ):
        self.level_f_lo = level_f_lo
        self.level_f_hi = level_f_hi
        self.level_v_lo = level_v_lo
        self.level_v_hi = level_v_hi
        self.conv_accuracy = conv_accuracy
        self.clear()
        self.v_level = (self.level_v_lo is not None) and (self.level_v_hi is not None)

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
        print_tuple += ("cvged",)
        spaces += [8]
        self.fmt_str = " ".join([f"%{ii}s" for ii in spaces])
        self.fmt_flt = "%.4f"
        self.header_str = "#" + self.fmt_str % print_tuple

    @staticmethod
    def args() -> List[Argument]:
        doc_level_f_lo = "The lower trust level of force model deviation"
        doc_level_f_hi = "The higher trust level of force model deviation"
        doc_level_v_lo = "The lower trust level of virial model deviation"
        doc_level_v_hi = "The higher trust level of virial model deviation"
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

    def record(
        self,
        md_f: List[np.ndarray],
        md_v_: Optional[List[np.ndarray]] = None,
    ):
        ntraj = len(md_f)
        if md_v_ is None:
            md_v = [None for ii in range(ntraj)]
        else:
            md_v = md_v_
        for ii in range(ntraj):
            id_f_cand, id_f_accu, id_f_fail = self._get_indexes(
                md_f[ii], self.level_f_lo, self.level_f_hi
            )
            id_v_cand, id_v_accu, id_v_fail = self._get_indexes(
                md_v[ii], self.level_v_lo, self.level_v_hi
            )
            self._record_one_traj(
                id_f_accu,
                id_f_cand,
                id_f_fail,
                id_v_accu,
                id_v_cand,
                id_v_fail,
            )
        assert len(self.traj_nframes) == ntraj
        assert len(self.traj_cand) == ntraj
        assert len(self.traj_accu) == ntraj
        assert len(self.traj_fail) == ntraj

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
    ):
        """
        Record one trajctory. inputs are the indexes of candidate, accurate and failed frames.

        """
        # check consistency
        novirial = id_v_cand is None
        if novirial:
            assert id_v_accu is None
            assert id_v_fail is None
        nframes = np.size(np.concatenate((id_f_cand, id_f_accu, id_f_fail)))
        if (not novirial) and nframes != np.size(
            np.concatenate((id_v_cand, id_v_accu, id_v_fail))
        ):
            raise FatalError("number of frames by virial ")
        # nframes
        # to sets
        set_f_accu = set(id_f_accu)
        set_f_cand = set(id_f_cand)
        set_f_fail = set(id_f_fail)
        set_v_accu = set([ii for ii in range(nframes)]) if novirial else set(id_v_accu)
        set_v_cand = set([]) if novirial else set(id_v_cand)
        set_v_fail = set([]) if novirial else set(id_v_fail)
        # accu, cand, fail
        set_accu = set_f_accu & set_v_accu
        set_cand = (
            (set_f_cand & set_v_accu)
            | (set_f_cand & set_v_cand)
            | (set_f_accu & set_v_cand)
        )
        set_fail = set_f_fail | set_v_fail
        # check size
        assert nframes == len(set_accu | set_cand | set_fail)
        assert 0 == len(set_accu & set_cand)
        assert 0 == len(set_accu & set_fail)
        assert 0 == len(set_cand & set_fail)
        # record
        self.traj_nframes.append(nframes)
        self.traj_cand.append(set_cand)
        self.traj_accu.append(set_accu)
        self.traj_fail.append(set_fail)

    def converged(
        self,
        reports: Optional[List[ExplorationReport]] = None,
    ) -> bool:
        r"""Check if the exploration is converged.

        Parameters
        ----------
        reports List[ExplorationReportTrustLevels]
                Historical reports

        Returns
        -------
        converged  bool
                If the exploration is converged.
        """
        return self.accurate_ratio() >= self.conv_accuracy

    def failed_ratio(
        self,
        tag=None,
    ):
        traj_nf = [len(ii) for ii in self.traj_fail]
        return float(sum(traj_nf)) / float(sum(self.traj_nframes))

    def accurate_ratio(
        self,
        tag=None,
    ):
        traj_nf = [len(ii) for ii in self.traj_accu]
        return float(sum(traj_nf)) / float(sum(self.traj_nframes))

    def candidate_ratio(
        self,
        tag=None,
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
        then randomly pick `max_nframes` frames from the candidates.

        Parameters
        ----------
        max_nframes    int
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
            random.shuffle(self.traj_cand_picked)
            ret = sorted(self.traj_cand_picked[:max_nframes])
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
            fmt_flt % (self.level_f_lo),
            fmt_flt % (self.level_f_hi),
        )
        if self.v_level:
            print_tuple += (
                fmt_flt % (self.level_v_lo),
                fmt_flt % (self.level_v_hi),
            )
        print_tuple += (str(self.converged()),)
        ret = " " + fmt_str % print_tuple
        return ret
