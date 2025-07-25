import random
import sys
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


class ExplorationReportAdaptiveLower(ExplorationReport):
    r"""The exploration report that adapts the lower trust level.

    This report will treat a fixed number of frames that has force
    model deviation lower than `level_f_hi`, and virial model deviation
    lower than `level_v_hi` as candidates.

    The number of force frames is given by max(`numb_candi_f`, `rate_candi_f` * nframes)
    The number of virial frames is given by max(`numb_candi_v`, `rate_candi_v` * nframes)

    The lower force trust level will be set to the lowest force model deviation
    of the force frames. The lower virial trust level will be set to the lowest
    virial model deviation of the virial frames

    The exploration will be treat as converged if the differences in model
    deviations in the neighboring steps are less than `conv_tolerance`
    in the last `n_checked_steps`.

    Parameters
    ----------
    level_f_hi          float
        The higher trust level of force model deviation
    numb_candi_f        int
        The number of force frames that has a model deviation lower than
        `level_f_hi` treated as candidate.
    rate_candi_f        float
        The ratio of force frames that has a model deviation lower than
        `level_f_hi` treated as candidate.
    level_v_hi          float
        The higher trust level of virial model deviation
    numb_candi_v        int
        The number of virial frames that has a model deviation lower than
        `level_v_hi` treated as candidate.
    rate_candi_v        float
        The ratio of virial frames that has a model deviation lower than
        `level_v_hi` treated as candidate.
    n_checked_steps     int
        The number of steps to check the convergence.
    conv_tolerance      float
        The convergence tolerance.
    candi_sel_prob     str
        The method for selecting candidates. It can be
        "uniform": all candidates are of the same probability.
        "inv_pop_f" or "inv_pop_f:nhist": the probability is inversely
        propotional to the population of a histogram between
        level_f_lo and level_f_hi. The number of bins in the histogram
        is set by nhist, which should be an integer. The default is 10.
    """

    def __init__(
        self,
        level_f_hi: float = 0.5,
        numb_candi_f: int = 200,
        rate_candi_f: float = 0.01,
        level_v_hi: Optional[float] = None,
        numb_candi_v: int = 0,
        rate_candi_v: float = 0.0,
        n_checked_steps: int = 2,
        conv_tolerance: float = 0.05,
        candi_sel_prob: str = "uniform",
    ):
        self.level_f_hi = level_f_hi
        self.level_v_hi = level_v_hi
        self.numb_candi_f = numb_candi_f
        self.rate_candi_f = rate_candi_f
        self.numb_candi_v = numb_candi_v
        self.rate_candi_v = rate_candi_v
        self.has_virial = self.level_v_hi is not None
        if not self.has_virial:
            self.level_v_hi = sys.float_info.max
            self.numb_candi_v = 0
            self.rate_candi_v = 0.0
        self.n_checked_steps = n_checked_steps
        self.conv_tolerance = conv_tolerance
        self.model_devi = None
        default_nhist = 10
        self.candi_sel_prob = candi_sel_prob.split(":")[0]
        if self.candi_sel_prob == "inv_pop_f":
            if len(candi_sel_prob.split(":")) == 2:
                self.nhist = int(candi_sel_prob.split(":")[1])
            else:
                self.nhist = default_nhist
        self.clear()

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
        spaces = [8, 8, 8, 10, 10, 10, 10]
        if self.has_virial:
            print_tuple += (
                "v_lo",
                "v_hi",
            )
            spaces += [10, 10]
        spaces += [8]
        self.fmt_str = " ".join([f"%{ii}s" for ii in spaces])
        self.fmt_flt = "%.4f"
        self.header_str = "#" + self.fmt_str % print_tuple
        self._no_candidate = False
        self._failed_ratio = None
        self._accurate_ratio = None
        self._candidate_ratio = None

    @staticmethod
    def doc() -> str:
        def make_class_doc_link(key):
            from dpgen2.entrypoint.args import (
                make_link,
            )

            return make_link(key, f"explore[lmp]/convergence[adaptive-lower]/{key}")

        numb_candi_f_link = make_class_doc_link("numb_candi_f")
        rate_candi_f_link = make_class_doc_link("rate_candi_f")
        numb_candi_v_link = make_class_doc_link("numb_candi_v")
        rate_candi_v_link = make_class_doc_link("rate_candi_v")
        numb_candi_s = f"{numb_candi_f_link} or {numb_candi_v_link}"
        rate_candi_s = f"{rate_candi_f_link} or {rate_candi_v_link}"
        level_f_hi_link = make_class_doc_link("level_f_hi")
        level_v_hi_link = make_class_doc_link("level_v_hi")
        conv_tolerance_link = make_class_doc_link("conv_tolerance")
        n_checked_steps_link = make_class_doc_link("n_checked_steps")
        return f"The method of adaptive adjust the lower trust levels. In each step of iterations, a number (set by {numb_candi_s}) or a ratio (set by {rate_candi_s}) of configurations with a model deviation lower than the higher trust level ({level_f_hi_link}, {level_v_hi_link}) are treated as candidates. The lowest model deviation of the candidates are treated as the lower trust level. If the lower trust level does not change significant (controlled by {conv_tolerance_link}) in {n_checked_steps_link}, the stage is treated as converged. "

    @staticmethod
    def args() -> List[Argument]:
        doc_level_f_hi = "The higher trust level of force model deviation"
        doc_numb_candi_f = "The number of force frames that has a model deviation lower than `level_f_hi` treated as candidate."
        doc_rate_candi_f = "The ratio of force frames that has a model deviation lower than `level_f_hi` treated as candidate."
        doc_level_v_hi = "The higher trust level of virial model deviation"
        doc_numb_candi_v = "The number of virial frames that has a model deviation lower than `level_v_hi` treated as candidate."
        doc_rate_candi_v = "The ratio of virial frames that has a model deviation lower than `level_v_hi` treated as candidate."
        doc_n_check_steps = "The number of steps to check the convergence."
        doc_conv_tolerance = "The convergence tolerance."
        doc_candi_sel_prob = (
            "The method for selecting candidates. It can be "
            "'uniform': all candidates are of the same probability. "
            "'inv_pop_f' or 'inv_pop_f:nhist': the probability is inversely "
            "propotional to the population of a histogram between "
            "leven_f_lo and level_f_hi. The number of bins in the histogram "
            "is set by nhist, which should be an integer. The default is 10."
        )

        return [
            Argument(
                "level_f_hi", float, optional=True, default=0.5, doc=doc_level_f_hi
            ),
            Argument(
                "numb_candi_f", int, optional=True, default=200, doc=doc_numb_candi_f
            ),
            Argument(
                "rate_candi_f", float, optional=True, default=0.01, doc=doc_rate_candi_f
            ),
            Argument(
                "level_v_hi", float, optional=True, default=None, doc=doc_level_v_hi
            ),
            Argument(
                "numb_candi_v", int, optional=True, default=0, doc=doc_numb_candi_v
            ),
            Argument(
                "rate_candi_v", float, optional=True, default=0.0, doc=doc_rate_candi_v
            ),
            Argument(
                "n_checked_steps", int, optional=True, default=2, doc=doc_n_check_steps
            ),
            Argument(
                "conv_tolerance",
                float,
                optional=True,
                default=0.05,
                doc=doc_conv_tolerance,
            ),
            Argument(
                "candi_sel_prob",
                str,
                optional=True,
                default="uniform",
                doc=doc_candi_sel_prob,
            ),
        ]

    def clear(
        self,
    ):
        self.ntraj = 0
        self.nframes = 0
        self.candi = set()
        self.accur = set()
        self.failed = []
        self.candi_picked = []
        self.model_devi = None
        self.md_f = []
        self.md_v = []

    def record(
        self,
        model_devi: DeviManager,
    ):
        ntraj = model_devi.ntraj
        self.ntraj += ntraj
        md_f = model_devi.get(DeviManager.MAX_DEVI_F)
        md_v = model_devi.get(DeviManager.MAX_DEVI_V)
        self.md_f += md_f
        self.md_v += md_v

        # inits
        coll_f = []
        coll_v = []
        # loop over trajs
        for ii in range(ntraj):
            add_nframes, add_accur, add_failed, add_f, add_v = self._record_one_traj(
                ii, md_f[ii], md_v[ii]
            )
            self.nframes += add_nframes
            self.accur.update(add_accur)
            self.failed += add_failed
            coll_f += add_f
            coll_v += add_v
        # sort
        coll_f.sort()
        coll_v.sort()
        assert len(coll_v) == len(coll_f)
        # calcuate numbers
        numb_candi_f = max(self.numb_candi_f, int(self.rate_candi_f * len(coll_f)))
        numb_candi_v = max(self.numb_candi_v, int(self.rate_candi_v * len(coll_v)))
        # adjust number of candidate
        if len(coll_f) < numb_candi_f:
            numb_candi_f = len(coll_f)
        if len(coll_v) < numb_candi_v:
            numb_candi_v = len(coll_v)
        # compute trust lo
        if numb_candi_v == 0:
            self.level_v_lo = self.level_v_hi
        else:
            self.level_v_lo = coll_v[-numb_candi_v][0]
        if not self.has_virial:
            self.level_v_lo = None
        if numb_candi_f == 0:
            self.level_f_lo = self.level_f_hi
        else:
            self.level_f_lo = coll_f[-numb_candi_f][0]
        # add to candidate set
        for ii in range(len(coll_f) - numb_candi_f, len(coll_f)):
            self.candi.add(tuple(coll_f[ii][1:]))
        for ii in range(len(coll_v) - numb_candi_v, len(coll_v)):
            self.candi.add(tuple(coll_v[ii][1:]))
        # accurate set is substracted by the candidate set
        self.accur = self.accur - self.candi
        self.model_devi = model_devi
        self._no_candidate = len(self.candi) == 0
        self._failed_ratio = float(len(self.failed)) / float(self.nframes)
        self._accurate_ratio = float(len(self.accur)) / float(self.nframes)
        self._candidate_ratio = float(len(self.candi)) / float(self.nframes)

    def _record_one_traj(
        self,
        tt,
        md_f,
        md_v,
    ):
        """
        Record one trajctory.

        tt:             traj index
        md_f, md_v:     model deviations of force and virial
        """
        # check consistency
        if self.has_virial and md_v is None:
            raise FatalError(
                "report requires virial model deviation, but no virial "
                "model deviation is provided."
            )
        # fake md_v as zeros if None is provided
        if md_v is None:
            md_v = np.zeros_like(md_f)
        # loop over frames
        nframes = md_f.shape[0]
        assert nframes == md_v.shape[0]
        failed = []
        accur = set()
        coll_f = []
        coll_v = []
        for ii in range(nframes):
            if md_f[ii] > self.level_f_hi or md_v[ii] > self.level_v_hi:
                failed.append((tt, ii))
            else:
                coll_f.append([md_f[ii], tt, ii])
                coll_v.append([md_v[ii], tt, ii])
                # now accur takes all non-failed frames,
                # will be substracted by candidate later
                accur.add((tt, ii))
        return nframes, accur, failed, coll_f, coll_v

    def _sequence_conv(
        self,
        seq,
    ) -> bool:
        if len(seq) <= 1:
            return False
        conv = [
            abs(seq[ii - 1] - seq[ii]) < self.conv_tolerance
            for ii in range(1, len(seq))
        ]
        return all(conv)

    def converged(
        self,
        reports,
    ) -> bool:
        if 1 + len(reports) < self.n_checked_steps:
            return False
        else:
            all_level_f = [ii.level_f_lo for ii in reports] + [self.level_f_lo]
            all_level_f = all_level_f[-self.n_checked_steps :]
            conv = self._sequence_conv(all_level_f)
            if self.has_virial:
                all_level_v = [ii.level_v_lo for ii in reports] + [self.level_v_lo]
                all_level_v = all_level_v[-self.n_checked_steps :]
                conv = conv and self._sequence_conv(all_level_v)
            return conv

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

    def get_candidate_ids(
        self,
        max_nframes: Optional[int] = None,
        clear: bool = True,
    ) -> List[List[int]]:
        ntraj = self.ntraj
        id_cand = self._get_candidates(max_nframes)
        id_cand_list = [[] for ii in range(ntraj)]
        for ii in id_cand:
            id_cand_list[ii[0]].append(ii[1])
        # free the memory, this method should only be called once
        if clear:
            self.clear()
        return id_cand_list

    def _get_candidates(
        self,
        max_nframes: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        if self.candi_sel_prob == "uniform":
            return self._get_candidates_uniform(max_nframes)
        elif self.candi_sel_prob == "inv_pop_f":
            return self._get_candidates_inv_pop_f(max_nframes)
        else:
            raise FatalError("unknown candidate selection style")

    def _get_candidates_uniform(
        self,
        max_nframes: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get candidates. If number of candidates is larger than `max_nframes`,
        then randomly pick `max_nframes` frames from the candidates.

        Parameters
        ----------
        max_nframes
            The maximal number of frames of candidates.

        Returns
        -------
        cand_frames   List[Tuple[int,int]]
            Candidate frames. A list of tuples: [(traj_idx, frame_idx), ...]
        """
        self.candi_picked = [(ii[0], ii[1]) for ii in self.candi]
        if max_nframes is not None and max_nframes < len(self.candi_picked):
            random.shuffle(self.candi_picked)
            ret = sorted(self.candi_picked[:max_nframes])
        else:
            ret = self.candi_picked
        return ret

    def _get_candidates_inv_pop_f(
        self,
        max_nframes: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """
        Get candidates. If number of candidates is larger than `max_nframes`,
        then randomly pick `max_nframes` frames from the candidates.
        The probability of chose a frame is propotional to the inverse
        population in force model deviation statistics.

        Parameters
        ----------
        max_nframes
            The maximal number of frames of candidates.

        Returns
        -------
        cand_frames   List[Tuple[int,int]]
            Candidate frames. A list of tuples: [(traj_idx, frame_idx), ...]
        """
        self.candi_picked = [(ii[0], ii[1]) for ii in self.candi]
        if max_nframes is not None and max_nframes < len(self.candi_picked):
            prob = self._choice_prob_inv_pop_f(self.candi_picked)
            indices = np.random.choice(
                len(self.candi_picked),
                size=max_nframes,
                replace=False,
                p=prob / np.sum(prob),
            )
            ret = [self.candi_picked[i] for i in indices]
        else:
            ret = self.candi_picked
        return ret

    def _choice_prob_inv_pop_f(
        self,
        candi: List,
    ):
        """Compute the probability of candi frames according to the inverse
        population in the model deviation statistics.

        Parameters
        ----------
        candi   List[Tuple[int,int]]
            Candidate frames. A list of tuples: [(traj_idx, frame_idx), ...]

        Returns
        -------
        prob    List[float]
            The probability of each candidate frame.

        """
        histo = np.zeros(self.nhist, dtype=int)
        for ii in candi:
            frame_md_f = self.md_f[ii[0]][ii[1]]
            hist_idx = self._histo_idx(frame_md_f)
            histo[hist_idx] += 1
        prob_tab = [1.0 / float(ii) if ii > 0 else 0.0 for ii in histo]
        prob = []
        for ii in candi:
            frame_md_f = self.md_f[ii[0]][ii[1]]
            hist_idx = self._histo_idx(frame_md_f)
            prob.append(prob_tab[hist_idx])
        return prob

    def _histo_idx(
        self,
        devi_f: float,
    ) -> int:
        """
        return the index in histogram given a force  model deviation.
        """
        dh = (self.level_f_hi - self.level_f_lo) / self.nhist
        hist_idx = int((devi_f - self.level_f_lo) / dh)
        if hist_idx < 0:
            hist_idx = 0
        elif hist_idx >= self.nhist:
            hist_idx = self.nhist - 1
        return hist_idx

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
        if self.has_virial:
            print_tuple += (
                fmt_flt % (self.level_v_lo),
                fmt_flt % (self.level_v_hi),
            )
        ret = " " + fmt_str % print_tuple
        return ret
