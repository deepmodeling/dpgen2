import sys
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
    """

    def __init__(
            self,
            level_f_hi : float = 0.5,
            numb_candi_f : int = 200,
            rate_candi_f : float = 0.01,
            level_v_hi : Optional[float] = None,
            numb_candi_v : int = 0,
            rate_candi_v : float = 0.,
            n_checked_steps: int = 2,
            conv_tolerance : float = 0.05,
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
            self.rate_candi_v = 0.
        self.n_checked_steps = n_checked_steps
        self.conv_tolerance = conv_tolerance
        self.clear()

        print_tuple = ('stage', 'id_stg.', 'iter.',
                       'accu.', 'cand.', 'fail.',
                       'lvl_f_lo', 'lvl_f_hi',
        )
        spaces = [8, 8, 8, 10, 10, 10, 10]
        if self.has_virial:
            print_tuple += ('v_lo', 'v_hi',)
            spaces += [10, 10]
        spaces += [8]
        self.fmt_str = ' '.join([f'%{ii}s' for ii in spaces])
        self.fmt_flt = '%.4f'
        self.header_str = '#' + self.fmt_str % print_tuple


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
        return [
            Argument("level_f_hi", float, optional=True, default=0.5, doc=doc_level_f_hi),
            Argument("numb_candi_f", int, optional=True, default=200, doc=doc_numb_candi_f),
            Argument("rate_candi_f", float, optional=True, default=0.01, doc=doc_rate_candi_v),
            Argument("level_v_hi", float, optional=True, default=None, doc=doc_level_v_hi),
            Argument("numb_candi_v", int, optional=True, default=0, doc=doc_numb_candi_v),
            Argument("rate_candi_v", float, optional=True, default=0.0, doc=doc_rate_candi_v),
            Argument("n_checked_steps", int, optional=True, default=2, doc=doc_n_check_steps),
            Argument("conv_tolerance", float, optional=True, default=0.05, doc=doc_conv_tolerance),
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

    def record(
            self,
            md_f : List[np.ndarray],
            md_v_: Optional[List[np.ndarray]] = None,
    ):
        ntraj = len(md_f)
        self.ntraj += ntraj
        if md_v_ is None:
            md_v = [None for ii in range(ntraj)]
        else:
            md_v = md_v_
        # inits
        coll_f = []
        coll_v = []
        # loop over trajs
        for ii in range(ntraj):
            add_nframes, add_accur, add_failed, add_f, add_v = \
                self._record_one_traj(ii, md_f[ii], md_v[ii])
            self.nframes += add_nframes
            self.accur = self.accur.union(add_accur)
            self.failed += add_failed
            coll_f += add_f
            coll_v += add_v
        # sort 
        coll_f.sort()
        coll_v.sort()
        assert(len(coll_v) == len(coll_f))
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


    def _record_one_traj(
            self,
            tt,
            md_f : np.ndarray,
            md_v : Optional[np.ndarray] = None,            
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
    )->bool:
        if len(seq) <= 1:
            return False
        conv = [ abs(seq[ii-1] - seq[ii]) < self.conv_tolerance 
                 for ii in range(1,len(seq)) ]
        return all(conv)

    def converged(
            self,
            reports,
    )->bool:
        if 1 + len(reports) < self.n_checked_steps:
            return False
        else:
            all_level_f = [ii.level_f_lo for ii in reports] + [self.level_f_lo]
            all_level_f = all_level_f[-self.n_checked_steps:]
            conv = self._sequence_conv(all_level_f)
            if self.has_virial:
                all_level_v = [ii.level_v_lo for ii in reports] + [self.level_v_lo]
                all_level_v = all_level_v[-self.n_checked_steps:]
                conv = conv and self._sequence_conv(all_level_v)
            return conv            

    def failed_ratio(
            self,
            tag = None,
    ):
        return float(len(self.failed)) / float(self.nframes)

    def accurate_ratio(
            self,
            tag = None,
    ):
        return float(len(self.accur)) / float(self.nframes)

    def candidate_ratio(
            self,
            tag = None,
    ):
        return float(len(self.candi)) / float(self.nframes)

    def get_candidate_ids(
            self,
            max_nframes : Optional[int] = None,
    ) -> List[List[int]]:
        ntraj = self.ntraj
        id_cand = self._get_candidates(max_nframes)
        id_cand_list = [[] for ii in range(ntraj)]
        for ii in id_cand:
            id_cand_list[ii[0]].append(ii[1])
        return id_cand_list

    def _get_candidates(
            self,
            max_nframes : Optional[int] = None,
    )->List[Tuple[int,int]]:
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
        self.candi_picked = [(ii[0], ii[1]) for ii in self.candi]
        if max_nframes is not None and max_nframes < len(self.candi_picked):
            random.shuffle(self.candi_picked)
            ret = sorted(self.candi_picked[:max_nframes])
        else:
            ret = self.candi_picked
        return ret
        
    def print_header(self) -> str:
        r"""Print the header of report"""
        return self.header_str

    def print(
            self, 
            stage_idx : int,
            idx_in_stage : int,
            iter_idx : int,
    ) -> str:
        r"""Print the report"""
        fmt_str = self.fmt_str
        fmt_flt = self.fmt_flt
        print_tuple = (
                str(stage_idx), str(idx_in_stage), str(iter_idx),
                fmt_flt%(self.accurate_ratio()),
                fmt_flt%(self.candidate_ratio()),
                fmt_flt%(self.failed_ratio()),
                fmt_flt%(self.level_f_lo),
                fmt_flt%(self.level_f_hi),
        )
        if self.has_virial:
            print_tuple += (
                fmt_flt%(self.level_v_lo),
                fmt_flt%(self.level_v_hi),
            )
        ret = ' ' + fmt_str % print_tuple
        return ret
