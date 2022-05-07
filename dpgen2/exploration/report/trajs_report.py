import numpy as np
import random
from . import ExplorationReport
from typing import (
    List,
    Tuple,
)

class TrajsExplorationReport(ExplorationReport):
    def __init__(
            self,
    ):
        self.clear()

    def clear(
            self,
    ):
        self.traj_nframes = []
        self.traj_cand = []
        self.traj_accu = []
        self.traj_fail = []        
        self.traj_cand_picked = []

    def record_traj(
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
            assert(id_v_accu is None)
            assert(id_v_fail is None)
        nframes = np.size(np.concatenate((id_f_cand, id_f_accu, id_f_fail)))
        if (not novirial) and nframes != np.size(np.concatenate((id_v_cand, id_v_accu, id_v_fail))):
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
        set_cand = ( (set_f_cand & set_v_accu) | 
                     (set_f_cand & set_v_cand) | 
                     (set_f_accu & set_v_cand) )
        set_fail = ( set_f_fail | set_v_fail)
        # check size
        assert(nframes == len(set_accu | set_cand | set_fail))
        assert(0 == len(set_accu & set_cand))
        assert(0 == len(set_accu & set_fail))
        assert(0 == len(set_cand & set_fail))
        # record
        self.traj_nframes.append(nframes)
        self.traj_cand.append(set_cand)
        self.traj_accu.append(set_accu)
        self.traj_fail.append(set_fail)

        
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

    def get_candidates(
            self,
            max_nframes : int = None,
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
        self.traj_cand_picked = []
        for tidx,tt in enumerate(self.traj_cand):
            for ff in tt:
                self.traj_cand_picked.append((tidx, ff))
        if max_nframes and max_nframes < len(self.traj_cand_picked):
            random.shuffle(self.traj_cand_picked)
            ret = sorted(self.traj_cand_picked[:max_nframes])
        else:
            ret = self.traj_cand_picked
        return ret
        
