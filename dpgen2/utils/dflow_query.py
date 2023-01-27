import numpy as np
import re
from typing import (
    List, Optional, Any,
)

def get_subkey(
        key : str, 
        idx : int = -1, 
):
    return key.split('--')[idx]

def get_iteration(
        key : str, 
):
    return get_subkey(key, 0)

def matched_step_key(
        all_keys : List[str],
        step_keys : Optional[List[str]] = None,
):
    """
    returns the keys in `all_keys` that matches any of the `step_keys`
    """
    if step_keys is None:
        return all_keys
    ret = []
    for kk in all_keys:
        for jj in step_keys:
            if re.match(f'iter-[0-9]*--{jj}-[0-9]*', kk) or\
               re.match(f'iter-[0-9]*--{jj}', kk) or\
               re.match(f'init--{jj}', kk):
                ret.append(kk)
                continue
    return ret

def get_last_scheduler(
        wf : Any, 
        keys : List[str],
):
    """
    get the output Scheduler of the last successful iteration
    """
    scheduler_keys_ = []
    for ii in keys:
        if get_subkey(ii) == 'scheduler':
            scheduler_keys_.append(ii)
    wf_info = wf.query()
    scheduler_keys = []
    for ii in scheduler_keys_:
        if wf_info.get_step(key=ii)[0]['phase'] == 'Succeeded':
            scheduler_keys.append(ii)
    if len(scheduler_keys) == 0:
        return None
    else:
        skey = sorted(scheduler_keys)[-1]
        step = wf_info.get_step(key=skey)[0]
        return step.outputs.parameters['exploration_scheduler'].value

def get_all_schedulers(
        wf : Any,
        keys : List[str],
):
    """
    get the output Scheduler of the all the iterations
    """
    scheduler_keys = sorted(matched_step_key(keys, ['scheduler']))
    if len(scheduler_keys) == 0:
        return None
    else:
        all_schedulers = [
            wf.query_step(key=skey)[0].outputs.parameters['exploration_scheduler'].value
            for skey in scheduler_keys
        ]
    return all_schedulers
        
def get_last_iteration(
        keys : List[str], 
):
    """
    get the index of the last iteraction from a list of step keys.
    """
    return int(sorted([get_subkey(ii,0) for ii in keys])[-1].split('-')[1])


def find_slice_ranges(
        keys : List[str], 
        sliced_subkey : str,
):
    """
    find range of sliced OPs that matches the pattern 'iter-[0-9]*--{sliced_subkey}-[0-9]*'
    """
    found_range = []
    tmp_range = []
    status = 'not-found'
    for idx,ii in enumerate(keys):
        if status == 'not-found':
            if re.match(f'iter-[0-9]*--{sliced_subkey}-[0-9]*', ii):
                status = 'found'
                tmp_range.append(idx)
        elif status == 'found':
            if not re.match(f'iter-[0-9]*--{sliced_subkey}-[0-9]*', ii):
                status = 'not-found'
                tmp_range.append(idx)                
                found_range.append(tmp_range)
                tmp_range = []
        else :
            raise RuntimeError(f'unknown status {status}, terrible error')
    return found_range


def _sort_slice_ops(keys, sliced_subkey):
    found_range = find_slice_ranges(keys, sliced_subkey)
    for ii in found_range:
        keys[ii[0]:ii[1]] = sorted(keys[ii[0]:ii[1]])
    return keys


def sort_slice_ops(
        keys : List[str], 
        sliced_subkey : List[str],
):
    """
    sort the keys of the sliced ops. the keys of the sliced ops contains sliced_subkey
    """
    if isinstance(sliced_subkey, str) :
        sliced_subkey = [sliced_subkey]
    for ii in sliced_subkey:
        keys = _sort_slice_ops(keys, ii)
    return keys


def print_keys_in_nice_format(
        keys : List[str], 
        sliced_subkey : List[str],
        idx_fmt_len : int = 8,
):
    keys = sort_slice_ops(keys, sliced_subkey)
    slice_range = []
    for ii in sliced_subkey:
        found_range = find_slice_ranges(keys, ii)
        slice_range += found_range
    slice_0 = [ii[0] for ii in slice_range]
    slice_1 = [ii[1] for ii in slice_range]
    
    normal_fmt = f'%{idx_fmt_len*2+4}d'
    range_fmt = f'%d -> %d'
    range_s_fmt = f'%{idx_fmt_len*2+4}s'
    
    idx = 0
    ret = []
    while(True):
        if idx >= len(keys):
            break
        try:
            idx_in_slice = slice_0.index(idx)
            range_0 = slice_0[idx_in_slice]
            range_1 = slice_1[idx_in_slice] - 1
            idx = range_1
            range_str = range_fmt % (range_0, range_1)
            ret.append((range_s_fmt + ' : ' + '%s -> %s') % (
                range_str, keys[range_0], keys[range_1]))
        except ValueError:
            ret.append((normal_fmt + ' : ' + '%s') % (
                idx, keys[idx]))
        idx += 1
    return '\n'.join(ret + [''])


