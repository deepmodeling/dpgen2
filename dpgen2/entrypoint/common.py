import dflow
from pathlib import Path
from dpgen2.utils import (
    dump_object_to_file,
    load_object_from_file,
    sort_slice_ops,
    print_keys_in_nice_format,
    workflow_config_from_dict,
    matched_step_key,
    bohrium_config_from_dict,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
from typing import (
    Union, List, Dict, Optional,
)


def global_config_workflow(
        wf_config,
        do_lebesgue : bool=False,
):
    # dflow_config, dflow_s3_config
    workflow_config_from_dict(wf_config)

    # lebesgue context
    lebesgue_context = None
    if do_lebesgue:
        from dflow.plugins.lebesgue import LebesgueContext
        lb_context_config = wf_config.get("lebesgue_context_config", None)
        if lb_context_config:
            lebesgue_context = LebesgueContext(
                **lb_context_config,
            )

    # bohrium configuration
    if wf_config.get("bohrium_config") is not None:
        assert(lebesgue_context is None), \
            "cannot use bohrium and lebesgue at the same time"
        bohrium_config_from_dict(wf_config["bohrium_config"])        

    return lebesgue_context


def expand_sys_str(root_dir: Union[str, Path]) -> List[str]:
    root_dir = Path(root_dir)
    matches = [str(d) for d in root_dir.rglob("*") if (d / "type.raw").is_file()]
    if (root_dir / "type.raw").is_file():
        matches.append(str(root_dir))
    return matches


def expand_idx (in_list) :
    ret = []
    for ii in in_list :
        if isinstance(ii, int) :
            ret.append(ii)
        elif isinstance(ii, str):
            step_str = ii.split(':')
            if len(step_str) > 1 :
                step = int(step_str[1])
            else :
                step = 1
            range_str = step_str[0].split('-')
            if len(range_str) == 2:
                ret += range(int(range_str[0]), int(range_str[1]), step)
            elif len(range_str) == 1 :
                ret += [int(range_str[0])]
            else:
                raise RuntimeError('not expected range string', step_str[0])
    ret = sorted(list(set(ret)))
    return ret


