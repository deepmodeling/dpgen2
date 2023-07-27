import os
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import dflow

from dpgen2.utils import (
    bohrium_config_from_dict,
    dump_object_to_file,
    load_object_from_file,
    matched_step_key,
    print_keys_in_nice_format,
    sort_slice_ops,
    workflow_config_from_dict,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


def global_config_workflow(
    wf_config,
):
    # dflow_config, dflow_s3_config
    workflow_config_from_dict(wf_config)

    if os.getenv("DFLOW_DEBUG"):
        dflow.config["mode"] = "debug"
        return None

    # bohrium configuration
    if wf_config.get("bohrium_config") is not None:
        bohrium_config_from_dict(wf_config["bohrium_config"])


def expand_sys_str(root_dir: Union[str, Path]) -> List[str]:
    root_dir = Path(root_dir)
    matches = [str(d) for d in root_dir.rglob("*") if (d / "type.raw").is_file()]
    if (root_dir / "type.raw").is_file():
        matches.append(str(root_dir))
    return matches


def expand_idx(in_list) -> List[int]:
    ret = []
    for ii in in_list:
        if isinstance(ii, int):
            ret.append(ii)
        elif isinstance(ii, str):
            step_str = ii.split(":")
            if len(step_str) > 1:
                step = int(step_str[1])
            else:
                step = 1
            range_str = step_str[0].split("-")
            if len(range_str) == 2:
                ret += range(int(range_str[0]), int(range_str[1]), step)
            elif len(range_str) == 1:
                ret += [int(range_str[0])]
            else:
                raise RuntimeError("not expected range string", step_str[0])
    ret = sorted(list(set(ret)))
    return ret
