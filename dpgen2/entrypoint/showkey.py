import glob, dpdata, os, pickle
from pathlib import Path
from dflow import (
    Workflow,
)
from dpgen2.entrypoint.submit import successful_step_keys
from dpgen2.utils import (
    sort_slice_ops,
    print_keys_in_nice_format,
    workflow_config_from_dict,
)

def showkey(
        wf_id,
        wf_config,
):
    workflow_config_from_dict(wf_config)    
    wf = Workflow(id=wf_id)
    all_step_keys = successful_step_keys(wf)
    all_step_keys = sort_slice_ops(
        all_step_keys, ['run-train', 'run-lmp', 'run-fp'],)
    prt_str = print_keys_in_nice_format(
        all_step_keys, ['run-train', 'run-lmp', 'run-fp'],)
    print(prt_str)

