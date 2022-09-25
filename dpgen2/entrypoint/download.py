import logging
from dflow import (
    Workflow,
)
from dpgen2.utils import (
    dflow_config,
)
from dpgen2.utils.download_dpgen2_artifacts import (
    download_dpgen2_artifacts,
)
from typing import (
    Optional, Dict, Union, List,
)

def download(
        workflow_id,
        wf_keys : Optional[List] = None,
        wf_config : Optional[Dict] = {}, 
):
    dflow_config_data = wf_config.get('dflow_config', None)
    dflow_config(dflow_config_data)

    wf = Workflow(id=workflow_id)

    if steps is None:
        wf_keys = wf.query_keys_of_steps()
    
    for kk in wf_keys:
        download_dpgen2_artifacts(wf, kk)
