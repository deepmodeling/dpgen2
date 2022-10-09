import logging
from dflow import (
    Workflow,
)
from dpgen2.utils import (
    workflow_config_from_dict,
)
from dpgen2.utils.dflow_query import (
    matched_step_key,
)
from dpgen2.utils.download_dpgen2_artifacts import (
    download_dpgen2_artifacts,
)
from typing import (
    Optional, Dict, Union, List,
)

def download(
        workflow_id,
        wf_config : Optional[Dict] = {}, 
        wf_keys : Optional[List] = None,
        prefix : Optional[str] = None,
):
    workflow_config_from_dict(wf_config)

    wf = Workflow(id=workflow_id)

    if wf_keys is None:
        wf_keys = wf.query_keys_of_steps()
    
    for kk in wf_keys:
        download_dpgen2_artifacts(wf, kk, prefix=prefix)
        logging.info(f'step {kk} downloaded')
