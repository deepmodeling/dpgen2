import logging
from dflow import (
    Workflow,
)
from dpgen2.utils import (
    dflow_config,
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
):
    dflow_config_data = wf_config.get('dflow_config', None)
    dflow_config(dflow_config_data)

    wf = Workflow(id=workflow_id)

    if wf_keys is None:
        wf_keys = wf.query_keys_of_steps()
    
    for kk in wf_keys:
        download_dpgen2_artifacts(wf, kk)
        logging.info(f'step {kk} downloaded')
