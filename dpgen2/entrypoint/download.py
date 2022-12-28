import logging
from dflow import (
    Workflow,
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
from dpgen2.entrypoint.common import (
    global_config_workflow,
)
from dpgen2.entrypoint.args import (
    normalize as normalize_args,
)


def download(
        workflow_id,
        wf_config : Optional[Dict] = {}, 
        wf_keys : Optional[List] = None,
        prefix : Optional[str] = None,
        chk_pnt : bool = False,
):
    wf_config = normalize_args(wf_config)

    global_config_workflow(wf_config)

    wf = Workflow(id=workflow_id)

    if wf_keys is None:
        wf_keys = wf.query_keys_of_steps()
    
    assert wf_keys is not None
    for kk in wf_keys:
        download_dpgen2_artifacts(wf, kk, prefix=prefix, chk_pnt=chk_pnt)
        logging.info(f'step {kk} downloaded')
