import logging, time
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

default_watching_keys = [
    "prep-run-train",
    "prep-run-lmp",
    "prep-run-fp",
]

def update_finished_steps(
        wf,
        finished_keys : List[str] = None,
        download : Optional[bool] = False,
):
    wf_keys = wf.query_keys_of_steps()
    if finished_keys is not None:
        diff_keys = []
        for kk in wf_keys:
            if not (kk in finished_keys):
                diff_keys.append(kk)
    else :
        diff_keys = wf_keys
    for kk in diff_keys:
        logging.info(f'steps {kk} finished')
        if download :
            download_dpgen2_artifacts(wf, kk)
            logging.info(f'steps {kk} downloaded')
    finished_keys = wf_keys
    return finished_keys
                

def watch(
        workflow_id,
        wf_config : Optional[Dict] = {},
        watching_keys : Optional[List] = default_watching_keys,
        frequence : Optional[float] = 600.,
        download : Optional[bool] = False,
):
    dflow_config_data = wf_config.get('dflow_config', None)
    dflow_config(dflow_config_data)

    wf = Workflow(id=workflow_id)

    finished_keys = None

    while wf.query_status() in ["Pending", "Running"]:
        finished_keys = update_finished_steps(wf, finished_keys, download)
        time.sleep(frequence)

    status = wf.query_status()
    if status == "Succeeded":
        logging.info("well done")
    elif status in ["Failed", "Error"]:
        logging.error("failed or error workflow")
