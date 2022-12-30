import logging, time
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

default_watching_keys = [
    "prep-run-train",
    "prep-run-lmp",
    "prep-run-fp",
    "collect-data",
]

def update_finished_steps(
        wf,
        finished_keys : Optional[List[str]] = None,
        download : Optional[bool] = False,
        watching_keys : Optional[List[str]] = None,
        prefix : Optional[str] = None,
        chk_pnt : bool = False,
):
    wf_keys = wf.query_keys_of_steps()
    wf_keys = matched_step_key(wf_keys, watching_keys)
    if finished_keys is not None:
        diff_keys = []
        for kk in wf_keys:
            if not (kk in finished_keys):
                diff_keys.append(kk)
    else :
        diff_keys = wf_keys
    for kk in diff_keys:
        logging.info(f'steps {kk.ljust(50,"-")} finished')
        if download :
            download_dpgen2_artifacts(wf, kk, prefix=prefix, chk_pnt=chk_pnt)
            logging.info(f'steps {kk.ljust(50,"-")} downloaded')
    finished_keys = wf_keys
    return finished_keys
                

def watch(
        workflow_id,
        wf_config : Optional[Dict] = {},
        watching_keys : Optional[List] = default_watching_keys,
        frequency : float = 600.,
        download : bool = False,
        prefix : Optional[str] = None,
        chk_pnt : bool = False,
):
    wf_config = normalize_args(wf_config)

    global_config_workflow(wf_config)

    wf = Workflow(id=workflow_id)

    finished_keys = None

    while wf.query_status() in ["Pending", "Running", "Failed", "Error"]:
        finished_keys = update_finished_steps(
            wf,
            finished_keys,
            download=download,
            watching_keys=watching_keys,
            prefix=prefix,
            chk_pnt=chk_pnt,
        )
        if wf.query_status() in ["Failed", "Error"]:
            break
        time.sleep(frequency)

    status = wf.query_status()
    if status == "Succeeded":
        finished_keys = update_finished_steps(
            wf,
            finished_keys,
            download=download,
            watching_keys=watching_keys,
            prefix=prefix,
            chk_pnt=chk_pnt,
        )
        logging.info("well done")
    elif status in ["Failed", "Error"]:
        logging.error("failed or error workflow")
