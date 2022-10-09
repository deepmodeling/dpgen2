import logging
from dflow import (
    Workflow,
)
from dpgen2.utils import (
    workflow_config_from_dict,
)
from dpgen2.utils.dflow_query import (
    get_last_scheduler,
)
from typing import (
    Optional, Dict, Union, List,
)

def status(
        workflow_id,
        wf_config : Optional[Dict] = {}, 
):
    workflow_config_from_dict(wf_config)

    wf = Workflow(id=workflow_id)

    wf_keys = wf.query_keys_of_steps()
    
    scheduler = get_last_scheduler(wf, wf_keys)

    if scheduler is not None:
        ptr_str = scheduler.print_convergence()
        print(ptr_str)
    else:
        logging.warn('no scheduler is finished')
