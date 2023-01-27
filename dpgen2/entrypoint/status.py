import logging
from dflow import (
    Workflow,
)
from dpgen2.utils.dflow_query import (
    get_last_scheduler,
    get_all_schedulers,
)
from typing import (
    Optional,
    Dict,
    Union,
    List,
)
from dpgen2.entrypoint.common import (
    global_config_workflow,
)
from dpgen2.entrypoint.args import (
    normalize as normalize_args,
)


def status(
    workflow_id,
    wf_config: Optional[Dict] = {},
):
    wf_config = normalize_args(wf_config)

    global_config_workflow(wf_config)

    wf = Workflow(id=workflow_id)

    wf_keys = wf.query_keys_of_steps()

    scheduler = get_last_scheduler(wf, wf_keys)

    if scheduler is not None:
        ptr_str = scheduler.print_convergence()
        print(ptr_str)
    else:
        logging.warn("no scheduler is finished")
