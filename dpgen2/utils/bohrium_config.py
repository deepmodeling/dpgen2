import importlib
import os

import dflow
from dflow.config import (
    config,
    s3_config,
)
from dflow.plugins import (
    bohrium,
)


def bohrium_config_from_dict(
    bohrium_config,
):
    config["host"] = bohrium_config["host"]
    config["k8s_api_server"] = bohrium_config["k8s_api_server"]
    bohrium.config["username"] = bohrium_config["username"]
    bohrium.config["password"] = bohrium_config["password"]
    bohrium.config["project_id"] = str(bohrium_config["project_id"])
    s3_config["repo_key"] = bohrium_config["repo_key"]
    module, cls = bohrium_config["storage_client"].rsplit(".", maxsplit=1)
    module = importlib.import_module(module)
    client = getattr(module, cls)
    s3_config["storage_client"] = client()
