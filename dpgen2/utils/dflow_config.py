import copy

from dflow.config import (
    config,
    s3_config,
)


def workflow_config_from_dict(
    wf_config,
):
    dflow_config_data = wf_config.get("dflow_config", None)
    dflow_config(dflow_config_data)
    dflow_s3_config_data = wf_config.get("dflow_s3_config", None)
    dflow_s3_config(dflow_s3_config_data)


def dflow_config_lower(
    dflow_config,
):
    dflow_s3_config = {}
    keys = [kk for kk in dflow_config.keys()]
    for kk in keys:
        if kk[:3] == "s3_":
            dflow_s3_config[kk[3:]] = dflow_config.pop(kk)
    for kk in dflow_config.keys():
        config[kk] = dflow_config[kk]
    for kk in dflow_s3_config.keys():
        s3_config[kk] = dflow_s3_config[kk]


def dflow_s3_config_lower(
    dflow_s3_config_data,
):
    for kk in dflow_s3_config_data.keys():
        s3_config[kk] = dflow_s3_config_data[kk]


def dflow_config(
    config_data,
):
    """
    set the dflow config by `config_data`

    the keys starting with "s3_" will be treated as s3_config keys,
    other keys are treated as config keys.

    """
    if config_data is not None:
        dflow_config_lower(config_data)


def dflow_s3_config(
    config_data,
):
    """
    set the s3 config by `config_data`

    """
    if config_data is not None:
        dflow_s3_config_lower(config_data)
