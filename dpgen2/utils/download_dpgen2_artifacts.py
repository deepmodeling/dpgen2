import itertools
import logging
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

import numpy as np
from dflow import (
    Workflow,
    download_artifact,
)

from dpgen2.utils.dflow_query import (
    get_iteration,
    get_subkey,
)

global_step_def_split = "/"


class DownloadDefinition:
    def __init__(self):
        self.input_def = {}
        self.output_def = {}

    def add_def(
        self,
        tdict,
        key,
        suffix=None,
    ):
        tdict[key] = suffix
        return self

    def add_input(
        self,
        input_key,
        suffix=None,
    ):
        return self.add_def(self.input_def, input_key, suffix)

    def add_output(
        self,
        output_key,
        suffix=None,
    ):
        return self.add_def(self.output_def, output_key, suffix)


op_download_setting = {
    "prep-run-train": DownloadDefinition()
    .add_input("init_models")
    .add_input("init_data")
    .add_input("iter_data")
    .add_output("scripts")
    .add_output("models")
    .add_output("logs")
    .add_output("lcurves"),
    "prep-run-explore": DownloadDefinition()
    .add_output("logs")
    .add_output("trajs")
    .add_output("model_devis"),
    "prep-run-fp": DownloadDefinition()
    .add_input("confs")
    .add_output("logs")
    .add_output("labeled_data"),
    "collect-data": DownloadDefinition().add_output("iter_data"),
}


def print_op_download_setting(op_download_setting=op_download_setting):
    ret = []
    for kk in op_download_setting.keys():
        ret.append(f"step: {kk}")
        idef = op_download_setting[kk].input_def
        if len(idef) > 0:
            ret.append("  input:")
            str_keys = []
            for ik in idef.keys():
                str_keys.append(f"{ik}")
            ret.append("    " + " ".join(str_keys))
        odef = op_download_setting[kk].output_def
        if len(odef) > 0:
            ret.append("  output:")
            str_keys = []
            for ik in odef.keys():
                str_keys.append(f"{ik}")
            ret.append("    " + " ".join(str_keys))
    return "\n".join(ret)


def download_dpgen2_artifacts(
    wf: Workflow,
    key: str,
    prefix: Optional[str] = None,
    chk_pnt: bool = False,
):
    """
    download the artifacts of a step.
    the key should be of format 'iter-xxxxxx--subkey-of-step-xxxxxx'
    the input and output artifacts will be downloaded to
    prefix/iter-xxxxxx/key-of-step/inputs/ and
    prefix/iter-xxxxxx/key-of-step/outputs/

    the downloaded input and output artifacts of steps are defined by
    `op_download_setting`

    """

    iteration = get_iteration(key)
    subkey = get_subkey(key)
    mypath = Path(iteration)
    if prefix is not None:
        mypath = Path(prefix) / mypath

    dsetting = op_download_setting.get(subkey)
    if dsetting is None:
        logging.warning(f"cannot find download settings for {key}")
        return

    input_def = dsetting.input_def
    output_def = dsetting.output_def

    dld_input = not (
        len(input_def) == 0
        or (chk_pnt and (mypath / subkey / "inputs" / "done").is_file())
    )
    dld_output = not (
        len(output_def) == 0
        or (chk_pnt and (mypath / subkey / "outputs" / "done").is_file())
    )

    step = None
    if dld_input or dld_output:
        step = wf.query_step(key=key)
        if len(step) == 0:
            raise RuntimeError(f"key {key} does not match any step")
        step = step[0]

    if dld_input:
        _dload_input_lower(step, mypath, key, subkey, input_def)
        if chk_pnt:
            (mypath / subkey / "inputs" / "done").touch()

    if dld_output:
        _dload_output_lower(step, mypath, key, subkey, output_def)
        if chk_pnt:
            (mypath / subkey / "outputs" / "done").touch()

    return


def download_dpgen2_artifacts_by_def(
    wf: Workflow,
    iterations: Optional[List[int]] = None,
    step_defs: Optional[List[str]] = None,
    prefix: Optional[str] = None,
    chk_pnt: bool = False,
):
    wf_step_keys = wf.query_keys_of_steps()

    if prefix is not None:
        prefix = prefix
    else:
        prefix = "."
    if iterations is None:
        iterations = _get_all_iterations(wf_step_keys)
    if step_defs is None:
        step_defs = _get_all_step_defs()
    step_defs = _filter_def_by_availability(step_defs)
    if len(step_defs) == 0:
        return

    # mk download items
    dld_items = _get_dld_items(iterations, step_defs)
    if chk_pnt:
        dld_items = _filter_if_complished(prefix, dld_items)
    if len(dld_items) == 0:
        return

    # get all steps
    step_keys = _get_all_queried_steps(wf_step_keys, dld_items)
    wf_steps = wf.query_step_by_key(step_keys)
    if not (len(wf_steps) == len(step_keys)):
        # fall back to the wf.query to get the steps.
        wf_info = wf.query()
        wf_steps = [wf_info.get_step(key=kk)[0] for kk in step_keys]
    # make step dict
    step_dict = {wf_step.key: wf_step for wf_step in wf_steps}

    # download all items
    for ii in dld_items:
        [step_key, io, item] = ii.split(global_step_def_split)
        step = step_dict.get(step_key)
        # skip all problematic steps
        if step is None or step["phase"] != "Succeeded":
            continue
        _dl_step_item(step, ii, prefix, chk_pnt)


def _dload_input_lower(
    step,
    mypath,
    key,
    subkey,
    input_def,
):
    for kk in input_def.keys():
        pref = mypath / subkey / "inputs"
        ksuff = input_def[kk]
        if ksuff is not None:
            pref = pref / ksuff
        try:
            download_artifact(
                step.inputs.artifacts[kk],
                path=pref,
                skip_exists=True,
            )
        except (NotImplementedError, FileNotFoundError):
            # NotImplementedError to be compatible with old versions of dflow
            logging.warning(
                f"cannot download input artifact  {kk}  of  {key}, it may be empty"
            )


def _dload_output_lower(
    step,
    mypath,
    key,
    subkey,
    output_def,
):
    for kk in output_def.keys():
        pref = mypath / subkey / "outputs"
        ksuff = output_def[kk]
        if ksuff is not None:
            pref = pref / ksuff
        try:
            download_artifact(
                step.outputs.artifacts[kk],
                path=pref,
                skip_exists=True,
            )
        except (NotImplementedError, FileNotFoundError):
            # NotImplementedError to be compatible with old versions of dflow
            logging.warning(
                f"cannot download input artifact  {kk}  of  {key}, it may be empty"
            )


def _get_all_step_defs(op_download_setting=op_download_setting):
    ret = []
    for kk, vv in op_download_setting.items():
        idef = vv.input_def
        for ik, iv in idef.items():
            ret.append(f"{kk}{global_step_def_split}input{global_step_def_split}{ik}")
        odef = vv.output_def
        for ik, iv in odef.items():
            ret.append(f"{kk}{global_step_def_split}output{global_step_def_split}{ik}")
    return ret


def _get_all_iterations(step_keys):
    ret = []
    for kk in step_keys:
        ii = get_iteration(kk)
        if ii.startswith("iter-"):
            ii = int(ii.split("-")[1])
            ret.append(ii)
    ret = sorted(list(set(ret)))
    return ret


def _get_all_queried_steps(wf_step_keys, dld_items):
    ret = []
    for ii in dld_items:
        ret.append(ii.split(global_step_def_split)[0])
    ret = set(ret)
    ret = ret.intersection(set(wf_step_keys))
    return sorted(list(ret))


def _get_dld_items(
    iterations,
    step_defs,
):
    items = []
    for ii, jj in itertools.product(iterations, step_defs):  # type: ignore
        items.append(f"iter-{ii:06d}--" + jj)
    return items


def _item_path(
    prefix,
    item,
):
    ret = Path(prefix)
    for ii in item.split(global_step_def_split):
        for jj in ii.split("--"):
            ret = ret / jj
    return ret


def _filter_if_complished(
    prefix,
    items,
):
    ret = []
    for tt in items:
        item_path = _item_path(prefix, tt)
        if not (item_path / "done").is_file():
            ret.append(tt)
        else:
            logging.info(f"{item_path} exists")
    return ret


def _filter_def_by_availability(
    defs,
):
    ret = []
    for dd in defs:
        splitted = dd.split(global_step_def_split)
        if len(splitted) != 3:
            raise RuntimeError(
                "the step definition should be in format "
                "stepkey/input_or_output/item_name.\n"
                "for example prep-run-dp-train/output/logs"
            )
        [subkey, io, name] = splitted
        if io in ["input"]:
            avail = (subkey in op_download_setting.keys()) and (
                name in op_download_setting[subkey].input_def.keys()
            )
        elif io in ["output"]:
            avail = (subkey in op_download_setting.keys()) and (
                name in op_download_setting[subkey].output_def.keys()
            )
        else:
            raise RuntimeError(f"unknown io style {io}")
        if not avail:
            logging.warning(f"cannot find download settings for {dd}")
        else:
            ret.append(dd)
    return ret


def _dl_step_item(
    step,
    item,
    prefix,
    ckpt,
):
    [step_key, io, name] = item.split(global_step_def_split)
    pref = _item_path(prefix, item)
    if io in ["input"]:
        target = step.inputs.artifacts[name]
    elif io in ["output"]:
        target = step.outputs.artifacts[name]
    else:
        raise RuntimeError("unknown io style {io}")
    try:
        download_artifact(
            target,
            path=pref,
            skip_exists=True,
        )
    except (NotImplementedError, FileNotFoundError):
        # NotImplementedError to be compatible with old versions of dflow
        logging.warning(f"cannot download item {pref}, it may be empty")
    if ckpt:
        pref.mkdir(parents=True, exist_ok=True)
        (pref / "done").touch()

    logging.info(f"{pref} downloaded")
