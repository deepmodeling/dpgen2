import copy
import glob
import json
import logging
import os
import pickle
import re
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import dpdata
from dflow import (
    ArgoStep,
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    S3Artifact,
    Step,
    Steps,
    Workflow,
    argo_range,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    FatalError,
    OPIOSign,
    PythonOPTemplate,
    TransientError,
    upload_packages,
)

from dpgen2.conf import (
    conf_styles,
)
from dpgen2.constants import (
    default_host,
    default_image,
)
from dpgen2.entrypoint.args import normalize as normalize_args
from dpgen2.entrypoint.common import (
    expand_idx,
    expand_sys_str,
    global_config_workflow,
)
from dpgen2.exploration.render import (
    TrajRenderLammps,
)
from dpgen2.exploration.report import (
    ExplorationReportTrustLevelsRandom,
    conv_styles,
)
from dpgen2.exploration.scheduler import (
    ConvergenceCheckStageScheduler,
    ExplorationScheduler,
)
from dpgen2.exploration.selector import (
    ConfFilters,
    ConfSelectorFrames,
    conf_filter_styles,
)
from dpgen2.exploration.task import (
    CustomizedLmpTemplateTaskGroup,
    ExplorationStage,
    ExplorationTask,
    LmpTemplateTaskGroup,
    NPTTaskGroup,
    caly_normalize,
    diffcsp_normalize,
    make_calypso_task_group_from_config,
    make_diffcsp_task_group_from_config,
    make_lmp_task_group_from_config,
    normalize_lmp_task_group_config,
)
from dpgen2.flow import (
    ConcurrentLearning,
)
from dpgen2.fp import (
    fp_styles,
)
from dpgen2.op import (
    CollectData,
    CollRunCaly,
    DiffCSPGen,
    PrepCalyDPOptim,
    PrepCalyInput,
    PrepCalyModelDevi,
    PrepDPTrain,
    PrepLmp,
    PrepRelax,
    RunCalyDPOptim,
    RunCalyModelDevi,
    RunDPTrain,
    RunLmp,
    RunLmpHDF5,
    RunNvNMD,
    RunNvNMDTrain,
    RunRelax,
    RunRelaxHDF5,
    SelectConfs,
)
from dpgen2.op.caly_evo_step_merge import (
    CalyEvoStepMerge,
)
from dpgen2.superop import (
    ConcurrentLearningBlock,
    PrepRunCaly,
    PrepRunDiffCSP,
    PrepRunDPTrain,
    PrepRunFp,
    PrepRunLmp,
)
from dpgen2.superop.caly_evo_step import (
    CalyEvoStep,
)
from dpgen2.utils import (
    BinaryFileInput,
    bohrium_config_from_dict,
    dump_object_to_file,
    get_artifact_from_uri,
    get_subkey,
    load_object_from_file,
    matched_step_key,
    print_keys_in_nice_format,
    sort_slice_ops,
    upload_artifact_and_print_uri,
    workflow_config_from_dict,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        }
    }
)


def make_concurrent_learning_op(
    train_style: str = "dp",
    explore_style: str = "lmp",
    fp_style: str = "vasp",
    prep_train_config: dict = default_config,
    run_train_config: dict = default_config,
    prep_explore_config: dict = default_config,
    run_explore_config: dict = default_config,
    prep_fp_config: dict = default_config,
    run_fp_config: dict = default_config,
    select_confs_config: dict = default_config,
    collect_data_config: dict = default_config,
    cl_step_config: dict = default_config,
    upload_python_packages: Optional[List[os.PathLike]] = None,
    valid_data: Optional[S3Artifact] = None,
    train_optional_files: Optional[List[str]] = None,
    explore_config: Optional[dict] = None,
):
    if train_style in ("dp", "dp-dist"):
        prep_run_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            PrepDPTrain,
            RunDPTrain,
            prep_config=prep_train_config,
            run_config=run_train_config,
            upload_python_packages=upload_python_packages,
            valid_data=valid_data,
            optional_files=train_optional_files,
        )
    elif train_style == "dp-nvnmd":
        prep_run_train_op = PrepRunDPTrain(
            "prep-run-nvnmd-train",
            PrepDPTrain,
            RunNvNMDTrain,
            prep_config=prep_train_config,
            run_config=run_train_config,
            upload_python_packages=upload_python_packages,
            valid_data=valid_data,
            optional_files=train_optional_files,
        )
    else:
        raise RuntimeError(f"unknown train_style {train_style}")
    if explore_style == "lmp":
        prep_run_explore_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            RunLmpHDF5 if explore_config["use_hdf5"] else RunLmp,  # type: ignore
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
    elif "lmp-nvnmd" in explore_style:
        prep_run_explore_op = PrepRunLmp(
            "prep-run-nvnmd",
            PrepLmp,
            RunNvNMD,
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
    elif "calypso" in explore_style:
        expl_mode = explore_style.split(":")[-1] if ":" in explore_style else "default"
        if expl_mode == "merge":
            caly_evo_step_op = CalyEvoStepMerge(
                name="caly-evo-step",
                collect_run_caly=CollRunCaly,
                prep_dp_optim=PrepCalyDPOptim,
                run_dp_optim=RunCalyDPOptim,
                expl_mode=expl_mode,
                prep_config=prep_explore_config,
                run_config=run_explore_config,
                upload_python_packages=None,
            )
        elif expl_mode == "default":
            caly_evo_step_op = CalyEvoStep(
                name="caly-evo-step",
                collect_run_caly=CollRunCaly,
                prep_dp_optim=PrepCalyDPOptim,
                run_dp_optim=RunCalyDPOptim,
                expl_mode=expl_mode,
                prep_config=prep_explore_config,
                run_config=run_explore_config,
                upload_python_packages=upload_python_packages,
            )
        else:
            raise KeyError(
                f"Unknown key: {explore_style}, support `calypso:default` and `calypso:merge`."
            )
        prep_run_explore_op = PrepRunCaly(
            "prep-run-calypso",
            prep_caly_input_op=PrepCalyInput,
            caly_evo_step_op=caly_evo_step_op,
            prep_caly_model_devi_op=PrepCalyModelDevi,
            run_caly_model_devi_op=RunCalyModelDevi,
            expl_mode=expl_mode,
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
    elif explore_style == "diffcsp":
        prep_run_explore_op = PrepRunDiffCSP(
            "prep-run-diffcsp",
            DiffCSPGen,
            PrepRelax,
            RunRelaxHDF5 if explore_config["use_hdf5"] else RunRelax,  # type: ignore
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
    else:
        raise RuntimeError(f"unknown explore_style {explore_style}")

    if fp_style in fp_styles.keys():
        prep_run_fp_op = PrepRunFp(
            "prep-run-fp",
            fp_styles[fp_style]["prep"],
            fp_styles[fp_style]["run"],
            prep_config=prep_fp_config,
            run_config=run_fp_config,
            upload_python_packages=upload_python_packages,
        )
    else:
        raise RuntimeError(f"unknown fp_style {fp_style}")

    # ConcurrentLearningBlock
    block_cl_op = ConcurrentLearningBlock(
        "concurrent-learning-block",
        prep_run_train_op,
        prep_run_explore_op,
        SelectConfs,
        prep_run_fp_op,
        CollectData,
        select_confs_config=select_confs_config,
        collect_data_config=collect_data_config,
        upload_python_packages=upload_python_packages,
    )
    # dpgen
    dpgen_op = ConcurrentLearning(
        "concurrent-learning",
        block_cl_op,
        upload_python_packages=upload_python_packages,
        step_config=cl_step_config,
    )

    return dpgen_op


def make_naive_exploration_scheduler(
    config,
):
    # use npt task group
    explore_style = config["explore"]["type"]

    if explore_style in ("lmp", "lmp-nvnmd"):
        return make_lmp_naive_exploration_scheduler(config)
    elif "calypso" in explore_style or explore_style == "diffcsp":
        return make_naive_exploration_scheduler_without_conf(config, explore_style)
    else:
        raise KeyError(f"Unknown explore_style `{explore_style}`")


def get_conf_filters(config):
    conf_filters = None
    if len(config) > 0:
        conf_filters = ConfFilters()
        for c in config:
            c = deepcopy(c)
            conf_filter = conf_filter_styles[c.pop("type")](**c)
            conf_filters.add(conf_filter)
    return conf_filters


def make_naive_exploration_scheduler_without_conf(config, explore_style):
    model_devi_jobs = config["explore"]["stages"]
    fp_task_max = config["fp"]["task_max"]
    max_numb_iter = config["explore"]["max_numb_iter"]
    fatal_at_max = config["explore"]["fatal_at_max"]
    convergence = config["explore"]["convergence"]
    output_nopbc = config["explore"]["output_nopbc"]
    conf_filters = get_conf_filters(config["explore"]["filters"])
    scheduler = ExplorationScheduler()
    # report
    conv_style = convergence.pop("type")
    report = conv_styles[conv_style](**convergence)
    # trajectory render, the format of the output trajs are assumed to be lammps/dump
    render = TrajRenderLammps(nopbc=output_nopbc)
    # selector
    selector = ConfSelectorFrames(
        render,
        report,
        fp_task_max,
        conf_filters,
    )

    for job_ in model_devi_jobs:
        if not isinstance(job_, list):
            job = [job_]
        else:
            job = job_
        # stage
        stage = ExplorationStage()
        for jj in job:
            if "calypso" in explore_style:
                jconf = caly_normalize(jj)
                # make task group
                tgroup = make_calypso_task_group_from_config(jconf)
            elif explore_style == "diffcsp":
                jconf = diffcsp_normalize(jj)
                # make task group
                tgroup = make_diffcsp_task_group_from_config(jconf)
            else:
                raise KeyError(f"Unknown explore_style `{explore_style}`")
            # add the list to task group
            tasks = tgroup.make_task()
            stage.add_task_group(tasks)
        # stage_scheduler
        stage_scheduler = ConvergenceCheckStageScheduler(
            stage,
            selector,
            max_numb_iter=max_numb_iter,
            fatal_at_max=fatal_at_max,
        )
        # scheduler
        scheduler.add_stage_scheduler(stage_scheduler)

    return scheduler


def make_lmp_naive_exploration_scheduler(config):
    model_devi_jobs = config["explore"]["stages"]
    sys_configs = config["explore"]["configurations"]
    mass_map = config["inputs"]["mass_map"]
    type_map = config["inputs"]["type_map"]
    numb_models = config["train"]["numb_models"]
    fp_task_max = config["fp"]["task_max"]
    max_numb_iter = config["explore"]["max_numb_iter"]
    fatal_at_max = config["explore"]["fatal_at_max"]
    convergence = config["explore"]["convergence"]
    output_nopbc = config["explore"]["output_nopbc"]
    conf_filters = get_conf_filters(config["explore"]["filters"])
    use_ele_temp = config["inputs"]["use_ele_temp"]
    config["explore"]["type"]
    scheduler = ExplorationScheduler()
    # report
    conv_style = convergence.pop("type")
    report = conv_styles[conv_style](**convergence)
    render = TrajRenderLammps(nopbc=output_nopbc, use_ele_temp=use_ele_temp)
    # selector
    selector = ConfSelectorFrames(
        render,
        report,
        fp_task_max,
        conf_filters,
    )

    sys_configs_lmp = []
    for sys_config in sys_configs:
        conf_style = sys_config.pop("type")
        generator = conf_styles[conf_style](**sys_config)
        sys_configs_lmp.append(generator.get_file_content(type_map))

    for job_ in model_devi_jobs:
        if not isinstance(job_, list):
            job = [job_]
        else:
            job = job_
        # stage
        stage = ExplorationStage()
        for jj in job:
            jconf = normalize_lmp_task_group_config(jj)
            n_sample = jconf.pop("n_sample")
            ##  ignore the expansion of sys_idx
            # get all file names of md initial configurations
            sys_idx = jconf.pop("conf_idx")
            conf_list = []
            for ii in sys_idx:
                conf_list += sys_configs_lmp[ii]
            # make task group
            tgroup = make_lmp_task_group_from_config(numb_models, mass_map, jconf)
            # add the list to task group
            tgroup.set_conf(
                conf_list,
                n_sample=n_sample,
                random_sample=True,
            )
            tasks = tgroup.make_task()
            stage.add_task_group(tasks)
        # stage_scheduler
        stage_scheduler = ConvergenceCheckStageScheduler(
            stage,
            selector,
            max_numb_iter=max_numb_iter,
            fatal_at_max=fatal_at_max,
        )
        # scheduler
        scheduler.add_stage_scheduler(stage_scheduler)

    return scheduler


def get_kspacing_kgamma_from_incar(
    fname,
):
    with open(fname) as fp:
        lines = fp.readlines()
    ks = None
    kg = None
    for ii in lines:
        if "KSPACING" in ii:
            ks = float(ii.split("=")[1])
        if "KGAMMA" in ii:
            if "T" in ii.split("=")[1]:
                kg = True
            elif "F" in ii.split("=")[1]:
                kg = False
            else:
                raise RuntimeError(f"invalid kgamma value {ii.split('=')[1]}")
    assert ks is not None and kg is not None
    return ks, kg


def make_optional_parameter(
    mixed_type=False,
    finetune_mode="no",
):
    return {"data_mixed_type": mixed_type, "finetune_mode": finetune_mode}


def get_systems_from_data(data, data_prefix=None):
    data = [data] if isinstance(data, str) else data
    assert isinstance(data, list)
    if data_prefix is not None:
        data = [os.path.join(data_prefix, ii) for ii in data]
    data = sum([expand_sys_str(ii) for ii in data], [])
    return data


def workflow_concurrent_learning(
    config: Dict,
) -> Step:
    default_config = config["default_step_config"]

    train_config = config["train"]["config"]
    explore_config = config["explore"]["config"]
    train_style = config["train"]["type"]
    explore_style = config["explore"]["type"]
    fp_style = config["fp"]["type"]
    prep_train_config = config["step_configs"]["prep_train_config"]
    run_train_config = config["step_configs"]["run_train_config"]
    prep_explore_config = config["step_configs"]["prep_explore_config"]
    run_explore_config = config["step_configs"]["run_explore_config"]
    prep_fp_config = config["step_configs"]["prep_fp_config"]
    run_fp_config = config["step_configs"]["run_fp_config"]
    select_confs_config = config["step_configs"]["select_confs_config"]
    collect_data_config = config["step_configs"]["collect_data_config"]
    cl_step_config = config["step_configs"]["cl_step_config"]
    upload_python_packages = config.get("upload_python_packages", None)
    train_optional_files = config["train"].get("optional_files", None)

    if train_style == "dp":
        init_models_paths = config["train"].get("init_models_paths", None)
        numb_models = config["train"]["numb_models"]
        if init_models_paths is not None and len(init_models_paths) != numb_models:
            raise RuntimeError(
                f"{len(init_models_paths)} init models provided, which does "
                "not match numb_models={numb_models}"
            )
    elif train_style == "dp-dist":
        init_models_paths = (
            [config["train"]["student_model_path"]]
            if "student_model_path" in config["train"]
            else None
        )
        config["train"]["numb_models"] = 1

    elif train_style == "dp-nvnmd":
        init_models_paths = config["train"].get("init_models_paths", None)
        numb_models = config["train"]["numb_models"]
        if init_models_paths is not None and len(init_models_paths) != numb_models:
            raise RuntimeError(
                f"{len(init_models_paths)} init models provided, which does "
                "not match numb_models={numb_models}"
            )

    else:
        raise RuntimeError(f"unknown params, train_style: {train_style}")

    if upload_python_packages is not None and isinstance(upload_python_packages, str):
        upload_python_packages = [upload_python_packages]
    if upload_python_packages is not None:
        _upload_python_packages: List[os.PathLike] = [
            Path(ii) for ii in upload_python_packages
        ]
        upload_python_packages = _upload_python_packages

    multitask = config["inputs"]["multitask"]
    valid_data = None
    if multitask:
        if config["inputs"]["multi_valid_data_uri"] is not None:
            valid_data = get_artifact_from_uri(config["inputs"]["multi_valid_data_uri"])
        elif config["inputs"]["multi_valid_data"] is not None:
            multi_valid_data = config["inputs"]["multi_valid_data"]
            valid_data = {}
            for k, v in multi_valid_data.items():
                sys = v["sys"]
                sys = get_systems_from_data(sys, v.get("prefix", None))
                valid_data[k] = sys
            valid_data = upload_artifact_and_print_uri(valid_data, "multi_valid_data")
    else:
        if config["inputs"]["valid_data_uri"] is not None:
            valid_data = get_artifact_from_uri(config["inputs"]["valid_data_uri"])
        elif config["inputs"]["valid_data_sys"] is not None:
            valid_data_prefix = config["inputs"]["valid_data_prefix"]
            valid_data = config["inputs"]["valid_data_sys"]
            valid_data = get_systems_from_data(valid_data, valid_data_prefix)
            valid_data = upload_artifact_and_print_uri(valid_data, "valid_data")

    concurrent_learning_op = make_concurrent_learning_op(
        train_style,
        explore_style,
        fp_style,
        prep_train_config=prep_train_config,
        run_train_config=run_train_config,
        prep_explore_config=prep_explore_config,
        run_explore_config=run_explore_config,
        prep_fp_config=prep_fp_config,
        run_fp_config=run_fp_config,
        select_confs_config=select_confs_config,
        collect_data_config=collect_data_config,
        cl_step_config=cl_step_config,
        upload_python_packages=upload_python_packages,
        valid_data=valid_data,
        train_optional_files=train_optional_files,
        explore_config=explore_config,
    )
    scheduler = make_naive_exploration_scheduler(config)

    type_map = config["inputs"]["type_map"]
    numb_models = config["train"]["numb_models"]
    template_script_ = config["train"]["template_script"]
    if isinstance(template_script_, list):
        template_script = [json.loads(Path(ii).read_text()) for ii in template_script_]
    else:
        template_script = json.loads(Path(template_script_).read_text())

    if (
        "teacher_model_path" in explore_config
        and explore_config["teacher_model_path"] is not None
    ):
        assert os.path.exists(
            explore_config["teacher_model_path"]
        ), f"No such file: {explore_config['teacher_model_path']}"
        explore_config["teacher_model_path"] = BinaryFileInput(
            explore_config["teacher_model_path"]
        )

    fp_config = {}
    fp_inputs_config = config["fp"]["inputs_config"]
    fp_inputs = fp_styles[fp_style]["inputs"](**fp_inputs_config)

    fp_config["inputs"] = fp_inputs
    fp_config["run"] = config["fp"]["run_config"]
    fp_config["extra_output_files"] = config["fp"]["extra_output_files"]
    if fp_style == "deepmd":
        assert (
            "teacher_model_path" in fp_config["run"]
        ), f"Cannot find 'teacher_model_path' in config['fp']['run_config'] when fp_style == 'deepmd'"
        assert os.path.exists(
            fp_config["run"]["teacher_model_path"]
        ), f"No such file: {fp_config['run']['teacher_model_path']}"
        fp_config["run"]["teacher_model_path"] = BinaryFileInput(
            fp_config["run"]["teacher_model_path"]
        )

    multitask = config["inputs"]["multitask"]
    if multitask:
        head = config["inputs"]["head"]
        if config["inputs"]["multi_init_data_uri"] is not None:
            init_data = get_artifact_from_uri(config["inputs"]["multi_init_data_uri"])
        else:
            multi_init_data = config["inputs"]["multi_init_data"]
            init_data = {}
            for k, v in multi_init_data.items():
                sys = v["sys"]
                sys = get_systems_from_data(sys, v.get("prefix", None))
                init_data[k] = sys
            init_data = upload_artifact_and_print_uri(init_data, "multi_init_data")
        train_config["multitask"] = True
        train_config["head"] = head
        explore_config["model_frozen_head"] = head
    else:
        if config["inputs"]["init_data_uri"] is not None:
            init_data = get_artifact_from_uri(config["inputs"]["init_data_uri"])
        else:
            init_data_prefix = config["inputs"]["init_data_prefix"]
            init_data = config["inputs"]["init_data_sys"]
            init_data = get_systems_from_data(init_data, init_data_prefix)
            init_data = upload_artifact_and_print_uri(init_data, "init_data")
    iter_data = upload_artifact([])
    if train_style == "dp" and config["train"]["init_models_uri"] is not None:
        init_models = get_artifact_from_uri(config["train"]["init_models_uri"])
    elif train_style == "dp-dist" and config["train"]["student_model_uri"] is not None:
        init_models = get_artifact_from_uri(config["train"]["student_model_uri"])
    elif init_models_paths is not None:
        init_models = upload_artifact_and_print_uri(init_models_paths, "init_models")
    else:
        init_models = None

    if config["inputs"]["use_ele_temp"]:
        explore_config["use_ele_temp"] = config["inputs"]["use_ele_temp"]

    optional_parameter = make_optional_parameter(
        config["inputs"]["mixed_type"],
    )

    if config["inputs"].get("do_finetune", False):
        if train_config["init_model_policy"] != "yes":
            logging.warning("In finetune mode, init_model_policy is forced to be 'yes'")
            train_config["init_model_policy"] = "yes"
        optional_parameter = make_optional_parameter(
            config["inputs"]["mixed_type"],
            finetune_mode="finetune",
        )

    # here the scheduler is passed as input parameter to the concurrent_learning_op
    dpgen_step = Step(
        "dpgen-step",
        template=concurrent_learning_op,
        parameters={
            "type_map": type_map,
            "numb_models": numb_models,
            "template_script": template_script,
            "train_config": train_config,
            "explore_config": explore_config,
            "fp_config": fp_config,
            "exploration_scheduler": scheduler,
            "optional_parameter": optional_parameter,
        },
        artifacts={
            "init_models": init_models,
            "init_data": init_data,
            "iter_data": iter_data,
        },
    )
    return dpgen_step


def get_scheduler_ids(
    reuse_step,
):
    scheduler_ids = []
    for idx, ii in enumerate(reuse_step):
        if get_subkey(ii.key, 1) == "scheduler":
            scheduler_ids.append(idx)
    scheduler_keys = [reuse_step[ii].key for ii in scheduler_ids]
    assert (
        sorted(scheduler_keys) == scheduler_keys
    ), "The scheduler keys are not properly sorted"

    if len(scheduler_ids) == 0:
        logging.warning(
            "No scheduler found in the workflow, " "does not do any replacement."
        )
    return scheduler_ids


def update_reuse_step_scheduler(
    reuse_step,
    scheduler_new,
):
    scheduler_ids = get_scheduler_ids(reuse_step)
    if len(scheduler_ids) == 0:
        return reuse_step

    # do replacement
    reuse_step[scheduler_ids[-1]].modify_output_parameter(
        "exploration_scheduler", scheduler_new
    )

    return reuse_step


def copy_scheduler_plans(
    scheduler_new,
    scheduler_old,
):
    if len(scheduler_old.stage_schedulers) == 0:
        return scheduler_new
    if len(scheduler_new.stage_schedulers) < len(scheduler_old.stage_schedulers):
        raise RuntimeError(
            "The new scheduler has less stages than the old scheduler, "
            "scheduler copy is not supported."
        )
    # the scheduler_old is planned. minic the init call of the scheduler
    if scheduler_old.get_iteration() > -1:
        scheduler_new.plan_next_iteration()
    for ii in range(len(scheduler_old.stage_schedulers)):
        old_stage = scheduler_old.stage_schedulers[ii]
        old_reports = old_stage.get_reports()
        if old_stage.next_iteration() > 0:
            if ii != scheduler_new.get_stage():
                raise RuntimeError(
                    f"The stage {scheduler_new.get_stage()} of the new "
                    f"scheduler does not match"
                    f"the stage {ii} of the old scheduler. "
                    f"scheduler, which should not happen"
                )
            for report in old_reports:
                scheduler_new.plan_next_iteration(report)
            if old_stage.complete() and (
                not scheduler_new.stage_schedulers[ii].complete()
            ):
                scheduler_new.force_stage_complete()
        else:
            break
    return scheduler_new


def submit_concurrent_learning(
    wf_config,
    reuse_step: Optional[List[ArgoStep]] = None,
    replace_scheduler: bool = False,
    no_submission: bool = False,
):
    # normalize args
    wf_config = normalize_args(wf_config)

    global_config_workflow(wf_config)

    dpgen_step = workflow_concurrent_learning(wf_config)

    if reuse_step is not None and replace_scheduler:
        scheduler_new = copy.deepcopy(
            dpgen_step.inputs.parameters["exploration_scheduler"].value
        )
        idx_old = get_scheduler_ids(reuse_step)[-1]
        scheduler_old = (
            reuse_step[idx_old].inputs.parameters["exploration_scheduler"].value
        )
        scheduler_new = copy_scheduler_plans(scheduler_new, scheduler_old)
        exploration_report = (
            reuse_step[idx_old].inputs.parameters["exploration_report"].value
        )
        # plan next
        # hack! trajs is set to None...
        conv, expl_task_grp, selector = scheduler_new.plan_next_iteration(
            exploration_report, trajs=None
        )
        # update output of the scheduler step
        reuse_step[idx_old].modify_output_parameter(
            "converged",
            conv,
        )
        reuse_step[idx_old].modify_output_parameter(
            "exploration_scheduler",
            scheduler_new,
        )
        reuse_step[idx_old].modify_output_parameter(
            "expl_task_grp",
            expl_task_grp,
        )
        reuse_step[idx_old].modify_output_parameter(
            "conf_selector",
            selector,
        )

    wf = Workflow(name=wf_config["name"], parallelism=wf_config["parallelism"])

    wf.add(dpgen_step)

    # for debug purpose, we may not really submit the wf
    if not no_submission:
        wf.submit(reuse_step=reuse_step)

    return wf


def print_list_steps(
    steps,
):
    ret = []
    for idx, ii in enumerate(steps):
        ret.append(f"{idx:8d}    {ii}")
    return "\n".join(ret)


def get_resubmit_keys(
    wf,
):
    wf_info = wf.query()
    all_steps = [
        step
        for step in wf_info.get_step(sort_by_generation=True)
        if step.key is not None
    ]
    super_keys = ["prep-run-train", "prep-run-explore", "prep-run-fp"]
    other_keys = [
        "select-confs",
        "collect-data",
        "scheduler",
        "id",
    ]

    folded_keys = {}
    for step in all_steps:
        if len(matched_step_key([step.key], super_keys)) > 0:
            sub_steps = wf_info.get_step(parent_id=step.id, sort_by_generation=True)
            sub_keys = [
                step.key
                for step in sub_steps
                if step.key is not None and step.phase == "Succeeded"
            ]
            sub_keys = sort_slice_ops(
                sub_keys,
                ["run-train", "run-lmp", "run-fp", "diffcsp-gen", "run-relax"],
            )
            if step.phase == "Succeeded":
                folded_keys[step.key] = sub_keys
            else:
                for key in sub_keys:
                    folded_keys[key] = [key]
        elif len(matched_step_key([step.key], other_keys)) > 0:
            folded_keys[step.key] = [step.key]
    return folded_keys


def resubmit_concurrent_learning(
    wf_config,
    wfid,
    list_steps=False,
    reuse=None,
    replace_scheduler=False,
    fold=False,
):
    wf_config = normalize_args(wf_config)

    global_config_workflow(wf_config)

    old_wf = Workflow(id=wfid)
    folded_keys = get_resubmit_keys(old_wf)
    all_step_keys = []
    super_keys = {}
    for super_key, keys in folded_keys.items():
        all_step_keys += keys
        for key in keys:
            super_keys[key] = super_key

    if list_steps:
        prt_str = print_keys_in_nice_format(
            all_step_keys,
            ["run-train", "run-lmp", "run-fp", "diffcsp-gen", "run-relax"],
        )
        print(prt_str)

    if reuse is None:
        return None
    reuse_idx = expand_idx(reuse)
    reused_keys = [all_step_keys[ii] for ii in reuse_idx]
    if fold:
        reused_folded_keys = {}
        for key in reused_keys:
            super_key = super_keys[key]
            if super_key not in reused_folded_keys:
                reused_folded_keys[super_key] = []
            reused_folded_keys[super_key].append(key)
        for k, v in reused_folded_keys.items():
            # reuse the super OP iif all steps within it are reused
            if set(v) == set(folded_keys[k]):
                reused_folded_keys[k] = [k]
        reused_keys = sum(reused_folded_keys.values(), [])
    reuse_step = old_wf.query_step(key=reused_keys, sort_by_generation=True)

    wf = submit_concurrent_learning(
        wf_config,
        reuse_step=reuse_step,
        replace_scheduler=replace_scheduler,
    )

    return wf
