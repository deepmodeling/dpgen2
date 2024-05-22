import copy
import glob
import json
import logging
import os
import pickle
import re
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
    ConfSelectorFrames,
)
from dpgen2.exploration.task import (
    CustomizedLmpTemplateTaskGroup,
    ExplorationStage,
    ExplorationTask,
    LmpTemplateTaskGroup,
    NPTTaskGroup,
    caly_normalize,
    make_calypso_task_group_from_config,
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
    PrepCalyDPOptim,
    PrepCalyInput,
    PrepCalyModelDevi,
    PrepDPTrain,
    PrepLmp,
    RunCalyDPOptim,
    RunCalyModelDevi,
    RunDPTrain,
    RunLmp,
    SelectConfs,
)
from dpgen2.superop import (
    CalyEvoStep,
    ConcurrentLearningBlock,
    PrepRunCaly,
    PrepRunDPTrain,
    PrepRunFp,
    PrepRunLmp,
)
from dpgen2.utils import (
    BinaryFileInput,
    bohrium_config_from_dict,
    dump_object_to_file,
    get_subkey,
    load_object_from_file,
    matched_step_key,
    print_keys_in_nice_format,
    sort_slice_ops,
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
        )
    else:
        raise RuntimeError(f"unknown train_style {train_style}")
    if explore_style == "lmp":
        prep_run_explore_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            RunLmp,
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
    elif explore_style == "calypso":
        caly_evo_step_op = CalyEvoStep(
            "caly-evo-step",
            collect_run_caly=CollRunCaly,
            prep_dp_optim=PrepCalyDPOptim,
            run_dp_optim=RunCalyDPOptim,
            prep_config=prep_explore_config,
            run_config=run_explore_config,
            upload_python_packages=upload_python_packages,
        )
        prep_run_explore_op = PrepRunCaly(
            "prep-run-calypso",
            prep_caly_input_op=PrepCalyInput,
            caly_evo_step_op=caly_evo_step_op,
            prep_caly_model_devi_op=PrepCalyModelDevi,
            run_caly_model_devi_op=RunCalyModelDevi,
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

    if explore_style == "lmp":
        return make_lmp_naive_exploration_scheduler(config)
    elif explore_style == "calypso":
        return make_calypso_naive_exploration_scheduler(config)


def make_calypso_naive_exploration_scheduler(config):
    model_devi_jobs = config["explore"]["stages"]
    fp_task_max = config["fp"]["task_max"]
    max_numb_iter = config["explore"]["max_numb_iter"]
    fatal_at_max = config["explore"]["fatal_at_max"]
    convergence = config["explore"]["convergence"]
    output_nopbc = config["explore"]["output_nopbc"]
    scheduler = ExplorationScheduler()
    # report
    conv_style = convergence.pop("type")
    report = conv_styles[conv_style](**convergence)
    render = TrajRenderLammps(nopbc=output_nopbc)
    # selector
    selector = ConfSelectorFrames(
        render,
        report,
        fp_task_max,
    )

    for job_ in model_devi_jobs:
        if not isinstance(job_, list):
            job = [job_]
        else:
            job = job_
        # stage
        stage = ExplorationStage()
        for jj in job:
            jconf = caly_normalize(jj)
            # make task group
            tgroup = make_calypso_task_group_from_config(jconf)
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
    scheduler = ExplorationScheduler()
    # report
    conv_style = convergence.pop("type")
    report = conv_styles[conv_style](**convergence)
    render = TrajRenderLammps(nopbc=output_nopbc)
    # selector
    selector = ConfSelectorFrames(
        render,
        report,
        fp_task_max,
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


def make_finetune_step(
    config,
    prep_train_config,
    run_train_config,
    upload_python_packages,
    numb_models,
    template_script,
    train_config,
    init_models,
    init_data,
    iter_data,
    valid_data=None,
):
    finetune_optional_parameter = {
        "mixed_type": config["inputs"]["mixed_type"],
        "finetune_mode": "finetune",
    }

    finetune_op = PrepRunDPTrain(
        "finetune",
        PrepDPTrain,
        RunDPTrain,
        prep_config=prep_train_config,
        run_config=run_train_config,
        upload_python_packages=upload_python_packages,
        finetune=True,
        valid_data=valid_data,
    )
    finetune_step = Step(
        "finetune-step",
        template=finetune_op,
        parameters={
            "block_id": "finetune",
            "numb_models": numb_models,
            "template_script": template_script,
            "train_config": train_config,
            "run_optional_parameter": finetune_optional_parameter,
        },
        artifacts={
            "init_models": init_models,
            "init_data": init_data,
            "iter_data": iter_data,
        },
    )
    return finetune_step


def workflow_concurrent_learning(
    config: Dict,
) -> Tuple[Step, Optional[Step]]:
    default_config = config["default_step_config"]

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
    else:
        raise RuntimeError(f"unknown params, train_style: {train_style}")

    if upload_python_packages is not None and isinstance(upload_python_packages, str):
        upload_python_packages = [upload_python_packages]
    if upload_python_packages is not None:
        _upload_python_packages: List[os.PathLike] = [
            Path(ii) for ii in upload_python_packages
        ]
        upload_python_packages = _upload_python_packages

    valid_data = config["inputs"]["valid_data_sys"]
    if valid_data is not None:
        valid_data_prefix = config["inputs"]["valid_data_prefix"]
        valid_data = [valid_data] if isinstance(valid_data, str) else valid_data
        assert isinstance(valid_data, list)
        if valid_data_prefix is not None:
            valid_data = [os.path.join(valid_data_prefix, ii) for ii in valid_data]
        valid_data = [expand_sys_str(ii) for ii in valid_data]
        valid_data = upload_artifact(valid_data)
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
    )
    scheduler = make_naive_exploration_scheduler(config)

    type_map = config["inputs"]["type_map"]
    numb_models = config["train"]["numb_models"]
    template_script_ = config["train"]["template_script"]
    if isinstance(template_script_, list):
        template_script = [json.loads(Path(ii).read_text()) for ii in template_script_]
    else:
        template_script = json.loads(Path(template_script_).read_text())
    train_config = config["train"]["config"]
    explore_config = config["explore"]["config"]
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
        multi_init_data = config["inputs"]["multi_init_data"]
        init_data = []
        multi_init_data_idx = {}
        for k, v in multi_init_data.items():
            sys = v["sys"]
            sys = [sys] if isinstance(sys, str) else sys
            assert isinstance(sys, list)
            if v["prefix"] is not None:
                sys = [os.path.join(v["prefix"], ii) for ii in sys]
            sys = [expand_sys_str(ii) for ii in sys]
            istart = len(init_data)
            init_data += sys
            iend = len(init_data)
            multi_init_data_idx[k] = list(range(istart, iend))
        train_config["multitask"] = True
        train_config["head"] = head
        train_config["multi_init_data_idx"] = multi_init_data_idx
        explore_config["head"] = head
    else:
        init_data_prefix = config["inputs"]["init_data_prefix"]
        init_data = config["inputs"]["init_data_sys"]
        if init_data_prefix is not None:
            init_data = [os.path.join(init_data_prefix, ii) for ii in init_data]
        if isinstance(init_data, str):
            init_data = expand_sys_str(init_data)
    init_data = upload_artifact(init_data)
    iter_data = upload_artifact([])
    if init_models_paths is not None:
        init_models = upload_artifact(init_models_paths)
    else:
        init_models = None

    finetune_step = None
    optional_parameter = make_optional_parameter(
        config["inputs"]["mixed_type"],
    )

    if config["inputs"].get("do_finetune", False):
        finetune_step = make_finetune_step(
            config,
            prep_train_config,
            run_train_config,
            upload_python_packages,
            numb_models,
            template_script,
            train_config,
            init_models,
            init_data,
            iter_data,
            valid_data=valid_data,
        )

        init_models = finetune_step.outputs.artifacts["models"]
        template_script = finetune_step.outputs.parameters["template_script"]

        optional_parameter = make_optional_parameter(
            config["inputs"]["mixed_type"],
            finetune_mode="train-init",
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
    return dpgen_step, finetune_step


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

    dpgen_step, finetune_step = workflow_concurrent_learning(
        wf_config,
    )

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
        # the modify-train-script step will be added as reuse step.
        # the following hack is not needed anymore.
        # wf_config["inputs"]["do_finetune"] = False
        # finetune will not be done again if the old process is reused.

    wf = Workflow(name=wf_config["name"])

    if wf_config["inputs"].get("do_finetune", False):
        assert finetune_step is not None
        wf.add(finetune_step)

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


def successful_step_keys(wf):
    all_step_keys = []
    steps = wf.query_step()
    # For reused steps whose startedAt are identical, sort them by key
    steps.sort(key=lambda x: "%s-%s" % (x.startedAt, x.key))
    for step in steps:
        if step.key is not None and step.phase == "Succeeded":
            all_step_keys.append(step.key)
    return all_step_keys


def get_superop(key):
    if "prep-train" in key:
        return key.replace("prep-train", "prep-run-train")
    elif "run-train-" in key:
        return re.sub("run-train-[0-9]*", "prep-run-train", key)
    elif "prep-lmp" in key:
        return key.replace("prep-lmp", "prep-run-explore")
    elif "run-lmp-" in key:
        return re.sub("run-lmp-[0-9]*", "prep-run-explore", key)
    elif "prep-fp" in key:
        return key.replace("prep-fp", "prep-run-fp")
    elif "run-fp-" in key:
        return re.sub("run-fp-[0-9]*", "prep-run-fp", key)
    elif "prep-caly-input" in key:
        return key.replace("prep-caly-input", "prep-run-explore")
    elif "collect-run-calypso-" in key:
        return re.sub("collect-run-calypso-[0-9]*-[0-9]*", "prep-run-explore", key)
    elif "prep-dp-optim-" in key:
        return re.sub("prep-dp-optim-[0-9]*-[0-9]*", "prep-run-explore", key)
    elif "run-dp-optim-" in key:
        return re.sub("run-dp-optim-[0-9]*-[0-9]*-[0-9]*", "prep-run-explore", key)
    elif "prep-caly-model-devi" in key:
        return key.replace("prep-caly-model-devi", "prep-run-explore")
    elif "run-caly-model-devi" in key:
        return re.sub("run-caly-model-devi-[0-9]*", "prep-run-explore", key)
    return None


def fold_keys(all_step_keys):
    folded_keys = {}
    for key in all_step_keys:
        is_superop = False
        for superop in ["prep-run-train", "prep-run-explore", "prep-run-fp"]:
            if superop in key:
                if key not in folded_keys:
                    folded_keys[key] = []
                is_superop = True
                break
        if is_superop:
            continue
        superop = get_superop(key)
        # if its super OP is succeeded, fold it into its super OP
        if superop is not None and superop in all_step_keys:
            if superop not in folded_keys:
                folded_keys[superop] = []
            folded_keys[superop].append(key)
        else:
            folded_keys[key] = [key]
    for k, v in folded_keys.items():
        if v == []:
            folded_keys[k] = [k]
    return folded_keys


def get_resubmit_keys(
    wf,
):
    all_step_keys = successful_step_keys(wf)
    all_step_keys = matched_step_key(
        all_step_keys,
        [
            "prep-run-train",
            "prep-train",
            "run-train",
            "modify-train-script",
            "prep-caly-input",
            "collect-run-calypso",
            "prep-dp-optim",
            "run-dp-optim",
            "prep-caly-model-devi",
            "run-caly-model-devi",
            "prep-run-explore",
            "prep-lmp",
            "run-lmp",
            "select-confs",
            "prep-run-fp",
            "prep-fp",
            "run-fp",
            "collect-data",
            "scheduler",
            "id",
        ],
    )
    all_step_keys = sort_slice_ops(
        all_step_keys,
        ["run-train", "run-lmp", "run-fp"],
    )
    folded_keys = fold_keys(all_step_keys)
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
    all_step_keys = sum(folded_keys.values(), [])

    if list_steps:
        prt_str = print_keys_in_nice_format(
            all_step_keys,
            ["run-train", "run-lmp", "run-fp"],
        )
        print(prt_str)

    if reuse is None:
        return None
    reuse_idx = expand_idx(reuse)
    reused_keys = [all_step_keys[ii] for ii in reuse_idx]
    if fold:
        reused_folded_keys = {}
        for key in reused_keys:
            superop = get_superop(key)
            if superop is not None:
                if superop not in reused_folded_keys:
                    reused_folded_keys[superop] = []
                reused_folded_keys[superop].append(key)
            else:
                reused_folded_keys[key] = [key]
        for k, v in reused_folded_keys.items():
            # reuse the super OP iif all steps within it are reused
            if v != [k] and k in folded_keys and set(v) == set(folded_keys[k]):
                reused_folded_keys[k] = [k]
        reused_keys = sum(reused_folded_keys.values(), [])
    reuse_step = old_wf.query_step(key=reused_keys)
    # For reused steps whose startedAt are identical, sort them by key
    reuse_step.sort(key=lambda x: "%s-%s" % (x.startedAt, x.key))

    wf = submit_concurrent_learning(
        wf_config,
        reuse_step=reuse_step,
        replace_scheduler=replace_scheduler,
    )

    return wf
