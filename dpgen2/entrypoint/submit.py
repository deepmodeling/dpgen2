import glob, dpdata, os, pickle
from pathlib import Path
from dflow import (
    InputParameter,
    OutputParameter,
    Inputs,
    InputArtifact,
    Outputs,
    OutputArtifact,
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
    S3Artifact,
    argo_range,
)
from dflow.python import (
    PythonOPTemplate,
    OP,
    OPIO,
    OPIOSign,
    Artifact,
    upload_packages,
    FatalError,
    TransientError,
)

from dpgen2.op import (
    PrepDPTrain,
    RunDPTrain,
    PrepLmp,
    RunLmp,
    SelectConfs,
    CollectData,
)
from dpgen2.superop import (
    PrepRunDPTrain,
    PrepRunLmp,
    PrepRunFp,
    ConcurrentLearningBlock,
)
from dpgen2.flow import (
    ConcurrentLearning,
)
from dpgen2.fp import (
    fp_styles,
)
from dpgen2.conf import (
    conf_styles,
)
from dpgen2.exploration.scheduler import (
    ExplorationScheduler,
    ConvergenceCheckStageScheduler,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTask,
    NPTTaskGroup,
    LmpTemplateTaskGroup,
    make_task_group_from_config,
)
from dpgen2.exploration.selector import (
    ConfSelectorLammpsFrames,
    TrustLevel,
)
from dpgen2.constants import (
    default_image,
    default_host,
)
from dpgen2.utils import (
    dump_object_to_file,
    load_object_from_file,
    sort_slice_ops,
    print_keys_in_nice_format,
    workflow_config_from_dict,
    matched_step_key,
    bohrium_config_from_dict,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
from dpgen2.entrypoint.common import (
    global_config_workflow,
    expand_sys_str,
    expand_idx,
)
from dpgen2.entrypoint.args import (
    normalize as normalize_args,
)
from typing import (
    Union, List, Dict, Optional,
)

default_config = normalize_step_dict(
    {
        "template_config" : {
            "image" : default_image,
        }
    }
)


def make_concurrent_learning_op (
        train_style : str = 'dp',
        explore_style : str = 'lmp',
        fp_style : str = 'vasp',
        prep_train_config : dict = default_config,
        run_train_config : dict = default_config,
        prep_explore_config : dict = default_config,
        run_explore_config : dict = default_config,
        prep_fp_config : dict = default_config,
        run_fp_config : dict = default_config,
        select_confs_config : dict = default_config,
        collect_data_config : dict = default_config,
        cl_step_config : dict = default_config,
        upload_python_packages : Optional[List[os.PathLike]] = None,
):
    if train_style == 'dp':
        prep_run_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            PrepDPTrain,
            RunDPTrain,
            prep_config = prep_train_config,
            run_config = run_train_config,
            upload_python_packages = upload_python_packages,
        )
    else:
        raise RuntimeError(f'unknown train_style {train_style}')
    if explore_style == 'lmp':
        prep_run_explore_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            RunLmp,
            prep_config = prep_explore_config,
            run_config = run_explore_config,
            upload_python_packages = upload_python_packages,
        )
    else:
        raise RuntimeError(f'unknown explore_style {explore_style}')

    if fp_style in fp_styles.keys():        
        prep_run_fp_op = PrepRunFp(
            f"prep-run-fp",
            fp_styles[fp_style]['prep'],
            fp_styles[fp_style]['run'],
            prep_config = prep_fp_config,
            run_config = run_fp_config,
            upload_python_packages = upload_python_packages,
        )
    else:
        raise RuntimeError(f'unknown fp_style {fp_style}')

    # ConcurrentLearningBlock
    block_cl_op = ConcurrentLearningBlock(
        "concurrent-learning-block", 
        prep_run_train_op,
        prep_run_explore_op,
        SelectConfs,
        prep_run_fp_op,
        CollectData,
        select_confs_config = select_confs_config,
        collect_data_config = collect_data_config,
        upload_python_packages = upload_python_packages,
    )    
    # dpgen
    dpgen_op = ConcurrentLearning(
        "concurrent-learning",
        block_cl_op,
        upload_python_packages = upload_python_packages,
        step_config = cl_step_config,
    )
        
    return dpgen_op


def make_naive_exploration_scheduler(
        config,
        old_style = False,
):
    # use npt task group
    model_devi_jobs = config['model_devi_jobs'] if old_style else config['explore']['stages']
    sys_configs = config['sys_configs'] if old_style else config['explore']['configurations']
    sys_prefix = config.get('sys_prefix')
    if sys_prefix is not None:
        for ii in range(len(sys_configs)):
            if isinstance(sys_configs[ii], list):
                sys_configs[ii] = [os.path.join(sys_prefix, jj) for jj in sys_prefix[ii]]
    mass_map = config['mass_map'] if old_style else config['inputs']['mass_map']
    type_map = config['type_map'] if old_style else config['inputs']['type_map']
    numb_models = config['numb_models'] if old_style else config['train']['numb_models']
    fp_task_max = config['fp_task_max'] if old_style else config['fp']['task_max']
    conv_accuracy = config['conv_accuracy'] if old_style else config['explore']['conv_accuracy']
    max_numb_iter = config['max_numb_iter'] if old_style else config['explore']['max_numb_iter']
    output_nopbc = False if old_style else config['explore']['output_nopbc']
    fatal_at_max = config.get('fatal_at_max', True) if old_style else config['explore']['fatal_at_max']
    scheduler = ExplorationScheduler()
    
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
            n_sample = jj.pop('n_sample')
            ##  ignore the expansion of sys_idx
            # get all file names of md initial configurations
            try:
                sys_idx = jj.pop('sys_idx')
            except KeyError:
                sys_idx = jj.pop('conf_idx')
            conf_list = []        
            for ii in sys_idx:
                conf_list += sys_configs_lmp[ii]
            # make task group
            tgroup = make_task_group_from_config(numb_models, mass_map, jj)
            # add the list to task group
            tgroup.set_conf(
                conf_list,
                n_sample=n_sample,
            )
            tasks = tgroup.make_task()
            stage.add_task_group(tasks)
        # trust level
        trust_level = TrustLevel(
            config['model_devi_f_trust_lo'] if old_style else config['explore']['f_trust_lo'],
            config['model_devi_f_trust_hi'] if old_style else config['explore']['f_trust_hi'],
            level_v_lo=config.get('model_devi_v_trust_lo') if old_style else config['explore']['v_trust_lo'],
            level_v_hi=config.get('model_devi_v_trust_hi') if old_style else config['explore']['v_trust_hi'],
        )
        # selector
        selector = ConfSelectorLammpsFrames(
            trust_level,
            fp_task_max,
            nopbc=output_nopbc,
        )
        # stage_scheduler
        stage_scheduler = ConvergenceCheckStageScheduler(
            stage,
            selector,
            conv_accuracy = conv_accuracy,
            max_numb_iter = max_numb_iter,
            fatal_at_max = fatal_at_max,
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
        if 'KSPACING' in ii:
            ks = float(ii.split('=')[1])
        if 'KGAMMA' in ii:
            if 'T' in ii.split('=')[1]:
                kg = True
            elif 'F' in ii.split('=')[1]:
                kg = False
            else:
                raise RuntimeError(f"invalid kgamma value {ii.split('=')[1]}")
    assert ks is not None and kg is not None
    return ks, kg


def workflow_concurrent_learning(
        config : Dict,
        old_style : bool = False,
):
    default_config = normalize_step_dict(config.get('default_config', {})) if old_style else config['default_step_config']

    train_style = config.get('train_style', 'dp') if old_style else config['train']['type']
    explore_style = config.get('explore_style', 'lmp') if old_style else config['explore']['type']
    fp_style = config.get('fp_style', 'vasp') if old_style else config['fp']['type']
    prep_train_config = normalize_step_dict(config.get('prep_train_config', default_config)) if old_style else config['step_configs']['prep_train_config']
    run_train_config = normalize_step_dict(config.get('run_train_config', default_config)) if old_style else config['step_configs']['run_train_config']
    prep_explore_config = normalize_step_dict(config.get('prep_explore_config', default_config)) if old_style else config['step_configs']['prep_explore_config']
    run_explore_config = normalize_step_dict(config.get('run_explore_config', default_config)) if old_style else config['step_configs']['run_explore_config']
    prep_fp_config = normalize_step_dict(config.get('prep_fp_config', default_config)) if old_style else config['step_configs']['prep_fp_config']
    run_fp_config = normalize_step_dict(config.get('run_fp_config', default_config)) if old_style else config['step_configs']['run_fp_config']
    select_confs_config = normalize_step_dict(config.get('select_confs_config', default_config)) if old_style else config['step_configs']['select_confs_config']
    collect_data_config = normalize_step_dict(config.get('collect_data_config', default_config)) if old_style else config['step_configs']['collect_data_config']
    cl_step_config = normalize_step_dict(config.get('cl_step_config', default_config)) if old_style else config['step_configs']['cl_step_config']
    upload_python_packages = config.get('upload_python_packages', None)
    init_models_paths = config.get('training_iter0_model_path', None) if old_style else config['train'].get('training_iter0_model_path', None)
    if upload_python_packages is not None and isinstance(upload_python_packages, str):
        upload_python_packages = [upload_python_packages]
    if upload_python_packages is not None:
        _upload_python_packages: List[os.PathLike] = [Path(ii) for ii in upload_python_packages]
        upload_python_packages = _upload_python_packages

    concurrent_learning_op = make_concurrent_learning_op(
        train_style,
        explore_style,
        fp_style,
        prep_train_config = prep_train_config,
        run_train_config = run_train_config,
        prep_explore_config = prep_explore_config,
        run_explore_config = run_explore_config,
        prep_fp_config = prep_fp_config,
        run_fp_config = run_fp_config,
        select_confs_config = select_confs_config,
        collect_data_config = collect_data_config,
        cl_step_config = cl_step_config,
        upload_python_packages = upload_python_packages,
    )
    scheduler = make_naive_exploration_scheduler(config, old_style=old_style)

    type_map = config['type_map'] if old_style else config['inputs']['type_map']
    numb_models = config['numb_models'] if old_style else config['train']['numb_models']
    template_script_ = config['default_training_param'] if old_style else config['train']['template_script']
    if isinstance(template_script_, list):
        template_script = [Path(ii).read_text() for ii in template_script_]
    else:
        template_script = Path(template_script_).read_text()
    train_config = {} if old_style else config['train']['config']
    lmp_config = config.get('lmp_config', {}) if old_style else config['explore']['config']
    fp_config = config.get('fp_config', {}) if old_style else {}
    if old_style:        
        potcar_names = config['fp_pp_files']
        incar_template_name = config['fp_incar']
        kspacing, kgamma = get_kspacing_kgamma_from_incar(incar_template_name)
        fp_inputs_config = {
            'kspacing' : kspacing,
            'kgamma' : kgamma,
            'incar_template_name' : incar_template_name,
            'potcar_names' : potcar_names,
        }
    else:
        fp_inputs_config = config['fp']['inputs_config']
    fp_inputs = fp_styles[fp_style]['inputs'](**fp_inputs_config)

    fp_config['inputs'] = fp_inputs
    fp_config['run'] = config['fp']['run_config']

    init_data_prefix = config.get('init_data_prefix') if old_style else config['inputs']['init_data_prefix']
    init_data = config['init_data_sys'] if old_style else config['inputs']['init_data_sys']
    if init_data_prefix is not None:
        init_data = [os.path.join(init_data_prefix, ii) for ii in init_data]
    if isinstance(init_data,str):
        init_data = expand_sys_str(init_data)
    init_data = upload_artifact(init_data)
    iter_data = upload_artifact([])
    if init_models_paths is not None:
        init_models = upload_artifact(init_models_paths)
    else:
        init_models = None

    # here the scheduler is passed as input parameter to the concurrent_learning_op
    dpgen_step = Step(
        'dpgen-step',
        template = concurrent_learning_op,
        parameters = {
            "type_map" : type_map,
            "numb_models" : numb_models,
            "template_script" : template_script,
            "train_config" : train_config,
            "lmp_config" : lmp_config,
            "fp_config" : fp_config,
            "exploration_scheduler" : scheduler,
        },
        artifacts = {
            "init_models" : init_models,
            "init_data" : init_data,
            "iter_data" : iter_data,
        },
    )
    return dpgen_step


def submit_concurrent_learning(
        wf_config,
        reuse_step = None,
        old_style = False,
):
    # normalize args
    wf_config = normalize_args(wf_config)

    do_lebesgue = wf_config.get("lebesgue_context_config", None) is not None

    context = global_config_workflow(wf_config, do_lebesgue=do_lebesgue)
    
    dpgen_step = workflow_concurrent_learning(wf_config, old_style=old_style)

    wf = Workflow(name="dpgen", context=context)
    wf.add(dpgen_step)

    wf.submit(reuse_step=reuse_step)

    return wf


def print_list_steps(
        steps,
):
    ret = []
    for idx,ii in enumerate(steps):
        ret.append(f'{idx:8d}    {ii}')
    return '\n'.join(ret)


def successful_step_keys(wf):
    all_step_keys_ = wf.query_keys_of_steps()
    wf_info = wf.query()
    all_step_keys = []
    for ii in all_step_keys_:
        if wf_info.get_step(key=ii)[0]['phase'] == 'Succeeded':
            all_step_keys.append(ii)
    return all_step_keys


def get_resubmit_keys(
        wf,
):
    all_step_keys = successful_step_keys(wf)
    all_step_keys = matched_step_key(
        all_step_keys,
        ['prep-train', 'run-train', 'prep-lmp', 'run-lmp', 'select-confs', 
         'prep-fp', 'run-fp', 'collect-data', 'scheduler', 'id'],
    )
    all_step_keys = sort_slice_ops(
        all_step_keys, ['run-train', 'run-lmp', 'run-fp'],)
    return all_step_keys


def resubmit_concurrent_learning(
        wf_config,
        wfid,
        list_steps = False,
        reuse = None,
        old_style = False,
):
    wf_config = normalize_args(wf_config)

    context = global_config_workflow(wf_config)

    old_wf = Workflow(id=wfid)
    all_step_keys = get_resubmit_keys(old_wf)

    if list_steps:
        prt_str = print_keys_in_nice_format(
            all_step_keys, ['run-train', 'run-lmp', 'run-fp'],)
        print(prt_str)

    if reuse is None:
        return None
    reuse_idx = expand_idx(reuse)
    reuse_step = []
    old_wf_info = old_wf.query()
    for ii in reuse_idx:
        reuse_step += old_wf_info.get_step(key=all_step_keys[ii])

    wf = submit_concurrent_learning(
        wf_config, 
        reuse_step=reuse_step,
        old_style=old_style,
    )

    return wf
