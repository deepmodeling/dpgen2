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
    PrepVasp,
    RunVasp,
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
    VaspInputs,
)
from dpgen2.exploration.scheduler import (
    ExplorationScheduler,
    ConvergenceCheckStageScheduler,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTask,
    NPTTaskGroup,
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
)
from dpgen2.utils.step_config import normalize as normalize_step_dict
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
        prep_train_config : str = default_config,
        run_train_config : str = default_config,
        prep_explore_config : str = default_config,
        run_explore_config : str = default_config,
        prep_fp_config : str = default_config,
        run_fp_config : str = default_config,
        select_confs_config : str = default_config,
        collect_data_config : str = default_config,
        upload_python_package : bool = None,
):
    if train_style == 'dp':
        prep_run_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            PrepDPTrain,
            RunDPTrain,
            prep_config = prep_train_config,
            run_config = run_train_config,
            upload_python_package = upload_python_package,
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
            upload_python_package = upload_python_package,
        )
    else:
        raise RuntimeError(f'unknown explore_style {explore_style}')
    if fp_style == 'vasp':
        prep_run_fp_op = PrepRunFp(
            "prep-run-vasp",
            PrepVasp,
            RunVasp,
            prep_config = prep_fp_config,
            run_config = run_fp_config,
            upload_python_package = upload_python_package,
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
        upload_python_package = upload_python_package,
    )    
    # dpgen
    dpgen_op = ConcurrentLearning(
        "concurrent-learning",
        block_cl_op,
        upload_python_package = upload_python_package,
        step_config = default_config,
    )
        
    return dpgen_op


def make_naive_exploration_scheduler(
        config,
):
    # use npt task group
    model_devi_jobs = config['model_devi_jobs']
    sys_configs = config['sys_configs']
    mass_map = config['mass_map']
    type_map = config['type_map']
    numb_models = config['numb_models']
    fp_task_max = config['fp_task_max']
    scheduler = ExplorationScheduler()

    for job in model_devi_jobs:
        # task group
        tgroup = NPTTaskGroup()
        ##  ignore the expansion of sys_idx
        # get all file names of md initial configuraitons
        sys_idx = job['sys_idx']
        conf_list_fname = []        
        for ii in sys_idx:
            for jj in sys_configs[ii]:                
                confs = sorted(glob.glob(jj))
                conf_list_fname = conf_list_fname + confs
        # make list of configuration file content
        conf_list = []
        for ii in conf_list_fname:
            ss = dpdata.System(ii, type_map=type_map)
            ss.to('lammps/lmp', 'tmp.lmp')
            conf_list.append(Path('tmp.lmp').read_text())
        if Path('tmp.lmp').is_file():
            os.remove('tmp.lmp')
        # add the list to task group
        tgroup.set_conf(
            conf_list,
            n_sample = 3,
        )
        temps = job['temps']
        press = job['press']
        trj_freq = job['trj_freq']
        nsteps = job['nsteps']
        ensemble = job['ensemble']
        # add md settings
        tgroup.set_md(
            numb_models,
            mass_map,
            temps = temps,
            press = press,
            ens = ensemble,
            nsteps = nsteps,
        )
        tasks = tgroup.make_task()
        # stage
        stage = ExplorationStage()
        stage.add_task_group(tasks)
        # trust level
        trust_level = TrustLevel(
            config['model_devi_f_trust_lo'],
            config['model_devi_f_trust_hi'],
        )
        # selector
        selector = ConfSelectorLammpsFrames(
            trust_level,
            fp_task_max,
        )
        # stage_scheduler
        stage_scheduler = ConvergenceCheckStageScheduler(
            stage,
            selector,
            conv_accuracy = 0.9,
            max_numb_iter = 3,
        )
        # scheduler
        scheduler.add_stage_scheduler(stage_scheduler)
        
    return scheduler


def get_kspacing_kgamma_from_incar(
        fname,
):
    with open(fname) as fp:
        lines = fp.readlines()
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
    return ks, kg


def workflow_concurrent_learning(
        config,
):
    train_style = config['train_style']
    explore_style = config['explore_style']
    fp_style = config['fp_style']
    prep_train_config = normalize_step_dict(config.get('prep_train_config', {}))
    run_train_config = normalize_step_dict(config.get('run_train_config', {}))
    prep_explore_config = normalize_step_dict(config.get('prep_explore_config', {}))
    run_explore_config = normalize_step_dict(config.get('run_explore_config', {}))
    prep_fp_config = normalize_step_dict(config.get('prep_fp_config', {}))
    run_fp_config = normalize_step_dict(config.get('run_fp_config', {}))
    upload_python_package = config.get('upload_python_package', None)
    init_models_paths = config.get('training_iter0_model_path')

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
        upload_python_package = upload_python_package,
    )
    scheduler = make_naive_exploration_scheduler(config)
    scheduler_file = Path('in_scheduler.dat')
    with open(scheduler_file, 'wb') as fp:
        pickle.dump(scheduler, fp)
    scheduler_arti = upload_artifact(scheduler_file)

    type_map = config['type_map']
    numb_models = config['numb_models']
    template_script = config['default_training_param']
    train_config = {}
    lmp_config = config.get('lmp_config', {})
    fp_config = config.get('fp_config', {})
    kspacing, kgamma = get_kspacing_kgamma_from_incar(config['fp_incar'])
    fp_pp_files = config['fp_pp_files']
    potcar_names = {}
    for kk,vv in zip(type_map, fp_pp_files):
        potcar_names[kk] = f"{config['fp_pp_path']}/{vv}"
    fp_inputs = VaspInputs(
        kspacing = kspacing,
        kgamma = kgamma,
        incar_template_name = config['fp_incar'],
        potcar_names = potcar_names,
    )
    fp_arti = upload_artifact(
        dump_object_to_file(fp_inputs, 'vasp_inputs.dat'))        
    init_data = upload_artifact(config['init_data_sys'])
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
        },
        artifacts = {
            "exploration_scheduler" : scheduler_arti,
            'fp_inputs' : fp_arti,
            "init_models" : init_models,
            "init_data" : init_data,
            "iter_data" : iter_data,
        },
    )
    return dpgen_step

def submit_concurrent_learning(
        wf_config,
):
    # set global config
    from dflow import config, s3_config
    dflow_config = wf_config.get('dflow_config', None)
    if dflow_config :
        config["host"] = dflow_config.get('host', None)
        s3_config["endpoint"] = dflow_config.get('s3_endpoint', None)
        config["k8s_api_server"] = dflow_config.get('k8s_api_server', None)
        config["token"] = dflow_config.get('token', None)    

    # lebesque context
    from dflow.plugins.lebesgue import LebesgueContext
    lb_context_config = wf_config.get("lebesque_context_config", None)
    if lb_context_config:
        lebesgue_context = LebesgueContext(
            **lb_context_config,
        )
    else :
        lebesgue_context = None

    # print('config:', config)
    # print('s3_config:',s3_config)
    # print('lebsque context:', lb_context_config)

    dpgen_step = workflow_concurrent_learning(wf_config)

    wf = Workflow(name="dpgen", context=lebesgue_context)
    wf.add(dpgen_step)

    wf.submit()

    return wf
