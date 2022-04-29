import glob
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
    ConstTrustLevelStageScheduler,
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

def make_concurrent_learning_op (
        train_style : str = 'dp',
        explore_style : str = 'lmp',
        fp_style : str = 'vasp',
        prep_train_image : str = 'dflow:v1.0',
        run_train_image : str = 'dflow:v1.0',
        prep_explore_image : str = 'dflow:v1.0',
        run_explore_image : str = 'dflow:v1.0',
        prep_fp_image : str = 'dflow:v1.0',
        run_fp_image : str = 'dflow:v1.0',
        upload_python_package : bool = False,
):
    if train_style == 'dp':
        prep_run_train_op = PrepRunDPTrain(
            "prep-run-dp-train",
            PrepDPTrain,
            RunDPTrain,
            prep_image = prep_train_image,
            run_image = run_train_image,
            upload_python_package = upload_python_package,
        )
    else:
        raise RuntimeError(f'unknown train_style {train_style}')
    if explore_style == 'lmp':
        prep_run_explore_op = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            RunLmp,
            prep_image = prep_explore_image,
            run_image = run_explore_image,
            upload_python_package = upload_python_package,
        )
    else:
        raise RuntimeError(f'unknown explore_style {explore_style}')
    if fp_style == 'vasp':
        prep_run_fp_op = PrepRunFp(
            "prep-run-vasp",
            PrepVasp,
            RunVasp,
            prep_image = prep_fp_image,
            run_image = run_fp_image,
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
        upload_python_package = upload_python_package,
    )    
    # dpgen
    dpgen_op = ConcurrentLearning(
        "concurrent-learning",
        block_cl_op,
        upload_python_package = upload_python_package,
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
    trust_level = TrustLevel(
        config['model_devi_f_trust_lo'],
        config['model_devi_f_trust_hi'],
    )
    scheduler = ExplorationScheduler()

    for job in model_devi_jobs:
        # task group
        tgroup = NPTTaskGroup()
        # ignore the expansion of sys_idx
        sys_idx = job['sys_idx']
        conf_list = []        
        for ii in sys_idx:
            for jj in sys_configs[ii]:                
                confs = glob.glob(jj)
                conf_list = conf_list + confs
        tgroup.set_conf(conf_list)
        temps = job['temps']
        trj_freq = job['trj_freq']
        nsteps = job['nsteps']
        ensemble = job['ensemble']
        tgroup.set_md(
            numb_models,
            mass_map,
            temps = temps,
            ens = ensemble,
            nsteps = nsteps,
        )
        tasks = tgroup.make_task()
        # stage
        stage = ExplorationStage()
        stage.add_task_group(tasks)
        # stage_scheduler
        stage_scheduler = ConstTrustLevelStageScheduler(
            stage,
            trust_level,
            conv_accuracy = 0.9,
            max_numb_iter = 10,
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
    prep_train_image = config['prep_train_image']
    run_train_image = config['prep_train_image']
    prep_explore_image = config['prep_explore_image']
    run_explore_image = config['prep_explore_image']
    prep_fp_image = config['prep_fp_image']
    run_fp_image = config['prep_fp_image']
    upload_python_package = config.get('upload_python_package', None)

    concurrent_learning_op = make_concurrent_learning_op(
        train_style,
        explore_style,
        fp_style,
        prep_train_image,
        run_train_image,
        prep_explore_image,
        run_explore_image,
        prep_fp_image,
        run_fp_image,
        upload_python_package,
    )
    scheduler = make_naive_exploration_scheduler(config)

    type_map = config['type_map']
    numb_models = config['numb_models']
    template_script = config['default_training_param']
    train_config = {}
    lmp_config = {}
    fp_config = {}
    kspacing, kgamma = get_kspacing_kgamma_from_incar(config['fp_incar'])
    potcar_names = config['fp_pp_files']
    for kk,vv in potcar_names.items():
        potcar_names[kk] = f"{config['fp_pp_path']}/{vv}"
    fp_inputs = VaspInputs(
        kspacing = kspacing,
        kgamma = kgamma,
        incar_template_name = config['fp_incar'],
        potcar_names = potcar_names,
    )
    init_data = upload_artifact(config['init_data'])
    iter_data = upload_artifact([])
    init_models = None
    
    dpgen_step = Step(
        'dpgen-step',
        template = concurrent_learning_op,
        parameters = {
            "type_map" : type_map,
            "numb_models" : numb_models,
            "template_script" : template_script,
            "train_config" : train_config,
            "lmp_config" : lmp_config,
            'fp_inputs' : fp_inputs,
            "fp_config" : fp_config,
            "exploration_scheduler" : scheduler,
        },
        artifacts = {
            "init_models" : init_models,
            "init_data" : init_data,
            "iter_data" : iter_data,
        },
    )
        
    wf = Workflow(name="dpgen")
    wf.add(dpgen_step)
    
    return wf

def submit_concurrent_learning(
        config,
):
    wf = workflow_concurrent_learning(config)
    wf.submit()
    return wf
