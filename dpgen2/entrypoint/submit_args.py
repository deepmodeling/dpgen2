import dargs
from dargs import (
    Argument,
    Variant,
)
from dpgen2.constants import default_image
from dflow.plugins.lebesgue import LebesgueExecutor
from dpgen2.op.run_dp_train import RunDPTrain
from dpgen2.op.run_lmp import RunLmp
from dpgen2.op.run_vasp import RunVasp
from dpgen2.utils import (
    step_conf_args,
    normalize_step_dict,
)

def dp_train_args():
    doc_numb_models = "Number of models trained for evaluating the model deviation"
    doc_config = "Configuration of training"
    doc_template_script = "Template training script. It can be a `List[Dict]`, the length of which is the same as `numb_models`. Each template script in the list is used to train a model. Can be a `Dict`, the models share the same template training script. "
    
    return [
        Argument("config", dict, RunDPTrain.training_args(), optional=True, default=RunDPTrain.normalize_config({}), doc=doc_numb_models),
        Argument("numb_models", int, optional=True, default=4, doc=doc_numb_models),
        Argument("template_script", [list,dict], optional=False, doc=doc_template_script),
    ]

def variant_train():
    doc = "the type of the training"
    return Variant("type", [
        Argument("dp", dict, dp_train_args()),
    ], doc=doc)

def lmp_args():
    doc_config = "Configuration of lmp exploration"
    doc_max_numb_iter = "Maximum number of iterations per stage"
    doc_conv_accuracy = "Convergence accuracy"
    doc_fatal_at_max = "Fatal when the number of iteration per stage reaches the `max_numb_iter`"
    doc_f_trust_lo = "Lower trust level of force model deviation"
    doc_f_trust_hi = "Higher trust level of force model deviation"
    doc_v_trust_lo = "Lower trust level of virial model deviation"
    doc_v_trust_hi = "Higher trust level of virial model deviation"
    doc_configuration_prefix = "The path prefix of lmp initial configurations"
    doc_configuration = "A list of initial configurations."
    doc_stages = "A list of exploration stages."

    return [
        Argument("config", dict, RunLmp.lmp_args(), optional=True, default=RunLmp.normalize_config({}), doc=doc_config),
        Argument("max_numb_iter", int, optional=True, default=10, doc=doc_max_numb_iter),
        Argument("conv_accuracy", float, optional=True, default=0.9, doc=doc_conv_accuracy),
        Argument("fatal_at_max", bool, optional=True, default=True, doc=doc_fatal_at_max),
        Argument("f_trust_lo", float, optional=False, doc=doc_f_trust_lo),
        Argument("f_trust_hi", float, optional=False, doc=doc_f_trust_hi),
        Argument("v_trust_lo", float, optional=True, default=None, doc=doc_v_trust_lo),
        Argument("v_trust_hi", float, optional=True, default=None, doc=doc_v_trust_hi),
        Argument("configuration_prefix", str, optional=True, default=None, doc=doc_configuration_prefix),
        Argument("configurations", list, optional=False, doc=doc_configuration, alias=["configuration"]),
        Argument("stages", list, optional=False, doc=doc_stages),
    ]

def variant_explore():
    doc = "the type of the exploration"
    return Variant("type", [
        Argument("lmp", dict, lmp_args()),
    ], doc=doc)

def vasp_args():
    doc_config = "Configuration of vasp runs"
    doc_task_max = "Maximum number of vasp tasks for each iteration"
    doc_pp_files = 'The pseudopotential files set by a dict, e.g. {"Al" : "path/to/the/al/pp/file", "Mg" : "path/to/the/mg/pp/file"}'
    doc_incar = "The path to the template incar file"

    return [
        Argument("config", dict, RunVasp.vasp_args(), optional=True, default=RunVasp.normalize_config({}), doc=doc_config),
        Argument("task_max", int, optional=True, default=10, doc=doc_task_max),
        Argument("pp_files", dict, optional=False, doc=doc_pp_files),
        Argument("incar", str, optional=False, doc=doc_pp_files),
    ]

def variant_fp():
    doc = "the type of the fp"
    return Variant("type", [
        Argument("vasp", dict, vasp_args()),
    ], doc=doc)

def input_args():
    doc_type_map = 'The type map. e.g. ["Al", "Mg"]. Al and Mg will have type 0 and 1, respectively.'
    doc_mass_map = "The mass map. e.g. [27., 24.]. Al and Mg will be set with mass 27. and 24. amu, respectively."
    doc_init_data_prefix = "The prefix of initial data systems"
    doc_init_sys = "The inital data systems"

    return [
        Argument("type_map", list, optional=False, doc=doc_type_map),
        Argument("mass_map", list, optional=False, doc=doc_mass_map),
        Argument("init_data_prefix", str, optional=True, default=None, doc=doc_init_data_prefix),
        Argument("init_data_sys", list, optional=False, default=None, doc=doc_init_sys),
    ]
    

def dflow_conf_args():
    doc_dflow_config = "The configuration passed to dflow"
    doc_s3_config = "The S3 configuration passed to dflow"
    
    return [
        Argument("dflow_config", dict, optional=True, default=None, doc=doc_dflow_config),
        Argument("s3_config", dict, optional=True, default=None, doc=doc_s3_config),
    ]

def lebesgue_conf_args():
    doc_lebesgue_context_config = "Configuration passed to dflow Lebesgue context"
    
    return [
        Argument("lebesgue_context_config", dict, optional=True, default=None, doc=doc_lebesgue_context_config),
    ]

def default_step_config_args():
    doc_default_step_config = "The default step configuration."
    
    return [
        Argument("default_step_config", dict, step_conf_args(), optional=True, default={}, doc=doc_default_step_config),
    ]


def dpgen_step_config_args(default_config):
    doc_prep_train_config = "Configuration for prepare train"
    doc_run_train_config = "Configuration for run train"
    doc_prep_explore_config = "Configuration for prepare exploration"
    doc_run_explore_config = "Configuration for run exploration"
    doc_prep_fp_config = "Configuration for prepare fp"
    doc_run_fp_config = "Configuration for run fp"
    doc_select_confs_config = "Configuration for the select confs"
    doc_collect_data_config = "Configuration for the collect data"
    doc_cl_step_config = "Configuration for the concurrent learning step"

    return [
        Argument("prep_train_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_prep_train_config),
        Argument("run_train_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_run_train_config),
        Argument("prep_explore_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_prep_explore_config),
        Argument("run_explore_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_run_explore_config),
        Argument("prep_fp_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_prep_fp_config),
        Argument("run_fp_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_run_fp_config),
        Argument("select_confs_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_select_confs_config),
        Argument("collect_data_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_collect_data_config),
        Argument("cl_step_config", dict, step_conf_args(), optional=True, default=default_config, doc=doc_cl_step_config),
    ]


def submit_args(default_step_config = normalize_step_dict({})):
    doc_step_configs = "Configurations for executing dflow steps"
    doc_upload_python_package = "Upload python package, for debug purpose"
    doc_inputs = "The input parameter and artifacts for dpgen2"
    doc_train = "The configuration for training"
    doc_explore = "The configuration for exploration"
    doc_fp = "The configuration for FP"

    return \
        dflow_conf_args() + \
        lebesgue_conf_args() + \
        default_step_config_args() + [
        Argument("step_configs", dict, dpgen_step_config_args(default_step_config), optional=True, default={}, doc=doc_step_configs),
        Argument("upload_python_package", str, optional=True, default=None, doc=doc_upload_python_package),
        Argument("inputs", dict, input_args(), optional=False, doc=doc_inputs),
        Argument("train", dict, [], [variant_train()], optional=False, doc=doc_train),
        Argument("explore", dict, [], [variant_explore()], optional=False, doc=doc_explore),
        Argument("fp", dict, [], [variant_fp()], optional=False, doc=doc_fp),
    ]


def normalize(data):
    default_step_config = normalize_step_dict(data.get('default_step_config', {}))
    defs = submit_args(default_step_config)

    base = Argument("base", dict, defs)
    data = base.normalize_value(data, trim_pattern="_*")
    # not possible to strictly check Lebesgue_executor_args, dirty hack!
    base.check_value(data, strict=False)
    return data


def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    sca = submit_args()
    base = Argument("submit", dict, sca)
    ptr = []
    ptr.append(base.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split('\n'):
        if 'argument path' in ii:
            key_words.append(ii.split(':')[1].replace('`','').strip())
    return "\n\n".join(ptr)


