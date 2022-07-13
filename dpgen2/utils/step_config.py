import dargs
from dargs import (
    Argument,
    Variant,
)
from dpgen2.constants import default_image
from dflow.plugins.lebesgue import LebesgueExecutor

def lebesgue_extra_args():
    # It is not possible to strictly check the keys in this section....
    doc_scass_type = "The machine configuraiton."
    doc_program_id = "The ID of the program."
    doc_job_type = "The type of job."
    doc_template_cover = "The key for hacking around a bug in Lebesgue."

    return [
        Argument("scass_type", str, optional=True, doc=doc_scass_type),
        Argument("program_id", str, optional=True, doc=doc_program_id),
        Argument("job_type", str, optional=True, default="container", doc=doc_job_type),
        Argument("template_cover_cmd_escape_bug", bool, optional=True, default=True, doc=doc_template_cover),
    ]

def lebesgue_executor_args():
    doc_extra = "The 'extra' key in the lebesgue executor. Note that we do not check if 'the `dict` provided to the 'extra' key is valid or not."
    return [
        Argument("extra", dict, lebesgue_extra_args(), optional = True, doc = doc_extra),
    ]

def variant_executor():
    doc = f'The type of the executor.'
    return Variant("type", [
        Argument("lebesgue_v2", dict, lebesgue_executor_args()),
    ], doc = doc)

def template_conf_args():
    doc_image = 'The image to run the step.'
    doc_timeout = 'The time limit of the OP. Unit is second.'
    doc_retry_on_transient_error = 'Retry the step if a TransientError is raised.'
    doc_timeout_as_transient_error = 'Treat the timeout as TransientError.'
    return [
        Argument("image", str, optional=True, default=default_image, doc=doc_image),
        Argument("timeout", int, optional=True, default=None, doc=doc_timeout),
        Argument("retry_on_transient_error", bool, optional=True, default=None, doc=doc_retry_on_transient_error),
        Argument("timeout_as_transient_error", bool, optional=True, default=False, doc=doc_timeout_as_transient_error),
    ]

def step_conf_args():
    doc_template = 'The configs passed to the PythonOPTemplate.'
    doc_executor = 'The executor of the step.'
    doc_continue_on_failed = 'If continue the the step is failed (FatalError, TransientError, A certain number of retrial is reached...).'
    doc_continue_on_num_success = 'Only in the sliced OP case. Continue the workflow if a certain number of the sliced jobs are successful.'
    doc_continue_on_success_ratio = 'Only in the sliced OP case. Continue the workflow if a certain ratio of the sliced jobs are successful.'

    return [
        Argument("template_config", dict, template_conf_args(), optional=True, default={'image':default_image}, doc=doc_template),
        Argument("continue_on_failed", bool, optional=True, default=False, doc=doc_continue_on_failed),
        Argument("continue_on_num_success", int, optional=True, default=None, doc=doc_continue_on_num_success),
        Argument("continue_on_success_ratio", float, optional=True, default=None, doc=doc_continue_on_success_ratio),
        Argument("executor", dict, [], [variant_executor()], optional=True, default=None, doc = doc_executor),
    ]

def normalize(data):
    sca = step_conf_args()
    base = Argument("base", dict, sca)
    data = base.normalize_value(data, trim_pattern="_*")
    # not possible to strictly check Lebesgue_executor_args, dirty hack!
    base.check_value(data, strict=False)
    return data

def gen_doc(*, make_anchor=True, make_link=True, **kwargs):
    if make_link:
        make_anchor = True
    sca = step_conf_args()
    base = Argument("step_config", dict, sca)
    ptr = []
    ptr.append(base.gen_doc(make_anchor=make_anchor, make_link=make_link, **kwargs))

    key_words = []
    for ii in "\n\n".join(ptr).split('\n'):
        if 'argument path' in ii:
            key_words.append(ii.split(':')[1].replace('`','').strip())
    return "\n\n".join(ptr)


def init_executor(
        executor_dict,
):
    if executor_dict is None:
        return None
    etype = executor_dict.pop('type')
    if etype == "lebesgue_v2":
        return LebesgueExecutor(**executor_dict)
    else:
        raise RuntimeError('unknown executor type', etype)    
    
