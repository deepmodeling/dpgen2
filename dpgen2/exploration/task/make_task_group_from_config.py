import dargs
from dargs import (
    Argument,
    Variant,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    model_name_pattern,
    plm_input_name,
)
from dpgen2.exploration.task.customized_lmp_template_task_group import (
    CustomizedLmpTemplateTaskGroup,
)
from dpgen2.exploration.task.lmp_template_task_group import (
    LmpTemplateTaskGroup,
)
from dpgen2.exploration.task.npt_task_group import (
    NPTTaskGroup,
)

doc_conf_idx = "The configurations of `configurations[conf_idx]` will be used to generate the initial configurations of the tasks. This key provides the index of selected item in the `configurations` array."
doc_n_sample = "Number of configurations. If this number is smaller than the number of configruations in `configruations[conf_idx]`, then `n_sample` configruations are randomly sampled from `configruations[conf_idx]`, otherwise all configruations in `configruations[conf_idx]` will be used. If not provided, all configruations in `configruations[conf_idx]` will be used."


def npt_task_group_args():
    doc_temps = "A list of temperatures in K. Also used to initialize the temperature"
    doc_press = "A list of pressures in bar."
    doc_ens = "The ensemble. Allowd options are 'nve', 'nvt', 'npt', 'npt-a', 'npt-t'. 'npt-a' stands for anisotrpic box sampling and 'npt-t' stands for triclinic box sampling."
    doc_dt = "The time step"
    doc_nsteps = "The number of steps"
    doc_traj_freq = "The frequency of dumping configurations and thermodynamic states"
    doc_tau_t = "The time scale of thermostat"
    doc_tau_p = "The time scale of barostat"
    doc_pka_e = "The energy of primary knock-on atom"
    doc_neidelay = "The delay of updating the neighbor list"
    doc_no_pbc = "Not using the periodic boundary condition"
    doc_use_clusters = "Calculate atomic model deviation"
    doc_relative_f_epsilon = "Calculate relative force model deviation"
    doc_relative_v_epsilon = "Calculate relative virial model deviation"

    return [
        Argument("conf_idx", list, optional=False, doc=doc_conf_idx, alias=["sys_idx"]),
        Argument(
            "n_sample",
            int,
            optional=True,
            default=None,
            doc=doc_n_sample,
        ),
        Argument("temps", list, optional=False, doc=doc_temps, alias=["Ts"]),
        Argument("press", list, optional=True, doc=doc_press, alias=["Ps"]),
        Argument(
            "ens", str, optional=True, default="nve", doc=doc_ens, alias=["ensemble"]
        ),
        Argument("dt", float, optional=True, default=1e-3, doc=doc_dt),
        Argument("nsteps", int, optional=True, default=100, doc=doc_nsteps),
        Argument(
            "trj_freq",
            int,
            optional=True,
            default=10,
            doc=doc_nsteps,
            alias=["t_freq", "trj_freq", "traj_freq"],
        ),
        Argument("tau_t", float, optional=True, default=5e-2, doc=doc_tau_t),
        Argument("tau_p", float, optional=True, default=5e-1, doc=doc_tau_p),
        Argument("pka_e", float, optional=True, default=None, doc=doc_pka_e),
        Argument("neidelay", int, optional=True, default=None, doc=doc_neidelay),
        Argument("no_pbc", bool, optional=True, default=False, doc=doc_no_pbc),
        Argument(
            "use_clusters", bool, optional=True, default=False, doc=doc_use_clusters
        ),
        Argument(
            "relative_f_epsilon",
            float,
            optional=True,
            default=None,
            doc=doc_relative_f_epsilon,
        ),
        Argument(
            "relative_v_epsilon",
            float,
            optional=True,
            default=None,
            doc=doc_relative_v_epsilon,
        ),
    ]


def lmp_template_task_group_args():
    doc_lmp_template_fname = "The file name of lammps input template"
    doc_plm_template_fname = "The file name of plumed input template"
    doc_revisions = "The revisions. Should be a dict providing the key - list of desired values pair. Key is the word to be replaced in the templates, and it may appear in both the lammps and plumed input templates. All values in the value list will be enmerated."
    doc_traj_freq = "The frequency of dumping configurations and thermodynamic states"

    return [
        Argument("conf_idx", list, optional=False, doc=doc_conf_idx, alias=["sys_idx"]),
        Argument(
            "n_sample",
            int,
            optional=True,
            default=None,
            doc=doc_n_sample,
        ),
        Argument(
            "lmp_template_fname",
            str,
            optional=False,
            doc=doc_lmp_template_fname,
            alias=["lmp_template", "lmp"],
        ),
        Argument(
            "plm_template_fname",
            str,
            optional=True,
            default=None,
            doc=doc_plm_template_fname,
            alias=["plm_template", "plm"],
        ),
        Argument("revisions", dict, optional=True, default={}),
        Argument(
            "traj_freq",
            int,
            optional=True,
            default=10,
            doc=doc_traj_freq,
            alias=["t_freq", "trj_freq", "trj_freq"],
        ),
    ]


def customized_lmp_template_task_group_args():
    doc_input_lmp_tmpl_name = "The file name of lammps input template"
    doc_input_plm_tmpl_name = "The file name of plumed input template"
    doc_revisions = "The revisions. Should be a dict providing the key - list of desired values pair. Key is the word to be replaced in the templates, and it may appear in both the lammps and plumed input templates. All values in the value list will be enmerated."
    doc_traj_freq = "The frequency of dumping configurations and thermodynamic states"
    doc_custom_shell_commands = (
        "Customized shell commands to be run for each configuration. "
        "The commands require `input_lmp_conf_name` as input conf file, "
        "`input_lmp_tmpl_name` and `input_plm_tmpl_name` as templates, "
        "and `input_extra_files` as extra input files. "
        "By running the commands a series folders in pattern "
        "`output_dir_pattern` are supposed to be generated, "
        "and each folder is supposed to contain a configuration file "
        "`output_lmp_conf_name`, a lammps template file `output_lmp_tmpl_name` "
        "and a plumed template file `output_plm_tmpl_name`."
    )
    doc_input_extra_files = (
        "Extra files that may be needed to execute the shell commands"
    )
    doc_output_dir_pattern = (
        "Pattern of resultant folders generated by the shell commands."
    )
    doc_input_lmp_conf_name = "Input conf file name for the shell commands."
    doc_output_lmp_conf_name = "Generated conf file name."
    doc_output_lmp_tmpl_name = "Generated lmp input file name."
    doc_output_plm_tmpl_name = "Generated plm input file name."

    return [
        Argument("conf_idx", list, optional=False, doc=doc_conf_idx, alias=["sys_idx"]),
        Argument(
            "n_sample",
            int,
            optional=True,
            default=None,
            doc=doc_n_sample,
        ),
        Argument(
            "custom_shell_commands", list, optional=False, doc=doc_custom_shell_commands
        ),
        Argument("revisions", dict, optional=True, default={}, doc=doc_revisions),
        Argument(
            "traj_freq",
            int,
            optional=True,
            default=10,
            doc=doc_traj_freq,
            alias=["t_freq", "trj_freq", "trj_freq"],
        ),
        Argument(
            "input_lmp_conf_name",
            str,
            optional=True,
            default=lmp_conf_name,
            doc=doc_input_lmp_conf_name,
        ),
        Argument(
            "input_lmp_tmpl_name",
            str,
            optional=True,
            default=lmp_input_name,
            doc=doc_input_lmp_tmpl_name,
            alias=["lmp_template", "lmp"],
        ),
        Argument(
            "input_plm_tmpl_name",
            str,
            optional=True,
            default=None,
            doc=doc_input_plm_tmpl_name,
            alias=["plm_template", "plm"],
        ),
        Argument(
            "input_extra_files",
            list,
            optional=True,
            default=[],
            doc=doc_input_extra_files,
        ),
        Argument(
            "output_dir_pattern",
            [str, list],
            optional=True,
            default="*",
            doc=doc_output_dir_pattern,
        ),
        Argument(
            "output_lmp_conf_name",
            str,
            optional=True,
            default=lmp_conf_name,
            doc=doc_output_lmp_conf_name,
        ),
        Argument(
            "output_lmp_tmpl_name",
            str,
            optional=True,
            default=lmp_input_name,
            doc=doc_output_lmp_tmpl_name,
        ),
        Argument(
            "output_plm_tmpl_name",
            str,
            optional=True,
            default=plm_input_name,
            doc=doc_output_plm_tmpl_name,
        ),
    ]


def variant_task_group():
    doc = "the type of the task group"
    doc_lmp_md = "Lammps MD tasks. DPGEN will generate the lammps input script"
    doc_lmp_template = "Lammps MD tasks defined by templates. User provide lammps (and plumed) template for lammps tasks. The variables in templates are revised by the revisions key. Notice that the lines for pair style, dump and plumed are reserved for the revision of dpgen2, and the users should not write these lines by themselves. Rather, users notify dpgen2 the poistion of the line for `pair_style` by writting 'pair_style deepmd', the line for `dump` by writting 'dump dpgen_dump'. If plumed is used, the line for `fix plumed` shouldbe written exactly as 'fix dpgen_plm'. "
    doc_customized_lmp_template = "Lammps MD tasks defined by user customized shell commands and templates. User provided shell script generates a series of folders, and each folder contains a lammps template task group. "
    return Variant(
        "type",
        [
            Argument(
                "lmp-md", dict, npt_task_group_args(), alias=["lmp-npt"], doc=doc_lmp_md
            ),
            Argument(
                "lmp-template",
                dict,
                lmp_template_task_group_args(),
                doc=doc_lmp_template,
            ),
            Argument(
                "customized-lmp-template",
                dict,
                customized_lmp_template_task_group_args(),
                doc=doc_customized_lmp_template,
            ),
        ],
        doc=doc,
    )


def task_group_args():
    return Argument("task_group", dict, [], [variant_task_group()])


def normalize(data):
    args = task_group_args()
    data = args.normalize_value(data, trim_pattern="_*")
    args.check_value(data, strict=False)
    return data


def config_strip_confidx(
    config,
):
    cc = config.copy()
    cc.pop("conf_idx") if "conf_idx" in cc else None
    cc.pop("n_sample") if "n_sample" in cc else None
    return cc


def make_task_group_from_config(
    numb_models,
    mass_map,
    config,
):
    # Work around the required conf_idx.
    # May not be a good design!!!
    config["conf_idx"] = [] if "conf_idx" not in config else None
    config = normalize(config)
    config = config_strip_confidx(config)
    if config["type"] == "lmp-md":
        tgroup = NPTTaskGroup()
        config.pop("type")
        tgroup.set_md(
            numb_models,
            mass_map,
            **config,
        )
    elif config["type"] == "lmp-template":
        tgroup = LmpTemplateTaskGroup()
        config.pop("type")
        lmp_template = config.pop("lmp_template_fname")
        tgroup.set_lmp(
            numb_models,
            lmp_template,
            **config,
        )
    elif config["type"] == "customized-lmp-template":
        tgroup = CustomizedLmpTemplateTaskGroup()
        config.pop("type")
        sh_cmd = config.pop("custom_shell_commands")
        tgroup.set_lmp(
            numb_models,
            sh_cmd,
            **config,
        )
    else:
        raise RuntimeError("unknown task group type: ", config["type"])
    return tgroup


if __name__ == "__main__":
    print(normalize({"type": "lmp-md"}))
