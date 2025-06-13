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
from dpgen2.exploration.task import (
    DiffCSPTaskGroup,
)
from dpgen2.exploration.task.caly_task_group import (
    CalyTaskGroup,
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
    doc_ele_temp_f = "The electron temperature set by frame style"
    doc_ele_temp_a = "The electron temperature set by atomistic style"
    doc_pimd_bead = "Bead index for PIMD, None for non-PIMD"

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
        Argument(
            "ele_temp_f",
            float,
            optional=True,
            default=None,
            doc=doc_ele_temp_f,
        ),
        Argument(
            "ele_temp_a",
            float,
            optional=True,
            default=None,
            doc=doc_ele_temp_a,
        ),
        Argument(
            "pimd_bead",
            str,
            optional=True,
            default=None,
            doc=doc_pimd_bead,
        ),
    ]


def lmp_template_task_group_args():
    doc_lmp_template_fname = "The file name of lammps input template"
    doc_plm_template_fname = "The file name of plumed input template"
    doc_revisions = "The revisions. Should be a dict providing the key - list of desired values pair. Key is the word to be replaced in the templates, and it may appear in both the lammps and plumed input templates. All values in the value list will be enmerated."
    doc_traj_freq = "The frequency of dumping configurations and thermodynamic states"
    doc_extra_pair_style_args = "The extra arguments for pair_style"
    doc_pimd_bead = "Bead index for PIMD, None for non-PIMD"

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
            "extra_pair_style_args",
            str,
            optional=True,
            default="",
            doc=doc_extra_pair_style_args,
        ),
        Argument(
            "pimd_bead",
            str,
            optional=True,
            default=None,
            doc=doc_pimd_bead,
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
                "lmp-nvnmd",
                dict,
                npt_task_group_args(),
                alias=["lmp-nvnmd-npt"],
                doc=doc_lmp_md,
            ),
            Argument(
                "lmp-template",
                dict,
                lmp_template_task_group_args(),
                doc=doc_lmp_template,
            ),
            Argument(
                "lmp-nvnmd-template",
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


def lmp_task_group_args():
    return Argument("task_group", dict, [], [variant_task_group()])


def lmp_normalize(data):
    args = lmp_task_group_args()
    data = args.normalize_value(data, trim_pattern="_*")
    args.check_value(data, strict=False)
    return data


def caly_task_grp_args():
    return [
        Argument("numb_of_species", int, optional=False, doc="number of species."),
        Argument(
            "name_of_atoms",
            list,
            optional=False,
            doc="name of atoms.",
        ),
        Argument(
            "atomic_number",
            list,
            optional=True,
            doc="atomic number of each element.",
        ),
        Argument(
            "numb_of_atoms",
            list,
            optional=False,
            doc="number of each atom.",
        ),
        Argument(
            "distance_of_ions",
            [list, dict],
            optional=True,
            doc="the distance matrix between different elements.",
        ),
        Argument(
            "pop_size",
            int,
            optional=True,
            default=30,
            doc="the number of structures would be generated in each step.",
        ),
        Argument(
            "max_step",
            int,
            optional=True,
            default=5,
            doc="the max iteration number of CALYPSO loop.",
        ),
        Argument(
            "system_name",
            str,
            optional=True,
            default="CALYPSO",
            doc="system name.",
        ),
        Argument(
            "numb_of_formula",
            list,
            optional=True,
            default=[1, 1],
            doc="the formula range of simulation cell.",
        ),
        Argument(
            "pressure",
            float,
            optional=True,
            default=0.001,
            doc="the aim pressure (in Kbar) when using MLP to optimize structures.",
        ),
        Argument(
            "fmax",
            float,
            optional=True,
            default=0.01,
            doc="the converge criterion. The force on all individual atoms should be less than `fmax`.",
        ),
        Argument(
            "opt_step",
            float,
            optional=True,
            default=1000,
            doc="the converge criterion. The force on all individual atoms should be less than `fmax`.",
        ),
        Argument(
            "volume",
            float,
            optional=True,
            default=0,
            doc="the volume of simulation cell in one formula.",
        ),
        Argument(
            "ialgo",
            int,
            optional=True,
            default=2,
            doc="the evolution algorithm of CALYPSO. 1: global pso, 2: local pso, 3: sabc.",
        ),
        Argument(
            "pso_ratio",
            float,
            optional=True,
            default=0.6,
            doc="the ratio of structures generated by evolution algorithm in one step.",
        ),
        Argument(
            "icode",
            int,
            optional=True,
            default=15,
            doc="the software of structure optimization. 1: VASP, 15: DP.",
        ),
        Argument(
            "numb_of_lbest",
            int,
            optional=True,
            default=4,
            doc="the number of evolution direction when using LPSO.",
        ),
        Argument(
            "numb_of_local_optim",
            int,
            optional=True,
            default=3,
            doc="the number of making structure optimization when using dft.",
        ),
        Argument(
            "command",
            str,
            optional=True,
            default="sh submit.sh",
            doc="the command of running structure optimization.",
        ),
        Argument(
            "max_time",
            int,
            optional=True,
            default=9000,
            doc="the max time (in second) of structure optimization. ",
        ),
        Argument(
            "pick_up",
            bool,
            optional=True,
            default=False,
            doc="whether to continue the calculation. ",
        ),
        Argument(
            "pick_step",
            int,
            optional=True,
            default=0,
            doc="from which step to continue the calculation. ",
        ),
        Argument(
            "parallel",
            bool,
            optional=True,
            default=False,
            doc="whether to run calypso in parallel.",
        ),
        Argument(
            "split",
            bool,
            optional=True,
            default=True,
            doc="sperate generating structures and structure optimizations. in dpgen2, split must be True.",
        ),
        Argument(
            "spec_space_group",
            list,
            optional=True,
            default=[2, 230],
            doc="the range of spacegroup.",
        ),
        Argument(
            "vsc",
            bool,
            optional=True,
            default=False,
            doc="whether to run calypso in variational stoichiometry way.",
        ),
        Argument(
            "ctrl_range",
            list,
            optional=True,
            default=[[1, 10]],
            doc="the atom range of each atoms.",
        ),
        Argument(
            "max_numb_atoms",
            int,
            optional=True,
            default=100,
            doc="the max number of atoms.",
        ),
    ]


def caly_task_group_args():
    doc_caly_task_grp = "CALYPSO structure prediction tasks. DPGEN will generate the calypso input script"
    return Argument(
        "task_group",
        dict,
        caly_task_grp_args(),
        doc=doc_caly_task_grp,
    )


def caly_normalize(data):
    args = caly_task_group_args()
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


def make_calypso_task_group_from_config(config):
    config.pop("type", None)
    config = caly_normalize(config)

    tgroup = CalyTaskGroup()
    tgroup.set_params(**config)
    return tgroup


def diffcsp_task_group_args():
    doc_diffcsp_task_grp = "DiffCSP exploration tasks"
    doc_trj_freq = "The frequency of dumping configurations and model devis"
    doc_fmax = "Force tolerence in relaxation"
    doc_steps = "Maximum number of steps in relaxation"
    doc_timeout = "Timeout (seconds) in relaxation"
    return Argument(
        "task_group",
        dict,
        [
            Argument(
                "trj_freq",
                int,
                optional=True,
                default=10,
                doc=doc_trj_freq,
                alias=["t_freq", "trj_freq", "traj_freq"],
            ),
            Argument(
                "fmax",
                float,
                optional=True,
                default=1e-4,
                doc=doc_fmax,
            ),
            Argument(
                "steps",
                int,
                optional=True,
                default=200,
                doc=doc_steps,
            ),
            Argument(
                "timeout",
                int,
                optional=True,
                default=None,
                doc=doc_timeout,
            ),
        ],
        doc=doc_diffcsp_task_grp,
    )


def diffcsp_normalize(config):
    args = diffcsp_task_group_args()
    config = args.normalize_value(config, trim_pattern="_*")
    args.check_value(config, strict=False)
    return config


def make_diffcsp_task_group_from_config(config):
    config = diffcsp_normalize(config)
    tgroup = DiffCSPTaskGroup(**config)
    return tgroup


def make_lmp_task_group_from_config(
    numb_models,
    mass_map,
    config,
):
    # Work around the required conf_idx.
    # May not be a good design!!!
    config["conf_idx"] = [] if "conf_idx" not in config else None
    config = lmp_normalize(config)
    config = config_strip_confidx(config)

    if config["type"] == "lmp-md":
        tgroup = NPTTaskGroup()
        config.pop("type")
        tgroup.set_md(
            numb_models,
            mass_map,
            **config,
        )
    elif config["type"] == "lmp-nvnmd":
        tgroup = NPTTaskGroup()
        config.pop("type")
        config["nvnmd_version"] = "0.0"
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
    elif config["type"] == "lmp-nvnmd-template":
        tgroup = LmpTemplateTaskGroup()
        config.pop("type")
        config["nvnmd_version"] = "0.0"
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
    print(lmp_normalize({"type": "lmp-md"}))
