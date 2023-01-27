import dargs
from dargs import (
    Argument,
    Variant,
)

from dpgen2.exploration.task import (
    LmpTemplateTaskGroup,
    NPTTaskGroup,
)


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


def variant_task_group():
    doc = "the type of the task group"
    return Variant(
        "type",
        [
            Argument("lmp-md", dict, npt_task_group_args(), alias=["lmp-npt"]),
            Argument("lmp-template", dict, lmp_template_task_group_args()),
        ],
        doc=doc,
    )


def task_group_args():
    return Argument("task_group_configs", dict, [], [variant_task_group()])


def normalize(data):
    args = task_group_args()
    data = args.normalize_value(data, trim_pattern="_*")
    args.check_value(data, strict=False)
    return data


def make_task_group_from_config(
    numb_models,
    mass_map,
    config,
):
    config = normalize(config)
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
    else:
        raise RuntimeError("unknown task group type: ", config["type"])
    return tgroup


if __name__ == "__main__":
    print(normalize({"type": "lmp-md"}))
