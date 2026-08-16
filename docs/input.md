(inputscript)=
# Guide on writing input scripts for dpgen2 commands

## Preliminaries

The reader of this doc is assumed to be familiar with the concurrent learning algorithm that the dpgen2 implements. If not, one may check [this paper](https://doi.org/10.1016/j.cpc.2020.107206).

## The input script for all dpgen2 commands

For all the dpgen2 commands, one need to provide `dflow2` global configurations. For example,
```json
    "dflow_config" : {
	"host" : "http://address.of.the.host:port"
    },
    "dflow_s3_config" : {
	"endpoint" : "address.of.the.s3.sever:port"
    },
```
The `dpgen` simply pass all keys of `"dflow_config"` to [`dflow.config`](https://deepmodeling.com/dflow/dflow.html#dflow.config.set_config) and all keys of `"dflow_s3_config"` to [`dflow.s3_config`](https://deepmodeling.com/dflow/dflow.html#dflow.utils.set_s3_config).


## The input script for `submit` and `resubmit`

The full documentation of the `submit` and `resubmit` script can be [found here](submitargs). This documentation provides a fast guide on how to write the input script.

In the input script of `dpgen2 submit` and `dpgen2 resubmit`, one needs to provide the definition of the workflow and how they are executed in the input script. One may find an example input script in the [dpgen2 Al-Mg alloy example](../examples/almg/input.json).

The definition of the workflow can be provided by the following sections:

### Inputs

This section provides the inputs to start a dpgen2 workflow. An example for the Al-Mg alloy
```json
"inputs": {
	"type_map":		["Al", "Mg"],
	"mass_map":		[27, 24],
	"init_data_sys":	[
		"path/to/init/data/system/0",
		"path/to/init/data/system/1"
	],
}
```
The key {dargs:argument}`"init_data_sys" <inputs/init_data_sys>` provides the initial training data to kick-off the training of deep potential (DP) models.


### Training

This section defines how a model is trained.
```json
"train" : {
	"type" : "dp",
	"numb_models" : 4,
	"config" : {},
	"template_script" : "/path/to/the/template/input.json",
	"_comment" : "all"
}
```
The `"type" : "dp"` tell the traning method is {dargs:argument}`"dp" <train>`, i.e. calling [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) to train DP models.
The `"config"` key defines the training configs, see {ref}`the full documentation<train[dp]/config>`.
The {dargs:argument}`"template_script" <train[dp]/template_script>` provides the template training script in `json` format.


### Exploration

This section defines how the configuration space is explored.
```json
"explore" : {
	"type" : "lmp",
	"config" : {
		"command": "lmp -var restart 0"
	},
	"convergence": {
	    "type" :	"fixed-levels",
	    "conv_accuracy" :	0.9,
	    "level_f_lo":	0.05,
	    "level_f_hi":	0.50,
	    "_comment" : "all"
	},
	"max_numb_iter" :	5,
	"fatal_at_max" :	false,
	"configurations":	[
		{
		"type": "alloy",
		"lattice" : ["fcc", 4.57],
		"replicate" : [2, 2, 2],
		"numb_confs" : 30,
		"concentration" : [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
		},
		{
		"type" : "file",
		"prefix": "/file/prefix",
		"files" : ["relpath/to/confs/*"],
		"fmt" : "deepmd/npy"
		}
	],
	"stages":	[
	    [
		{
		    "_comment" : "stage 0, task group 0",
		    "type" : "lmp-md",
		    "ensemble": "nvt", "nsteps":  50, "temps": [50, 100], "trj_freq": 10,
		    "conf_idx": [0], "n_sample" : 3
		},
		{
		    "_comment" : "stage 0, task group 1",
		    "type" : "lmp-template",
		    "lmp" : "template.lammps", "plm" : "template.plumed",
		    "trj_freq" : 10, "revisions" : {"V_NSTEPS" : [40], "V_TEMP" : [150, 200]},
		    "conf_idx": [0], "n_sample" : 3
		}
	    ],
	    [
		{
		    "_comment" : "stage 1, task group 0",
		    "type" : "lmp-md",
		    "ensemble": "npt", "nsteps":  50, "press": [1e0], "temps": [50, 100, 200], "trj_freq": 10,
		    "conf_idx": [1], "n_sample" : 3
		}
	    ]
	]
}
```
The {dargs:argument}`"type" : "lmp"<explore>` means that configurations are explored by LAMMPS DPMD runs.
The {dargs:argument}`"config"<explore[lmp]/config>` key defines the lmp configs.
The {dargs:argument}`"configurations"<explore[lmp]/configurations>` provides the initial configurations (coordinates of atoms and the simulation cell) of the DPMD simulations. It is a list. The elements of the list are `dict`s that defines how the configurations are generated

- Automatic alloy configuration generator. See {ref}`the detailed doc<explore[lmp]/configurations[alloy]>` for the allowed keys.
- Configurations load from files. See {ref}`the detailed doc<explore[lmp]/configurations[file]>` for the allowed keys.

The {dargs:argument}`"stages"<explore[lmp]/stages>` defines the exploration stages. It is of type `list[list[dict]]`. The outer `list` enumerate the exploration stages, the inner list enumerate the task groups of the stage. Each `dict` defines a stage. See {ref}`the full documentation of the task group<task_group_sec>` for writting task groups.

The {dargs:argument}`"n_sample"<task_group[lmp-md]/n_sample>` tells the number of confgiruations randomly sampled from the set picked by {dargs:argument}`"conf_idx"<task_group[lmp-md]/conf_idx>` from {dargs:argument}`"configurations"<explore[lmp]/configurations>` for each exploration task. All configurations has the equal possibility to be sampled. The default value of `"n_sample"` is `null`, in this case all picked configurations are sampled. In the example, we have 3 samples for stage 0 task group 0 and 2 thermodynamic states (NVT, T=50 and 100K), then the task group has 3x2=6 NVT DPMD tasks.

#### PLUMED CV candidate filtering

LAMMPS exploration candidates can be restricted to a union of named PLUMED CV
regions after the model-deviation trust window and before final CV-space
coverage or an explicitly configured selection policy:

```json
"explore": {
    "config": {
        "plm_output_file": "COLVAR"
    },
    "cv_filter": {
        "regions": [
            {"d": [0.08, 0.12]},
            {"v": [1.8, 2.2]}
        ],
        "sampling": {"mode": "report"}
    }
}
```

Each region is a mapping from a field in the PLUMED `#! FIELDS` header to a
lower-inclusive, upper-exclusive interval. Conditions within a region are
combined by AND and regions are combined by OR. The PLUMED input must write the
selected fields with the same stride as `trj_freq`, for example:

```plumed
LOAD FILE=/absolute/path/ReactiveVoronoi.so
d: DISTANCE ATOMS=1,2
v: VORONOI_COORDINATION ...
PRINT ARG=d,v STRIDE=10 FILE=COLVAR
```

`LOAD` is only needed for CVs that are not built into the active PLUMED. Build
such a plugin with that same PLUMED installation (for example, `plumed mklib
ReactiveVoronoi.cpp`); shared libraries from a different compiler or PLUMED
build may be ABI-incompatible.

The region bounds use the units written to `COLVAR` (PLUMED's default length
unit is nm). The filter fails if the file, field, finite values, strictly
increasing `time`, or row-to-trajectory alignment is invalid. Model-deviation
trust levels are applied first, followed by the CV regions and then the existing
candidate limit and selection policy.

By default, DPGEN2 prevents candidates from clustering where the trajectory
spends most of its time. If all regions share one CV field, it uses 10
equal-width bins along that CV. If they share two CV fields, it uses a 10 by 10
grid. In both cases it covers separated non-empty bins or cells and selects the
largest force model deviation within each one. An explicit policy can override
these defaults:

```json
"cv_filter": {
    "regions": [
        {"d": [0.08, 0.12]},
        {"d": [0.16, 0.24], "v": [1.8, 2.2]}
    ],
    "sampling": {
        "mode": "uniform",
        "field": "d",
        "n_bins": 10,
        "within_bin": "max_deviation",
        "seed": 20260815
    }
}
```

`uniform` divides the configured interval for `field` in each region into
equal-width bins. Regions receive a balanced share of the FP task limit and
the selected non-empty bins span the available interval. `within_bin` is
either `random` or `max_deviation`; `seed` makes random choices reproducible.
Every region must bound the primary `field`; its other CVs remain AND
constraints. Use `{"mode": "random", "seed": 20260815}` for reproducible
candidate-frame random selection after the CV filter; this follows the
trajectory's CV density and can therefore cluster in a highly populated CV
region. Use `{"mode": "report"}` to retain the
convergence report's original random or maximum-deviation selection. If regions
do not share exactly one or two CV fields, sampling must be specified because
DPGEN2 cannot infer an unambiguous coverage space.

For two-CV coverage and auditable reaction windows, regions may also have
names and weights:

```json
"cv_filter": {
    "regions": [
        {
            "name": "incipient_contact",
            "conditions": {
                "iondistance": [0.2, 2.0],
                "ionization": [0.2, 2.0]
            }
        },
        {
            "name": "separated",
            "conditions": {
                "iondistance": [2.0, 10.0],
                "ionization": [0.2, 2.0]
            },
            "weight": 1.0
        }
    ],
    "sampling": {
        "mode": "grid",
        "grid": {"iondistance": 8, "ionization": 4},
        "within_bin": "max_deviation",
        "seed": 20260815,
        "min_frame_gap": 5
    },
    "time_alignment": {
        "start": 0.0,
        "step": 0.01,
        "atol": 1e-8
    }
}
```

`grid` currently requires exactly two CV fields, both bounded by every region.
It allocates the candidate limit across regions (using `weight` when supplied),
covers separated non-empty cells before adding extra frames, and then uses
`within_bin` inside each cell. `min_frame_gap` is a minimum frame-index
separation within each trajectory. A spacing constraint may leave the result
underfilled; DPGEN2 reports this instead of silently relaxing the constraint.
`time_alignment` additionally verifies `time = start + frame * step` with the
configured absolute tolerance.

When a CV filter is active, the selected DeepMD data directory contains
`cv_selection.csv` and `cv_selection_summary.json`. They record trajectory and
frame IDs, time, maximum force model deviation, CV values, matching regions,
grid cells, population counts, spacing rejections, and any underfilled quota.


### FP

This section defines the first-principle (FP) calculation .

```json
"fp" : {
	"type": "vasp",
	"task_max":	2,
	"run_config": {
		"command": "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
	},
	"inputs_config": {
		"pp_files":	{"Al" : "vasp/POTCAR.Al", "Mg" : "vasp/POTCAR.Mg"},
		"kspacing":	0.32,
		"incar": "vasp/INCAR"
	}
}
```
The {dargs:argument}`"type" : "vasp"<fp>` means that first-principles are VASP calculations.
The {dargs:argument}`"run_config"<fp[vasp]/run_config>` key defines the configs for running VASP tasks.
The {dargs:argument}`"task_max"<fp[vasp]/task_max>` key defines the maximal number of vasp calculations in each dpgen2 iteration.
The {dargs:argument}`"pp_files"<fp[vasp]/inputs_config/pp_files>`, {dargs:argument}`"kspacing"<fp[vasp]/inputs_config/kspacing>` and {dargs:argument}`"incar"<fp[vasp]/inputs_config/incar>` keys provides the pseudopotential files, spacing for kspace sampling and the template incar file, respectively.


### Configuration of dflow step

The execution units of the dpgen2 are the dflow `Step`s. How each step is executed is defined by the {dargs:argument}`"step_configs"<step_configs>`.
```json
"step_configs":{
	"prep_train_config" : {
		"_comment" : "content omitted"
	},
	"run_train_config" : {
		"_comment" : "content omitted"
	},
	"prep_explore_config" : {
		"_comment" : "content omitted"
	},
	"run_explore_config" : {
		"_comment" : "content omitted"
	},
	"prep_fp_config" : {
		"_comment" : "content omitted"
	},
	"run_fp_config" : {
		"_comment" : "content omitted"
	},
	"select_confs_config" : {
		"_comment" : "content omitted"
	},
	"collect_data_config" : {
		"_comment" : "content omitted"
	},
	"cl_step_config" : {
		"_comment" : "content omitted"
	},
	"_comment" : "all"
},
```
The configs for prepare training, run training, prepare exploration, run exploration, prepare fp, run fp, select configurations, collect data and concurrent learning steps are given correspondingly.

Any of the config in the {dargs:argument}`"step_configs"<step_configs>` can be ommitted. If so, the configs of the step is set to the default step configs, which is provided by the following section, for example,
```json
"default_step_config" : {
	"template_config" : {
	    "image" : "dpgen2:x.x.x"
	}
},
```
The way of writing the {dargs:argument}`"default_step_config"<default_step_config>` is the same as any step config in the {dargs:argument}`"step_configs"<step_configs>`.
