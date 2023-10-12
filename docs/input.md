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
