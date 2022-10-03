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
	"s3_endpoint" : "address.of.the.s3.sever:port"
    },
```
The `dpgen` simply pass all keys of `"dflow_config"` to `dflow.config` and all keys of `"dflow_s3_config"` to `dflow.s3_config`. 


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
The key `"init_data_sys"` provides the initial training data to kick-off the training of deep potential (DP) models.


### Training

This section defines how a model is trained. 
```json
"train" : {
	"type" : "dp",
	"numb_models" : 4,
	"config" : {},
	"template_script" : {
		"_comment" : "omitted content of tempalte script"
	},
	"_comment" : "all"
}
```
The `"type" : "dp"` tell the traning method is `"dp"`, i.e. calling [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) to train DP models. 
The `"config"` key defines the training configs, see [the full documentation](rundptrainargs). 
The `"template_script"` provides the template training script in `json` format. 


### Exploration

This section defines how the configuration space is explored. 
```json
"explore" : {
	"type" : "lmp",
	"config" : {
		"command": "lmp -var restart 0"
	},
	"max_numb_iter" :	5,
	"conv_accuracy" :	0.9,
	"fatal_at_max" :	false,
	"f_trust_lo":		0.05,
	"f_trust_hi":		0.50,
	"configurations":	[
		{
		"lattice" : ["fcc", 4.57],
		"replicate" : [2, 2, 2],
		"numb_confs" : 30,
		"concentration" : [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
		}
		{
		"lattice" : ["fcc", 4.57],
		"replicate" : [3, 3, 3],
		"numb_confs" : 30,
		"concentration" : [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
		}
	],
	"stages":	[
		{ "_idx": 0, "ensemble": "nvt", "nsteps": 20, "press": null, "conf_idx": [0], "temps": [50,100], "trj_freq": 10, "n_sample" : 3 },
		{ "_idx": 1, "ensemble": "nvt", "nsteps": 20, "press": null, "conf_idx": [1], "temps": [50,100], "trj_freq": 10, "n_sample" : 3 }
	],
}
```
The `"type" : "lmp"` means that configurations are explored by LAMMPS DPMD runs. 
The `"config"` key defines the lmp configs, see [the full documentation](runlmpargs). 
The `"configurations"` provides the initial configurations (coordinates of atoms and the simulation cell) of the DPMD simulations. It is a list. The elements of the list can be 

- `list[str]`: The strings provides the path to the configuration files.
- `dict`: Automatic alloy configuration generator. See [the detailed doc](alloy_configs) of the allowed keys.

The `"stages"` defines the exploration stages. It is a list of `dict`s, with each `dict` defining a stage. The `"ensemble"`, `"nsteps"`, `"press"`, `"temps"`, `"traj_freq"` keys are self-explanatory. `"conf_idx"` pickes initial configurations of DPMD simulations from the `"configurations"` list, it provides the index of the element in the `"configurations"` list. `"n_sample"` tells the number of confgiruations randomly sampled from the set picked by `"conf_idx"` for each thermodynamic state. All configurations picked by `"conf_idx"` has the same possibility to be sampled. The default value of `"n_sample"` is `null`, in this case all picked configurations are sampled. In the example, each stage have 3 samples and 2 thermodynamic states (NVT, T=50 and 100K), then each iteration run 3x2=6 NVT DPMD simulatins.


### FP

This section defines the first-principle (FP) calculation . 

```json
"fp" : {
	"type" :	"vasp",
	"config" : {
		"command": "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
	},
	"task_max":	2,
	"pp_files":	{"Al" : "vasp/POTCAR.Al", "Mg" : "vasp/POTCAR.Mg"},
	"incar":         "vasp/INCAR",
	"_comment" : "all"
}
```
The `"type" : "vasp"` means that first-principles are VASP calculations. 
The `"config"` key defines the vasp configs, see [the full documentation](runvaspargs). 
The `"task_max"` key defines the maximal number of vasp calculations in each dpgen2 iteration.
The `"pp_files"` and `"incar"` keys provides the pseudopotential files and the template incar file.


### Configuration of dflow step

The execution units of the dpgen2 are the dflow `Step`s. How each step is executed is defined by the `"step_configs"`.
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

The readers are refered to [this page](stepconfigargs) for a full documentation of the step configs.

Any of the config in the `step_configs` can be ommitted. If so, the configs of the step is set to the default step configs, which is provided by the following section, for example,
```json
"default_step_config" : {
	"template_config" : {
	    "image" : "dpgen2:x.x.x"
	}
},
```
The way of writing the `default_step_config` is the same as any step config in the `step_configs`. One may refer to [this page](stepconfigargs) for full documentation.
