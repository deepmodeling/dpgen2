import unittest, json, shutil, os
import dflow
from dflow import (
    Workflow,
)
import mock, textwrap
from dpgen2.entrypoint.workflow import (
    execute_workflow_subcommand,
)


class ParserTest(unittest.TestCase):
    @mock.patch("dflow.Workflow.terminate")
    def test_terminate(self, mocked_f):
        config = json.loads(foo_str)
        execute_workflow_subcommand("terminate", "foo", config)
        mocked_f.assert_called_with()

    @mock.patch("dflow.Workflow.stop")
    def test_stop(self, mocked_f):
        config = json.loads(foo_str)
        execute_workflow_subcommand("stop", "foo", config)
        mocked_f.assert_called_with()

    @mock.patch("dflow.Workflow.suspend")
    def test_suspend(self, mocked_f):
        config = json.loads(foo_str)
        execute_workflow_subcommand("suspend", "foo", config)
        mocked_f.assert_called_with()

    @mock.patch("dflow.Workflow.delete")
    def test_delete(self, mocked_f):
        config = json.loads(foo_str)
        execute_workflow_subcommand("delete", "foo", config)
        mocked_f.assert_called_with()

    @mock.patch("dflow.Workflow.retry")
    def test_retry(self, mocked_f):
        config = json.loads(foo_str)
        execute_workflow_subcommand("retry", "foo", config)
        mocked_f.assert_called_with()

    @mock.patch("dflow.Workflow.resume")
    def test_resume(self, mocked_f):
        config = json.loads(foo_str)
        execute_workflow_subcommand("resume", "foo", config)
        mocked_f.assert_called_with()


foo_str = textwrap.dedent(
    """
{
    "default_step_config" : {
	"template_config" : {
	    "image" : "dflow:1.1.4",
	    "_comment" : "all"
	},
	"_comment" : "all"
    },

    "step_configs":{
	"_comment" : "all"
    },

    "upload_python_packages" : "/path/to/dpgen2",

    "inputs": {
	"type_map":		["Al", "Mg"],
	"mass_map":		[27, 24],
	"init_data_prefix":	"",
	"init_data_sys":	[
	    "init/al.fcc.01x01x01/02.md/sys-0004/deepmd",
	    "init/mg.fcc.01x01x01/02.md/sys-0004/deepmd"
	],
	"_comment" : "all"
    },
    "train":{
	"type" :	"dp",
	"numb_models" : 4,
	"config" : {},
	"template_script" : "dp_input_template",
	"_comment" : "all"
    },

    "explore" : {
	"type" : "lmp",
	"config" : {
	    "command": "lmp -var restart 0"
	},
	"max_numb_iter" :	5,
	"fatal_at_max" :	false,
        "convergence":{
                "type": "fixed-levels",
                "level_f_lo":		0.05,
                "level_f_hi":		0.50,
                "conv_accuracy" :	0.9
        },
	"configuration_prefix": null, 
	"configuration":	[
	],
	"stages":	[
	],
	"_comment" : "all"
    },
    "fp" : {
	"type" :	"vasp",
	"run_config" : {
	    "command": "source /opt/intel/oneapi/setvars.sh && mpirun -n 16 vasp_std"
	},
	"task_max":	2,
	"inputs_config" : {
	    "pp_files":	{"Al" : "vasp/POTCAR.Al", "Mg" : "vasp/POTCAR.Mg"},
	    "incar":    "vasp/INCAR",
	    "kspacing":	0.32,
	    "kgamma":	true
	},
	"_comment" : "all"
    }
}
"""
)
