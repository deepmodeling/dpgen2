from utils.context import dpgen2
import numpy as np
import unittest, json, shutil, os
from pathlib import Path
from dpgen2.constants import default_image
# from dpgen2.utils.step_config import normalize, gen_doc, init_executor
from dpgen2.utils import (
    normalize_step_dict as normalize,
    gen_doc_step_dict as gen_doc,
    init_executor,
)
import dflow

class TestStepConfig(unittest.TestCase):
    def test_success(self):
        idict = {
            "template_config":{
                "image" : "bula",
            },
            "executor" : {
                "type" : "lebesgue_v2",
                "extra" : {
                    "scass_type" : "foo",
                    "program_id" : "bar",
                },
            },
        }
        expected_odict = {
            "template_config":{
                "image" : "bula",
                "timeout" : None,
                "retry_on_transient_error" : None,
                "timeout_as_transient_error" : False,
            },
            "continue_on_failed" : False,
            "continue_on_num_success" : None,
            "continue_on_success_ratio" : None,
            "executor" : {
                "type" : "lebesgue_v2",
                "extra" : {
                    "scass_type" : "foo",
                    "program_id" : "bar",
                    "job_type" : "container",
                    "template_cover_cmd_escape_bug" : True,
                },
            },
        }
        odict = normalize(idict)
        self.assertEqual(odict, expected_odict)

    def test_empty(self):
        idict = { }
        expected_odict = {
            "template_config":{
                "image" : default_image,
                "timeout" : None,
                "retry_on_transient_error" : None,
                "timeout_as_transient_error" : False,
            },
            "continue_on_failed" : False,
            "continue_on_num_success" : None,
            "continue_on_success_ratio" : None,
            "executor" : None,
        }
        odict = normalize(idict)
        self.assertEqual(odict, expected_odict)


    def test_init_executor(self):
        idict = {
            "template_config":{
                "image" : "bula",
            },
            "executor" : {
                "type" : "lebesgue_v2",
                "extra" : {
                    "scass_type" : "foo",
                    "program_id" : "bar",
                },
            },
        }
        odict = normalize(idict)
        ret = init_executor(odict.pop('executor'))
        self.assertEqual(type(ret), dflow.plugins.lebesgue.LebesgueExecutor)


    def test_init_executor_notype(self):
        idict = {
            "template_config":{
                "image" : "bula",
            },
        }
        odict = normalize(idict)
        ret = init_executor(odict.pop('executor'))
        self.assertEqual(ret, None)
