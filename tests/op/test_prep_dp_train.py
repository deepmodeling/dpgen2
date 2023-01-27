from op.context import dpgen2
import numpy as np
import unittest, json, shutil
from mock import mock

from dflow.python import OP, OPIO, OPIOSign, Artifact
from pathlib import Path
from dpgen2.constants import (
    train_task_pattern,
    train_script_name,
)
from dpgen2.op.prep_dp_train import PrepDPTrain

template_script_se_e2_a = {
    "model": {"descriptor": {"type": "se_e2_a", "seed": 1}, "fitting_net": {"seed": 1}},
    "training": {
        "systems": [],
        "set_prefix": "set",
        "stop_batch": 2000,
        "batch_size": "auto",
        "seed": 1,
    },
}

template_script_hybrid = {
    "model": {
        "descriptor": {
            "type": "hybrid",
            "list": [{"type": "se_e2_a", "seed": 1}, {"type": "se_e3", "seed": 1}],
        },
        "fitting_net": {"seed": 1},
    },
    "training": {
        "systems": [],
        "set_prefix": "set",
        "stop_batch": 2000,
        "batch_size": "auto",
        "seed": 1,
    },
}


class faked_rg:
    faked_random = -1

    @classmethod
    def randrange(cls, xx):
        cls.faked_random += 1
        return cls.faked_random


class TestPrepDPTrain(unittest.TestCase):
    def setUp(self):
        self.numb_models = 2
        self.ptrain = PrepDPTrain()

    def tearDown(self):
        for ii in range(self.numb_models):
            if Path(train_task_pattern % ii).exists():
                shutil.rmtree(train_task_pattern % ii)

    def _check_output_dir_and_file_exist(self, op, numb_models):
        task_names = op["task_names"]
        task_paths = op["task_paths"]
        for ii in range(self.numb_models):
            self.assertEqual(train_task_pattern % ii, task_names[ii])
            self.assertEqual(Path(train_task_pattern % ii), task_paths[ii])
            self.assertTrue(task_paths[ii].is_dir())
            self.assertTrue((task_paths[ii] / train_script_name).is_file())

    def test_template_str_hybrid(self):
        ip = OPIO(
            {"template_script": template_script_hybrid, "numb_models": self.numb_models}
        )

        faked_rg.faked_random = -1
        with mock.patch("random.randrange", faked_rg.randrange):
            op = self.ptrain.execute(ip)

        self._check_output_dir_and_file_exist(op, self.numb_models)

        for ii in range(self.numb_models):
            with open(Path(train_task_pattern % ii) / train_script_name) as fp:
                jdata = json.load(fp)
                self.assertEqual(
                    jdata["model"]["descriptor"]["list"][0]["seed"], 4 * ii + 0
                )
                self.assertEqual(
                    jdata["model"]["descriptor"]["list"][1]["seed"], 4 * ii + 1
                )
                self.assertEqual(jdata["model"]["fitting_net"]["seed"], 4 * ii + 2)
                self.assertEqual(jdata["training"]["seed"], 4 * ii + 3)

    def test_template_str_se_e2_a(self):

        ip = OPIO(
            {
                "template_script": template_script_se_e2_a,
                "numb_models": self.numb_models,
            }
        )

        faked_rg.faked_random = -1
        with mock.patch("random.randrange", faked_rg.randrange):
            op = self.ptrain.execute(ip)

        self._check_output_dir_and_file_exist(op, self.numb_models)

        for ii in range(self.numb_models):
            with open(Path(train_task_pattern % ii) / train_script_name) as fp:
                jdata = json.load(fp)
                self.assertEqual(jdata["model"]["descriptor"]["seed"], 3 * ii + 0)
                self.assertEqual(jdata["model"]["fitting_net"]["seed"], 3 * ii + 1)
                self.assertEqual(jdata["training"]["seed"], 3 * ii + 2)

    def test_template_list_hyb_sea(self):

        ip = OPIO(
            {
                "template_script": [template_script_hybrid, template_script_se_e2_a],
                "numb_models": self.numb_models,
            }
        )

        faked_rg.faked_random = -1
        with mock.patch("random.randrange", faked_rg.randrange):
            op = self.ptrain.execute(ip)

        self._check_output_dir_and_file_exist(op, self.numb_models)

        ii = 0
        with open(Path(train_task_pattern % ii) / train_script_name) as fp:
            jdata = json.load(fp)
            self.assertEqual(
                jdata["model"]["descriptor"]["list"][0]["seed"], 4 * ii + 0
            )
            self.assertEqual(
                jdata["model"]["descriptor"]["list"][1]["seed"], 4 * ii + 1
            )
            self.assertEqual(jdata["model"]["fitting_net"]["seed"], 4 * ii + 2)
            self.assertEqual(jdata["training"]["seed"], 4 * ii + 3)
        ii = 1
        with open(Path(train_task_pattern % ii) / train_script_name) as fp:
            jdata = json.load(fp)
            self.assertEqual(jdata["model"]["descriptor"]["seed"], 4 * ii + 0)
            self.assertEqual(jdata["model"]["fitting_net"]["seed"], 4 * ii + 1)
            self.assertEqual(jdata["training"]["seed"], 4 * ii + 2)

    def test_template_raise_wrong_list_length(self):

        ip = OPIO(
            {
                "template_script": [
                    template_script_hybrid,
                    template_script_hybrid,
                    template_script_se_e2_a,
                ],
                "numb_models": self.numb_models,
            }
        )

        with self.assertRaises(RuntimeError) as context:
            faked_rg.faked_random = -1
            with mock.patch("random.randrange", faked_rg.randrange):
                op = self.ptrain.execute(ip)
        self.assertTrue(
            "length of the template list should be equal to 2" in str(context.exception)
        )
