import json
import shutil
import unittest
from pathlib import (
    Path,
)

import numpy as np
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
)
from mock import (
    mock,
)

# isort: off
from .context import (
    dpgen2,
)
from dpgen2.constants import (
    train_script_name,
    train_task_pattern,
)
from dpgen2.op.prep_nvnmd_train import (
    PrepNvNMDTrain,
)

# isort: on

template_script_nvnmd_v0 = {
    "nvnmd": {"version": 0, "seed": 1},
    "training": {
        "systems": [],
        "stop_batch": 2000,
        "batch_size": "auto",
        "seed": 1,
    },
}


template_script_nvnmd_v1 = {
    "nvnmd": {"version": 1, "seed": 1},
    "training": {
        "systems": [],
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


class TestPrepNvNMDTrain(unittest.TestCase):
    def setUp(self):
        self.numb_models = 2
        self.ptrain = PrepNvNMDTrain()

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

    def test_template_nvnmd_v1(self):
        ip = OPIO(
            {
                "template_script": template_script_nvnmd_v1,
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
                self.assertEqual(jdata["nvnmd"]["version"], 1)
                self.assertEqual(jdata["nvnmd"]["seed"], 2 * ii + 0)
                self.assertEqual(jdata["training"]["seed"], 2 * ii + 1)

    def test_template_nvnmd_v0(self):
        ip = OPIO(
            {
                "template_script": template_script_nvnmd_v0,
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
                self.assertEqual(jdata["nvnmd"]["version"], 0)
                self.assertEqual(jdata["nvnmd"]["seed"], 2 * ii + 0)
                self.assertEqual(jdata["training"]["seed"], 2 * ii + 1)

    def test_template_list_nvnmd_v0_v1(self):
        ip = OPIO(
            {
                "template_script": [template_script_nvnmd_v0, template_script_nvnmd_v1],
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
            self.assertEqual(jdata["nvnmd"]["version"], 0)
            self.assertEqual(jdata["nvnmd"]["seed"], 2 * ii)
            self.assertEqual(jdata["training"]["seed"], 2 * ii + 1)
        ii = 1
        with open(Path(train_task_pattern % ii) / train_script_name) as fp:
            jdata = json.load(fp)
            self.assertEqual(jdata["nvnmd"]["version"], 1)
            self.assertEqual(jdata["nvnmd"]["seed"], 2 * ii)
            self.assertEqual(jdata["training"]["seed"], 2 * ii + 1)

    def test_template_raise_wrong_list_length(self):
        ip = OPIO(
            {
                "template_script": [
                    template_script_nvnmd_v1,
                    template_script_nvnmd_v0,
                    template_script_nvnmd_v1,
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
