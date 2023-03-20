import json
import shutil
import unittest
import os
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
from op.context import (
    dpgen2,
)

from dpgen2.constants import (
    train_script_name,
    train_task_pattern,
)
from dpgen2.op.modify_train_script import (
    ModifyTrainScript,
)

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


class TestModifyTrainScript(unittest.TestCase):
    def setUp(self):
        self.numb_models = 2
        self.ptrain = ModifyTrainScript()

    def tearDown(self):
        for ii in range(self.numb_models):
            if Path(train_task_pattern % ii).exists():
                shutil.rmtree(train_task_pattern % ii)

    def test_template_str_se_e2_a(self):
        for ii in range(self.numb_models):
            os.mkdir(Path(train_task_pattern % ii))
            with open(Path(train_task_pattern % ii) / train_script_name, "w") as fp:
                json.dump(template_script_se_e2_a, fp, indent=4)
        ip = OPIO(
            {
                "numb_models": self.numb_models,
                "scripts": Path("."),
            }
        )

        op = self.ptrain.execute(ip)

        template_script = op["template_script"]

        assert isinstance(template_script, list)
        for ii in range(self.numb_models):
            with open(Path(train_task_pattern % ii) / train_script_name) as fp:
                jdata = json.load(fp)
                self.assertEqual(jdata["model"], template_script[ii]["model"])
        
