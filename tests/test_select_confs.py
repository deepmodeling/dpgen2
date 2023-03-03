import json
import os
import shutil
import time
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
    Tuple,
)

import jsonpickle
import numpy as np
from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    S3Artifact,
    Step,
    Steps,
    Workflow,
    argo_range,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    FatalError,
    OPIOSign,
    PythonOPTemplate,
)

try:
    from context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from mocked_ops import (
    MockedConfSelector,
    MockedExplorationReport,
    MockedSelectConfs,
)

from dpgen2.op.select_confs import (
    SelectConfs,
)


class TestMockedSelectConfs(unittest.TestCase):
    def setUp(self):
        self.conf_selector = MockedConfSelector()
        self.traj_fmt = "foo"
        self.type_map = []
        self.trajs = [Path("traj.foo"), Path("traj.bar")]
        self.model_devis = [Path("md.foo"), Path("md.bar")]

    def tearDown(self):
        for ii in ["conf.0", "conf.1"]:
            ii = Path(ii)
            if ii.is_file():
                os.remove(ii)

    def test(self):
        op = MockedSelectConfs()
        out = op.execute(
            OPIO(
                {
                    "conf_selector": self.conf_selector,
                    "type_map": self.type_map,
                    "trajs": self.trajs,
                    "model_devis": self.model_devis,
                }
            )
        )
        confs = out["confs"]
        report = out["report"]

        # self.assertTrue(report.converged())
        self.assertTrue(confs[0].is_file())
        self.assertTrue(confs[1].is_file())
        self.assertTrue(confs[0].read_text(), "conf of conf.0")
        self.assertTrue(confs[1].read_text(), "conf of conf.1")


class TestSelectConfs(unittest.TestCase):
    def setUp(self):
        self.conf_selector = MockedConfSelector()
        self.type_map = []
        self.trajs = [Path("traj.foo"), Path("traj.bar")]
        self.model_devis = [Path("md.foo"), Path("md.bar")]

    def tearDown(self):
        for ii in ["conf.0", "conf.1"]:
            ii = Path(ii)
            if ii.is_file():
                os.remove(ii)

    def test(self):
        op = SelectConfs()
        out = op.execute(
            OPIO(
                {
                    "conf_selector": self.conf_selector,
                    "type_map": self.type_map,
                    "trajs": self.trajs,
                    "model_devis": self.model_devis,
                }
            )
        )
        confs = out["confs"]
        report = out["report"]

        # self.assertTrue(report.converged())
        self.assertTrue(confs[0].is_file())
        self.assertTrue(confs[1].is_file())
        self.assertTrue(confs[0].read_text(), "conf of conf.0")
        self.assertTrue(confs[1].read_text(), "conf of conf.1")

    def test_validate_trajs(self):
        trajs = ["foo", "bar", None, "tar"]
        model_devis = ["zar", "par", None, "mar"]
        trajs, model_devis = SelectConfs.validate_trajs(trajs, model_devis)
        self.assertEqual(trajs, ["foo", "bar", "tar"])
        self.assertEqual(model_devis, ["zar", "par", "mar"])

        trajs = ["foo", "bar", None, "tar"]
        model_devis = ["zar", "par", None]
        with self.assertRaises(FatalError) as context:
            trajs, model_devis = SelectConfs.validate_trajs(trajs, model_devis)

        trajs = ["foo", "bar"]
        model_devis = ["zar", None]
        with self.assertRaises(FatalError) as context:
            trajs, model_devis = SelectConfs.validate_trajs(trajs, model_devis)

        trajs = ["foo", None]
        model_devis = ["zar", "par"]
        with self.assertRaises(FatalError) as context:
            trajs, model_devis = SelectConfs.validate_trajs(trajs, model_devis)
