import os
import unittest
from pathlib import (
    Path,
)

from dflow.python import (
    OPIO,
    FatalError,
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
        trajs, model_devis, _ = SelectConfs.validate_trajs(trajs, model_devis, None)
        self.assertEqual(trajs, ["foo", "bar", "tar"])
        self.assertEqual(model_devis, ["zar", "par", "mar"])

        trajs = ["foo", "bar", None, "tar"]
        model_devis = ["zar", "par", None]
        with self.assertRaises(FatalError) as context:
            trajs, model_devis, _ = SelectConfs.validate_trajs(trajs, model_devis, None)

        trajs = ["foo", "bar"]
        model_devis = ["zar", None]
        with self.assertRaises(FatalError) as context:
            trajs, model_devis, _ = SelectConfs.validate_trajs(trajs, model_devis, None)

        trajs = ["foo", None]
        model_devis = ["zar", "par"]
        with self.assertRaises(FatalError) as context:
            trajs, model_devis, _ = SelectConfs.validate_trajs(trajs, model_devis, None)

        trajs = ["foo", "bar", None, "tar"]
        model_devis = ["zar", "par", None, "mar"]
        optional_outputs = ["dar", "far", None, "gar"]
        trajs, model_devis, optional_outputs = SelectConfs.validate_trajs(
            trajs, model_devis, optional_outputs
        )
        self.assertEqual(trajs, ["foo", "bar", "tar"])
        self.assertEqual(model_devis, ["zar", "par", "mar"])
        self.assertEqual(optional_outputs, ["dar", "far", "gar"])

        trajs = ["foo", "bar"]
        model_devis = ["zar", "par"]
        optional_outputs = ["dar", None]
        with self.assertRaises(FatalError) as context:
            trajs, model_devis, optional_outputs = SelectConfs.validate_trajs(
                trajs, model_devis, optional_outputs
            )
