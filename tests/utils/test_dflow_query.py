import os
import textwrap
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
)

import numpy as np

try:
    from exploration.context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dflow.python import (
    FatalError,
)

from dpgen2.exploration.report import (
    ExplorationReport,
)
from dpgen2.exploration.scheduler import (
    ConvergenceCheckStageScheduler,
    ExplorationScheduler,
)
from dpgen2.exploration.selector import (
    ConfSelectorFrames,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTaskGroup,
)
from dpgen2.utils.dflow_query import (
    find_slice_ranges,
    get_all_schedulers,
    get_iteration,
    get_last_iteration,
    get_last_scheduler,
    get_subkey,
    matched_step_key,
    print_keys_in_nice_format,
    sort_slice_ops,
)

# isort: off
import utils.context
from mocked_ops import (
    MockedExplorationReport,
    MockedExplorationTaskGroup,
    MockedExplorationTaskGroup1,
    MockedStage,
    MockedStage1,
)

# isort: on

dpgen_keys = [
    "finetune--prep-train",
    "finetune--run-train-0002",
    "finetune--run-train-0000",
    "finetune--run-train-0001",
    "finetune--modify-train-script",
    "finetune--prep-run-train",
    "init--scheduler",
    "init--id",
    "iter-000000--prep-train",
    "iter-000000--run-train-0002",
    "iter-000000--run-train-0000",
    "iter-000000--run-train-0001",
    "iter-000000--prep-run-train",
    "iter-000000--prep-lmp",
    "iter-000000--run-lmp-000001",
    "iter-000000--run-lmp-000004",
    "iter-000000--run-lmp-000005",
    "iter-000000--run-lmp-000002",
    "iter-000000--run-lmp-000003",
    "iter-000000--run-lmp-000000",
    "iter-000000--prep-run-explore",
    "iter-000000--select-confs",
    "iter-000000--prep-fp",
    "iter-000000--run-fp-000001",
    "iter-000000--run-fp-000000",
    "iter-000000--prep-run-fp",
    "iter-000000--collect-data",
    "iter-000000--block",
    "iter-000000--scheduler",
    "iter-000000--id",
    "iter-000001--prep-train",
    "iter-000001--run-train-0000",
    "iter-000001--run-train-0001",
    "iter-000001--run-train-0002",
    "iter-000001--prep-run-train",
    "iter-000001--prep-lmp",
    "iter-000001--run-lmp-000003",
    "iter-000001--run-lmp-000000",
    "iter-000001--run-lmp-000001",
    "iter-000001--run-lmp-000005",
    "iter-000001--run-lmp-000002",
    "iter-000001--run-lmp-000004",
    "iter-000001--prep-run-explore",
    "iter-000001--select-confs",
    "iter-000001--prep-fp",
    "iter-000001--run-fp-000001",
    "iter-000001--run-fp-000000",
    "iter-000001--prep-run-fp",
    "iter-000001--collect-data",
    "iter-000001--block",
    "iter-000001--scheduler",
    "iter-000001--id",
    "iter-000000--loop",
]


class MockedTar:
    def __init__(self):
        self.value = 10


class MockedFoo:
    def __init__(self):
        self.parameters = {"exploration_scheduler": MockedTar()}


class MockedBar:
    def __init__(self, xx, kk):
        self.key = kk
        self.outputs = MockedFoo()
        self.outputs.parameters["exploration_scheduler"].value = xx * 10

    def __getitem__(self, key):
        assert key == "phase"
        return "Succeeded"


def _get_step(key=None):
    if key == "init--scheduler":
        return [MockedBar(2, key)]
    elif key == "iter-0--scheduler":
        return [MockedBar(0, key)]
    elif key == "iter-1--scheduler":
        return [MockedBar(1, key)]
    else:
        raise RuntimeError("unexpected key")


class MockedWFInfo:
    def get_step(self, key=None):
        return _get_step(key)


class MockedWF:
    def __init__(
        self,
        none_global=True,
    ):
        self.none_global = none_global

    def query_step(self, key=None):
        return _get_step(key)

    def query(self):
        return MockedWFInfo()

    def query_global_outputs(self):
        # mocked return None: non-global scheduler output
        if self.none_global:
            return None
        else:
            return MockedFoo()

    def query_step_by_key(self, keys):
        ret = [_get_step(kk)[0] for kk in keys]
        return ret


class TestDflowQuery(unittest.TestCase):
    def test_get_subkey(self):
        self.assertEqual(get_subkey("aa--bb--cc", 0), "aa")
        self.assertEqual(get_subkey("aa--bb--cc", 1), "bb")
        self.assertEqual(get_subkey("aa--bb--cc", 2), "cc")
        self.assertEqual(get_subkey("aa--bb--cc"), "cc")
        self.assertEqual(get_subkey("aa"), "aa")
        self.assertEqual(get_subkey("aa---bb"), "-bb")
        self.assertEqual(get_subkey("aa----bb", 1), "")
        self.assertEqual(get_subkey(""), "")
        self.assertEqual(get_iteration("aa--bb--cc"), "aa")

    def test_matched_step_key(self):
        all_keys = ["iter-000--foo", "iter-111--bar", "iter-222--foo-001"]
        step_keys = ["foo"]
        ret = matched_step_key(all_keys, step_keys)
        self.assertEqual(ret, ["iter-000--foo", "iter-222--foo-001"])

    def test_get_last_scheduler(self):
        value = get_last_scheduler(
            MockedWF(),
            ["iter-1--scheduler", "foo", "bar", "iter-0--scheduler", "init--scheduler"],
        )
        self.assertEqual(value, 10)

    def test_get_last_scheduler(self):
        value = get_last_scheduler(
            MockedWF(none_global=False),
            ["iter-1--scheduler", "foo", "bar", "iter-0--scheduler", "init--scheduler"],
        )
        self.assertEqual(value, 10)

    def test_get_all_schedulers(self):
        value = get_all_schedulers(
            MockedWF(),
            ["iter-1--scheduler", "foo", "bar", "iter-0--scheduler", "init--scheduler"],
        )
        self.assertEqual(value, [20, 0, 10])

    def test_get_last_iteration(self):
        last = get_last_iteration(dpgen_keys)
        self.assertEqual(last, 1)

    def test_sort_slice_ops(self):
        idxes = find_slice_ranges(dpgen_keys, "run-lmp")
        self.assertEqual(idxes, [[8, 14], [30, 36]])

    def test_sort_slice_ops(self):
        expected_output = [
            "finetune--prep-train",
            "finetune--run-train-0000",
            "finetune--run-train-0001",
            "finetune--run-train-0002",
            "finetune--modify-train-script",
            "finetune--prep-run-train",
            "init--scheduler",
            "init--id",
            "iter-000000--prep-train",
            "iter-000000--run-train-0000",
            "iter-000000--run-train-0001",
            "iter-000000--run-train-0002",
            "iter-000000--prep-run-train",
            "iter-000000--prep-lmp",
            "iter-000000--run-lmp-000000",
            "iter-000000--run-lmp-000001",
            "iter-000000--run-lmp-000002",
            "iter-000000--run-lmp-000003",
            "iter-000000--run-lmp-000004",
            "iter-000000--run-lmp-000005",
            "iter-000000--prep-run-explore",
            "iter-000000--select-confs",
            "iter-000000--prep-fp",
            "iter-000000--run-fp-000000",
            "iter-000000--run-fp-000001",
            "iter-000000--prep-run-fp",
            "iter-000000--collect-data",
            "iter-000000--block",
            "iter-000000--scheduler",
            "iter-000000--id",
            "iter-000001--prep-train",
            "iter-000001--run-train-0000",
            "iter-000001--run-train-0001",
            "iter-000001--run-train-0002",
            "iter-000001--prep-run-train",
        ]
        ncheck = len(expected_output)
        self.assertEqual(
            sort_slice_ops(dpgen_keys[:ncheck], ["run-train", "run-lmp", "run-fp"]),
            expected_output,
        )

    def test_print_keys(self):
        expected_output = [
            "                   0 : finetune--prep-train",
            "              1 -> 3 : finetune--run-train-0000 -> finetune--run-train-0002",
            "                   4 : finetune--modify-train-script",
            "                   5 : finetune--prep-run-train",
            "                   6 : init--scheduler",
            "                   7 : init--id",
            "                   8 : iter-000000--prep-train",
            "             9 -> 11 : iter-000000--run-train-0000 -> iter-000000--run-train-0002",
            "                  12 : iter-000000--prep-run-train",
        ]
        expected_output = "\n".join(expected_output + [""])

        ret = print_keys_in_nice_format(
            dpgen_keys[:13],
            ["run-train", "run-lmp", "run-fp"],
            idx_fmt_len=8,
        )

        self.assertEqual(expected_output, ret)
