import json
import unittest
from pathlib import (
    Path,
)

from dpgen2.entrypoint.args import (
    normalize,
)

p_examples = Path(__file__).parent.parent / "examples"

input_files = (
    p_examples / "almg" / "input.json",
    # p_examples / "almg" / "input-v005.json",
    # p_examples / "almg" / "dp_template.json",
    p_examples / "calypso" / "input.test.json",
    p_examples / "water" / "input_distill.json",
    p_examples / "water" / "input_dpgen.json",
    p_examples / "water" / "input_multitask.json",
    p_examples / "ch4" / "input_dist.json",
    # p_examples / "chno" / "dpa_manyi.json",
    p_examples / "chno" / "input.json",
)


class TestExamples(unittest.TestCase):
    def test_arguments(self):
        for fn in input_files:
            with self.subTest(fn=fn):
                with open(fn) as f:
                    jdata = json.load(f)
                normalize(jdata)
