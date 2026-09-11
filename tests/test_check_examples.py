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
    p_examples / "water" / "input_dpgen_abacus.json",
    p_examples / "water" / "input_dpgen_cp2k.json",
    p_examples / "diffcsp" / "dpgen.json",
)


class TestExamples(unittest.TestCase):
    def test_arguments(self):
        for fn in input_files:
            with self.subTest(fn=fn):
                with open(fn) as f:
                    jdata = json.load(f)
                normalize(jdata)

                template_scripts = jdata["train"]["template_script"]
                if isinstance(template_scripts, str):
                    template_scripts = [template_scripts]
                for template_script in template_scripts:
                    template_path = fn.parent / template_script
                    self.assertTrue(
                        template_path.is_file(),
                        f"Missing training template referenced by {fn}: {template_path}",
                    )
                    with open(template_path) as f:
                        json.load(f)
