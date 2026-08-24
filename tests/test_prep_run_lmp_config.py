import unittest

from dflow import (
    Step,
)

from dpgen2.op.prep_lmp import (
    PrepLmp,
)
from dpgen2.op.run_lmp import (
    RunLmp,
)
from dpgen2.superop.prep_run_lmp import (
    PrepRunLmp,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict


class TestPrepRunLmpConfig(unittest.TestCase):
    def test_non_sliced_model_preparation_ignores_slice_success_controls(self):
        config = normalize_step_dict(
            {
                "continue_on_num_success": 1,
                "continue_on_success_ratio": 0.5,
            }
        )
        steps = PrepRunLmp(
            "prep-run-lmp",
            PrepLmp,
            RunLmp,
            prep_config=normalize_step_dict({}),
            run_config=config,
        )
        prepare_models = next(
            step for step in steps.steps if step.name == "prepare-models"
        )
        run_lmp = next(step for step in steps.steps if step.name == "run-lmp")

        self.assertIsNone(prepare_models.continue_on_num_success)
        self.assertIsNone(prepare_models.continue_on_success_ratio)
        self.assertEqual(run_lmp.continue_on_num_success, 1)
        self.assertEqual(run_lmp.continue_on_success_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
