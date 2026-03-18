"""Unit tests for PrepRunAse super-OP.

These tests verify the structural properties of :class:`PrepRunAse`
by inspecting the source code and module attributes, without importing
dflow (which would trigger S3/MinIO connection attempts).

Tests that require a live dflow/Argo infrastructure are in a separate
class guarded by ``SKIP_UT_WITH_DFLOW``.
"""

import ast
import os
import sys
import unittest

# isort: off
# Add repo root to path without importing dflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# isort: on


class TestPrepRunAseSourceCode(unittest.TestCase):
    """Verify PrepRunAse structure by inspecting source code.

    These tests do NOT import dflow and therefore do NOT trigger S3
    connection attempts.  They verify the implementation by reading
    the source of the module.
    """

    @classmethod
    def setUpClass(cls):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "dpgen2", "superop", "prep_run_ase.py"
        )
        with open(src_path) as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    # ------------------------------------------------------------------ #
    # Class definition
    # ------------------------------------------------------------------ #

    def test_PrepRunAse_class_defined(self):
        class_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef)
        ]
        self.assertIn("PrepRunAse", class_names)

    def test_prep_run_ase_function_defined(self):
        func_names = [
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn("_prep_run_ase", func_names)

    # ------------------------------------------------------------------ #
    # Keys
    # ------------------------------------------------------------------ #

    def test_keys_list_contains_prep_ase(self):
        self.assertIn('"prep-ase"', self.source)

    def test_keys_list_contains_run_ase(self):
        self.assertIn('"run-ase"', self.source)

    def test_run_ase_step_key_has_item_placeholder(self):
        self.assertIn("{{item}}", self.source)

    # ------------------------------------------------------------------ #
    # Input parameters
    # ------------------------------------------------------------------ #

    def test_input_parameter_block_id(self):
        self.assertIn('"block_id"', self.source)

    def test_input_parameter_explore_config(self):
        self.assertIn('"explore_config"', self.source)

    def test_input_parameter_expl_task_grp(self):
        self.assertIn('"expl_task_grp"', self.source)

    def test_input_parameter_type_map(self):
        self.assertIn('"type_map"', self.source)

    # ------------------------------------------------------------------ #
    # Input artifacts
    # ------------------------------------------------------------------ #

    def test_input_artifact_models(self):
        self.assertIn('"models"', self.source)

    # ------------------------------------------------------------------ #
    # Output parameters
    # ------------------------------------------------------------------ #

    def test_output_parameter_task_names(self):
        self.assertIn('"task_names"', self.source)

    # ------------------------------------------------------------------ #
    # Output artifacts
    # ------------------------------------------------------------------ #

    def test_output_artifact_logs(self):
        self.assertIn('"logs"', self.source)

    def test_output_artifact_trajs(self):
        self.assertIn('"trajs"', self.source)

    def test_output_artifact_model_devis(self):
        self.assertIn('"model_devis"', self.source)

    def test_no_plm_output_artifact(self):
        """ASE does not produce plumed output."""
        self.assertNotIn("plm_output", self.source)

    def test_no_optional_outputs_artifact(self):
        """ASE does not produce optional_outputs (no ele_temp)."""
        self.assertNotIn("optional_outputs", self.source)

    def test_no_extra_outputs_artifact(self):
        """ASE does not produce extra_outputs."""
        self.assertNotIn("extra_outputs", self.source)

    # ------------------------------------------------------------------ #
    # Implementation details
    # ------------------------------------------------------------------ #

    def test_ase_index_pattern_used(self):
        """ase_index_pattern must be used for the run-ase fan-out sequence."""
        self.assertIn("ase_index_pattern", self.source)

    def test_prep_ase_input_parameter_name(self):
        """PrepAse expects 'ase_task_grp', not 'lmp_task_grp'."""
        self.assertIn("ase_task_grp", self.source)
        self.assertNotIn("lmp_task_grp", self.source)

    def test_prep_ase_step_name(self):
        """The prep step must be named 'prep-ase'."""
        self.assertIn('"prep-ase"', self.source)

    def test_run_ase_step_name(self):
        """The run step must be named 'run-ase'."""
        self.assertIn('"run-ase"', self.source)

    def test_slices_used_for_run_step(self):
        """RunAse must use Slices for fan-out."""
        self.assertIn("Slices", self.source)

    def test_output_artifacts_wired_from_run_step(self):
        """trajs and model_devis must be wired from run_ase outputs."""
        # The source may have line breaks inside the subscript expression,
        # so we check for the key names independently.
        self.assertIn('run_ase.outputs.artifacts', self.source)
        self.assertIn('"traj"', self.source)
        self.assertIn('"model_devi"', self.source)
        # Verify the wiring lines are present (may span multiple lines)
        self.assertIn('outputs.artifacts["trajs"]._from', self.source)
        self.assertIn('outputs.artifacts["model_devis"]._from', self.source)


class TestPrepRunAseModuleExports(unittest.TestCase):
    """Verify that dpgen2/superop/__init__.py exports PrepRunAse.

    Reads the __init__.py source without importing it.
    """

    @classmethod
    def setUpClass(cls):
        init_path = os.path.join(
            os.path.dirname(__file__), "..", "dpgen2", "superop", "__init__.py"
        )
        with open(init_path) as f:
            cls.init_source = f.read()

    def test_PrepRunAse_exported_from_superop(self):
        self.assertIn("PrepRunAse", self.init_source)

    def test_prep_run_ase_module_imported(self):
        self.assertIn("prep_run_ase", self.init_source)


if __name__ == "__main__":
    unittest.main()
