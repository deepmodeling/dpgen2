"""Tests for Round 4: ASE support in dpgen2/entrypoint/args.py and submit.py.

These tests use source-code inspection (string search) to avoid importing
dflow, which triggers S3/MinIO connection attempts even without Argo running.
"""
import os
import sys
import unittest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dpgen2", "entrypoint", "args.py"
)
SUBMIT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dpgen2", "entrypoint", "submit.py"
)


def _read(path):
    with open(os.path.abspath(path)) as f:
        return f.read()


ARGS_SRC = _read(ARGS_PATH)
SUBMIT_SRC = _read(SUBMIT_PATH)


# ===========================================================================
# args.py tests
# ===========================================================================


class TestArgsAseVariant(unittest.TestCase):
    """Check that args.py registers the 'ase' explore variant."""

    def test_ase_in_variant_explore(self):
        """variant_explore() must include Argument('ase', ...)."""
        self.assertIn('Argument("ase"', ARGS_SRC)

    def test_ase_args_function_defined(self):
        """ase_args() function must be defined."""
        self.assertIn("def ase_args(", ARGS_SRC)

    def test_run_ase_args_function_defined(self):
        """run_ase_args() function must be defined."""
        self.assertIn("def run_ase_args(", ARGS_SRC)

    def test_ase_args_has_config(self):
        """ase_args() must include a 'config' argument."""
        # Check that 'config' appears in ase_args context
        self.assertIn('"config"', ARGS_SRC)

    def test_ase_args_has_max_numb_iter(self):
        """ase_args() must include 'max_numb_iter'."""
        self.assertIn('"max_numb_iter"', ARGS_SRC)

    def test_ase_args_has_fatal_at_max(self):
        """ase_args() must include 'fatal_at_max'."""
        self.assertIn('"fatal_at_max"', ARGS_SRC)

    def test_ase_args_has_output_nopbc(self):
        """ase_args() must include 'output_nopbc'."""
        self.assertIn('"output_nopbc"', ARGS_SRC)

    def test_ase_args_has_convergence(self):
        """ase_args() must include 'convergence'."""
        self.assertIn('"convergence"', ARGS_SRC)

    def test_ase_args_has_configurations(self):
        """ase_args() must include 'configurations'."""
        self.assertIn('"configurations"', ARGS_SRC)

    def test_ase_args_has_stages(self):
        """ase_args() must include 'stages'."""
        self.assertIn('"stages"', ARGS_SRC)

    def test_ase_args_has_filters(self):
        """ase_args() must include 'filters'."""
        self.assertIn('"filters"', ARGS_SRC)

    def test_run_ase_args_has_use_hdf5(self):
        """run_ase_args() must include 'use_hdf5'."""
        self.assertIn('"use_hdf5"', ARGS_SRC)

    def test_run_ase_import(self):
        """args.py must import RunAse for ase_args()."""
        self.assertIn("from dpgen2.op.run_ase import", ARGS_SRC)

    def test_variant_explore_doc_ase(self):
        """variant_explore() must have a doc string for 'ase'."""
        self.assertIn("doc_ase", ARGS_SRC)

    def test_ase_args_uses_run_ase_normalize_config(self):
        """ase_args() config default must use RunAse.normalize_config."""
        self.assertIn("RunAse.normalize_config", ARGS_SRC)

    def test_ase_args_uses_run_ase_ase_args(self):
        """ase_args() config sub-args must use RunAse.ase_args()."""
        self.assertIn("RunAse.ase_args()", ARGS_SRC)


# ===========================================================================
# submit.py tests
# ===========================================================================


class TestSubmitAseImports(unittest.TestCase):
    """Check that submit.py imports the ASE-related symbols."""

    def test_imports_prep_ase(self):
        self.assertIn("PrepAse", SUBMIT_SRC)

    def test_imports_run_ase(self):
        self.assertIn("RunAse", SUBMIT_SRC)

    def test_imports_run_ase_hdf5(self):
        self.assertIn("RunAseHDF5", SUBMIT_SRC)

    def test_imports_prep_run_ase(self):
        self.assertIn("PrepRunAse", SUBMIT_SRC)

    def test_imports_ase_normalize(self):
        self.assertIn("ase_normalize", SUBMIT_SRC)

    def test_imports_make_ase_task_group_from_config(self):
        self.assertIn("make_ase_task_group_from_config", SUBMIT_SRC)


class TestSubmitMakeConcurrentLearningOpAse(unittest.TestCase):
    """Check that make_concurrent_learning_op handles explore_style == 'ase'."""

    def test_ase_branch_exists(self):
        """make_concurrent_learning_op must have an 'ase' branch."""
        self.assertIn('explore_style == "ase"', SUBMIT_SRC)

    def test_ase_branch_uses_prep_run_ase(self):
        """The 'ase' branch must instantiate PrepRunAse."""
        self.assertIn("PrepRunAse(", SUBMIT_SRC)

    def test_ase_branch_uses_prep_ase(self):
        """The 'ase' branch must pass PrepAse to PrepRunAse."""
        self.assertIn("PrepAse,", SUBMIT_SRC)

    def test_ase_branch_uses_run_ase_hdf5(self):
        """The 'ase' branch must use RunAseHDF5 when use_hdf5 is True."""
        self.assertIn("RunAseHDF5", SUBMIT_SRC)

    def test_ase_branch_uses_run_ase(self):
        """The 'ase' branch must use RunAse when use_hdf5 is False."""
        self.assertIn("RunAse,", SUBMIT_SRC)

    def test_ase_branch_name(self):
        """PrepRunAse must be named 'prep-run-ase'."""
        self.assertIn('"prep-run-ase"', SUBMIT_SRC)


class TestSubmitMakeNaiveExplorationSchedulerAse(unittest.TestCase):
    """Check that make_naive_exploration_scheduler dispatches to ASE."""

    def test_ase_dispatch_in_make_naive_scheduler(self):
        """make_naive_exploration_scheduler must dispatch to ASE."""
        self.assertIn('explore_style == "ase"', SUBMIT_SRC)

    def test_make_ase_naive_exploration_scheduler_defined(self):
        """make_ase_naive_exploration_scheduler must be defined."""
        self.assertIn("def make_ase_naive_exploration_scheduler(", SUBMIT_SRC)

    def test_ase_scheduler_uses_traj_render_lammps(self):
        """make_ase_naive_exploration_scheduler must use TrajRenderLammps."""
        self.assertIn("TrajRenderLammps", SUBMIT_SRC)

    def test_ase_scheduler_uses_make_ase_task_group_from_config(self):
        """make_ase_naive_exploration_scheduler must call make_ase_task_group_from_config."""
        self.assertIn("make_ase_task_group_from_config(", SUBMIT_SRC)

    def test_ase_scheduler_uses_ase_normalize(self):
        """make_ase_naive_exploration_scheduler must call ase_normalize."""
        self.assertIn("ase_normalize(", SUBMIT_SRC)

    def test_ase_scheduler_uses_convergence_check_stage_scheduler(self):
        """make_ase_naive_exploration_scheduler must use ConvergenceCheckStageScheduler."""
        self.assertIn("ConvergenceCheckStageScheduler(", SUBMIT_SRC)

    def test_ase_scheduler_uses_conf_selector_frames(self):
        """make_ase_naive_exploration_scheduler must use ConfSelectorFrames."""
        self.assertIn("ConfSelectorFrames(", SUBMIT_SRC)

    def test_ase_scheduler_reads_type_map(self):
        """make_ase_naive_exploration_scheduler must read type_map from config."""
        self.assertIn('type_map = config["inputs"]["type_map"]', SUBMIT_SRC)

    def test_ase_scheduler_reads_mass_map(self):
        """make_ase_naive_exploration_scheduler must read mass_map from config."""
        self.assertIn('mass_map = config["inputs"]["mass_map"]', SUBMIT_SRC)

    def test_ase_scheduler_reads_numb_models(self):
        """make_ase_naive_exploration_scheduler must read numb_models from config."""
        self.assertIn('numb_models = config["train"]["numb_models"]', SUBMIT_SRC)

    def test_ase_scheduler_sets_conf(self):
        """make_ase_naive_exploration_scheduler must call tgroup.set_conf."""
        self.assertIn("tgroup.set_conf(", SUBMIT_SRC)


class TestSubmitGetResubmitKeysAse(unittest.TestCase):
    """Check that get_resubmit_keys includes 'run-ase' in slice ops."""

    def test_run_ase_in_sort_slice_ops(self):
        """get_resubmit_keys must include 'run-ase' in sort_slice_ops call."""
        self.assertIn('"run-ase"', SUBMIT_SRC)

    def test_run_ase_in_print_keys_in_nice_format(self):
        """resubmit_concurrent_learning must include 'run-ase' in print_keys_in_nice_format."""
        # Both sort_slice_ops and print_keys_in_nice_format should have run-ase
        count = SUBMIT_SRC.count('"run-ase"')
        self.assertGreaterEqual(
            count, 2, "Expected 'run-ase' to appear at least twice in submit.py"
        )


# ===========================================================================
# Functional import test (no dflow instantiation)
# ===========================================================================


class TestArgsImportable(unittest.TestCase):
    """Verify that args.py can be imported and key functions are callable."""

    def test_ase_args_importable(self):
        """ase_args() must be importable from dpgen2.entrypoint.args."""
        from dpgen2.entrypoint.args import (
            ase_args,
        )

        result = ase_args()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_run_ase_args_importable(self):
        """run_ase_args() must be importable from dpgen2.entrypoint.args."""
        from dpgen2.entrypoint.args import (
            run_ase_args,
        )

        result = run_ase_args()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_variant_explore_includes_ase(self):
        """variant_explore() must include 'ase' as a choice."""
        from dpgen2.entrypoint.args import (
            variant_explore,
        )

        v = variant_explore()
        choice_names = [a.name for a in v.choice_dict.values()]
        self.assertIn("ase", choice_names)

    def test_ase_args_config_key(self):
        """ase_args() must have a 'config' key."""
        from dpgen2.entrypoint.args import (
            ase_args,
        )

        keys = [a.name for a in ase_args()]
        self.assertIn("config", keys)

    def test_ase_args_stages_key(self):
        """ase_args() must have a 'stages' key."""
        from dpgen2.entrypoint.args import (
            ase_args,
        )

        keys = [a.name for a in ase_args()]
        self.assertIn("stages", keys)

    def test_ase_args_configurations_key(self):
        """ase_args() must have a 'configurations' key."""
        from dpgen2.entrypoint.args import (
            ase_args,
        )

        keys = [a.name for a in ase_args()]
        self.assertIn("configurations", keys)

    def test_run_ase_args_use_hdf5_key(self):
        """run_ase_args() must have a 'use_hdf5' key."""
        from dpgen2.entrypoint.args import (
            run_ase_args,
        )

        keys = [a.name for a in run_ase_args()]
        self.assertIn("use_hdf5", keys)


if __name__ == "__main__":
    unittest.main()
