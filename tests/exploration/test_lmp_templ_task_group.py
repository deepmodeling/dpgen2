import itertools
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
from unittest.mock import (
    Mock,
    patch,
)

from dflow.python import (
    FatalError,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    plm_input_name,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    LmpTemplateTaskGroup,
)
from dpgen2.exploration.task.lmp_template_task_group import (
    check_revisions_completeness,
    find_unreplaced_variables,
    revise_by_keys,
)

in_lmp_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd ../graph.003.pb ../graph.001.pb ../graph.002.pb ../graph.000.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)

expected_lmp_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd model.000.pb model.001.pb model.002.pb model.003.pb out_freq 20 out_file model_devi.out
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump all custom 20 traj.dump id type x y z

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)


in_lmp_plm_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd ../graph.003.pb ../graph.001.pb ../graph.002.pb ../graph.000.pb  out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      * *

fix             dpgen_plm

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)

expected_lmp_plm_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd model.000.pb model.001.pb model.002.pb model.003.pb out_freq 20 out_file model_devi.out
pair_coeff      * *

fix             dpgen_plm all plumed plumedfile input.plumed outfile output.plumed

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump all custom 20 traj.dump id type x y z

velocity        all create ${TEMP} 826513
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS}
"""
)

in_plm_template = textwrap.dedent(
    """FOO V_TEMP
DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
RESTRAINT ARG=d1,d2 AT=V_DIST0,bar KAPPA=150.0,150.0 LABEL=restraint
PRINT ARG=restraint.bias
"""
)


in_lmp_pimd_template = textwrap.dedent(
    """variable        NSTEPS          equal V_NSTEPS
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal V_TEMP
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000
variable        ibead           uloop 4 pad

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump

velocity        all create ${TEMP} 826513
fix             1 all pimd/langevin ensemble npt integrator baoab temp ${TEMP} thermostat PILE_L 1234 tau ${TAU_T} iso ${PRES} barostat BZP taup ${TAU_P}

timestep        0.001
run             ${NSTEPS}
"""
)


expected_lmp_pimd_template = textwrap.dedent(
    """variable        NSTEPS          equal 1000
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal 300
variable        PRES            equal 0.0
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000
variable        ibead           uloop 4 pad

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box             tilt large
read_data       conf.lmp
change_box      all triclinic
mass            1 27.000000
mass            2 24.000000

pair_style      deepmd model.000.pb model.001.pb model.002.pb model.003.pb out_freq 20 out_file model_devi.${ibead}.out
pair_coeff      * *

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}

dump            dpgen_dump all custom 20 traj.${ibead}.dump id type x y z

velocity        all create ${TEMP} 826513
fix             1 all pimd/langevin ensemble npt integrator baoab temp ${TEMP} thermostat PILE_L 1234 tau ${TAU_T} iso ${PRES} barostat BZP taup ${TAU_P}

timestep        0.001
run             ${NSTEPS}
"""
)


class TestLmpTemplateTaskGroup(unittest.TestCase):
    def setUp(self):
        self.lmp_template_fname = Path("lmp.template")
        self.lmp_template_fname.write_text(in_lmp_template)
        self.lmp_plm_template_fname = Path("lmp.plm.template")
        self.lmp_plm_template_fname.write_text(in_lmp_plm_template)
        self.plm_template_fname = Path("plm.template")
        self.plm_template_fname.write_text(in_plm_template)
        self.numb_models = 4
        self.confs = ["foo", "bar"]
        self.lmp_rev_mat = {"V_NSTEPS": [1000], "V_TEMP": [50, 100]}
        self.lmp_plm_rev_mat = {
            "V_NSTEPS": [1000],
            "V_TEMP": [50, 100],
            "V_DIST0": [3, 4],
        }
        self.rev_empty = {}
        self.traj_freq = 20
        self.lmp_pimd_template_fname = Path("lmp.pimd.template")
        self.lmp_pimd_template_fname.write_text(in_lmp_pimd_template)

    def tearDown(self):
        os.remove(self.lmp_template_fname)
        os.remove(self.lmp_plm_template_fname)
        os.remove(self.plm_template_fname)
        os.remove(self.lmp_pimd_template_fname)

    def test_lmp(self):
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions=self.lmp_rev_mat,
            traj_freq=self.traj_freq,
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs)
            * len(self.lmp_rev_mat["V_NSTEPS"])
            * len(self.lmp_rev_mat["V_TEMP"]),
        )

        idx = 0
        for cc, ii, jj in itertools.product(
            range(len(self.confs)),
            range(len(self.lmp_rev_mat["V_NSTEPS"])),
            range(len(self.lmp_rev_mat["V_TEMP"])),
        ):
            ee = expected_lmp_template.split("\n")
            ee[0] = ee[0].replace("V_NSTEPS", str(self.lmp_rev_mat["V_NSTEPS"][ii]))
            ee[3] = ee[3].replace("V_TEMP", str(self.lmp_rev_mat["V_TEMP"][jj]))
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                ee,
            )
            idx += 1

    def test_set_lmp_preserves_positional_traj_freq(self):
        task_group = LmpTemplateTaskGroup()
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            None,
            self.lmp_rev_mat,
            self.traj_freq,
        )

        self.assertEqual(task_group.traj_freq, self.traj_freq)
        self.assertFalse(task_group.strict_revisions)

    def test_lmp_plm(self):
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_plm_template_fname,
            plm_template_fname=self.plm_template_fname,
            revisions=self.lmp_plm_rev_mat,
            traj_freq=self.traj_freq,
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            len(self.confs)
            * len(self.lmp_plm_rev_mat["V_NSTEPS"])
            * len(self.lmp_plm_rev_mat["V_TEMP"])
            * len(self.lmp_plm_rev_mat["V_DIST0"]),
        )
        idx = 0
        for cc, ii, jj, kk in itertools.product(
            range(len(self.confs)),
            range(len(self.lmp_plm_rev_mat["V_NSTEPS"])),
            range(len(self.lmp_plm_rev_mat["V_TEMP"])),
            range(len(self.lmp_plm_rev_mat["V_DIST0"])),
        ):
            eel = expected_lmp_plm_template.split("\n")
            eel[0] = eel[0].replace(
                "V_NSTEPS", str(self.lmp_plm_rev_mat["V_NSTEPS"][ii])
            )
            eel[3] = eel[3].replace("V_TEMP", str(self.lmp_plm_rev_mat["V_TEMP"][jj]))
            eep = in_plm_template.split("\n")
            eep[0] = eep[0].replace("V_TEMP", str(self.lmp_plm_rev_mat["V_TEMP"][jj]))
            eep[3] = eep[3].replace("V_DIST0", str(self.lmp_plm_rev_mat["V_DIST0"][kk]))
            self.assertEqual(
                task_group[idx].files()[lmp_conf_name],
                self.confs[cc],
            )
            self.assertEqual(
                task_group[idx].files()[lmp_input_name].split("\n"),
                eel,
            )
            self.assertEqual(
                task_group[idx].files()[plm_input_name].split("\n"),
                eep,
            )
            idx += 1

    def test_lmp_empty(self):
        """Non-strict empty revisions should warn but succeed."""
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions=self.rev_empty,
            traj_freq=self.traj_freq,
            strict_revisions=False,
        )
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            task_group.make_task()
            var_warnings = [x for x in w if "V_NSTEPS" in str(x.message)]
            self.assertGreater(len(var_warnings), 0)
        # Should still produce tasks
        ngroup = len(task_group)
        self.assertEqual(ngroup, len(self.confs))

    def test_lmp_pimd(self):
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(["foo"])
        task_group.set_lmp(
            self.numb_models,
            self.lmp_pimd_template_fname,
            revisions={"V_NSTEPS": [1000], "V_TEMP": [300]},
            traj_freq=self.traj_freq,
            pimd_bead="${ibead}",
        )
        task_group.make_task()
        ngroup = len(task_group)
        self.assertEqual(
            ngroup,
            1,
        )
        ee = expected_lmp_pimd_template.split("\n")
        self.assertEqual(
            task_group[0].files()[lmp_input_name].split("\n"),
            ee,
        )


class TestRevisionVariablePrecheck(unittest.TestCase):
    """Test PR6: validation of revision variables in LAMMPS templates."""

    def setUp(self):
        self.lmp_template_fname = Path("lmp_precheck.template")
        self.numb_models = 4
        self.confs = ["foo"]
        self.traj_freq = 10

    def tearDown(self):
        if self.lmp_template_fname.exists():
            os.remove(self.lmp_template_fname)

    def _write_template(self, content):
        self.lmp_template_fname.write_text(content)

    def test_undefined_variable_raises(self):
        """Template has V_PRESS but revisions only define V_NSTEPS and V_TEMP."""
        template = textwrap.dedent(
            """\
            variable        NSTEPS          equal V_NSTEPS
            variable        TEMP            equal V_TEMP
            variable        PRESS           equal V_PRESS

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            run             ${NSTEPS}
            """
        )
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_NSTEPS": [1000], "V_TEMP": [300]},
            traj_freq=self.traj_freq,
            strict_revisions=True,
        )
        with self.assertRaises(FatalError) as ctx:
            task_group.make_task()
        self.assertIn("V_PRESS", str(ctx.exception))
        self.assertIn("undefined revision variable", str(ctx.exception).lower())

    def test_no_revisions_strict_mode_raises(self):
        """Strict mode rejects V_* variables when no revisions are provided."""
        template = textwrap.dedent(
            """\
            variable        NSTEPS          equal V_NSTEPS
            variable        TEMP            equal V_TEMP

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            run             ${NSTEPS}
            """
        )
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={},
            traj_freq=self.traj_freq,
            strict_revisions=True,
        )

        with self.assertRaisesRegex(FatalError, "V_NSTEPS"):
            task_group.make_task()

    def test_all_variables_defined_no_error(self):
        """All V_* variables are covered by revisions — should succeed."""
        template = textwrap.dedent(
            """\
            variable        NSTEPS          equal V_NSTEPS
            variable        TEMP            equal V_TEMP

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            run             ${NSTEPS}
            """
        )
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_NSTEPS": [1000], "V_TEMP": [300, 600]},
            traj_freq=self.traj_freq,
        )
        # Should not raise
        task_group.make_task()
        self.assertEqual(len(task_group), 2)  # 1 conf * 2 V_TEMP values

    def test_unused_revision_key_warns(self):
        """Revision defines V_TYPO that doesn't appear in template — should warn."""
        template = textwrap.dedent(
            """\
            variable        NSTEPS          equal V_NSTEPS

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            run             ${NSTEPS}
            """
        )
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_NSTEPS": [1000], "V_TYPO": [42]},
            traj_freq=self.traj_freq,
        )
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            task_group.make_task()
            # Should have at least one warning about V_TYPO
            typo_warnings = [x for x in w if "V_TYPO" in str(x.message)]
            self.assertGreater(len(typo_warnings), 0)

    def test_lammps_variables_without_revision_prefix_are_not_flagged(self):
        """LAMMPS references without the reserved V_* prefix remain untouched."""
        template = textwrap.dedent(
            """\
            variable        NSTEPS          equal V_NSTEPS

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            velocity        all create ${TEMP} 12345
            run             ${NSTEPS}
            """
        )
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_NSTEPS": [1000]},
            traj_freq=self.traj_freq,
        )
        # ${TEMP} is LAMMPS syntax, not a dpgen revision variable — should not raise
        task_group.make_task()
        self.assertEqual(len(task_group), 1)

    def test_v_prefixed_lammps_identifier_is_reserved_in_strict_mode(self):
        """Native V_* identifiers warn by default and fail in strict mode."""
        template = textwrap.dedent(
            f"""
            variable        NSTEPS          equal V_NSTEPS
            variable        V_MAX           equal 3.0

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            velocity        all create {chr(36)}{{V_MAX}} 12345
            run             {chr(36)}{{NSTEPS}}
            """
        ).lstrip()
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_NSTEPS": [1000]},
            traj_freq=self.traj_freq,
            strict_revisions=True,
        )
        with self.assertRaisesRegex(FatalError, "V_MAX"):
            task_group.make_task()

        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_NSTEPS": [1000]},
            traj_freq=self.traj_freq,
        )
        with self.assertWarnsRegex(UserWarning, "V_MAX"):
            task_group.make_task()
        self.assertFalse(task_group.strict_revisions)
        self.assertEqual(len(task_group), 1)

    def test_plumed_template_undefined_variable_raises(self):
        """V_* in PLUMED template but not in revisions should also be caught."""
        lmp_template = textwrap.dedent(
            """\
            variable        NSTEPS          equal V_NSTEPS
            variable        TEMP            equal V_TEMP

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            fix             dpgen_plm
            run             ${NSTEPS}
            """
        )
        plm_template = textwrap.dedent(
            """\
            DISTANCE ATOMS=3,5 LABEL=d1
            RESTRAINT ARG=d1 AT=V_DIST0 KAPPA=150.0 LABEL=restraint
            """
        )
        self._write_template(lmp_template)
        plm_fname = Path("plm_precheck.template")
        plm_fname.write_text(plm_template)
        try:
            task_group = LmpTemplateTaskGroup()
            task_group.set_conf(self.confs)
            task_group.set_lmp(
                self.numb_models,
                self.lmp_template_fname,
                plm_template_fname=str(plm_fname),
                # V_DIST0 is used in PLUMED template but NOT defined here
                revisions={"V_NSTEPS": [1000], "V_TEMP": [300]},
                traj_freq=self.traj_freq,
                strict_revisions=True,
            )
            with self.assertRaises(FatalError) as ctx:
                task_group.make_task()
            self.assertIn("V_DIST0", str(ctx.exception))
        finally:
            plm_fname.unlink(missing_ok=True)

    def test_commented_variables_not_flagged(self):
        """V_* in LAMMPS comments should NOT trigger errors."""
        template = textwrap.dedent(
            """\
            variable        NSTEPS          equal V_NSTEPS
            # TODO: add V_PRESS support later
            # variable      PRESS           equal V_PRESS

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            run             ${NSTEPS}
            """
        )
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_NSTEPS": [1000]},
            traj_freq=self.traj_freq,
        )
        # V_PRESS is only in comments — should NOT raise
        task_group.make_task()
        self.assertEqual(len(task_group), 1)

    def test_undefined_longer_placeholder_raises_before_substitution(self):
        """A defined prefix must not erase a longer undefined placeholder."""
        template = textwrap.dedent(
            """\
            variable        TEMP            equal V_TEMPERATURE

            pair_style      deepmd
            pair_coeff      * *
            dump            dpgen_dump
            run             1
            """
        )
        self._write_template(template)
        task_group = LmpTemplateTaskGroup()
        task_group.set_conf(self.confs)
        task_group.set_lmp(
            self.numb_models,
            self.lmp_template_fname,
            revisions={"V_TEMP": [300]},
            traj_freq=self.traj_freq,
            strict_revisions=True,
        )
        with self.assertRaisesRegex(FatalError, "V_TEMPERATURE"):
            task_group.make_task()

    def test_overlapping_defined_placeholders_are_replaced_as_tokens(self):
        lines = ["print V_TEMP V_TEMPERATURE"]
        revised = revise_by_keys(lines, ["V_TEMP", "V_TEMPERATURE"], [300, 450])
        self.assertEqual(revised, ["print 300 450"])

    def test_quoted_hashes_preserve_revision_placeholders(self):
        content = textwrap.dedent(
            r"""\
            print "# target V_MISSING"
            print 'hash # target V_OTHER'
            print "escaped \"# target V_ESCAPED"
            # V_COMMENT_ONLY
            """
        )
        self.assertEqual(
            find_unreplaced_variables(content),
            {"V_MISSING", "V_OTHER", "V_ESCAPED"},
        )

    def test_unused_revision_key_uses_complete_comment_aware_tokens(self):
        template = "print V_TEMPERATURE # V_TEMP"
        with self.assertWarnsRegex(UserWarning, "Revision key 'V_TEMP'"):
            check_revisions_completeness(
                templates_content=["print 450"],
                revision_keys=["V_TEMP", "V_TEMPERATURE"],
                template_raw=template,
            )
