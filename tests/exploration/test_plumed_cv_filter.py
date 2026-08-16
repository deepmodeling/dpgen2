import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np
from dargs import (
    Argument,
)
from dflow.python import (
    FatalError,
)

from dpgen2.exploration.selector import (
    PlumedCVFilter,
)


class TestPlumedCVFilter(unittest.TestCase):
    def test_config_schema(self):
        schema = Argument("cv_filter", dict, PlumedCVFilter.args())
        config = schema.normalize_value(
            {
                "regions": [{"distance": [0.8, 1.2]}],
                "sampling": {
                    "mode": "uniform",
                    "field": "distance",
                    "n_bins": 8,
                },
            }
        )
        schema.check_value(config, strict=True)
        self.assertEqual(config["regions"][0]["distance"], [0.8, 1.2])
        self.assertEqual(config["sampling"]["within_bin"], "random")
        self.assertEqual(config["sampling"]["seed"], 0)

    def test_union_of_regions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text(
                "#! FIELDS time d1 d2\n" "0.0 0.5 2.0\n" "1.0 0.5 4.0\n" "2.0 2.5 9.0\n"
            )
            cv_filter = PlumedCVFilter(
                regions=[{"d1": [0.0, 1.0], "d2": [1.0, 3.0]}, {"d1": [2.0, 3.0]}]
            )
            self.assertEqual(cv_filter.get_selected_ids([output], [3]), [[0, 2]])

    def test_intervals_are_lower_inclusive_and_upper_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text("#! FIELDS time cv\n0.0 1.0\n1.0 2.0\n2.0 3.0\n")
            cv_filter = PlumedCVFilter(regions=[{"cv": [1.0, 3.0]}])
            self.assertEqual(cv_filter.get_selected_ids([output], [3]), [[0, 1]])

    def test_random_sampling_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text(
                "#! FIELDS time cv\n"
                + "".join(f"{ii}.0 {ii / 20:.3f}\n" for ii in range(20))
            )
            kwargs = {
                "regions": [{"cv": [0.25, 0.75]}],
                "sampling": {"mode": "random", "seed": 17},
            }
            selected = PlumedCVFilter(**kwargs).select_candidate_ids(
                [output], [20], [list(range(20))], 5
            )
            repeated = PlumedCVFilter(**kwargs).select_candidate_ids(
                [output], [20], [list(range(20))], 5
            )
            self.assertEqual(selected, repeated)
            self.assertEqual(len(selected[0]), 5)
            self.assertTrue(all(5 <= frame < 15 for frame in selected[0]))

    def test_uniform_sampling_spans_nonempty_bins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            values = [0.05, 0.06] + [ii / 10 + 0.05 for ii in range(1, 10)]
            output.write_text(
                "#! FIELDS time cv\n"
                + "".join(f"{ii}.0 {value:.3f}\n" for ii, value in enumerate(values))
            )
            deviations = np.arange(len(values), dtype=float)
            deviations[0] = 100.0
            deviations[1] = 200.0
            cv_filter = PlumedCVFilter(
                regions=[{"cv": [0.0, 1.0]}],
                sampling={
                    "mode": "uniform",
                    "field": "cv",
                    "n_bins": 10,
                    "within_bin": "max_deviation",
                    "seed": 3,
                },
            )
            selected = cv_filter.select_candidate_ids(
                [output], [len(values)], [list(range(len(values)))], 3, [deviations]
            )
            self.assertEqual(selected, [[1, 5, 10]])

    def test_uniform_random_within_bin_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text(
                "#! FIELDS time cv\n"
                + "".join(f"{ii}.0 {ii / 20:.3f}\n" for ii in range(20))
            )
            kwargs = {
                "regions": [{"cv": [0.0, 1.0]}],
                "sampling": {
                    "mode": "uniform",
                    "field": "cv",
                    "n_bins": 5,
                    "within_bin": "random",
                    "seed": 29,
                },
            }
            selected = PlumedCVFilter(**kwargs).select_candidate_ids(
                [output], [20], [list(range(20))], 5
            )
            repeated = PlumedCVFilter(**kwargs).select_candidate_ids(
                [output], [20], [list(range(20))], 5
            )
            self.assertEqual(selected, repeated)
            self.assertEqual(len(selected[0]), 5)

    def test_uniform_sampling_balances_regions_and_honors_and_conditions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text(
                "#! FIELDS time cv gate\n"
                "0.0 0.1 0.5\n"
                "1.0 0.9 0.5\n"
                "2.0 2.1 0.5\n"
                "3.0 2.9 0.5\n"
                "4.0 0.5 1.5\n"
            )
            cv_filter = PlumedCVFilter(
                regions=[
                    {"cv": [0.0, 1.0], "gate": [0.0, 1.0]},
                    {"cv": [2.0, 3.0], "gate": [0.0, 1.0]},
                ],
                sampling={
                    "mode": "uniform",
                    "field": "cv",
                    "n_bins": 10,
                    "within_bin": "random",
                    "seed": 9,
                },
            )
            selected = cv_filter.select_candidate_ids(
                [output], [5], [list(range(5))], 2
            )[0]
            self.assertEqual(len(selected), 2)
            self.assertTrue(any(frame in {0, 1} for frame in selected))
            self.assertTrue(any(frame in {2, 3} for frame in selected))
            self.assertNotIn(4, selected)

    def test_uniform_sampling_handles_empty_bins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text("#! FIELDS time cv\n0.0 0.05\n1.0 0.45\n2.0 0.95\n")
            cv_filter = PlumedCVFilter(
                regions=[{"cv": [0.0, 1.0]}],
                sampling={"mode": "uniform", "field": "cv", "n_bins": 10},
            )
            self.assertEqual(
                cv_filter.select_candidate_ids([output], [3], [list(range(3))], 2),
                [[0, 2]],
            )

    def test_uniform_sampling_spreads_a_small_limit_across_regions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text("#! FIELDS time cv\n0.0 0.5\n1.0 2.5\n2.0 4.5\n")
            cv_filter = PlumedCVFilter(
                regions=[{"cv": [0.0, 1.0]}, {"cv": [2.0, 3.0]}, {"cv": [4.0, 5.0]}],
                sampling={"mode": "uniform", "field": "cv", "n_bins": 4},
            )
            self.assertEqual(
                cv_filter.select_candidate_ids([output], [3], [list(range(3))], 2),
                [[0, 2]],
            )

    def test_alignment_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "COLVAR"
            output.write_text("#! FIELDS time cv\n0.0 0.5\n")
            with self.assertRaises(FatalError):
                PlumedCVFilter(regions=[{"cv": [0.0, 1.0]}]).get_selected_ids(
                    [output], [2]
                )

    def test_invalid_config_and_file_fail_closed(self):
        invalid_regions = [
            [],
            [{}],
            [{"cv": [1.0, 1.0]}],
            [{"cv": [0.0, float("inf")]}],
            [{"cv": "01"}],
        ]
        for regions in invalid_regions:
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                PlumedCVFilter(regions=regions)

        invalid_sampling = [
            {"mode": "weighted"},
            {"mode": "uniform"},
            {"mode": "uniform", "field": "other"},
            {"mode": "uniform", "field": "cv", "n_bins": 0},
            {"mode": "uniform", "field": "cv", "within_bin": "first"},
            {"mode": "random", "seed": True},
            {"mode": "random", "seed": -1},
        ]
        for sampling in invalid_sampling:
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                PlumedCVFilter(regions=[{"cv": [0.0, 1.0]}], sampling=sampling)

        invalid_outputs = [
            "0.0 0.5\n#! FIELDS time cv\n",
            "#! FIELDS time cv\n0.0 nan\n",
            "#! FIELDS time cv\n0.0\n",
            "#! FIELDS time cv\n#! FIELDS time other\n0.0 0.5\n",
            "#! FIELDS cv\n0.5\n",
            "#! FIELDS time cv\n1.0 0.5\n0.0 0.5\n",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FatalError):
                PlumedCVFilter(regions=[{"cv": [0.0, 1.0]}]).get_selected_ids(
                    [Path(tmpdir) / "missing"], [1]
                )
            for index, content in enumerate(invalid_outputs):
                output = Path(tmpdir) / f"COLVAR.{index}"
                output.write_text(content)
                with self.subTest(content=content), self.assertRaises(FatalError):
                    PlumedCVFilter(regions=[{"cv": [0.0, 1.0]}]).get_selected_ids(
                        [output], [1]
                    )
