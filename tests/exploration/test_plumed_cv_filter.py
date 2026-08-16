import tempfile
import unittest
from pathlib import (
    Path,
)

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
            {"regions": [{"distance": [0.8, 1.2]}]}
        )
        schema.check_value(config, strict=True)
        self.assertEqual(config["regions"][0]["distance"], [0.8, 1.2])

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
            output.write_text(
                "#! FIELDS time cv\n0.0 1.0\n1.0 2.0\n2.0 3.0\n"
            )
            cv_filter = PlumedCVFilter(regions=[{"cv": [1.0, 3.0]}])
            self.assertEqual(cv_filter.get_selected_ids([output], [3]), [[0, 1]])

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
