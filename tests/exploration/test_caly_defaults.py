"""Numerical defaults and explicit values in CALYPSO task generation."""

import unittest

import numpy as np

from dpgen2.exploration.task import (
    make_calypso_task_group_from_config,
)


class TestCalyDefaults(unittest.TestCase):
    def setUp(self):
        self.config = {
            "name_of_atoms": ["Li", "La"],
            "numb_of_atoms": [10, 10],
            "numb_of_species": 2,
        }

    def test_infers_optional_atomic_numbers_and_distances(self):
        """Pin physical values, including the serialized CALYPSO input."""
        group = make_calypso_task_group_from_config(self.config)
        self.assertEqual(group.atomic_number, [3, 57])
        np.testing.assert_allclose(group.distance_of_ions, [[1.79, 2.34], [2.34, 2.90]])
        tasks = group.make_task()
        self.assertEqual(len(tasks), 1)
        contents = tasks[0].files()["input.dat"]
        self.assertIn("AtomicNumber = 3 57", contents)
        self.assertIn("@DistanceOfIon\n1.79 2.34\n2.34 2.9\n@End", contents)

    def test_radius_override_changes_only_the_requested_element(self):
        group = make_calypso_task_group_from_config(
            dict(self.config, distance_of_ions={"Li": 2.0})
        )
        self.assertEqual(group.atomic_number, [3, 57])
        np.testing.assert_allclose(group.distance_of_ions, [[2.8, 2.85], [2.85, 2.9]])
        self.assertEqual(len(group.make_task()), 1)
        # Overrides must not mutate the shared default-radius table.
        defaults = make_calypso_task_group_from_config(self.config)
        np.testing.assert_allclose(
            defaults.distance_of_ions, [[1.79, 2.34], [2.34, 2.9]]
        )

    def test_each_optional_field_can_be_omitted_independently(self):
        matrix = [[2.0, 3.0], [3.0, 4.0]]
        group = make_calypso_task_group_from_config(
            dict(self.config, distance_of_ions=matrix)
        )
        self.assertEqual(group.atomic_number, [3, 57])
        np.testing.assert_allclose(group.distance_of_ions, matrix)
        self.assertEqual(len(group.make_task()), 1)
        group = make_calypso_task_group_from_config(
            dict(self.config, atomic_number=[3, 57])
        )
        self.assertEqual(group.atomic_number, [3, 57])
        np.testing.assert_allclose(group.distance_of_ions, [[1.79, 2.34], [2.34, 2.9]])
        self.assertEqual(len(group.make_task()), 1)

    def test_explicit_lists_are_preserved(self):
        matrix = [[2.0, 3.0], [3.0, 4.0]]
        group = make_calypso_task_group_from_config(
            dict(self.config, atomic_number=[3, 57], distance_of_ions=matrix)
        )
        self.assertEqual(group.atomic_number, [3, 57])
        np.testing.assert_allclose(group.distance_of_ions, matrix)

    def test_species_count_reports_both_inputs(self):
        with self.assertRaisesRegex(ValueError, "numb_of_species=3.*name_of_atoms"):
            make_calypso_task_group_from_config(dict(self.config, numb_of_species=3))

    def test_unknown_element_reports_the_configuration_key(self):
        for names in (["Li", "Xx"], ["Li", "X"], [["Li", "Xx"], ["La"]]):
            with self.subTest(names=names):
                with self.assertRaisesRegex(
                    ValueError, "unknown element.*name_of_atoms"
                ):
                    make_calypso_task_group_from_config(
                        dict(self.config, name_of_atoms=names)
                    )
        with self.assertRaisesRegex(ValueError, "unknown element.*distance_of_ions"):
            make_calypso_task_group_from_config(
                dict(self.config, distance_of_ions={"Xx": 2.0})
            )

    def test_unknown_radius_requires_an_explicit_value(self):
        config = {"name_of_atoms": ["Cf"], "numb_of_atoms": [1], "numb_of_species": 1}
        for distance in (None, {}):
            with self.subTest(distance=distance):
                with self.assertRaisesRegex(ValueError, "no known covalent radius.*Cf"):
                    make_calypso_task_group_from_config(
                        config
                        if distance is None
                        else dict(config, distance_of_ions=distance)
                    )
        for distance, expected in (({"Cf": 2.0}, [[2.8]]), ([[3.0]], [[3.0]])):
            with self.subTest(distance=distance):
                group = make_calypso_task_group_from_config(
                    dict(config, distance_of_ions=distance)
                )
                self.assertEqual(group.atomic_number, [98])
                np.testing.assert_allclose(group.distance_of_ions, expected)
                self.assertEqual(len(group.make_task()), 1)
