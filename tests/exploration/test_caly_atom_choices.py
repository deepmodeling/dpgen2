"""Exact, terminating assignment of CALYPSO's nested species choices."""

import copy
import itertools
import unittest
from unittest.mock import (
    patch,
)

from dpgen2.exploration.task import (
    make_calypso_task_group_from_config,
)
from dpgen2.exploration.task.caly_task_group import (
    _choose_distinct_species,
)


class TestCalyAtomChoices(unittest.TestCase):
    def assert_valid_choice(self, candidates, choice):
        self.assertEqual(len(choice), len(candidates))
        self.assertEqual(len(set(choice)), len(choice))
        for atom, allowed in zip(choice, candidates):
            self.assertIn(atom, allowed)

    def test_matches_exhaustive_search(self):
        """Check all 399 nonempty families over three symbols independently."""
        symbols = ("Li", "Na", "K")
        subsets = [
            list(subset)
            for size in range(1, 4)
            for subset in itertools.combinations(symbols, size)
        ]
        counts = [0, 0]
        for size in range(1, 4):
            for family in itertools.product(subsets, repeat=size):
                candidates = copy.deepcopy(list(family))
                expected = any(
                    len(set(choice)) == size
                    for choice in itertools.product(*candidates)
                )
                counts[expected] += 1
                with self.subTest(candidates=candidates):
                    if expected:
                        self.assert_valid_choice(
                            candidates, _choose_distinct_species(candidates)
                        )
                    else:
                        with self.assertRaisesRegex(ValueError, "distinct species"):
                            _choose_distinct_species(candidates)
                    self.assertEqual(candidates, list(family))
        self.assertEqual(sum(counts), 399)
        self.assertEqual(counts, [99, 300])

    def test_reassigns_an_earlier_choice(self):
        """A greedy pick must be displaced to accommodate the final slot."""
        with patch("dpgen2.exploration.task.caly_task_group.random.shuffle"):
            self.assertEqual(
                _choose_distinct_species([["Li", "Na", "K"], ["Na", "K"], ["K"]]),
                ["Li", "Na", "K"],
            )
            self.assertEqual(
                _choose_distinct_species([["Li", "Na"], ["Li"]]), ["Na", "Li"]
            )

    def test_randomized_candidate_order(self):
        """Controlled shuffle orders can produce different valid assignments."""
        candidates = [["Li", "Na"], ["Li", "Na"]]
        with patch("dpgen2.exploration.task.caly_task_group.random.shuffle"):
            first = _choose_distinct_species(candidates)
        with patch(
            "dpgen2.exploration.task.caly_task_group.random.shuffle",
            side_effect=lambda atoms: atoms.reverse(),
        ):
            second = _choose_distinct_species(candidates)
        self.assert_valid_choice(candidates, first)
        self.assert_valid_choice(candidates, second)
        self.assertNotEqual(first, second)
        self.assertEqual(candidates, [["Li", "Na"], ["Li", "Na"]])

    def test_empty_and_repeated_candidates(self):
        with self.assertRaisesRegex(ValueError, "name_of_atoms"):
            _choose_distinct_species([["Li"], []])
        self.assertEqual(_choose_distinct_species([["Li", "Li"]]), ["Li"])

    @patch(
        "dpgen2.exploration.task.caly_task_group.random.choice",
        side_effect=AssertionError("unbounded rejection sampling must not be used"),
    )
    def test_task_group_accepts_satisfiable_families(self, unused_choice):
        for candidates in (
            [["Li"]],
            [["Li", "Na"]],
            [["Li", "Na"], ["Li", "Na"]],
            [["Li", "Na", "K"], ["Na", "K"], ["K"]],
        ):
            size = len(candidates)
            with self.subTest(candidates=candidates):
                group = make_calypso_task_group_from_config(
                    {
                        "name_of_atoms": candidates,
                        "numb_of_atoms": [1] * size,
                        "numb_of_species": size,
                        "distance_of_ions": [[1.0] * size for _ in range(size)],
                    }
                )
                self.assert_valid_choice(candidates, group.name_of_atoms)
                self.assertEqual(len(group.make_task()), 1)

    @patch(
        "dpgen2.exploration.task.caly_task_group.random.choice",
        side_effect=AssertionError("unbounded rejection sampling must not be used"),
    )
    def test_task_group_rejects_impossible_families(self, unused_choice):
        for candidates in (
            [["Li"], ["Li"]],
            [["Li"], ["Li"], ["Na", "K"]],
            [["Li", "Na"], ["Li", "Na"], ["Li", "Na"], ["K"]],
            [["Li"], []],
        ):
            with self.subTest(candidates=candidates):
                with self.assertRaisesRegex(ValueError, "distinct species") as error:
                    make_calypso_task_group_from_config(
                        {
                            "name_of_atoms": candidates,
                            "numb_of_atoms": [1] * len(candidates),
                            "numb_of_species": len(candidates),
                            "distance_of_ions": {},
                        }
                    )
                self.assertIn(repr(candidates), str(error.exception))
