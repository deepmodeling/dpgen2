import os
import shutil
import unittest
from pathlib import (
    Path,
)

from dflow.python import (
    OPIO,
)

from dpgen2.op import (
    PrepRelax,
)


class TestPrepRelax(unittest.TestCase):
    def setUp(self):
        self.cifs = []

    def _make_cifs(self, count):
        cifs = []
        for i in range(count):
            p = Path("%i.cif" % i)
            p.write_text("Mocked cif.")
            cifs.append(p)
        self.cifs.extend(cifs)
        return cifs

    def _run_prep_relax(self, ncifs, group_size):
        op_in = OPIO(
            {
                "expl_config": {
                    "relax_group_size": group_size,
                },
                "cifs": self._make_cifs(ncifs),
            }
        )
        op = PrepRelax()
        return op.execute(op_in)

    def test_prep_relax(self):
        op_out = self._run_prep_relax(4, 2)
        self.assertEqual(op_out["ntasks"], 2)
        self.assertEqual(len(op_out["task_paths"]), 2)
        for i, task_path in enumerate(op_out["task_paths"]):
            self.assertEqual(str(task_path), "task.%06d" % i)
            self.assertEqual(len(list(task_path.iterdir())), 2)

    def test_keeps_partial_final_group(self):
        """Assign every CIF when the final task is not a full group."""
        op_out = self._run_prep_relax(5, 2)

        self.assertEqual(op_out["ntasks"], 3)
        self.assertEqual(
            [len(list(task_path.iterdir())) for task_path in op_out["task_paths"]],
            [2, 2, 1],
        )

    def test_creates_task_when_group_is_larger_than_input(self):
        op_out = self._run_prep_relax(1, 2)

        self.assertEqual(op_out["ntasks"], 1)
        self.assertEqual(len(list(op_out["task_paths"][0].iterdir())), 1)

    def test_rejects_non_positive_group_size(self):
        for group_size in (0, -1):
            with self.subTest(group_size=group_size):
                with self.assertRaisesRegex(ValueError, "greater than zero"):
                    self._run_prep_relax(1, group_size)

    def tearDown(self):
        for task_path in Path().glob("task.[0-9][0-9][0-9][0-9][0-9][0-9]"):
            if task_path.is_dir():
                shutil.rmtree(task_path)
        for cif in self.cifs:
            if cif.is_file():
                os.remove(cif)
