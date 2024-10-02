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
    def testPrepRelax(self):
        cifs = []
        for i in range(4):
            p = Path("%i.cif" % i)
            p.write_text("Mocked cif.")
            cifs.append(p)
        op_in = OPIO(
            {
                "expl_config": {
                    "relax_group_size": 2,
                },
                "cifs": cifs,
            }
        )
        op = PrepRelax()
        op_out = op.execute(op_in)
        self.assertEqual(op_out["ntasks"], 2)
        self.assertEqual(len(op_out["task_paths"]), 2)
        for i, task_path in enumerate(op_out["task_paths"]):
            self.assertEqual(str(task_path), "task.%06d" % i)
            self.assertEqual(len(list(task_path.iterdir())), 2)

    def tearDown(self):
        for i in range(2):
            if os.path.isdir("task.%06d" % i):
                shutil.rmtree("task.%06d" % i)
        for i in range(4):
            if os.path.isfile(f"{i}.cif"):
                os.remove(f"{i}.cif")
