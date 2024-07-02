import os
import shutil
import unittest
from pathlib import (
    Path,
)

import dflow
import dpdata
from dflow import (
    Step,
    Workflow,
    download_artifact,
    upload_artifact,
)

from dpgen2.fp import (
    FpOpCp2kInputs,
    PrepFpOpCp2k,
    RunFpOpCp2k,
)
from dpgen2.superop import (
    PrepRunFp,
)


class TestFpOpCp2k(unittest.TestCase):
    def tearDown(self):
        if Path("upload").is_dir():
            shutil.rmtree("upload")
        if Path("output").is_dir():
            shutil.rmtree("output")
        for p in Path(".").glob("cp2k-dpgen-*"):
            shutil.rmtree(p)

    def test_cp2k(self):
        data_path = Path(__file__).parent / "data.cp2k"
        fp_config = {
            "inputs": FpOpCp2kInputs(data_path / "input.inp"),
            "run": {
                "command": "cp -r %s output.log && cat %s"
                % (data_path / "output.log", data_path / "output.log"),
            },
        }
        confs = [data_path / "sys-3"]
        type_map = ["Na"]

        dflow.config["mode"] = "debug"
        prep_run_fp_op = PrepRunFp(
            "prep-run-fp",
            PrepFpOpCp2k,
            RunFpOpCp2k,
        )
        prep_run_fp = Step(
            name="prep-run-fp",
            template=prep_run_fp_op,
            parameters={
                "block_id": "iter-000000",
                "fp_config": fp_config,
                "type_map": type_map,
            },
            artifacts={
                "confs": upload_artifact(confs),
            },
            key="iter-000000--prep-run-fp",
        )
        wf = Workflow(name="cp2k-dpgen")
        wf.add(prep_run_fp)
        wf.submit()
        self.assertEqual(wf.query_status(), "Succeeded")

        step = wf.query_step(key="iter-000000--run-fp-000000")[0]
        log = download_artifact(step.outputs.artifacts["log"], path="output")[0]
        self.assertTrue(os.path.isfile(log))
        labeled_data = download_artifact(
            step.outputs.artifacts["labeled_data"], path="output"
        )[0]
        s = dpdata.LabeledSystem(labeled_data, fmt="deepmd/npy")
        self.assertEqual(len(s), 1)
