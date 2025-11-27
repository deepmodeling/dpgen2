import glob
import os
import shutil
import unittest
from pathlib import (
    Path,
)

from dflow import (
    Step,
    Workflow,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
)

from dpgen2.exploration.task import (
    DiffCSPTaskGroup,
)
from dpgen2.op import (
    DiffCSPGen,
    PrepRelax,
    RunRelax,
)
from dpgen2.superop import (
    PrepRunDiffCSP,
)


class MockedDiffCSPGen(DiffCSPGen):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_dir = Path("diffcsp.%s" % ip["task_id"])
        task_dir.mkdir(exist_ok=True)
        for i in range(2):
            fpath = task_dir / ("%s.cif" % i)
            fpath.write_text("Mocked cif.")
        return OPIO(
            {
                "cifs": list(Path(task_dir).glob("*.cif")),
            }
        )


class MockedRunRelax(RunRelax):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        cifs = os.listdir(ip["task_path"])
        assert len(cifs) == 2
        trajs = []
        model_devis = []
        for cif in cifs:
            name = cif[:-4]
            traj = ip["task_path"] / ("traj.%s.dump" % name)
            traj.write_text("Mocked traj.")
            trajs.append(traj)
            model_devi = ip["task_path"] / ("model_devi.%s.out" % name)
            model_devi.write_text("Mocked model_devi.")
            model_devis.append(model_devi)
        return OPIO(
            {
                "trajs": trajs,
                "model_devis": model_devis,
            }
        )


class TestPrepRunDiffCSP(unittest.TestCase):
    def testPrepRunDiffCSP(self):
        task_group = DiffCSPTaskGroup()
        task_group.make_task()

        wf = Workflow("test-prep-run-diffcsp")
        upload_packages = []
        if "__file__" in globals():
            upload_packages.append(__file__)
            upload_packages.append(os.path.dirname(__file__))
        steps = PrepRunDiffCSP(
            "prep-run-diffcsp",
            MockedDiffCSPGen,
            PrepRelax,
            MockedRunRelax,
            upload_python_packages=upload_packages,
        )
        step = Step(
            "main",
            template=steps,
            parameters={
                "block_id": "iter-000000",
                "expl_task_grp": task_group,
                "explore_config": {
                    "gen_tasks": 2,
                    "gen_command": "echo 'mocked generation' --model_path .",
                    "relax_group_size": 2,
                },
                "type_map": [],
            },
            artifacts={
                "models": upload_artifact([]),
            },
        )
        wf.add(step)
        wf.submit()
        wf.wait()
        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step("main")[0]
        trajs = download_artifact(step.outputs.artifacts["trajs"])
        self.assertEqual(len(trajs), 4)
        model_devis = download_artifact(step.outputs.artifacts["model_devis"])
        self.assertEqual(len(model_devis), 4)

    def tearDown(self):
        for d in glob.glob("test-prep-run-diffcsp-*") + ["task.000000", "task.000001"]:
            if os.path.isdir(d):
                shutil.rmtree(d)
