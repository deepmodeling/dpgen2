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
from dpgen2.utils.step_config import normalize as normalize_step_dict


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

    def test_step_configs(self):
        prep_config = normalize_step_dict(
            {
                "continue_on_failed": True,
                "continue_on_num_success": 1,
                "continue_on_success_ratio": 0.5,
                "parallelism": 3,
            }
        )
        run_config = normalize_step_dict(
            {
                "continue_on_failed": True,
                "continue_on_num_success": 2,
                "continue_on_success_ratio": 0.8,
                "parallelism": 7,
            }
        )
        steps = PrepRunDiffCSP(
            "prep-run-diffcsp",
            MockedDiffCSPGen,
            PrepRelax,
            MockedRunRelax,
            prep_config=prep_config,
            run_config=run_config,
        )
        diffcsp_gen = next(step for step in steps.steps if step.name == "diffcsp-gen")
        prep_relax = next(step for step in steps.steps if step.name == "prep-relax")
        run_relax = next(step for step in steps.steps if step.name == "run-relax")

        self.assertTrue(diffcsp_gen.continue_on_failed)
        self.assertEqual(diffcsp_gen.continue_on_num_success, 1)
        self.assertEqual(diffcsp_gen.continue_on_success_ratio, 0.5)
        self.assertEqual(diffcsp_gen.parallelism, 3)
        self.assertTrue(prep_relax.continue_on_failed)
        self.assertIsNone(prep_relax.continue_on_num_success)
        self.assertIsNone(prep_relax.continue_on_success_ratio)
        self.assertEqual(prep_relax.parallelism, 3)
        self.assertTrue(run_relax.continue_on_failed)
        self.assertEqual(run_relax.continue_on_num_success, 2)
        self.assertEqual(run_relax.continue_on_success_ratio, 0.8)
        self.assertEqual(run_relax.parallelism, 7)

    def tearDown(self):
        for d in glob.glob("test-prep-run-diffcsp-*") + ["task.000000", "task.000001"]:
            if os.path.isdir(d):
                shutil.rmtree(d)
