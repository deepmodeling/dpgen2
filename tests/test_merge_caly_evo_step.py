import json
import os
import pickle
import shutil
import time
import unittest
from pathlib import (
    Path,
)
from typing import (
    List,
    Set,
)

import jsonpickle
import numpy as np
from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    S3Artifact,
    Step,
    Steps,
    Workflow,
    argo_range,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_index_pattern,
    calypso_run_opt_file,
)

try:
    from context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from context import (
    default_host,
    default_image,
    skip_ut_with_dflow,
    skip_ut_with_dflow_reason,
    upload_python_packages,
)
from mocked_ops import (
    MockedCollRunCaly,
    MockedPrepCalyDPOptim,
    MockedRunCalyDPOptim,
    mocked_numb_models,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_model_devi_name,
    lmp_task_pattern,
    lmp_traj_name,
    model_name_pattern,
    train_log_name,
    train_script_name,
    train_task_pattern,
)
from dpgen2.exploration.task import (
    ExplorationTask,
    ExplorationTaskGroup,
)
from dpgen2.op import (
    PrepCalyDPOptim,
    RunCalyDPOptim,
)
from dpgen2.op.caly_evo_step_merge import (
    CalyEvoStepMerge,
)
from dpgen2.op.collect_run_caly import (
    CollRunCaly,
)
from dpgen2.op.prep_caly_input import (
    PrepCalyInput,
)
from dpgen2.superop.caly_evo_step import (
    CalyEvoStep,
)
from dpgen2.utils.step_config import normalize as normalize_step_dict

default_config = normalize_step_dict(
    {
        "template_config": {
            "image": default_image,
        },
        "template_slice_config": {
            "group_size": 2,
            "pool_size": 1,
        },
    }
)


class TestMockedCollRunCaly(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {}
        self.task_name = "task_name"
        self.file_storage = Path("storge_files")
        self.file_storage.mkdir(parents=True, exist_ok=True)
        self.input_file = self.file_storage.joinpath("input.dat")
        self.input_file.write_text("5")
        self.step_file = None
        self.results_dir = None
        self.opt_results_dir = None
        self.cnt_num = 0
        self.finished = str(False)

    def tearDown(self) -> None:
        shutil.rmtree(self.file_storage, ignore_errors=True)
        shutil.rmtree(Path(self.task_name), ignore_errors=True)

    def test_mocked_coll_run_caly_00(self):
        op = MockedCollRunCaly()
        out = op.execute(
            OPIO(
                {
                    "config": self.config,
                    "task_name": self.task_name,
                    "cnt_num": self.cnt_num,
                    "input_file": self.input_file,
                    "step": self.step_file,
                    "results": self.results_dir,
                    "opt_results_dir": self.opt_results_dir,
                }
            )
        )

        self.assertTrue(out["task_name"] == self.task_name)
        self.assertTrue(out["finished"] == "false")
        self.assertTrue(
            Path("task_name/poscar_dir").joinpath("POSCAR_1")
            in list(out["poscar_dir"].glob("POSCAR_*"))
        )
        self.assertTrue(len(list(out["poscar_dir"].rglob("POSCAR_*"))) == 5)
        self.assertTrue(
            out["input_file"] == Path(self.task_name).joinpath(self.input_file.name)
        )
        self.assertTrue(out["input_file"].read_text() == str(5))
        self.assertTrue(out["step"] == Path(self.task_name).joinpath("step"))
        self.assertTrue(out["step"].read_text() == str(2))
        self.assertTrue(out["results"] == Path(self.task_name).joinpath("results"))


class TestMockedRunDPOptim(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {}
        self.task_name = "task_name"
        self.file_storage = Path("storge_files")
        self.file_storage.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            self.file_storage.joinpath(f"POSCAR_{i}").write_text(f"POSCAR_{i}")
        self.file_storage.joinpath(f"frozen_model.pb").write_text(f"model.{i}.pb")
        self.caly_run_opt_file = self.file_storage.joinpath(calypso_run_opt_file)
        self.caly_run_opt_file.write_text("caly_run_opt_script")
        self.caly_check_opt_file = self.file_storage.joinpath(calypso_check_opt_file)
        self.caly_check_opt_file.write_text("caly_check_opt_script")

    def tearDown(self) -> None:
        shutil.rmtree(self.file_storage, ignore_errors=True)
        shutil.rmtree(Path(self.task_name), ignore_errors=True)

    def test_mocked_run_dp_optim(self):
        op = MockedRunCalyDPOptim()
        out = op.execute(
            OPIO(
                {
                    "config": self.config,
                    "finished": "false",
                    "cnt_num": 0,
                    "task_name": self.task_name,
                    "task_dir": self.file_storage,
                }
            )
        )

        # check output
        self.assertEqual(out["task_name"], self.task_name)

        optim_results_dir = out["optim_results_dir"]
        list_optim_results_dir = list(optim_results_dir.iterdir())
        counts_optim_results_dir = len(list_optim_results_dir)
        counts_outcar_in_optim_results_dir = len(
            list(optim_results_dir.rglob("OUTCAR_*"))
        )

        self.assertTrue(optim_results_dir, Path(self.task_name) / "optim_results_dir")
        self.assertEqual(counts_optim_results_dir, 15)
        self.assertEqual(counts_outcar_in_optim_results_dir, 5)
        self.assertTrue(
            Path(self.task_name) / "optim_results_dir" / "CONTCAR_4"
            in list_optim_results_dir
        )

        traj_results_dir = out["traj_results"]
        list_traj_results_dir = list(traj_results_dir.glob("*.traj"))
        counts_traj = len(list_traj_results_dir)
        self.assertEqual(traj_results_dir, Path(self.task_name) / "traj_results")
        self.assertEqual(counts_traj, 5)
        self.assertTrue(
            Path(self.task_name) / "traj_results" / "3.traj", list_traj_results_dir
        )


# @unittest.skip("temporary pass")
@unittest.skipIf(skip_ut_with_dflow, skip_ut_with_dflow_reason)
class TestCalyEvoStepMerge(unittest.TestCase):
    def setUp(self):
        self.expl_config = {}
        self.work_dir = Path("storge_files")
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.max_step = 2
        self.nmodels = mocked_numb_models
        self.model_list = []
        for ii in range(self.nmodels):
            model_path = self.work_dir.joinpath(f"task.{ii}")
            model_path.mkdir(exist_ok=True, parents=True)
            model = model_path.joinpath(f"model.ckpt.pt")
            model.write_text(f"model {ii}")
            self.model_list.append(model)
        self.models = upload_artifact(self.model_list)

        self.block_id = "id123id"
        temp_name_pattern = "caly_task." + calypso_index_pattern
        self.task_name = temp_name_pattern % 1
        self.task_name_list = [self.task_name, temp_name_pattern % 2]

        input_file = self.work_dir.joinpath("input.dat")
        input_file.write_text(str(self.max_step))
        self.input_file = upload_artifact(input_file)
        self.input_file_list = upload_artifact([input_file, input_file])

        self.step = None
        self.results = None
        self.opt_results_dir = None

        caly_run_opt_file = self.work_dir.joinpath("caly_run_opt.py")
        caly_run_opt_file.write_text("caly_run_opt")
        self.caly_run_opt_file = upload_artifact(caly_run_opt_file)
        self.caly_run_opt_files = upload_artifact(
            [caly_run_opt_file, caly_run_opt_file]
        )

        caly_check_opt_file = self.work_dir.joinpath("caly_check_opt.py")
        caly_check_opt_file.write_text("caly_check_opt")
        self.caly_check_opt_file = upload_artifact(caly_check_opt_file)
        self.caly_check_opt_files = upload_artifact(
            [caly_check_opt_file, caly_check_opt_file]
        )

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)
        for i in Path().glob("caly-evo-step-*"):
            shutil.rmtree(i, ignore_errors=True)
        for i in Path().glob("caly_task*"):
            shutil.rmtree(i, ignore_errors=True)

    def test_caly_evo_step(self):
        steps = CalyEvoStepMerge(
            name="caly-evo-step",
            collect_run_caly=MockedCollRunCaly,
            prep_dp_optim=PrepCalyDPOptim,
            run_dp_optim=MockedRunCalyDPOptim,
            prep_config=default_config,
            run_config=default_config,
            upload_python_packages=None,
        )
        caly_evo_step = Step(
            "caly-evo-step",
            template=PythonOPTemplate(
                steps,
                python_packages=upload_python_packages,
                **default_config["template_config"],
            ),
            slices=Slices(
                input_parameter=[
                    "task_name",
                ],
                input_artifact=[
                    "input_file",
                    "results",
                    "step",
                    "opt_results_dir",
                    "caly_run_opt_file",
                    "caly_check_opt_file",
                ],
                output_artifact=["traj_results"],
            ),
            parameters={
                "block_id": self.block_id,
                "task_name": self.task_name_list,
                "iter_num": "{{item}}",
                "expl_config": self.expl_config,
            },
            artifacts={
                "models": self.models,
                "input_file": self.input_file_list,
                "caly_run_opt_file": self.caly_run_opt_files,
                "caly_check_opt_file": self.caly_check_opt_files,
                "results": None,
                "step": None,
                "opt_results_dir": None,
                "qhull_input": None,
            },
        )

        wf = Workflow(name="caly-evo-step", host=default_host)
        wf.add(caly_evo_step)
        wf.submit()

        while wf.query_status() in ["Pending", "Running"]:
            time.sleep(4)

        self.assertEqual(wf.query_status(), "Succeeded")
        step = wf.query_step(name="caly-evo-step")[0]
        self.assertEqual(step.phase, "Succeeded")

        download_artifact(step.outputs.artifacts["traj_results"])

        cwd = Path().cwd()
        for idx, name in enumerate(self.task_name_list):
            cwd = Path().cwd()
            os.chdir(Path(name))
            traj_list = list(Path().rglob("*.traj"))
            self.assertEqual(len(traj_list), 5 * self.max_step)
            self.assertTrue(
                Path("opt_path_0/traj_results").joinpath(f"{idx}.0.traj") in traj_list
            )
            os.chdir(cwd)
