from dflow.python import (
    OP,
    OPIO,
    Artifact,
    FatalError,
    OPIOSign,
    upload_packages,
)

upload_packages.append(__file__)

import json
import os
import pickle
import re
import shutil
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Tuple,
)

try:
    from flow.context import (
        dpgen2,
    )
except ModuleNotFoundError:
    # case of upload everything to argo, no context needed
    pass
from dpgen2.constants import (
    calypso_check_opt_file,
    calypso_run_opt_file,
    fp_task_pattern,
    lmp_conf_name,
    lmp_input_name,
    lmp_log_name,
    lmp_model_devi_name,
    lmp_task_pattern,
    lmp_traj_name,
    model_name_pattern,
    nvnmd_model_name_pattern,
    train_log_name,
    train_script_name,
    train_task_pattern,
)
from dpgen2.exploration.report import (
    ExplorationReport,
)
from dpgen2.exploration.scheduler import (
    ConvergenceCheckStageScheduler,
)
from dpgen2.exploration.selector import (
    ConfSelector,
)
from dpgen2.exploration.task import (
    ExplorationStage,
    ExplorationTask,
    ExplorationTaskGroup,
)
from dpgen2.fp import (
    PrepVasp,
    RunVasp,
)
from dpgen2.fp.vasp import (
    VaspInputs,
    vasp_conf_name,
    vasp_input_name,
)
from dpgen2.op.collect_data import (
    CollectData,
)
from dpgen2.op.collect_run_caly import (
    CollRunCaly,
)
from dpgen2.op.prep_caly_dp_optim import (
    PrepCalyDPOptim,
)
from dpgen2.op.prep_dp_train import (
    PrepDPTrain,
)
from dpgen2.op.prep_lmp import (
    PrepExplorationTaskGroup,
)
from dpgen2.op.run_caly_dp_optim import (
    RunCalyDPOptim,
)
from dpgen2.op.run_caly_model_devi import (
    RunCalyModelDevi,
)
from dpgen2.op.run_dp_train import (
    RunDPTrain,
)
from dpgen2.op.run_lmp import (
    RunLmp,
)
from dpgen2.op.run_nvnmd import (
    RunNvNMD,
)
from dpgen2.op.run_nvnmd_train import (
    RunNvNMDTrain,
)
from dpgen2.op.select_confs import (
    SelectConfs,
)

mocked_template_script = {"seed": 1024, "data": []}
mocked_numb_models = 3
mocked_numb_lmp_tasks = 6
mocked_numb_select = 2
mocked_incar_template = "incar template"


def make_mocked_init_models(numb_models):
    tmp_models = []
    for ii in range(numb_models):
        ff = Path(model_name_pattern % ii)
        ff.write_text(f"This is init model {ii}")
        tmp_models.append(ff)
    return tmp_models


def make_mocked_init_nvnmd_models(numb_models):
    tmp_models = []
    for ii in range(numb_models):
        nvnmd_models_dir = Path(nvnmd_model_name_pattern % ii)
        nvnmd_models_dir.mkdir(exist_ok=True, parents=True)
        for jj in (
            "frozen_model.pb",
            "model.ckpt.meta",
            "model.ckpt.data",
            "model.ckpt.index",
        ):
            ff = nvnmd_models_dir / jj
            ff.write_text(f"This is init {jj} {ii}")
        tmp_models.append(nvnmd_models_dir)
    return tmp_models


def make_mocked_init_data():
    tmp_init_data = [Path("init_data/foo"), Path("init_data/bar")]
    for ii in tmp_init_data:
        ii.mkdir(exist_ok=True, parents=True)
        (ii / "a").write_text("data a")
        (ii / "b").write_text("data b")
    return tmp_init_data


class MockedPrepDPTrain(PrepDPTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        template = ip["template_script"]
        numb_models = ip["numb_models"]
        ofiles = []
        osubdirs = []

        assert template == mocked_template_script
        assert numb_models == mocked_numb_models

        for ii in range(numb_models):
            jtmp = template
            jtmp["seed"] = ii
            subdir = Path(train_task_pattern % ii)
            subdir.mkdir(exist_ok=True, parents=True)
            fname = subdir / "input.json"
            with open(fname, "w") as fp:
                json.dump(jtmp, fp, indent=4)
            osubdirs.append(str(subdir))
            ofiles.append(fname)

        op = OPIO(
            {
                "task_names": osubdirs,
                "task_paths": [Path(ii) for ii in osubdirs],
            }
        )
        return op


class MockedRunDPTrain(RunDPTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        work_dir = Path(ip["task_name"])
        script = ip["task_path"] / "input.json"
        init_model = Path(ip["init_model"])
        init_data = ip["init_data"]
        iter_data = ip["iter_data"]

        assert script.is_file()
        assert ip["task_path"].is_dir()
        assert init_model.is_file()
        assert len(init_data) == 2
        assert re.match("task.[0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert ip["task_name"] in str(ip["task_path"])
        assert "model" in str(ip["init_model"])
        assert ".pb" in str(ip["init_model"])
        list_init_data = sorted([str(ii) for ii in init_data])
        assert "init_data/bar" in list_init_data[0]
        assert "init_data/foo" in list_init_data[1]
        assert Path(list_init_data[0]).is_dir()
        assert Path(list_init_data[1]).is_dir()

        script = Path(script).resolve()
        init_model = init_model.resolve()
        init_model_str = str(init_model)
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp["data"] = data
        with open(script, "w") as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        oscript = Path("input.json")
        if not oscript.exists():
            from shutil import (
                copyfile,
            )

            copyfile(script, oscript)
        model = Path("model.pb")
        lcurve = Path("lcurve.out")
        log = Path("log")

        assert init_model.exists()
        with log.open("w") as f:
            f.write(f"init_model {str(init_model)} OK\n")
        for ii in jtmp["data"]:
            assert Path(ii).exists()
            assert (ii in init_data_str) or (ii in iter_data_str)
            with log.open("a") as f:
                f.write(f"data {str(ii)} OK\n")
        assert script.exists()
        with log.open("a") as f:
            f.write(f"script {str(script)} OK\n")

        with model.open("w") as f:
            f.write("read from init model: \n")
            f.write(init_model.read_text() + "\n")
        with lcurve.open("w") as f:
            f.write("read from train_script: \n")
            f.write(script.read_text() + "\n")

        os.chdir(cwd)

        return OPIO(
            {
                "script": work_dir / oscript,
                "model": work_dir / model,
                "lcurve": work_dir / lcurve,
                "log": work_dir / log,
            }
        )


class MockedRunDPTrainCheckOptParam(RunDPTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        if not ip["optional_parameter"]["mixed_type"]:
            raise FatalError(
                f"the value of mixed_type is {ip['optional_parameter']['mixed_type']} "
            )
        if not ip["optional_parameter"]["finetune_mode"]:
            raise FatalError(
                f"the value of finetune_mode is {ip['optional_parameter']['finetune_mode']} "
            )
        return MockedRunDPTrain.execute(self, ip)


class MockedRunDPTrainNoneInitModel(RunDPTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        work_dir = Path(ip["task_name"])
        script = ip["task_path"] / "input.json"
        if ip["init_model"] is not None:
            raise FatalError("init model is not None")
        init_data = ip["init_data"]
        iter_data = ip["iter_data"]

        assert script.is_file()
        assert ip["task_path"].is_dir()
        assert len(init_data) == 2
        assert re.match("task.[0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert ip["task_name"] in str(ip["task_path"])
        list_init_data = sorted([str(ii) for ii in init_data])
        assert "init_data/bar" in list_init_data[0]
        assert "init_data/foo" in list_init_data[1]
        assert Path(list_init_data[0]).is_dir()
        assert Path(list_init_data[1]).is_dir()

        script = Path(script).resolve()
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp["data"] = data
        with open(script, "w") as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        oscript = Path("input.json")
        if not oscript.exists():
            from shutil import (
                copyfile,
            )

            copyfile(script, oscript)
        model = Path("model.pb")
        lcurve = Path("lcurve.out")
        log = Path("log")

        for ii in jtmp["data"]:
            assert Path(ii).exists()
            assert (ii in init_data_str) or (ii in iter_data_str)
            with log.open("a") as f:
                f.write(f"data {str(ii)} OK\n")
        assert script.exists()
        with log.open("a") as f:
            f.write(f"script {str(script)} OK\n")

        with model.open("w") as f:
            f.write("read from init model: \n")
        with lcurve.open("w") as f:
            f.write("read from train_script: \n")
            f.write(script.read_text() + "\n")

        os.chdir(cwd)

        return OPIO(
            {
                "script": work_dir / oscript,
                "model": work_dir / model,
                "lcurve": work_dir / lcurve,
                "log": work_dir / log,
            }
        )


class MockedPrepNvNMDTrain(PrepDPTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        template = ip["template_script"]
        numb_models = ip["numb_models"]
        ofiles = []
        osubdirs = []

        assert template == mocked_template_script
        assert numb_models == mocked_numb_models

        for ii in range(numb_models):
            jtmp = template
            jtmp["seed"] = ii
            subdir = Path(train_task_pattern % ii)
            subdir.mkdir(exist_ok=True, parents=True)
            fname = subdir / "input.json"
            with open(fname, "w") as fp:
                json.dump(jtmp, fp, indent=4)
            osubdirs.append(str(subdir))
            ofiles.append(fname)

        op = OPIO(
            {
                "task_names": osubdirs,
                "task_paths": [Path(ii) for ii in osubdirs],
            }
        )
        return op


class MockedRunNvNMDTrain(RunNvNMDTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        work_dir = Path(ip["task_name"])
        script = ip["task_path"] / "input.json"
        init_model = ip["init_model"]
        init_data = ip["init_data"]
        iter_data = ip["iter_data"]

        assert script.is_file()
        assert ip["task_path"].is_dir()
        assert ip["init_model"].is_dir()
        assert len(init_data) == 2
        assert re.match("task.[0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert ip["task_name"] in str(ip["task_path"])
        init_frz_model = ip["init_model"] / "frozen_model.pb"
        init_model_ckpt_data = ip["init_model"] / "model.ckpt.data"
        init_model_ckpt_meta = ip["init_model"] / "model.ckpt.meta"
        init_model_ckpt_index = ip["init_model"] / "model.ckpt.index"

        assert ".pb" in str(init_frz_model)
        assert "ckpt.meta" in str(init_model_ckpt_meta)
        assert "ckpt.data" in str(init_model_ckpt_data)
        assert "ckpt.index" in str(init_model_ckpt_index)
        list_init_data = sorted([str(ii) for ii in init_data])
        assert "init_data/bar" in list_init_data[0]
        assert "init_data/foo" in list_init_data[1]
        assert Path(list_init_data[0]).is_dir()
        assert Path(list_init_data[1]).is_dir()

        script = Path(script).resolve()
        init_model = init_model.resolve()
        init_frz_model = init_frz_model.resolve()
        init_model_ckpt_data = init_model_ckpt_data.resolve()
        init_model_ckpt_meta = init_model_ckpt_meta.resolve()
        init_model_ckpt_index = init_model_ckpt_index.resolve()
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp["data"] = data
        with open(script, "w") as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        for script_str in ["input.json", "input_cnn.json", "input_qnn.json"]:
            oscript = Path(script_str)
            if not oscript.exists():
                from shutil import (
                    copyfile,
                )

                copyfile(script, oscript)

        oscript = Path("input.json")
        cnn_dir = Path("nvnmd_cnn")
        qnn_dir = Path("nvnmd_qnn")
        cnn_model = cnn_dir / Path("frozen_model.pb")
        qnn_model = qnn_dir / Path("model.pb")
        model_ckpt_meta_file = cnn_dir / Path("model.ckpt.meta")
        model_ckpt_data_file = cnn_dir / Path("model.ckpt.data-00000-of-00001")
        model_ckpt_index_file = cnn_dir / Path("model.ckpt.index")
        lcurve = cnn_dir / Path("lcurve.out")
        log = Path("log")

        assert init_frz_model.exists()
        assert init_model_ckpt_meta.exists()
        assert init_model_ckpt_data.exists()
        assert init_model_ckpt_index.exists()
        with log.open("w") as f:
            f.write(f"init_model {str(init_model)} OK\n")
        for ii in jtmp["data"]:
            assert Path(ii).exists()
            assert (ii in init_data_str) or (ii in iter_data_str)
            with log.open("a") as f:
                f.write(f"data {str(ii)} OK\n")
        assert script.exists()
        with log.open("a") as f:
            f.write(f"script {str(script)} OK\n")

        cnn_dir.mkdir(exist_ok=True, parents=True)
        with cnn_model.open("w") as f:
            f.write("read from init model: \n")
            f.write(init_frz_model.read_text() + "\n")
        with model_ckpt_meta_file.open("w") as f:
            f.write("read from init model: \n")
            f.write(init_model_ckpt_meta.read_text() + "\n")
        with model_ckpt_data_file.open("w") as f:
            f.write("read from init model: \n")
            f.write(init_model_ckpt_data.read_text() + "\n")
        with model_ckpt_index_file.open("w") as f:
            f.write("read from init model: \n")
            f.write(init_model_ckpt_index.read_text() + "\n")

        qnn_dir.mkdir(exist_ok=True, parents=True)
        with qnn_model.open("w") as f:
            f.write("read from init model: \n")
            f.write(init_frz_model.read_text() + "\n")
        with lcurve.open("w") as f:
            f.write("read from train_script: \n")
            f.write(script.read_text() + "\n")

        model_files = "nvnmd_models"
        os.makedirs(model_files, exist_ok=True)
        shutil.copy(cnn_model, "nvnmd_models")
        shutil.copy(qnn_model, "nvnmd_models")
        shutil.copy(model_ckpt_meta_file, "nvnmd_models")
        shutil.copy(model_ckpt_data_file, "nvnmd_models")
        shutil.copy(model_ckpt_index_file, "nvnmd_models")

        os.chdir(cwd)

        return OPIO(
            {
                "script": work_dir / oscript,
                "model": work_dir / model_files,
                "lcurve": work_dir / lcurve,
                "log": work_dir / log,
            }
        )


class MockedRunNvNMDTrainCheckOptParam(RunDPTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        if not ip["optional_parameter"]["mixed_type"]:
            raise FatalError(
                f"the value of mixed_type is {ip['optional_parameter']['mixed_type']} "
            )
        if not ip["optional_parameter"]["finetune_mode"]:
            raise FatalError(
                f"the value of finetune_mode is {ip['optional_parameter']['finetune_mode']} "
            )
        return MockedRunDPTrain.execute(self, ip)


class MockedRunNvNMDTrainNoneInitModel(RunNvNMDTrain):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        work_dir = Path(ip["task_name"])
        script = ip["task_path"] / "input.json"
        if ip["init_model"] is not None:
            raise FatalError("init model is not None")
        init_data = ip["init_data"]
        iter_data = ip["iter_data"]

        assert script.is_file()
        assert ip["task_path"].is_dir()
        assert len(init_data) == 2
        assert re.match("task.[0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert ip["task_name"] in str(ip["task_path"])
        list_init_data = sorted([str(ii) for ii in init_data])
        assert "init_data/bar" in list_init_data[0]
        assert "init_data/foo" in list_init_data[1]
        assert Path(list_init_data[0]).is_dir()
        assert Path(list_init_data[1]).is_dir()

        script = Path(script).resolve()
        init_data = [ii.resolve() for ii in init_data]
        iter_data = [ii.resolve() for ii in iter_data]
        init_data_str = [str(ii) for ii in init_data]
        iter_data_str = [str(ii) for ii in iter_data]

        with open(script) as fp:
            jtmp = json.load(fp)
        data = []
        for ii in sorted(init_data_str):
            data.append(ii)
        for ii in sorted(iter_data_str):
            data.append(ii)
        jtmp["data"] = data
        with open(script, "w") as fp:
            json.dump(jtmp, fp, indent=4)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        for script_str in ["input.json", "input_cnn.json", "input_qnn.json"]:
            oscript = Path(script_str)
            if not oscript.exists():
                from shutil import (
                    copyfile,
                )

                copyfile(script, oscript)

        oscript = Path("input.json")
        cnn_dir = Path("nvnmd_cnn")
        qnn_dir = Path("nvnmd_qnn")
        cnn_model = cnn_dir / Path("frozen_model.pb")
        qnn_model = qnn_dir / Path("model.pb")
        model_ckpt_meta_file = cnn_dir / Path("model.ckpt.meta")
        model_ckpt_data_file = cnn_dir / Path("model.ckpt.data-00000-of-00001")
        model_ckpt_index_file = cnn_dir / Path("model.ckpt.index")
        lcurve = cnn_dir / Path("lcurve.out")
        log = Path("log")

        for ii in jtmp["data"]:
            assert Path(ii).exists()
            assert (ii in init_data_str) or (ii in iter_data_str)
            with log.open("a") as f:
                f.write(f"data {str(ii)} OK\n")
        assert script.exists()
        with log.open("a") as f:
            f.write(f"script {str(script)} OK\n")

        cnn_dir.mkdir(exist_ok=True, parents=True)
        with cnn_model.open("w") as f:
            f.write("read from init model: \n")
        with model_ckpt_meta_file.open("w") as f:
            f.write("read from init model ckpt: \n")
        with model_ckpt_data_file.open("w") as f:
            f.write("read from init model ckpt: \n")
        with model_ckpt_index_file.open("w") as f:
            f.write("read from init model ckpt: \n")

        qnn_dir.mkdir(exist_ok=True, parents=True)
        with qnn_model.open("w") as f:
            f.write("read from init model: \n")
        with lcurve.open("w") as f:
            f.write("read from train_script: \n")
            f.write(script.read_text() + "\n")

        model_files = "nvnmd_models"
        os.makedirs(model_files, exist_ok=True)
        shutil.copy(cnn_model, "nvnmd_models")
        shutil.copy(qnn_model, "nvnmd_models")
        shutil.copy(model_ckpt_meta_file, "nvnmd_models")
        shutil.copy(model_ckpt_data_file, "nvnmd_models")
        shutil.copy(model_ckpt_index_file, "nvnmd_models")

        os.chdir(cwd)

        return OPIO(
            {
                "script": work_dir / oscript,
                "model": work_dir / model_files,
                "lcurve": work_dir / lcurve,
                "log": work_dir / log,
            }
        )


class MockedRunLmp(RunLmp):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        models = ip["models"]

        assert ip["task_path"].is_dir()
        assert re.match("task.[0-9][0-9][0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert task_path.is_dir()
        assert ip["task_name"] in str(ip["task_path"])
        assert (
            len(models) == mocked_numb_models
        ), f"{len(models)} == {mocked_numb_models}"
        for ii in range(mocked_numb_models):
            assert ip["models"][ii].is_file()
            assert "model" in str(ip["models"][ii])
            assert ".pb" in str(ip["models"][ii])
        assert (task_path / lmp_conf_name).is_file()
        assert (task_path / lmp_input_name).is_file()

        task_path = task_path.resolve()
        models = [ii.resolve() for ii in models]
        models_str = [str(ii) for ii in models]

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob

        ifiles = glob.glob(str(task_path / "*"))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        for ii in models:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)

        log = Path(lmp_log_name)
        traj = Path(lmp_traj_name)
        model_devi = Path(lmp_model_devi_name)

        # fc = ['log of {task_name}']
        # for ii in ['conf.lmp', 'in.lammps'] + models_str:
        #     if Path(ii).exists():
        #         fc.append(f'{ii} OK')
        # log.write_text('\n'.join(fc))
        # log.write_text('log of {task_name}')
        fc = []
        for ii in [lmp_conf_name, lmp_input_name] + [ii.name for ii in models]:
            fc.append(Path(ii).read_text())
        log.write_text("\n".join(fc))
        model_devi.write_text(f"model_devi of {task_name}")
        traj_out = []
        traj_out.append(f"traj of {task_name}")
        traj_out.append(Path(lmp_conf_name).read_text())
        traj_out.append(Path(lmp_input_name).read_text())
        traj.write_text("\n".join(traj_out))

        os.chdir(cwd)

        return OPIO(
            {
                "log": work_dir / log,
                "traj": work_dir / traj,
                "model_devi": work_dir / model_devi,
            }
        )


class MockedRunNvNMD(RunNvNMD):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        models = ip["models"]

        assert ip["task_path"].is_dir()
        assert re.match("task.[0-9][0-9][0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert task_path.is_dir()
        assert ip["task_name"] in str(ip["task_path"])
        assert (
            len(models) == mocked_numb_models
        ), f"{len(models)} == {mocked_numb_models}"
        for ii in range(mocked_numb_models):
            assert ip["models"][ii].is_file()
            assert "model" in str(ip["models"][ii])
            assert ".pb" in str(ip["models"][ii])
        assert (task_path / lmp_conf_name).is_file()
        assert (task_path / lmp_input_name).is_file()

        task_path = task_path.resolve()
        models = [ii.resolve() for ii in models]
        models_str = [str(ii) for ii in models]

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob

        ifiles = glob.glob(str(task_path / "*"))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)
        for ii in models:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)

        log = Path(lmp_log_name)
        traj = Path(lmp_traj_name)
        model_devi = Path(lmp_model_devi_name)

        # fc = ['log of {task_name}']
        # for ii in ['conf.lmp', 'in.lammps'] + models_str:
        #     if Path(ii).exists():
        #         fc.append(f'{ii} OK')
        # log.write_text('\n'.join(fc))
        # log.write_text('log of {task_name}')
        fc = []
        for ii in [lmp_conf_name, lmp_input_name] + [ii.name for ii in models]:
            fc.append(Path(ii).read_text())
        log.write_text("\n".join(fc))
        model_devi.write_text(f"model_devi of {task_name}")
        traj_out = []
        traj_out.append(f"traj of {task_name}")
        traj_out.append(Path(lmp_conf_name).read_text())
        traj_out.append(Path(lmp_input_name).read_text())
        traj.write_text("\n".join(traj_out))

        os.chdir(cwd)

        return OPIO(
            {
                "log": work_dir / log,
                "traj": work_dir / traj,
                "model_devi": work_dir / model_devi,
            }
        )


class MockedPrepVasp(PrepVasp):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        confs = ip["confs"]
        # incar_temp = ip['incar_temp']
        # potcars = ip['potcars']
        vasp_input = ip["config"]["inputs"]
        type_map = ip["type_map"]
        if not (type_map == ["H", "O"]):
            raise FatalError

        incar_temp = vasp_input.incar_template
        potcars = vasp_input.potcars

        for ii in confs:
            assert ii.is_file()
        assert vasp_input.incar_template == mocked_incar_template

        nconfs = len(confs)
        task_paths = []

        for ii in range(nconfs):
            task_path = Path(fp_task_pattern % ii)
            task_path.mkdir(exist_ok=True, parents=True)
            from shutil import (
                copyfile,
            )

            copyfile(confs[ii], task_path / vasp_conf_name)
            (task_path / vasp_input_name).write_text(incar_temp)
            task_paths.append(task_path)

        task_names = [str(ii) for ii in task_paths]
        return OPIO(
            {
                "task_names": task_names,
                "task_paths": task_paths,
            }
        )


class MockedRunVasp(RunVasp):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_name = ip["task_name"]
        task_path = ip["task_path"]

        assert ip["task_path"].is_dir()
        assert re.match("task.[0-9][0-9][0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert ip["task_name"] in str(ip["task_path"])
        assert (ip["task_path"] / vasp_conf_name).is_file()
        assert (ip["task_path"] / vasp_input_name).is_file()

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob

        ifiles = glob.glob(str(task_path / "*"))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)

        log = Path("log")
        # labeled_data = Path('labeled_data')
        labeled_data = Path("data_" + task_name)

        fc = []
        for ii in [vasp_conf_name, vasp_input_name]:
            fc.append(Path(ii).read_text())
        log.write_text("\n".join(fc))
        labeled_data.mkdir(exist_ok=True, parents=True)

        fc = []
        fc.append(f"labeled_data of {task_name}")
        fc.append(Path(vasp_conf_name).read_text())
        (labeled_data / "data").write_text("\n".join(fc))

        os.chdir(cwd)

        return OPIO(
            {
                "log": work_dir / log,
                "labeled_data": work_dir / labeled_data,
                "extra_outputs": [],
            }
        )


class MockedRunVaspFail1(RunVasp):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_name = ip["task_name"]
        task_path = ip["task_path"]

        assert ip["task_path"].is_dir()
        assert re.match("task.[0-9][0-9][0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert ip["task_name"] in str(ip["task_path"])
        assert (ip["task_path"] / vasp_conf_name).is_file()
        assert (ip["task_path"] / vasp_input_name).is_file()

        if task_id == 1:
            raise FatalError

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob

        ifiles = glob.glob(str(task_path / "*"))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)

        log = Path("log")
        # labeled_data = Path('labeled_data')
        labeled_data = Path("data_" + task_name)

        fc = []
        for ii in [vasp_conf_name, vasp_input_name]:
            fc.append(Path(ii).read_text())
        log.write_text("\n".join(fc))
        labeled_data.mkdir(exist_ok=True, parents=True)

        fc = []
        fc.append(f"labeled_data of {task_name}")
        fc.append(Path(vasp_conf_name).read_text())
        (labeled_data / "data").write_text("\n".join(fc))

        os.chdir(cwd)

        return OPIO(
            {
                "log": work_dir / log,
                "labeled_data": work_dir / labeled_data,
                "extra_outputs": [],
            }
        )


class MockedRunVaspRestart(RunVasp):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_name = ip["task_name"]
        task_path = ip["task_path"]

        assert ip["task_path"].is_dir()
        assert re.match("task.[0-9][0-9][0-9][0-9][0-9][0-9]", ip["task_name"])
        task_id = int(ip["task_name"].split(".")[1])
        assert ip["task_name"] in str(ip["task_path"])
        assert (ip["task_path"] / vasp_conf_name).is_file()
        assert (ip["task_path"] / vasp_input_name).is_file()

        work_dir = Path(task_name)

        cwd = os.getcwd()
        work_dir.mkdir(exist_ok=True, parents=True)
        os.chdir(work_dir)

        import glob

        ifiles = glob.glob(str(task_path / "*"))
        for ii in ifiles:
            if not Path(Path(ii).name).exists():
                Path(Path(ii).name).symlink_to(ii)

        log = Path("log")
        # labeled_data = Path('labeled_data')
        labeled_data = Path("data_" + task_name)

        fc = []
        for ii in [vasp_conf_name, vasp_input_name]:
            fc.append(Path(ii).read_text())
        log.write_text("\n".join(fc))
        labeled_data.mkdir(exist_ok=True, parents=True)

        fc = []
        fc.append("restarted")
        fc.append(f"labeled_data of {task_name}")
        fc.append(Path(vasp_conf_name).read_text())
        (labeled_data / "data").write_text("\n".join(fc))

        os.chdir(cwd)

        return OPIO(
            {
                "log": work_dir / log,
                "labeled_data": work_dir / labeled_data,
                "extra_outputs": [],
            }
        )


class MockedCollectData(CollectData):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        name = ip["name"]
        labeled_data = ip["labeled_data"]
        iter_data = ip["iter_data"]

        new_iter_data = []
        # copy iter_data
        for ii in iter_data:
            iiname = ii.name
            shutil.copytree(ii, iiname)
            new_iter_data.append(Path(iiname))

        # collect labled data
        name = Path(name)
        name.mkdir(exist_ok=True, parents=True)

        for ii in labeled_data:
            iiname = ii.name
            shutil.copytree(ii, name / iiname)
        new_iter_data.append(name)

        return OPIO(
            {
                "iter_data": new_iter_data,
            }
        )


class MockedCollectDataCheckOptParam(CollectData):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        if not ip["optional_parameter"]["mixed_type"]:
            raise FatalError(
                f"the value of mixed_type is {ip['optional_parameter']['mixed_type']} "
            )
        return MockedCollectData.execute(self, ip)


class MockedCollectDataFailed(CollectData):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        name = ip["name"]
        labeled_data = ip["labeled_data"]
        name = Path(name)
        for ii in labeled_data:
            iiname = ii.name

        raise FatalError


class MockedCollectDataRestart(CollectData):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        name = ip["name"]
        labeled_data = ip["labeled_data"]
        iter_data = ip["iter_data"]

        new_iter_data = []
        # copy iter_data
        for ii in iter_data:
            iiname = ii.name
            shutil.copytree(ii, iiname)
            new_iter_data.append(Path(iiname))

        # collect labled data
        name = Path(name)
        name.mkdir(exist_ok=True, parents=True)

        for ii in labeled_data:
            iiname = ii.name
            shutil.copytree(ii, name / iiname)
            fc = (name / iiname / "data").read_text()
            fc = "restart\n" + fc
            (name / iiname / "data").write_text(fc)

        new_iter_data.append(name)

        return OPIO(
            {
                "iter_data": new_iter_data,
            }
        )


class MockedExplorationReport(ExplorationReport):
    def __init__(self, conv_accuracy=0.9):
        self.conv_accuracy = conv_accuracy
        self.failed = 0.1
        self.candidate = 0.1
        self.accurate = 0.8

    def clear(self):
        raise NotImplementedError

    def record(self, mdf, mdv):
        raise NotImplementedError

    def print(
        self,
        stage_idx: int,
        idx_in_stage: int,
        iter_idx: int,
    ):
        # raise NotImplementedError
        return f"{self.failed} {self.candidate} {self.accurate} {self.conv_accuracy}"

    def print_header(self):
        # raise NotImplementedError
        return "header"

    def converged(self, reports):
        return self.accurate >= self.conv_accuracy

    def no_candidate(self):
        return self.candidate_ratio() == 0.0

    def get_candidate_ids(self, max_nframes):
        raise NotImplementedError

    def failed_ratio(
        self,
        tag=None,
    ) -> float:
        return self.failed

    def accurate_ratio(
        self,
        tag=None,
    ) -> float:
        return self.accurate

    def candidate_ratio(
        self,
        tag=None,
    ) -> float:
        return self.candidate


class MockedExplorationTaskGroup(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()
        ntask = mocked_numb_lmp_tasks
        for jj in range(ntask):
            tt = ExplorationTask()
            tt.add_file(lmp_conf_name, f"mocked conf {jj}").add_file(
                lmp_input_name, f"mocked input {jj}"
            )
            self.add_task(tt)

    def make_task(self):
        raise NotImplementedError


class MockedExplorationTaskGroup1(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()
        ntask = mocked_numb_lmp_tasks
        for jj in range(ntask):
            tt = ExplorationTask()
            tt.add_file(lmp_conf_name, f"mocked 1 conf {jj}").add_file(
                lmp_input_name, f"mocked 1 input {jj}"
            )
            self.add_task(tt)

    def make_task(self):
        raise NotImplementedError


class MockedExplorationTaskGroup2(ExplorationTaskGroup):
    def __init__(self):
        super().__init__()
        ntask = mocked_numb_lmp_tasks
        for jj in range(ntask):
            tt = ExplorationTask()
            tt.add_file(lmp_conf_name, f"mocked 2 conf {jj}").add_file(
                lmp_input_name, f"mocked 2 input {jj}"
            )
            self.add_task(tt)

    def make_task(self):
        raise NotImplementedError


class MockedStage(ExplorationStage):
    def make_task(self):
        return MockedExplorationTaskGroup()


class MockedStage1(ExplorationStage):
    def make_task(self):
        return MockedExplorationTaskGroup1()


class MockedStage2(ExplorationStage):
    def make_task(self):
        return MockedExplorationTaskGroup2()


class MockedConfSelector(ConfSelector):
    def __init__(
        self,
        conv_accuracy: float = 0.9,
    ):
        self.conv_accuracy = conv_accuracy

    def select(
        self,
        trajs: List[Path],
        model_devis: List[Path],
        type_map: List[str] = None,
        optional_outputs: Optional[List[Path]] = None,
    ) -> Tuple[List[Path], ExplorationReport]:
        confs = []
        if len(trajs) == mocked_numb_lmp_tasks:
            # get output from prep_run_lmp
            for ii in range(mocked_numb_select):
                ftraj = trajs[ii].read_text().strip().split("\n")
                fcont = []
                fcont.append(f"select conf.{ii}")
                fcont.append(ftraj[1])
                fcont.append(ftraj[2])
                fname = Path(f"conf.{ii}")
                fname.write_text("\n".join(fcont))
                confs.append(fname)
        else:
            # for the case of artificial input. trajs of length 2
            fname = Path("conf.0")
            fname.write_text("conf of conf.0")
            confs.append(fname)
            fname = Path("conf.1")
            fname.write_text("conf of conf.1")
            confs.append(fname)
        report = MockedExplorationReport(conv_accuracy=self.conv_accuracy)
        return confs, report


class MockedSelectConfs(SelectConfs):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        conf_selector = ip["conf_selector"]
        trajs = ip["trajs"]
        model_devis = ip["model_devis"]
        confs, report = conf_selector.select(trajs, model_devis)

        # get lmp output. check if all trajs and model devis are files
        if len(trajs) == mocked_numb_lmp_tasks:
            for ii in range(mocked_numb_lmp_tasks):
                assert trajs[ii].is_file()
                assert model_devis[ii].is_file()

        return OPIO(
            {
                "report": report,
                "confs": confs,
            }
        )


class MockedConstTrustLevelStageScheduler(ConvergenceCheckStageScheduler):
    def __init__(
        self,
        stage: ExplorationStage,
        conv_accuracy: float = 0.9,
        max_numb_iter: int = None,
    ):
        self.selector = MockedConfSelector(
            conv_accuracy=conv_accuracy,
        )
        super().__init__(stage, self.selector, max_numb_iter=max_numb_iter)


class MockedCollRunCaly(CollRunCaly):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        cnt_num = ip["cnt_num"]
        config = ip["config"] if ip["config"] is not None else {}
        # config = CollRunCaly.normalize_config(config)
        command = config.get("run_calypso_command", "calypso.x")
        # input.dat
        input_file = ip["input_file"].resolve()
        max_step = Path(input_file).read_text().strip()

        # work_dir name: calypso_task.idx
        work_dir = Path(ip["task_name"])
        work_dir.mkdir(exist_ok=True, parents=True)

        qhull_input = (
            ip["qhull_input"].resolve()
            if ip["qhull_input"] is not None
            else ip["qhull_input"]
        )
        step = ip["step"].resolve() if ip["step"] is not None else ip["step"]
        results = (
            ip["results"].resolve() if ip["results"] is not None else ip["results"]
        )

        opt_results_dir = []
        if ip["opt_results_dir"] is not None:
            for temp in ip["opt_results_dir"]:
                opt_results_dir.append(Path(temp).resolve())

        os.chdir(work_dir)
        Path(input_file.name).symlink_to(input_file)
        if step is not None and results is not None and opt_results_dir is not None:
            step = ip["step"].resolve()
            results = ip["results"].resolve()

            for opt_results_name in opt_results_dir:
                for file_name in opt_results_name.iterdir():
                    Path(file_name.name).symlink_to(file_name)
            # opt_results_dir = ip["opt_results_dir"].resolve()

            Path(step.name).symlink_to(step)
            Path(results.name).symlink_to(results)
            # Path(opt_results_dir.name).symlink_to(opt_results_dir)

        finished = "true" if int(cnt_num) == int(max_step) else "false"
        if finished == "false":
            for i in range(5):
                Path(f"POSCAR_{str(i)}").write_text(f"POSCAR_{str(i)}")

            if step is None:
                Path("step").write_text("2")
            else:
                step_num = Path("step").read_text().strip()
                Path("step").write_text(f"{int(step_num)+1}")

            if qhull_input is None:
                Path("test_qconvex.in").write_text("")

            step_num = int(Path("step").read_text().strip())

            if results is None:
                Path("results").mkdir(parents=True, exist_ok=True)
                for i in range(1, step_num):
                    Path(f"results/pso_ini_{i}").write_text(f"pso_ini_{i}")
                    Path(f"results/pso_opt_{i}").write_text(f"pso_opt_{i}")
                    Path(f"results/pso_sor_{i}").write_text(f"pso_sor_{i}")
            else:
                i = step_num
                Path(f"results/pso_ini_{i}").write_text(f"pso_ini_{i}")
                Path(f"results/pso_opt_{i}").write_text(f"pso_opt_{i}")
                Path(f"results/pso_sor_{i}").write_text(f"pso_sor_{i}")

        poscar_dir = Path("poscar_dir")
        poscar_dir.mkdir(parents=True, exist_ok=True)
        for poscar in Path().glob("POSCAR_*"):
            target = poscar_dir.joinpath(poscar.name)
            shutil.copyfile(poscar, target)
        # finished = "true" if int(cnt_num) == int(max_step) else "false"

        os.chdir(cwd)
        ret_dict = {
            "task_name": work_dir.name,
            "finished": finished,
            "poscar_dir": work_dir.joinpath(poscar_dir),
            "input_file": work_dir.joinpath(input_file.name),
            "results": work_dir.joinpath("results"),
            "step": work_dir.joinpath("step"),
            "qhull_input": work_dir.joinpath("test_qconvex.in"),
        }
        return OPIO(ret_dict)


class MockedPrepCalyDPOptim(PrepCalyDPOptim):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()

        finished = ip["finished"]
        work_dir = Path(ip["task_name"])
        work_dir.mkdir(parents=True, exist_ok=True)

        poscar_dir = ip["poscar_dir"]
        models_dir = ip["models_dir"]
        caly_run_opt_file = ip["caly_run_opt_file"].resolve()
        caly_check_opt_file = ip["caly_check_opt_file"].resolve()
        poscar_list = [poscar.resolve() for poscar in poscar_dir.rglob("POSCAR_*")]
        model_list = [model.resolve() for model in models_dir.rglob("*model*pb")]
        model_list = sorted(model_list, key=lambda x: str(x).split(".")[1])
        model_file = model_list[0]

        os.chdir(work_dir)

        group_size = 2

        if finished == "false":
            grouped_poscar_list = [
                poscar_list[i : i + group_size]
                for i in range(0, len(poscar_list), group_size)
            ]
            task_dirs = []
            for idx, poscar_list in enumerate(grouped_poscar_list):
                opt_path = Path(f"opt_path_{idx}")
                task_dirs.append(work_dir / opt_path)

                cwd = os.getcwd()
                os.chdir(opt_path)
                for poscar in poscar_list:
                    Path(poscar.name).symlink_to(poscar)
                Path(model_file.name).symlink_to(model_file)
                Path(caly_run_opt_file.name).symlink_to(caly_run_opt_file)
                Path(caly_check_opt_file.name).symlink_to(caly_check_opt_file)
                os.chdir(cwd)

            task_names = [str(task_dir) for task_dir in task_dirs]
        else:
            temp_dir = work_dir / "opt_path"
            temp_dir.mkdir(parents=True, exist_ok=True)
            task_dirs = [temp_dir]
            task_names = [str(task_dir) for task_dir in task_dirs]

        os.chdir(cwd)
        return OPIO(
            {
                "task_name": task_names,
                "task_dirs": task_dirs,
            }
        )


class MockedRunCalyDPOptim(RunCalyDPOptim):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()

        finished = ip["finished"]
        work_dir = Path(ip["task_name"])
        cnt_num = ip["cnt_num"]
        work_dir.mkdir(parents=True, exist_ok=True)

        config = ip["config"] if ip["config"] is not None else {}
        command = config.get("run_opt_command", "python -u calypso_run_opt.py")

        task_path = ip["task_dir"]
        if task_path is not None:
            input_files = [ii.resolve() for ii in Path(task_path).iterdir()]
        else:
            pass

        os.chdir(work_dir)

        if finished == "false":
            for ii in input_files:
                iname = ii.name
                Path(iname).symlink_to(ii)

            poscar_list = sorted(Path().rglob("POSCAR_*"))
            cnt = 0
            for poscar in poscar_list:
                cnt += 1
                poscar_name = poscar.name
                num = poscar_name.strip("POSCAR_")
                Path(poscar_name.replace("POSCAR", "CONTCAR")).write_text("")
                Path(poscar_name.replace("POSCAR", "OUTCAR")).write_text("")
                Path(f"{num}.traj").write_text("")

            optim_results_dir = Path("optim_results_dir")
            optim_results_dir.mkdir(parents=True, exist_ok=True)
            for poscar in Path().glob("POSCAR_*"):
                target = optim_results_dir.joinpath(poscar.name)
                shutil.copyfile(poscar, target)
            for contcar in Path().glob("CONTCAR_*"):
                target = optim_results_dir.joinpath(contcar.name)
                shutil.copyfile(contcar, target)
            for outcar in Path().glob("OUTCAR_*"):
                target = optim_results_dir.joinpath(outcar.name)
                shutil.copyfile(outcar, target)

            traj_results_dir = Path("traj_results")
            traj_results_dir.mkdir(parents=True, exist_ok=True)
            for traj in Path().glob("*.traj"):
                target = traj_results_dir.joinpath(str(cnt_num) + "." + traj.name)
                shutil.copyfile(traj, target)
        else:
            optim_results_dir = Path("optim_results_dir")
            optim_results_dir.mkdir(parents=True, exist_ok=True)

            traj_results_dir = Path("traj_results")
            traj_results_dir.mkdir(parents=True, exist_ok=True)

        os.chdir(cwd)
        return OPIO(
            {
                "task_name": str(work_dir),
                "optim_results_dir": work_dir / optim_results_dir,
                "traj_results": work_dir / traj_results_dir,
            }
        )


class MockedRunCalyModelDevi(RunCalyModelDevi):
    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()

        work_dir = Path(ip["task_name"])
        work_dir.mkdir(parents=True, exist_ok=True)

        type_map = ip["type_map"]

        traj_dirs = ip["traj_dirs"]
        traj_dirs = [traj_dir.resolve() for traj_dir in traj_dirs]

        models_dir = ip["models"]
        models_dir = [model.resolve() for model in models_dir]

        dump_file_name = "traj.dump"
        model_devi_file_name = "model_devi.out"

        ref_dump_str = """ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
        0.0000000000        10.0000000000         0.0000000000
        0.0000000000        10.0000000000         0.0000000000
        0.0000000000        10.0000000000         0.0000000000
ITEM: ATOMS id type x y z fx fy fz
    1     1        0.0000000000         0.0000000000         0.0000000000        0.0000000000         0.0000000000         0.0000000000
    2     2        5.0000000000         5.0000000000         5.0000000000        0.0000000000         0.0000000000         0.0000000000"""
        os.chdir(work_dir)
        f = open(dump_file_name, "a")
        f.write(ref_dump_str)
        f.close()

        f = open(model_devi_file_name, "a")
        f.write("# \n0 1 1 1 1 1 1")
        f.close()

        os.chdir(cwd)
        return OPIO(
            {
                "task_name": str(work_dir),
                "traj": [work_dir / dump_file_name],
                "model_devi": [work_dir / model_devi_file_name],
            }
        )
