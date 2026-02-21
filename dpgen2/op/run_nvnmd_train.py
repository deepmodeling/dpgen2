import copy
import glob
import json
import logging
import os
import shutil
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import dpdata
from dargs import (
    Argument,
    ArgumentEncoder,
    Variant,
    dargs,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    FatalError,
    NestedDict,
    OPIOSign,
    Parameter,
    TransientError,
)

from dpgen2.constants import (
    train_cnn_script_name,
    train_qnn_script_name,
    train_script_name,
    train_task_pattern,
)
from dpgen2.op.run_dp_train import (
    RunDPTrain,
    _expand_all_multi_sys_to_sys,
)
from dpgen2.utils.chdir import (
    set_directory,
)
from dpgen2.utils.run_command import (
    run_command,
)


def _make_train_command(
    dp_command,
    train_script_name,
    do_init_model,
    init_model,
    train_args="",
):
    # find checkpoint
    if os.path.isfile("nvnmd_cnn/checkpoint") and not os.path.isfile(
        "nvnmd_cnn/frozen_model.pb"
    ):
        checkpoint = "nvnmd_cnn/model.ckpt"
    else:
        checkpoint = None

    # case of restart
    if checkpoint is not None:
        command = dp_command + [
            "train-nvnmd",
            "--restart",
            checkpoint,
            train_script_name,
        ]
        return command

    # case of init model
    assert checkpoint is None
    case_init_model = do_init_model
    if case_init_model:
        if isinstance(init_model, list):  # initialize from model.ckpt
            # init_model = ".".join(str(init_model[0]).split('.')[:-1])
            for i in init_model:
                if os.path.exists(i):
                    shutil.copy(i, ".")
            init_model = "model.ckpt"
            init_flag = "--init-model"
        else:  # initialize from frozen model
            init_flag = "--init-frz-model"

        command = dp_command + [
            "train-nvnmd",
            init_flag,
            str(init_model),
            train_script_name,
        ]
    else:
        command = dp_command + ["train-nvnmd", train_script_name]

    command += train_args.split()
    return command


class RunNvNMDTrain(OP):
    r"""Execute a DP training task. Train and freeze a DP model.

    A working directory named `task_name` is created. All input files
    are copied or symbol linked to directory `task_name`. The
    DeePMD-kit training and freezing commands are exectuted from
    directory `task_name`.

    """

    default_optional_parameter = {
        "mixed_type": False,
    }

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "config": dict,
                "task_name": BigParameter(str),
                "optional_parameter": Parameter(
                    dict,
                    default=RunNvNMDTrain.default_optional_parameter,
                ),
                "task_path": Artifact(Path),
                "init_model": Artifact(Path, optional=True),
                "init_data": Artifact(NestedDict[Path]),
                "iter_data": Artifact(List[Path]),
                "valid_data": Artifact(NestedDict[Path], optional=True),
                "optional_files": Artifact(List[Path], optional=True),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "script": Artifact(Path),
                "model": Artifact(Path),
                "lcurve": Artifact(Path),
                "log": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        r"""Execute the OP.

        Parameters
        ----------
        ip : dict
            Input dict with components:

            - `config`: (`dict`) The config of training task. Check `RunDPTrain.training_args` for definitions.
            - `task_name`: (`str`) The name of training task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepDPTrain`.
            - `init_model`: (`Artifact(Path)`) The checkpoint and frozen model to initialize the training.
            - `init_data`: (`Artifact(NestedDict[Path])`) Initial training data.
            - `iter_data`: (`Artifact(List[Path])`) Training data generated in the DPGEN iterations.

        Returns
        -------
        Any
            Output dict with components:
            - `script`: (`Artifact(Path)`) The training script.
            - `model`: (`Artifact(Path)`) The trained continuous and quantized frozen model, the checkpoint model.
            - `lcurve`: (`Artifact(Path)`) The learning curve file.
            - `log`: (`Artifact(Path)`) The log file of training.

        Raises
        ------
        FatalError
            On the failure of training or freezing. Human intervention needed.
        """
        mixed_type = ip["optional_parameter"]["mixed_type"]
        config = ip["config"] if ip["config"] is not None else {}
        dp_command = ip["config"].get("command", "dp").split()
        train_args = config.get("train_args", "")
        config = RunDPTrain.normalize_config(config)
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        init_model = ip["init_model"]
        init_frz_model = ip["init_model"] / "frozen_model.pb" if init_model else None
        init_model_ckpt_data = (
            ip["init_model"] / "model.ckpt.data-00000-of-00001" if init_model else None
        )
        init_model_ckpt_meta = (
            ip["init_model"] / "model.ckpt.meta" if init_model else None
        )
        init_model_ckpt_index = (
            ip["init_model"] / "model.ckpt.index" if init_model else None
        )
        init_data = ip["init_data"]
        iter_data = ip["iter_data"]
        valid_data = ip["valid_data"]
        iter_data_old_exp = _expand_all_multi_sys_to_sys(iter_data[:-1])
        iter_data_new_exp = _expand_all_multi_sys_to_sys(iter_data[-1:])
        iter_data_exp = iter_data_old_exp + iter_data_new_exp
        work_dir = Path(task_name)

        # update the input script
        input_script = Path(task_path) / train_script_name
        with open(input_script) as fp:
            train_dict = json.load(fp)
        if "systems" in train_dict["training"]:
            major_version = "1"
        else:
            major_version = "2"

        # auto prob style
        init_model_ckpt = [
            init_model_ckpt_meta,
            init_model_ckpt_data,
            init_model_ckpt_index,
        ]
        do_init_model = RunDPTrain.decide_init_model(
            config,
            init_model_ckpt if init_model_ckpt_data is not None else init_frz_model,
            init_data,
            iter_data,
            mixed_type=mixed_type,
        )
        auto_prob_str = "prob_sys_size"
        if do_init_model:
            old_ratio = config["init_model_old_ratio"]
            len_init = len(init_data)
            numb_old = len_init + len(iter_data_old_exp)
            numb_new = numb_old + len(iter_data_new_exp)
            auto_prob_str = f"prob_sys_size; 0:{numb_old}:{old_ratio}; {numb_old}:{numb_new}:{1.-old_ratio:g}"

        # update the input dict
        train_dict = RunDPTrain.write_data_to_input_script(
            train_dict,
            config,
            init_data,
            iter_data_exp,
            auto_prob_str,
            major_version,
            valid_data,
        )
        train_cnn_dict = RunDPTrain.write_other_to_input_script(
            train_dict, config, do_init_model, major_version, False
        )
        train_qnn_dict = RunDPTrain.write_other_to_input_script(
            train_dict,
            config,
            do_init_model,
            major_version,
            True,
        )

        with set_directory(work_dir):
            # open log
            fplog = open("train.log", "w")

            def clean_before_quit():
                fplog.close()

            # dump train script

            with open(train_script_name, "w") as fp:
                json.dump(train_cnn_dict, fp, indent=4)

            with open(train_cnn_script_name, "w") as fp:
                json.dump(train_cnn_dict, fp, indent=4)

            with open(train_qnn_script_name, "w") as fp:
                json.dump(train_qnn_dict, fp, indent=4)

            if ip["optional_files"] is not None:
                for f in ip["optional_files"]:
                    Path(f.name).symlink_to(f)

            # train cnn model
            command = _make_train_command(
                dp_command,
                train_cnn_script_name,
                do_init_model,
                init_model_ckpt if init_model_ckpt_data is not None else init_model,
                train_args="-s s1",
            )

            if not RunDPTrain.skip_training(
                work_dir, train_dict, init_model, iter_data, None
            ):
                ret, out, err = run_command(command)
                if ret != 0:
                    clean_before_quit()
                    logging.error(
                        "".join(
                            (
                                "dp train-nvnmd -s s1 failed\n",
                                "out msg: ",
                                out,
                                "\n",
                                "err msg: ",
                                err,
                                "\n",
                            )
                        )
                    )
                    raise FatalError("dp train-nvnmd -s s1 failed")
                fplog.write(
                    "#=================== train_cnn std out ===================\n"
                )
                fplog.write(out)
                fplog.write(
                    "#=================== train_cnn std err ===================\n"
                )
                fplog.write(err)

                cnn_model_file = "nvnmd_cnn/frozen_model.pb"
                model_ckpt_data_file = "nvnmd_cnn/model.ckpt.data-00000-of-00001"
                model_ckpt_index_file = "nvnmd_cnn/model.ckpt.index"
                model_ckpt_meta_file = "nvnmd_cnn/model.ckpt.meta"
                lcurve_file = "nvnmd_cnn/lcurve.out"

                if os.path.exists("input_v2_compat.json"):
                    shutil.copy2("input_v2_compat.json", train_script_name)

            else:
                cnn_model_file = init_model
                model_ckpt_data_file = ""
                model_ckpt_index_file = ""
                model_ckpt_meta_file = ""
                lcurve_file = "nvnmd_qnn/lcurve.out"

            # train qnn model
            command = _make_train_command(
                dp_command,
                train_qnn_script_name,
                do_init_model,
                init_model_ckpt if init_model_ckpt_data is not None else init_model,
                train_args="-s s2",
            )

            ret, out, err = run_command(command)
            if ret != 0:
                clean_before_quit()
                logging.error(
                    "".join(
                        (
                            "dp train-nvnmd -s s2 failed\n",
                            "out msg: ",
                            out,
                            "\n",
                            "err msg: ",
                            err,
                            "\n",
                        )
                    )
                )
                raise FatalError("dp train-nvnmd -s s2 failed")
            fplog.write("#=================== train_qnn std out ===================\n")
            fplog.write(out)
            fplog.write("#=================== train_qnn std err ===================\n")
            fplog.write(err)

            qnn_model_file = "nvnmd_qnn/model.pb"

            clean_before_quit()

            # copy all models files to the output directory
            os.makedirs("nvnmd_models", exist_ok=True)
            if os.path.exists(cnn_model_file):
                shutil.copy(cnn_model_file, "nvnmd_models")
            if os.path.exists(qnn_model_file):
                shutil.copy(qnn_model_file, "nvnmd_models")
            if os.path.exists(model_ckpt_meta_file):
                shutil.copy(model_ckpt_meta_file, "nvnmd_models")
            if os.path.exists(model_ckpt_data_file):
                shutil.copy(model_ckpt_data_file, "nvnmd_models")
            if os.path.exists(model_ckpt_index_file):
                shutil.copy(model_ckpt_index_file, "nvnmd_models")

            model_files = "nvnmd_models"

        return OPIO(
            {
                "script": work_dir / train_script_name,
                "model": work_dir / model_files,
                "lcurve": work_dir / lcurve_file,
                "log": work_dir / "train.log",
            }
        )


config_args = RunDPTrain.training_args
