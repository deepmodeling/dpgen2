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
                "init_model_ckpt_meta": Artifact(Path, optional=True),
                "init_model_ckpt_data": Artifact(Path, optional=True),
                "init_model_ckpt_index": Artifact(Path, optional=True),
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
                "cnn_model": Artifact(Path),
                "qnn_model": Artifact(Path),
                "model_ckpt_data": Artifact(Path),
                "model_ckpt_meta": Artifact(Path),
                "model_ckpt_index": Artifact(Path),
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

            - `config`: (`dict`) The config of training task. Check `RunNvNMDTrain.training_args` for definitions.
            - `task_name`: (`str`) The name of training task.
            - `task_path`: (`Artifact(Path)`) The path that contains all input files prepareed by `PrepDPTrain`.
            - `init_model`: (`Artifact(Path)`) A frozen model to initialize the training.
            - `init_model_ckpt_meta`: (`Artifact(Path)`, optional) The meta file of the frozen model.
            - `init_model_ckpt_data`: (`Artifact(Path)`, optional) The data file of the frozen model.
            - `init_model_ckpt_index`: (`Artifact(Path)`, optional) The index file of the frozen model.
            - `init_data`: (`Artifact(NestedDict[Path])`) Initial training data.
            - `iter_data`: (`Artifact(List[Path])`) Training data generated in the DPGEN iterations.
            - `valid_data`: (`Artifact(NestedDict[Path])`, optional) Validation data.
            - `optional_files`: (`Artifact(List[Path])`, optional) Optional files to be copied to the working directory.

        Returns
        -------
        Any
            Output dict with components:
            - `script`: (`Artifact(Path)`) The training script.
            - `cnn_model`: (`Artifact(Path)`) The trained continuous frozen model.
            - `qnn_model`: (`Artifact(Path)`) The trained quantized  frozen model.
            - `model_ckpt_data`: (`Artifact(Path)`) The data file of the trained model.
            - `model_ckpt_meta`: (`Artifact(Path)`) The meta file of the trained model.
            - `model_ckpt_index`: (`Artifact(Path)`) The index file of the trained model.
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
        config = RunNvNMDTrain.normalize_config(config)
        task_name = ip["task_name"]
        task_path = ip["task_path"]
        init_model = ip["init_model"]
        init_model_ckpt_data = ip["init_model_ckpt_data"]
        init_model_ckpt_meta = ip["init_model_ckpt_meta"]
        init_model_ckpt_index = ip["init_model_ckpt_index"]
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
        do_init_model = RunNvNMDTrain.decide_init_model(
            config,
            init_model_ckpt if init_model_ckpt_data is not None else init_model,
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
        train_dict = RunNvNMDTrain.write_data_to_input_script(
            train_dict,
            config,
            init_data,
            iter_data_exp,
            auto_prob_str,
            major_version,
            valid_data,
        )
        train_cnn_dict = RunNvNMDTrain.write_other_to_input_script(
            train_dict,
            config,
            do_init_model,
            False,
            major_version,
        )
        train_qnn_dict = RunNvNMDTrain.write_other_to_input_script(
            train_dict,
            config,
            do_init_model,
            True,
            major_version,
        )

        with set_directory(work_dir):
            # open log
            fplog = open("train.log", "w")

            def clean_before_quit():
                fplog.close()

            # dump train script
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

            if not RunNvNMDTrain.skip_training(
                work_dir, train_dict, init_model, iter_data
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

            if os.path.exists("input_v2_compat.json"):
                shutil.copy2("input_v2_compat.json", train_script_name)

            clean_before_quit()

        return OPIO(
            {
                "script": work_dir / train_cnn_script_name,
                "cnn_model": work_dir / cnn_model_file,
                "qnn_model": work_dir / qnn_model_file,
                "model_ckpt_data": work_dir / model_ckpt_data_file,
                "model_ckpt_meta": work_dir / model_ckpt_meta_file,
                "model_ckpt_index": work_dir / model_ckpt_index_file,
                "lcurve": work_dir / lcurve_file,
                "log": work_dir / "train.log",
            }
        )

    @staticmethod
    def write_data_to_input_script(
        idict: dict,
        config,
        init_data: Union[List[Path], Dict[str, List[Path]]],
        iter_data: List[Path],
        auto_prob_str: str = "prob_sys_size",
        major_version: str = "2",
        valid_data: Optional[Union[List[Path], Dict[str, List[Path]]]] = None,
    ):
        odict = idict.copy()

        data_list = [str(ii) for ii in init_data] + [str(ii) for ii in iter_data]
        if major_version == "1":
            # v1 behavior
            odict["training"]["systems"] = data_list
            odict["training"].setdefault("batch_size", "auto")
            odict["training"]["auto_prob_style"] = auto_prob_str
            if valid_data is not None:
                odict["training"]["validation_data"] = {
                    "systems": [str(ii) for ii in valid_data],
                    "batch_size": 1,
                }
        elif major_version == "2":
            # v2 behavior
            odict["training"]["training_data"]["systems"] = data_list
            odict["training"]["training_data"].setdefault("batch_size", "auto")
            odict["training"]["training_data"]["auto_prob"] = auto_prob_str
            if valid_data is None:
                odict["training"].pop("validation_data", None)
            else:
                odict["training"]["validation_data"] = {
                    "systems": [str(ii) for ii in valid_data],
                    "batch_size": 1,
                }
        else:
            raise RuntimeError("unsupported DeePMD-kit major version", major_version)
        return odict

    @staticmethod
    def write_other_to_input_script(
        idict,
        config,
        do_init_model,
        train_qnn_model: bool = False,
        major_version: str = "1",
    ):
        odict = copy.deepcopy(idict)
        odict["training"]["disp_file"] = "lcurve.out"
        odict["training"]["save_ckpt"] = "model.ckpt"
        if do_init_model:
            odict["learning_rate"]["start_lr"] = config["init_model_start_lr"]
            if "loss_dict" in odict:
                for v in odict["loss_dict"].values():
                    if isinstance(v, dict):
                        v["start_pref_e"] = config["init_model_start_pref_e"]
                        v["start_pref_f"] = config["init_model_start_pref_f"]
                        v["start_pref_v"] = config["init_model_start_pref_v"]
            else:
                odict["loss"]["start_pref_e"] = config["init_model_start_pref_e"]
                odict["loss"]["start_pref_f"] = config["init_model_start_pref_f"]
                odict["loss"]["start_pref_v"] = config["init_model_start_pref_v"]
            if major_version == "1":
                odict["training"]["stop_batch"] = config["init_model_numb_steps"]
            elif major_version == "2":
                odict["training"]["numb_steps"] = config["init_model_numb_steps"]
            else:
                raise RuntimeError(
                    "unsupported DeePMD-kit major version", major_version
                )
        if train_qnn_model:
            odict["learning_rate"]["start_lr"] = config["init_model_start_lr"]
            if "loss_dict" in odict:
                for v in odict["loss_dict"].values():
                    if isinstance(v, dict):
                        v["start_pref_e"] = 1
                        v["start_pref_f"] = 1
                        v["start_pref_v"] = 1
            if major_version == "1":
                odict["training"]["stop_batch"] = 0
            elif major_version == "2":
                odict["training"]["numb_steps"] = 0
        return odict

    @staticmethod
    def skip_training(
        work_dir,
        train_dict,
        init_model,
        iter_data,
    ):
        # we have init model and no iter data, skip training
        if (init_model is not None) and (iter_data is None or len(iter_data) == 0):
            with set_directory(work_dir):
                with open(train_script_name, "w") as fp:
                    json.dump(train_dict, fp, indent=4)
                Path("train.log").write_text(
                    f"We have init model {init_model} and "
                    f"no iteration training data. "
                    f"The training is skipped.\n"
                )
                Path("lcurve.out").touch()
            return True
        else:
            return False

    @staticmethod
    def decide_init_model(
        config,
        init_model,
        init_data,
        iter_data,
        mixed_type=False,
    ):
        do_init_model = False
        # decide if we do init-model
        ## cases we do definitely not
        if init_model is None or iter_data is None or len(iter_data) == 0:
            do_init_model = False
        ## cases controlled by the policy
        else:
            if config["init_model_policy"] == "no":
                do_init_model = False
            elif config["init_model_policy"] == "yes":
                do_init_model = True
            elif "old_data_larger_than" in config["init_model_policy"]:
                old_data_size_level = int(config["init_model_policy"].split(":")[-1])
                if isinstance(init_data, dict):
                    init_data_size = _get_data_size_of_all_systems(
                        sum(init_data.values(), [])
                    )
                else:
                    init_data_size = _get_data_size_of_all_systems(init_data)
                iter_data_old_size = _get_data_size_of_all_mult_sys(
                    iter_data[:-1], mixed_type=mixed_type
                )
                old_data_size = init_data_size + iter_data_old_size
                if old_data_size > old_data_size_level:
                    do_init_model = True
        return do_init_model

    @staticmethod
    def training_args():
        doc_command = "The command for DP, 'dp' for default"
        doc_init_model_policy = "The policy of init-model training. It can be\n\n\
    - 'no': No init-model training. Traing from scratch.\n\n\
    - 'yes': Do init-model training.\n\n\
    - 'old_data_larger_than:XXX': Do init-model if the training data size of the previous model is larger than XXX. XXX is an int number."
        doc_init_model_old_ratio = "The frequency ratio of old data over new data"
        doc_init_model_numb_steps = "The number of training steps when init-model"
        doc_init_model_start_lr = "The start learning rate when init-model"
        doc_init_model_start_pref_e = (
            "The start energy prefactor in loss when init-model"
        )
        doc_init_model_start_pref_f = (
            "The start force prefactor in loss when init-model"
        )
        doc_init_model_start_pref_v = (
            "The start virial prefactor in loss when init-model"
        )
        doc_train_args = "Extra arguments for dp train"
        return [
            Argument(
                "command",
                str,
                optional=True,
                default="dp",
                doc=doc_command,
            ),
            Argument(
                "init_model_policy",
                str,
                optional=True,
                default="no",
                doc=doc_init_model_policy,
            ),
            Argument(
                "init_model_old_ratio",
                float,
                optional=True,
                default=0.9,
                doc=doc_init_model_old_ratio,
            ),
            Argument(
                "init_model_numb_steps",
                int,
                optional=True,
                default=400000,
                doc=doc_init_model_numb_steps,
                alias=["init_model_stop_batch"],
            ),
            Argument(
                "init_model_start_lr",
                float,
                optional=True,
                default=1e-4,
                doc=doc_init_model_start_lr,
            ),
            Argument(
                "init_model_start_pref_e",
                float,
                optional=True,
                default=0.1,
                doc=doc_init_model_start_pref_e,
            ),
            Argument(
                "init_model_start_pref_f",
                float,
                optional=True,
                default=100,
                doc=doc_init_model_start_pref_f,
            ),
            Argument(
                "init_model_start_pref_v",
                float,
                optional=True,
                default=0.0,
                doc=doc_init_model_start_pref_v,
            ),
            Argument(
                "train_args",
                str,
                optional=True,
                default="",
                doc=doc_train_args,
            ),
        ]

    @staticmethod
    def normalize_config(data={}):
        ta = RunNvNMDTrain.training_args()

        base = Argument("base", dict, ta)
        data = base.normalize_value(data, trim_pattern="_*")
        base.check_value(data, strict=True)

        return data


def _get_data_size_of_system(data_dir):
    ss = dpdata.System(data_dir, fmt="deepmd/npy")
    return ss.get_nframes()


def _get_data_size_of_all_systems(data_dirs):
    count = 0
    for ii in data_dirs:
        count += _get_data_size_of_system(ii)
    return count


def _get_data_size_of_mult_sys(data_dir, mixed_type=False):
    ms = dpdata.MultiSystems()
    if mixed_type:
        ms.from_deepmd_npy_mixed(data_dir)  # type: ignore
    else:
        ms.from_deepmd_npy(data_dir)  # type: ignore
    return ms.get_nframes()


def _get_data_size_of_all_mult_sys(data_dirs, mixed_type=False):
    count = 0
    for ii in data_dirs:
        count += _get_data_size_of_mult_sys(ii, mixed_type)
    return count


def _expand_multi_sys_to_sys(multi_sys_dir):
    all_type_raws = sorted(glob.glob(os.path.join(multi_sys_dir, "*", "type.raw")))
    all_sys_dirs = [str(Path(ii).parent) for ii in all_type_raws]
    return all_sys_dirs


def _expand_all_multi_sys_to_sys(list_multi_sys):
    all_sys_dirs = []
    for ii in list_multi_sys:
        all_sys_dirs = all_sys_dirs + _expand_multi_sys_to_sys(ii)
    return all_sys_dirs


config_args = RunNvNMDTrain.training_args
