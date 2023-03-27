import json
import random
import sys
from pathlib import (
    Path,
)
from typing import (
    List,
    Tuple,
    Union,
)

from dflow import (
    InputArtifact,
    InputParameter,
    OutputParameter,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
)

from dpgen2.constants import (
    train_script_name,
    train_task_pattern,
)


class ModifyTrainScript(OP):
    r"""Modify the training scripts to prepare them for training
    tasks in dpgen step.

    Read the training scripts modified by finetune, and replace
    the original template scripts to be compatible with pre-trained models.
    New templates are returned as `op["template_script"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "numb_models": int,
                "scripts": Artifact(Path),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "template_script": List[dict],
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

            - `scripts`: (`Artifact(Path)`) Training scripts from finetune.
            - `numb_models`: (`int`) Number of DP models to train.

        Returns
        -------
        op : dict
            Output dict with components:

            - `template_script`: (`List[dict]`) One template from one finetuning task. The length of the list should be the same as `numb_models`.

        """
        scripts = ip["scripts"]
        new_template_script = []
        numb_models = ip["numb_models"]

        for ii in range(numb_models):
            subdir = Path(train_task_pattern % ii)
            train_script = Path(scripts) / subdir / train_script_name
            with open(train_script, "r") as fp:
                train_dict = json.load(fp)

            if "systems" in train_dict["training"]:
                major_version = "1"
            else:
                major_version = "2"
            if major_version == "1":
                train_dict["training"]["systems"] = []
            elif major_version == "2":
                train_dict["training"]["training_data"]["systems"] = []

            new_template_script.append(train_dict)

        op = OPIO(
            {
                "template_script": new_template_script,
            }
        )
        return op
