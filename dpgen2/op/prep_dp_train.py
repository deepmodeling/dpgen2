from dflow.python import OP, OPIO, OPIOSign, Artifact

import json, random, sys
from typing import Tuple, List, Union
from pathlib import Path
from dpgen2.constants import (
    train_task_pattern,
    train_script_name,
)


class PrepDPTrain(OP):
    r"""Prepares the working directories for DP training tasks.

    A list of (`numb_models`) working directories containing all files
    needed to start training tasks will be created. The paths of the
    directories will be returned as `op["task_paths"]`. The identities
    of the tasks are returned as `op["task_names"]`.

    """

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "template_script": Union[dict, List[dict]],
                "numb_models": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "task_names": List[str],
                "task_paths": Artifact(List[Path]),
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

            - `template_script`: (`str` or `List[str]`) A template of the training script. Can be a `str` or `List[str]`. In the case of `str`, all training tasks share the same training input template, the only difference is the random number used to initialize the network parameters. In the case of `List[str]`, one training task uses one template from the list. The random numbers used to initialize the network parameters are differnt. The length of the list should be the same as `numb_models`.
            - `numb_models`: (`int`) Number of DP models to train.

        Returns
        -------
        op : dict
            Output dict with components:

            - `task_names`: (`List[str]`) The name of tasks. Will be used as the identities of the tasks. The names of different tasks are different.
            - `task_paths`: (`Artifact(List[Path])`) The parepared working paths of the tasks. The order fo the Paths should be consistent with `op["task_names"]`

        """
        template = ip["template_script"]
        numb_models = ip["numb_models"]
        osubdirs = []
        if type(template) != list:
            template = [template for ii in range(numb_models)]
        else:
            if not (len(template) == numb_models):
                raise RuntimeError(
                    f"length of the template list should be equal to {numb_models}"
                )

        for ii in range(numb_models):
            # mkdir
            subdir = Path(train_task_pattern % ii)
            subdir.mkdir(exist_ok=True, parents=True)
            osubdirs.append(str(subdir))
            # change random seed in template
            idict = self._script_rand_seed(template[ii])
            # write input script
            fname = subdir / train_script_name
            with open(fname, "w") as fp:
                json.dump(idict, fp, indent=4)

        op = OPIO(
            {
                "task_names": osubdirs,
                "task_paths": [Path(ii) for ii in osubdirs],
            }
        )
        return op

    def _script_rand_seed(
        self,
        input_dict,
    ):
        jtmp = input_dict.copy()
        if jtmp["model"]["descriptor"]["type"] == "hybrid":
            for desc in jtmp["model"]["descriptor"]["list"]:
                desc["seed"] = random.randrange(sys.maxsize) % (2**32)
        else:
            jtmp["model"]["descriptor"]["seed"] = random.randrange(sys.maxsize) % (
                2**32
            )
        jtmp["model"]["fitting_net"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        jtmp["training"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        return jtmp
