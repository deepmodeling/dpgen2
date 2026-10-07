import copy
import json
import tempfile
import unittest
from pathlib import (
    Path,
)

from deepmd.utils.argcheck import normalize as normalize_training
from dflow.python import (
    OPIO,
)

from dpgen2.constants import (
    train_script_name,
)
from dpgen2.entrypoint.args import (
    normalize,
)
from dpgen2.op.prep_dp_train import (
    PrepDPTrain,
)
from dpgen2.op.run_dp_train import (
    RunDPTrain,
    _make_train_command,
)
from dpgen2.utils.chdir import (
    set_directory,
)

p_examples = Path(__file__).parent.parent / "examples"

input_files = (
    p_examples / "almg" / "input.json",
    # p_examples / "almg" / "input-v005.json",
    # p_examples / "almg" / "dp_template.json",
    p_examples / "calypso" / "input.test.json",
    p_examples / "water" / "input_distill.json",
    p_examples / "water" / "input_dpgen.json",
    p_examples / "water" / "input_multitask.json",
    p_examples / "ch4" / "input_dist.json",
    # p_examples / "chno" / "dpa_manyi.json",
    p_examples / "chno" / "input.json",
    p_examples / "water" / "input_dpgen_abacus.json",
    p_examples / "water" / "input_dpgen_cp2k.json",
    p_examples / "water" / "input_dpgen_slurm.json",
    p_examples / "diffcsp" / "dpgen.json",
)


class TestExamples(unittest.TestCase):
    def _load_templates(self, filename, config):
        templates = config["train"]["template_script"]
        is_list = isinstance(templates, list)
        if not is_list:
            templates = [templates]
        loaded = []
        # Match the documented invocation and production's CWD-relative paths.
        with set_directory(filename.parent):
            for template in templates:
                path = Path(template)
                self.assertTrue(
                    path.is_file(),
                    f"Missing training template referenced by {filename}: {path}",
                )
                loaded.append(json.loads(path.read_text()))
        return loaded if is_list else loaded[0]

    def test_arguments(self):
        for filename in input_files:
            with self.subTest(filename=filename):
                config = normalize(json.loads(filename.read_text()))
                self._load_templates(filename, config)

    def test_training_templates(self):
        for filename in input_files:
            with self.subTest(filename=filename):
                config = normalize(json.loads(filename.read_text()))
                templates = self._load_templates(filename, config)
                train_config = config["train"]["config"]
                multitask = config["inputs"]["multitask"]
                train_config.update(multitask=multitask, head=config["inputs"]["head"])
                # submit() fixes distillation to a single student model.
                numb_models = (
                    1
                    if config["train"]["type"] == "dp-dist"
                    else config["train"]["numb_models"]
                )
                with tempfile.TemporaryDirectory() as tmp, set_directory(Path(tmp)):
                    prepared = PrepDPTrain().execute(
                        OPIO(
                            {
                                "template_script": templates,
                                "numb_models": numb_models,
                            }
                        )
                    )
                    self.assertEqual(len(prepared["task_paths"]), numb_models)
                    for task_path in prepared["task_paths"]:
                        script = json.loads((task_path / train_script_name).read_text())
                        models = script["model"].get(
                            "model_dict", {"single": script["model"]}
                        )
                        for model in models.values():
                            self.assertEqual(
                                model["type_map"], config["inputs"]["type_map"]
                            )
                            if model["descriptor"]["type"] == "dpa2":
                                self.assertEqual(train_config["impl"], "pytorch")
                        if multitask:
                            heads = set(config["inputs"]["multi_init_data"])
                            self.assertEqual(set(models), heads)
                            self.assertEqual(
                                set(script["training"]["data_dict"]), heads
                            )
                            self.assertEqual(set(script["loss_dict"]), heads)
                            self.assertIn(train_config["head"], heads)
                            init_data = {head: [Path(f"init/{head}")] for head in heads}
                        else:
                            init_data = [Path("init/data")]
                        script = RunDPTrain.write_data_to_input_script(
                            script,
                            train_config,
                            init_data,
                            [Path("iter/data")],
                            major_version="2",
                        )
                        if multitask:
                            for head, data in script["training"]["data_dict"].items():
                                expected = [f"init/{head}"]
                                if head == train_config["head"]:
                                    expected.append("iter/data")
                                self.assertEqual(
                                    data["training_data"]["systems"], expected
                                )
                        else:
                            self.assertEqual(
                                script["training"]["training_data"]["systems"],
                                ["init/data", "iter/data"],
                            )
                        # Exercise both fresh and initialized training paths.
                        for initialized in [False, True]:
                            rewritten = RunDPTrain.write_other_to_input_script(
                                copy.deepcopy(script),
                                train_config,
                                initialized,
                                major_version="2",
                            )
                            normalize_training(rewritten, multi_task=multitask)

    def test_distillation_student_backend(self):
        filename = p_examples / "ch4" / "input_dist.json"
        config = normalize(json.loads(filename.read_text()))
        template = self._load_templates(filename, config)
        self.assertEqual(config["train"]["config"]["impl"], "tensorflow")
        self.assertEqual(template["model"]["descriptor"]["type"], "se_e2_a")
        self.assertEqual(template["model"]["type_map"], ["H", "C"])
        with tempfile.TemporaryDirectory() as tmp, set_directory(Path(tmp)):
            command = _make_train_command(
                ["dp"],
                train_script_name,
                "tensorflow",
                True,
                config["train"]["student_model_path"],
                None,
                "",
                False,
            )
        self.assertIn("--init-frz-model", command)
        self.assertIn("student_model.pb", command)
