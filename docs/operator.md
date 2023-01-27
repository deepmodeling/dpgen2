
# Operators

There are two types of OPs in DPGEN2

- [OP](#the-op-rundptrain). An execution unit the the workflow. It can be roughly viewed as a piece of Python script taking some input and gives some outputs. An OP cannot be used in the `dflow` until it is embedded in a super-OP.
- [Super-OP](#the-super-op-preprundptrain). An execution unite that is composed by one or more OP and/or super-OPs.

Techinically, OP is a Python class derived from [`dflow.python.OP`](https://github.com/dptech-corp/dflow/blob/master/README.md#13--interface-layer). It serves as the `PythonOPTemplate` of `dflow.Step`.

The super-OP is a Python class derived from `dflow.Steps`. It contains `dflow.Step`s as building blocks, and can be used as OP template to generate a `dflow.Step`. The explanation of the concepts `dflow.Step` and `dflow.Steps`, one may refer to the [manual of dflow](https://github.com/dptech-corp/dflow/blob/master/README.md#123--workflow).

## The super-OP `PrepRunDPTrain`

In the following we will take the `PrepRunDPTrain` super-OP as an example to illustrate how to write OPs in DPGEN2.

`PrepRunDPTrain` is a super-OP that prepares several DeePMD-kit training tasks, and submit all of them. This super-OP is composed by two `dflow.Step`s building from `dflow.python.OP`s `PrepDPTrain` and `RunDPTrain`.

```python
from dflow import (
    Step,
    Steps,
)
from dflow.python import(
    PythonOPTemplate,
    OP,
    Slices,
)

class PrepRunDPTrain(Steps):
    def __init__(
            self,
            name : str,
            prep_train_op : OP,
            run_train_op : OP,
            prep_train_image : str = "dflow:v1.0",
            run_train_image : str = "dflow:v1.0",
    ):
		...
        self = _prep_run_dp_train(
            self,
            self.step_keys,
            prep_train_op,
            run_train_op,
            prep_train_image = prep_train_image,
            run_train_image = run_train_image,
        )
```
The construction of the `PrepRunDPTrain` takes prepare-training `OP` and run-training `OP` and their docker images as input, and implemented in internal method `_prep_run_dp_train`.
```python
def _prep_run_dp_train(
        train_steps,
        step_keys,
        prep_train_op : OP = PrepDPTrain,
        run_train_op : OP = RunDPTrain,
        prep_train_image : str = "dflow:v1.0",
        run_train_image : str = "dflow:v1.0",
):
    prep_train = Step(
        ...
        template=PythonOPTemplate(
            prep_train_op,
            image=prep_train_image,
            ...
        ),
        ...
    )
    train_steps.add(prep_train)

    run_train = Step(
        ...
        template=PythonOPTemplate(
            run_train_op,
            image=run_train_image,
            ...
        ),
        ...
    )
    train_steps.add(run_train)

    train_steps.outputs.artifacts["scripts"]._from = run_train.outputs.artifacts["script"]
    train_steps.outputs.artifacts["models"]._from = run_train.outputs.artifacts["model"]
    train_steps.outputs.artifacts["logs"]._from = run_train.outputs.artifacts["log"]
    train_steps.outputs.artifacts["lcurves"]._from = run_train.outputs.artifacts["lcurve"]

    return train_steps
```

In `_prep_run_dp_train`, two instances of `dflow.Step`, i.e. `prep_train` and `run_train`, generated from `prep_train_op` and `run_train_op`, respectively, are added to `train_steps`. Both of `prep_train_op` and `run_train_op` are OPs (python classes derived from `dflow.python.OP`s) that will be illustrated later. `train_steps` is an instance of `dflow.Steps`. The outputs of the second OP `run_train` are assigned to the outputs of the `train_steps`.

The `prep_train` prepares a list of paths, each of which contains all necessary files to start a DeePMD-kit training tasks.

The `run_train` slices the list of paths, and assign each item in the list to a DeePMD-kit task. The task is executed by `run_train_op`. This is a very nice feature of `dflow`, because the developer only needs to implement how one DeePMD-kit task is executed, and then all the items in the task list will be executed [in parallel](https://github.com/dptech-corp/dflow/blob/master/README.md#315-produce-parallel-steps-using-loop). See the following code to see how it works
```python
    run_train = Step(
        'run-train',
        template=PythonOPTemplate(
            run_train_op,
            image=run_train_image,
            slices = Slices(
                "int('{{item}}')",
                input_parameter = ["task_name"],
                input_artifact = ["task_path", "init_model"],
                output_artifact = ["model", "lcurve", "log", "script"],
            ),
        ),
        parameters={
            "config" : train_steps.inputs.parameters["train_config"],
            "task_name" : prep_train.outputs.parameters["task_names"],
        },
        artifacts={
            'task_path' : prep_train.outputs.artifacts['task_paths'],
            "init_model" : train_steps.inputs.artifacts['init_models'],
            "init_data": train_steps.inputs.artifacts['init_data'],
            "iter_data": train_steps.inputs.artifacts['iter_data'],
        },
        with_sequence=argo_sequence(argo_len(prep_train.outputs.parameters["task_names"]), format=train_index_pattern),
        key = step_keys['run-train'],
    )
```
The input parameter `"task_names"` and artifacts `"task_paths"` and `"init_model"` are sliced and supplied to each DeePMD-kit task. The output artifacts of the tasks (`"model"`, `"lcurve"`, `"log"` and `"script"`) are stacked in the same order as the input lists. These lists are assigned as the outputs of `train_steps` by
```python
    train_steps.outputs.artifacts["scripts"]._from = run_train.outputs.artifacts["script"]
    train_steps.outputs.artifacts["models"]._from = run_train.outputs.artifacts["model"]
    train_steps.outputs.artifacts["logs"]._from = run_train.outputs.artifacts["log"]
    train_steps.outputs.artifacts["lcurves"]._from = run_train.outputs.artifacts["lcurve"]
```


## The OP `RunDPTrain`

We will take `RunDPTrain` as an example to illustrate how to implement an OP in DPGEN2.
The source code of this OP is found [here](https://github.com/wanghan-iapcm/dpgen2/blob/master/dpgen2/op/run_dp_train.py)

Firstly of all, an OP should be implemented as a derived class of `dflow.python.OP`.

The `dflow.python.OP` requires static type define for the input and output variables, i.e. the signatures of an OP. The input and output signatures of the `dflow.python.OP` are given by `classmethods` `get_input_sign` and `get_output_sign`.


```python
from dflow.python import (
    OP,
    OPIO,
    OPIOSign,
    Artifact,
)
class RunDPTrain(OP):
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "config" : dict,
            "task_name" : str,
            "task_path" : Artifact(Path),
            "init_model" : Artifact(Path),
            "init_data" : Artifact(List[Path]),
            "iter_data" : Artifact(List[Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "script" : Artifact(Path),
            "model" : Artifact(Path),
            "lcurve" : Artifact(Path),
            "log" : Artifact(Path),
        })
```

All items not defined as `Artifact` are treated as parameters of the `OP`. The concept of parameter and artifact are explained in the [dflow document](https://github.com/dptech-corp/dflow/blob/master/README.md#Parametersandartifacts). To be short, the artifacts can be `pathlib.Path` or a list of `pathlib.Path`. The artifacts are passed by the file system. Other data structures are treated as parameters, they are passed as variables encoded in `str`. Therefore, a large amout of information should be stored in artifacts, otherwise they can be considered as parameters.

The operation of the `OP` is implemented in method `execute`, and are run in docker containers. Again taking the `execute` method of `RunDPTrain` as an example

```python
    @OP.exec_sign_check
    def execute(
            self,
            ip : OPIO,
    ) -> OPIO:
        ...
        task_name = ip['task_name']
        task_path = ip['task_path']
        init_model = ip['init_model']
        init_data = ip['init_data']
        iter_data = ip['iter_data']
        ...
        work_dir = Path(task_name)
        ...
        # here copy all files in task_path to work_dir
        ...
        with set_directory(work_dir):
            fplog = open('train.log', 'w')
            def clean_before_quit():
                fplog.close()
            # train model
            command = ['dp', 'train', train_script_name]
            ret, out, err = run_command(command)
            if ret != 0:
                clean_before_quit()
                raise FatalError('dp train failed')
            fplog.write(out)
            # freeze model
            ret, out, err = run_command(['dp', 'freeze', '-o', 'frozen_model.pb'])
            if ret != 0:
                clean_before_quit()
                raise FatalError('dp freeze failed')
            fplog.write(out)
            clean_before_quit()

        return OPIO({
            "script" : work_dir / train_script_name,
            "model" : work_dir / "frozen_model.pb",
            "lcurve" : work_dir / "lcurve.out",
            "log" : work_dir / "train.log",
        })
```

The inputs and outputs variables are recorded in data structure `dflow.python.OPIO`, which is initialized by a Python dict. The keys in the input/output `dict`, and the types of the input/output variables will be checked against their signatures by decorator `OP.exec_sign_check`. If any key or type does not match, an exception will be raised.

It is noted that all input artifacts of the `OP` are read-only, therefore, the first step of the `RunDPTrain.execute` is to copy all necessary input files from the directory `task_path` prepared by `PrepDPTrain` to the working directory `work_dir`.

`with_directory` method creates the `work_dir` and swithes to the directory before the execution, and then exits the directoy when the task finishes or an error is raised.

In what follows, the training and model frozen bash commands are executed consecutively. The return code is check and a `FatalError` is raised if a non-zero code is detected.

Finally the trained model file, input script, learning curve file and the log file are recored in a `dflow.python.OPIO` and returned.
