# Running the example configurations

Run `dpgen2 submit` from the directory containing the chosen input file. Training
and other file paths are relative to the process working directory. For example:

```sh
cd examples/water
dpgen2 submit input_dpgen.json
```

The training templates are shipped next to their inputs and their `model.type_map`
matches the corresponding `inputs.type_map`, including element order. DPA-2
examples use the DeePMD-kit 3.x PyTorch backend and its nested `repinit` /
`repformer` descriptor schema. The water multitask template defines both
`water_1` and `water_2` in `model.model_dict`, `loss_dict`, and
`training.data_dict`; new iteration data is added only to `water_1`.
`input_distill.json` keeps the TensorFlow backend and uses a compatible
`se_atten` template. These are training-input examples, not bundled datasets or
pretrained models: replace the initial-data, model, potential, executor, and
machine paths/settings for your deployment before submission.

## Methane distillation student

`ch4/input_dist.json` uses TensorFlow and initializes one student from
`student_model.pb`. Its small `student_train.json` uses a `se_e2_a` descriptor,
64-wide fitting layers, and the type map `["H", "C"]`. The frozen student must be
built from that same model definition; an unrelated `.pb` or a PyTorch `.pt`
model is not interchangeable. After providing the initial labeled methane data
at the paths in the template, generate the matching student with DeePMD-kit's
TensorFlow backend:

```sh
cd examples/ch4
dp --tf train student_train.json
dp --tf freeze -o student_model.pb
dpgen2 submit input_dist.json
```

Supply a separately trained `teacher_model.pb` and configure the remaining
example paths before the final command. If using an existing frozen student,
replace the student template with the architecture and type map used to train
that exact student. The workflow forces the distillation model count to one.

## Template checks

The test extra includes the DeePMD-kit input-schema validator without requiring
a TensorFlow/PyTorch training runtime. Run:

```sh
pip install -e '.[test]'
PYTHONPATH=.:tests python -m unittest -v tests.test_check_examples
```

These checks load paths from each example directory, prepare the actual number
of training tasks, inject initial and iteration data, exercise fresh and
initialized training rewrites, and validate the resulting DeePMD-kit schemas.
They do not execute full model training or external simulation/labeling jobs.
