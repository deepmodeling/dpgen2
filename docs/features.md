# Supported features

This page records the features wired into the main `dpgen2 submit` workflow on
the current branch. It replaces the historical project status that described
DPGEN2 as supporting only DeePMD-kit, LAMMPS, and VASP.

The support levels used below are:

- **Supported**: integrated into the configuration schema and concurrent-learning workflow, with tests or an example in this repository.
- **Partial**: a useful lower-level capability exists, but the complete DP-GEN behavior still requires additional implementation.
- **Not implemented**: no built-in workflow integration exists yet.

## Workflows and training

| Capability | Status | Implementation and examples |
| --- | --- | --- |
| Concurrent learning | Supported | The [DP-GEN loop](../dpgen2/flow/dpgen_loop.py) is exposed through the [submit and resubmit commands](quickcli.md). |
| DeePMD-kit training | Supported | The `dp` training style uses [PrepRunDPTrain](../dpgen2/superop/prep_run_dp_train.py); the [Al-Mg example](../examples/almg/input.json) provides a compact configuration. |
| Knowledge distillation | Supported | The `dp-dist` training style and DeepMD labeling backend are demonstrated by the [distillation example](../examples/water/input_distill.json). |
| Multitask training and finetuning | Supported | See the [multitask example](../examples/water/input_multitask.json) and the [pretrained-model finetuning example](../examples/water/input_dpgen.json). |
| Simplify workflow | Not implemented | Open a focused feature request describing the desired CLI, artifacts, and compatibility with the DP-GEN simplify workflow. |
| Initial-data generation workflow | Not implemented | Configuration files and existing datasets can be supplied as inputs, but DPGEN2 does not yet generate an initial dataset as a standalone workflow. |

## Exploration and selection

| Engine or feature | Status | Configuration and examples |
| --- | --- | --- |
| LAMMPS molecular dynamics | Supported | Use exploration type `lmp` and task type `lmp-md`; see the [Al-Mg example](../examples/almg/input.json). |
| LAMMPS templates and PLUMED | Supported | The `lmp-template` and `customized-lmp-template` task groups support user-provided LAMMPS and PLUMED inputs; see the [CHNO LAMMPS-template example](../examples/chno/input.json) and the [PLUMED template tests](../tests/exploration/test_lmp_templ_task_group.py). |
| CALYPSO structure search | Supported | Use `calypso`, `calypso:default`, or `calypso:merge`; see the [CALYPSO example](../examples/calypso/input.test.json) and [PrepRunCaly](../dpgen2/superop/prep_run_calypso.py). |
| DiffCSP structure generation | Supported | Use exploration type `diffcsp`; see the [DiffCSP example](../examples/diffcsp/dpgen.json) and [PrepRunDiffCSP](../dpgen2/superop/prep_run_diffcsp.py). |
| Atomic model deviation and cluster extraction | Partial | `lmp-md` can request atomic model-deviation output with `use_clusters`, but the built-in selector still labels complete frames rather than extracting local clusters. |
| Gromacs molecular dynamics | Not implemented | No Gromacs exploration OP or submit configuration is registered. |
| AMBER molecular dynamics | Not implemented | No AMBER exploration OP or submit configuration is registered. |

## Labeling backends

| Backend | Status | Configuration and examples |
| --- | --- | --- |
| VASP | Supported | Use FP type `vasp`; see the [Al-Mg example](../examples/almg/input.json) and [VASP implementation](../dpgen2/fp/vasp.py). |
| Gaussian | Supported | Use FP type `gaussian`; see the [CHNO example](../examples/chno/input.json) and [Gaussian implementation](../dpgen2/fp/gaussian.py). |
| CP2K | Supported | Use FP type `fpop_cp2k`; see the [CP2K example](../examples/water/input_dpgen_cp2k.json) and [CP2K adapter](../dpgen2/fp/cp2k.py). This adapter requires the optional FPOP dependency. |
| ABACUS | Supported | Use FP type `fpop_abacus`; see the [ABACUS example](../examples/water/input_dpgen_abacus.json) and [ABACUS adapter](../dpgen2/fp/abacus.py). This adapter requires the optional FPOP dependency. |
| DeePMD-kit inference | Supported | Use FP type `deepmd` for teacher-model labeling in distillation workflows; see the [DeepMD implementation](../dpgen2/fp/deepmd.py). |
| Siesta | Not implemented | No Siesta preparation, execution, or output-collection adapter is registered. |

## Adding a missing capability

The remaining entries are intentionally listed as focused gaps instead of one
umbrella implementation task. A feature request for one engine or workflow
should define its configuration schema, required executables and files,
prepared task layout, collected output format, unit tests, and a runnable
example. The [operator guide](operator.md) and [exploration extension guide](exploration.md)
describe the integration points.
