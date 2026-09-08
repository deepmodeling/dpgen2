# Guide on dpgen2 commands

One may use dpgen2 through command line interface. A full documentation of the cli is found [here](fullcli)

## Submit a workflow
The dpgen2 workflow can be submitted via the `submit` command
```bash
dpgen2 submit input.json
```
where `input.json` is the input script. A guide of writing the script is found [here](inputscript).
When a workflow is submitted, a ID (WFID) of the workflow will be printed for later reference.

## Check the convergence of a workflow
The convergence of stages of the workflow can be checked by the `status` command. It prints the indexes of the finished stages, iterations, and the accurate, candidate and failed ratio of explored configurations of each iteration.
```bash
$ dpgen2 status input.json WFID
#   stage  id_stg.    iter.      accu.      cand.      fail.
# Stage    0  --------------------
        0        0        0     0.8333     0.1667     0.0000
        0        1        1     0.7593     0.2407     0.0000
        0        2        2     0.7778     0.2222     0.0000
        0        3        3     1.0000     0.0000     0.0000
# Stage    0  converged YES  reached max numb iterations NO
# All stages converged
```

## Watch the progress of a workflow
The progress of a workflow can be watched on-the-fly
```bash
$ dpgen2 watch input.json WFID
INFO:root:steps iter-000000--prep-run-train----------------------- finished
INFO:root:steps iter-000000--prep-run-explore--------------------- finished
INFO:root:steps iter-000000--prep-run-fp-------------------------- finished
INFO:root:steps iter-000000--collect-data------------------------- finished
INFO:root:steps iter-000001--prep-run-train----------------------- finished
INFO:root:steps iter-000001--prep-run-explore--------------------- finished
...
```
The artifacts can be downloaded on-the-fly with `-d` flag. Note that the existing files are automatically skipped if one sets `dflow_config["archive_mode"] = None`.


## Download workflow results

The `download` command retrieves training, exploration, and labeling artifacts without requiring direct access to the workflow storage backend. List the supported artifact names first:

```bash
dpgen2 download input.json WFID --list-supported
```

Running without filters downloads every supported artifact from every successful iteration. Use iteration and artifact filters for a smaller result set:

```bash
dpgen2 download input.json WFID \
  --iterations 0-2 \
  --step-definitions \
    prep-run-train/output/models \
    prep-run-train/output/lcurves \
    prep-run-train/output/logs \
    prep-run-explore/output/trajs \
    prep-run-explore/output/model_devis \
    prep-run-fp/output/labeled_data \
  --prefix results
```

Files are organized below `results/iter-000000/<step>/<input-or-output>/<artifact>`. Existing completed downloads are skipped by default; pass `--no-check-point` to request them again. The corresponding result groups are:

- training: models, learning curves, logs, and generated scripts;
- exploration: trajectories, model deviations, logs, and extra outputs;
- labeling: input configurations, labeled data, logs, and extra outputs.


## Show the keys of steps

Each dpgen2 step is assigned a unique key. The keys of the finished steps can be checked with `showkey` command
```bash                                                                                                                                                                              $ dpgen2 showkey input.json WFID
                   0 : iter-000000--prep-train
              1 -> 4 : iter-000000--run-train-0000 -> iter-000000--run-train-0003
                   5 : iter-000000--prep-lmp
             6 -> 14 : iter-000000--run-lmp-000000 -> iter-000000--run-lmp-000008
                  15 : iter-000000--select-confs
                  16 : iter-000000--prep-fp
            17 -> 20 : iter-000000--run-fp-000000 -> iter-000000--run-fp-000003
                  21 : iter-000000--collect-data
                  22 : iter-000000--scheduler
                  23 : iter-000000--id
                  24 : iter-000001--prep-train
            25 -> 28 : iter-000001--run-train-0000 -> iter-000001--run-train-0003
                  29 : iter-000001--prep-lmp
            30 -> 38 : iter-000001--run-lmp-000000 -> iter-000001--run-lmp-000008
                  39 : iter-000001--select-confs
                  40 : iter-000001--prep-fp
            41 -> 44 : iter-000001--run-fp-000000 -> iter-000001--run-fp-000003
                  45 : iter-000001--collect-data
                  46 : iter-000001--scheduler
                  47 : iter-000001--id
```


## Resubmit a workflow

If a workflow stopped abnormally, one may submit a new workflow with some steps of the old workflow reused.
```bash
dpgen2 resubmit input.json WFID --reuse 0-41
```
The steps of workflow WDID 0-41 (0<=id<41, note that 41 is not included) will be reused in the new workflow. The indexes of the steps are printed by `dpgen2 showkey`. In the example, all the steps before the `iter-000001--run-fp-000000` will be used in the new workflow.
