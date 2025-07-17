import itertools
import random
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    lmp_model_devi_name,
    lmp_pimd_model_devi_name,
    lmp_pimd_traj_name,
    lmp_traj_name,
    model_name_pattern,
    plm_input_name,
    plm_output_name,
)

from .conf_sampling_task_group import (
    ConfSamplingTaskGroup,
)
from .lmp import (
    make_lmp_input,
)
from .task import (
    ExplorationTask,
)


class LmpTemplateTaskGroup(ConfSamplingTaskGroup):
    def __init__(
        self,
    ):
        super().__init__()
        self.lmp_set = False
        self.plm_set = False

    def set_lmp(
        self,
        numb_models: int,
        lmp_template_fname: str,
        plm_template_fname: Optional[str] = None,
        revisions: dict = {},
        traj_freq: int = 10,
        extra_pair_style_args: str = "",
        nvnmd_version: Optional[str] = None,
        pimd_bead: Optional[str] = None,
    ) -> None:
        self.lmp_template = Path(lmp_template_fname).read_text().split("\n")
        self.revisions = revisions
        self.traj_freq = traj_freq
        self.extra_pair_style_args = extra_pair_style_args
        self.nvnmd_version = nvnmd_version
        self.pimd_bead = pimd_bead
        self.lmp_set = True
        self.model_list = sorted([model_name_pattern % ii for ii in range(numb_models)])
        self.lmp_template = revise_lmp_input_model(
            self.lmp_template,
            self.model_list,
            self.traj_freq,
            self.extra_pair_style_args,
            self.pimd_bead,
            nvnmd_version=self.nvnmd_version,
        )
        self.lmp_template = revise_lmp_input_dump(
            self.lmp_template,
            self.traj_freq,
            self.pimd_bead,
            nvnmd_version=self.nvnmd_version,
        )
        if nvnmd_version is not None:
            self.lmp_template = revise_lmp_input_rerun(self.lmp_template)
        if plm_template_fname is not None:
            self.plm_template = Path(plm_template_fname).read_text().split("\n")
            self.plm_set = True

    def make_task(
        self,
    ) -> "LmpTemplateTaskGroup":
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.lmp_set:
            raise RuntimeError("Lammps template and revisions are not set")
        if self.plm_set:
            lmp_template = revise_lmp_input_plm(
                self.lmp_template,
                plm_input_name,
                out_plm=plm_output_name,
            )
        else:
            lmp_template = self.lmp_template
        # clear all existing tasks
        self.clear()
        confs = self._sample_confs()
        templates = [lmp_template]
        if self.plm_set:
            templates.append(self.plm_template)
        conts = self.make_cont(templates, self.revisions)
        nconts = len(conts[0])
        for cc, ii in itertools.product(confs, range(nconts)):  # type: ignore
            if not self.plm_set:
                self.add_task(self._make_lmp_task(cc, conts[0][ii]))
            else:
                self.add_task(self._make_lmp_task(cc, conts[0][ii], conts[1][ii]))
        return self

    def make_cont(
        self,
        templates: list,
        revisions: dict,
    ):
        keys = revisions.keys()
        prod_vv = [revisions[kk] for kk in keys]
        ntemplate = len(templates)
        ret = [[] for ii in range(ntemplate)]
        for vv in itertools.product(*prod_vv):
            for ii in range(ntemplate):
                tt = templates[ii].copy()
                ret[ii].append("\n".join(revise_by_keys(tt, keys, vv)))
        return ret

    def _make_lmp_task(
        self,
        conf: str,
        lmp_cont: str,
        plm_cont: Optional[str] = None,
    ) -> ExplorationTask:
        task = ExplorationTask()
        task.add_file(
            lmp_conf_name,
            conf,
        ).add_file(
            lmp_input_name,
            lmp_cont,
        )
        if plm_cont is not None:
            task.add_file(
                plm_input_name,
                plm_cont,
            )
        return task


def find_only_one_key(lmp_lines, key):
    found = []
    for idx in range(len(lmp_lines)):
        words = lmp_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key:
            found.append(idx)
    if len(found) > 1:
        raise RuntimeError("found %d keywords %s" % (len(found), key))
    if len(found) == 0:
        raise RuntimeError("failed to find keyword %s" % (key))
    return found[0]


def revise_lmp_input_model(
    lmp_lines,
    task_model_list,
    trj_freq,
    extra_pair_style_args="",
    pimd_bead=None,
    deepmd_version="1",
    nvnmd_version=None,
):
    if extra_pair_style_args:
        extra_pair_style_args = " " + extra_pair_style_args
    graph_list = " ".join(task_model_list)
    model_devi_file_name = (
        lmp_pimd_model_devi_name % pimd_bead
        if pimd_bead is not None
        else lmp_model_devi_name
    )
    if nvnmd_version is None:
        idx = find_only_one_key(lmp_lines, ["pair_style", "deepmd"])
        lmp_lines[idx] = "pair_style      deepmd %s out_freq %d out_file %s%s" % (
            graph_list,
            trj_freq,
            model_devi_file_name,
            extra_pair_style_args,
        )
    else:
        idx = find_only_one_key(lmp_lines, ["pair_style", "nvnmd"])
        lmp_lines[idx] = "pair_style      nvnmd %s %s" % (
            "model.pb",
            extra_pair_style_args,
        )

    return lmp_lines


def revise_lmp_input_dump(lmp_lines, trj_freq, pimd_bead=None, nvnmd_version=None):
    idx = find_only_one_key(lmp_lines, ["dump", "dpgen_dump"])
    lmp_traj_file_name = (
        lmp_pimd_traj_name % pimd_bead if pimd_bead is not None else lmp_traj_name
    )
    if nvnmd_version is None:
        lmp_lines[
            idx
        ] = f"dump            dpgen_dump all custom {trj_freq} {lmp_traj_file_name} id type x y z"
    else:
        lmp_lines[idx] = (
            "dump            dpgen_dump all custom %s ${rerun}_%s id type x y z fx fy fz"
            % (trj_freq, lmp_traj_file_name)
        )
        lmp_lines.insert(idx + 1, 'if "${rerun} > 0" then "jump SELF rerun"')
    return lmp_lines


def revise_lmp_input_plm(lmp_lines, in_plm, out_plm="output.plumed"):
    idx = find_only_one_key(lmp_lines, ["fix", "dpgen_plm"])
    lmp_lines[idx] = "fix             dpgen_plm all plumed plumedfile %s outfile %s" % (
        in_plm,
        out_plm,
    )
    return lmp_lines


def revise_lmp_input_rerun(lmp_lines):
    lmp_lines.append("jump SELF end")
    lmp_lines.append("label rerun")
    lmp_lines.append(f"rerun 0_{lmp_traj_name} dump x y z fx fy fz add yes")
    lmp_lines.append("label end")
    return lmp_lines


def revise_by_keys(lmp_lines, keys, values):
    for kk, vv in zip(keys, values):  # type: ignore
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(kk, str(vv))
    return lmp_lines
