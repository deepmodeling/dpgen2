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
    ) -> None:
        self.lmp_template = Path(lmp_template_fname).read_text().split("\n")
        self.revisions = revisions
        self.traj_freq = traj_freq
        self.lmp_set = True
        self.model_list = sorted([model_name_pattern % ii for ii in range(numb_models)])
        self.lmp_template = revise_lmp_input_model(
            self.lmp_template, self.model_list, self.traj_freq
        )
        self.lmp_template = revise_lmp_input_dump(self.lmp_template, self.traj_freq)
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


def revise_lmp_input_model(lmp_lines, task_model_list, trj_freq, deepmd_version="1"):
    idx = find_only_one_key(lmp_lines, ["pair_style", "deepmd"])
    graph_list = " ".join(task_model_list)
    lmp_lines[idx] = "pair_style      deepmd %s out_freq %d out_file model_devi.out" % (
        graph_list,
        trj_freq,
    )
    return lmp_lines


def revise_lmp_input_dump(lmp_lines, trj_freq):
    idx = find_only_one_key(lmp_lines, ["dump", "dpgen_dump"])
    lmp_lines[idx] = (
        f"dump            dpgen_dump all custom %d {lmp_traj_name} id type x y z"
        % trj_freq
    )
    return lmp_lines


def revise_lmp_input_plm(lmp_lines, in_plm, out_plm="output.plumed"):
    idx = find_only_one_key(lmp_lines, ["fix", "dpgen_plm"])
    lmp_lines[idx] = "fix             dpgen_plm all plumed plumedfile %s outfile %s" % (
        in_plm,
        out_plm,
    )
    return lmp_lines


def revise_by_keys(lmp_lines, keys, values):
    for kk, vv in zip(keys, values):  # type: ignore
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(kk, str(vv))
    return lmp_lines
