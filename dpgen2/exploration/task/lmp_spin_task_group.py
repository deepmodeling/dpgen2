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


class LmpSpinTaskGroup(ConfSamplingTaskGroup):
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
    ) -> None:
        self.lmp_template = Path(lmp_template_fname).read_text().split("\n")
        self.revisions = revisions
        self.lmp_set = True
        self.model_list = sorted([model_name_pattern % ii for ii in range(numb_models)])
        if plm_template_fname is not None:
            self.plm_template = Path(plm_template_fname).read_text().split("\n")
            self.plm_set = True

    def make_task(
        self,
    ) -> "LmpSpinTaskGroup":
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.lmp_set:
            raise RuntimeError("Lammps SPIN template and revisions are not set")
        # clear all existing tasks
        self.clear()
        confs = self._sample_confs()
        templates = [self.lmp_template]
        conts = self.make_cont(templates, self.revisions)
        nconts = len(conts[0])
        for cc, ii in itertools.product(confs, range(nconts)):  # type: ignore
            self.add_task(self._make_lmp_task(cc, conts[0][ii]))
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


def revise_by_keys(lmp_lines, keys, values):
    for kk, vv in zip(keys, values):  # type: ignore
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(kk, str(vv))
    return lmp_lines
