import itertools
import random
from typing import (
    List,
    Optional,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    model_name_pattern,
)

from .task import (
    ExplorationTask,
)
from .task_group import (
    ExplorationTaskGroup,
)


class ConfSamplingTaskGroup(ExplorationTaskGroup):
    def __init__(
        self,
    ):
        super().__init__()
        self.conf_set = False

    def set_conf(
        self,
        conf_list: List[str],
        n_sample: Optional[int] = None,
        random_sample: bool = False,
    ):
        """
        Set the configurations of exploration

        Parameters
        ----------
        conf_list       str
            A list of file contents
        n_sample        int
            Number of samples drawn from the conf list each time
            `make_task` is called. If set to `None`,
            `n_sample` is set to length of the conf_list.
        random_sample   bool
            If true the confs are randomly sampled, otherwise are
            consecutively sampled from the conf_list
        """
        self.conf_list = conf_list
        if n_sample is None:
            self.n_sample = len(self.conf_list)
        else:
            self.n_sample = n_sample
        self.random_sample = random_sample
        self.conf_queue = []
        self.conf_set = True

    def _sample_confs(
        self,
    ) -> list:
        confs = []
        for ii in range(self.n_sample):
            if len(self.conf_queue) == 0:
                add_list = self.conf_list.copy()
                if self.random_sample:
                    random.shuffle(add_list)
                self.conf_queue += add_list
            confs.append(self.conf_queue.pop(0))
        return confs
