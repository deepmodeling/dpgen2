from typing import (
    Optional,
)

from .task import (
    ExplorationTask,
)
from .task_group import (
    ExplorationTaskGroup,
)


class DiffCSPTaskGroup(ExplorationTaskGroup):
    def __init__(
        self,
        trj_freq: int = 10,
        fmax: float = 1e-4,
        steps: int = 200,
        timeout: Optional[int] = None,
    ):
        super().__init__()
        self.trj_freq = trj_freq
        self.fmax = fmax
        self.steps = steps
        self.timeout = timeout

    def make_task(self) -> "DiffCSPTaskGroup":
        """
        Make the DiffCSP task group.

        Returns
        -------
        task_grp: DiffCSPTaskGroup
            Return one DiffCSP task group.
        """
        # clear all existing tasks
        self.clear()
        self.add_task(self._make_diffcsp_task())
        return self

    def _make_diffcsp_task(self) -> ExplorationTask:
        task = ExplorationTask()
        task.trj_freq = self.trj_freq  # type: ignore
        task.fmax = self.fmax  # type: ignore
        task.steps = self.steps  # type: ignore
        task.timeout = self.timeout  # type: ignore
        return task
