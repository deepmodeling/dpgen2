from .task_group import (
    BaseExplorationTaskGroup,
    ExplorationTaskGroup,
)


class ExplorationStage:
    """
    The exploration stage.

    """

    def __init__(self):
        self.clear()

    def clear(self):
        """
        Clear all exploration group.

        """
        self.explor_groups = []

    def add_task_group(
        self,
        grp: ExplorationTaskGroup,
    ):
        """
        Add an exploration group

        Parameters
        ----------
        grp : ExplorationTaskGroup
            The added exploration task group

        """
        self.explor_groups.append(grp)
        return self

    def make_task(
        self,
    ) -> BaseExplorationTaskGroup:
        """
        Make the LAMMPS task group.

        Returns
        -------
        task_grp: BaseExplorationTaskGroup
            The returned lammps task group. The number of tasks is equal to
            the summation of task groups defined by all the exploration groups
            added to the stage.

        """

        lmp_task_grp = BaseExplorationTaskGroup()
        for ii in self.explor_groups:
            # lmp_task_grp.add_group(ii.make_task())
            lmp_task_grp += ii.make_task()
        return lmp_task_grp
