import logging
import os
import re
import tempfile
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Union,
)

from dpgen2.constants import (
    lmp_conf_name,
    lmp_input_name,
    model_name_pattern,
    plm_input_name,
)
from dpgen2.utils import (
    run_command,
    set_directory,
)

from .conf_sampling_task_group import (
    ConfSamplingTaskGroup,
)
from .lmp import (
    make_lmp_input,
)
from .lmp_template_task_group import (
    LmpTemplateTaskGroup,
)
from .task import (
    ExplorationTask,
    ExplorationTaskGroup,
)


class CustomizedLmpTemplateTaskGroup(ConfSamplingTaskGroup):
    def __init__(
        self,
    ):
        super().__init__()
        self.lmp_set = False
        self.plm_set = False

    def set_lmp(
        self,
        numb_models: int,
        custom_shell_commands: List[str],
        revisions: dict = {},
        traj_freq: int = 10,
        input_lmp_conf_name: str = lmp_conf_name,
        input_lmp_tmpl_name: str = lmp_input_name,
        input_plm_tmpl_name: Optional[str] = None,
        input_extra_files: List[str] = [],
        output_dir_pattern: Union[str, List[str]] = "*",
        output_lmp_conf_name: str = lmp_conf_name,
        output_lmp_tmpl_name: str = lmp_input_name,
        output_plm_tmpl_name: Optional[str] = None,
    ) -> None:
        r"""Set lammps task.

        Parameters
        ----------
        numb_models : int
            Number of models
        custom_shell_commands : str
            Customized shell commands to be run for each configuration.
            The commands require `input_lmp_conf_name` as input conf file,
            `input_lmp_tmpl_name` and `input_plm_tmpl_name` as templates,
            and `input_extra_files` as extra input files.
            By running the commands a series folders in pattern
            `output_dir_pattern` are supposed to be generated,
            and each folder is supposed to contain a configuration file
            `output_lmp_conf_name`, a lammps template file `output_lmp_tmpl_name`
            and a plumed template file `output_plm_tmpl_name`.
        revisions : dict
            Revision dictionary. Provided in {key: [enumerated values]} format
        traj_freq : int
            Frequency along trajectory of checking model deviation
        input_lmp_conf_name : str
            Input conf file name for the shell commands.
        input_lmp_tmpl_name : str
            Template file name of lammps input
        input_plm_tmpl_name : str
            Template file name of the plumed input
        input_extra_files : List[str]
            Extra files that may be needed to execute the shell commands
        output_dir_pattern : Union[str, List[str]]
            Pattern of resultant folders generated by the shell commands.
        output_lmp_conf_name : str
            Generated conf file name.
        output_lmp_tmpl_name : str
            Generated lmp input file name.
        output_plm_tmpl_name : str
            Generated plm input file name.

        """
        self.numb_models = numb_models
        self.lmp_template = Path(input_lmp_tmpl_name).read_text().split("\n")
        self.revisions = revisions
        self.traj_freq = traj_freq
        self.has_plm = input_plm_tmpl_name is not None
        self.do_custom = (
            custom_shell_commands is not None and custom_shell_commands != []
        )
        if not self.do_custom:
            raise RuntimeError(
                "no customized shell command is found, please provide at least 1 "
                "or use LmpTemplateTaskGroup instead"
            )

        self.lmp_template_fn = Path(input_lmp_tmpl_name)
        self.lmp_template_fc = Path(input_lmp_tmpl_name).read_text()
        if self.has_plm:
            tmp_input_plm_tmpl_name = (
                input_plm_tmpl_name if input_plm_tmpl_name is not None else ""
            )
            self.plm_template_fn = Path(tmp_input_plm_tmpl_name)
            self.plm_template_fc = Path(tmp_input_plm_tmpl_name).read_text()
        tmp_input_extra_files = [Path(ii) for ii in input_extra_files]
        tmp_input_extra_files.append(self.lmp_template_fn)
        if self.has_plm:
            tmp_input_extra_files.append(self.plm_template_fn)
        if custom_shell_commands is not None:
            self.input_extra_files = [ii.name for ii in tmp_input_extra_files]
            self.input_extra_file_contents = [
                Path(ii).read_text() for ii in tmp_input_extra_files
            ]
        self.custom_shell_commands = custom_shell_commands
        self.output_dir_pattern = output_dir_pattern
        if type(self.output_dir_pattern) is str:
            self.output_dir_pattern = [self.output_dir_pattern]
        self.input_lmp_conf_name = input_lmp_conf_name
        self.output_lmp_conf_name = output_lmp_conf_name
        self.output_lmp_tmpl_name = output_lmp_tmpl_name
        self.output_plm_tmpl_name = output_plm_tmpl_name

        self.lmp_set = True
        self.plm_set = True if self.has_plm else False

    def make_task(
        self,
    ) -> ExplorationTaskGroup:
        if not self.conf_set:
            raise RuntimeError("confs are not set")
        if not self.lmp_set:
            raise RuntimeError("Lammps template and revisions are not set")

        confs = self._sample_confs()
        for cc in confs:
            self._make_customized_task_group(cc)

        return self

    def _make_customized_task_group(
        self,
        conf,
    ) -> ExplorationTaskGroup:
        with tempfile.TemporaryDirectory() as tmpdir:
            with set_directory(Path(tmpdir)):
                Path(self.input_lmp_conf_name).write_text(conf)
                # copy all customized files
                for ff, cc in zip(
                    self.input_extra_files, self.input_extra_file_contents
                ):
                    Path(ff).write_text(cc)
                # run all customized shell commands
                for ss in self.custom_shell_commands:
                    # run shell command with os.system
                    ret = os.system(ss)
                    if ret != 0:
                        raise RuntimeError(
                            f"execution of {ss} returns a non-zero value {ret}"
                        )
                # loop over all pattern matched result dirs
                for ff in [
                    ii for ii in sorted(os.listdir(os.getcwd())) if Path(ii).is_dir()
                ]:
                    matched_ff = None
                    for pp in self.output_dir_pattern:
                        if re.match(pp, ff):
                            matched_ff = ff
                            break
                    # no matched continue
                    if matched_ff is None:
                        logging.info(
                            "No output dir matches the patter {self.output_dir_pattern} "
                        )
                        continue
                    with set_directory(Path(matched_ff)):
                        lmp_tgroup = LmpTemplateTaskGroup()
                        lmp_tgroup.set_lmp(
                            self.numb_models,
                            self.output_lmp_tmpl_name,
                            self.output_plm_tmpl_name if self.has_plm else None,
                            revisions=self.revisions,
                            traj_freq=self.traj_freq,
                        )
                        conf_fc = Path(self.output_lmp_conf_name).read_text()
                        lmp_tgroup.set_conf(
                            [conf_fc],
                            1,
                        )
                        self.add_group(lmp_tgroup.make_task())
        return self
