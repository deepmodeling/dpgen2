import itertools
import random
import re
import warnings
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
)

from dflow.python import (
    FatalError,
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
        pimd_bead: Optional[str] = None,
        input_extra_files: Optional[List[str]] = None,
        strict_revisions: bool = True,
    ) -> None:
        self.lmp_template = Path(lmp_template_fname).read_text().split("\n")
        self.revisions = revisions
        self.strict_revisions = strict_revisions
        self.traj_freq = traj_freq
        self.extra_pair_style_args = extra_pair_style_args
        self.pimd_bead = pimd_bead
        if input_extra_files is not None:
            self.input_extra_files = [Path(ii).name for ii in input_extra_files]
            self.input_extra_file_contents = [
                Path(ii).read_text() for ii in input_extra_files
            ]
        else:
            self.input_extra_files = []
            self.input_extra_file_contents = []
        self.lmp_set = True
        self.model_list = sorted([model_name_pattern % ii for ii in range(numb_models)])
        self.lmp_template = revise_lmp_input_model(
            self.lmp_template,
            self.model_list,
            self.traj_freq,
            self.extra_pair_style_args,
            self.pimd_bead,
        )
        self.lmp_template = revise_lmp_input_dump(
            self.lmp_template, self.traj_freq, self.pimd_bead
        )
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
        # Validate: check for unreplaced V_* variables in substituted templates
        if self.revisions:
            template_raw = "\n".join(self.lmp_template)
            if self.plm_set:
                template_raw += "\n" + "\n".join(self.plm_template)
            # Flatten all template variants (LAMMPS + PLUMED) for validation
            all_conts = [c for c_list in conts for c in c_list]
            try:
                check_revisions_completeness(
                    all_conts,
                    list(self.revisions.keys()),
                    template_raw=template_raw,
                    strict=self.strict_revisions,
                )
            except ValueError as exc:
                raise FatalError(str(exc)) from exc
        else:
            # Warn (but don't error) if template has V_* but no revisions provided.
            # This can be legitimate for customized-lmp-template workflows or tests.
            combined = "\n".join(self.lmp_template)
            if self.plm_set:
                combined += "\n" + "\n".join(self.plm_template)
            unreplaced = find_unreplaced_variables(combined)
            if unreplaced:
                warnings.warn(
                    f"LAMMPS template contains revision variable(s) {sorted(unreplaced)} "
                    f"but no 'revisions' dict was provided. "
                    f"These variables will NOT be substituted. "
                    f"If this is unintentional, add them to 'revisions' in your "
                    f"exploration config.",
                    stacklevel=2,
                )
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

        # Add extra files to the task
        for file_name, file_content in zip(
            self.input_extra_files, self.input_extra_file_contents
        ):
            task.add_file(file_name, file_content)

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
):
    idx = find_only_one_key(lmp_lines, ["pair_style", "deepmd"])
    if extra_pair_style_args:
        extra_pair_style_args = " " + extra_pair_style_args
    graph_list = " ".join(task_model_list)
    model_devi_file_name = (
        lmp_pimd_model_devi_name % pimd_bead
        if pimd_bead is not None
        else lmp_model_devi_name
    )
    lmp_lines[idx] = "pair_style      deepmd %s out_freq %d out_file %s%s" % (
        graph_list,
        trj_freq,
        model_devi_file_name,
        extra_pair_style_args,
    )
    return lmp_lines


def revise_lmp_input_dump(lmp_lines, trj_freq, pimd_bead=None):
    idx = find_only_one_key(lmp_lines, ["dump", "dpgen_dump"])
    lmp_traj_file_name = (
        lmp_pimd_traj_name % pimd_bead if pimd_bead is not None else lmp_traj_name
    )
    lmp_lines[
        idx
    ] = f"dump            dpgen_dump all custom {trj_freq} {lmp_traj_file_name} id type x y z"
    return lmp_lines


def revise_lmp_input_plm(lmp_lines, in_plm, out_plm="output.plumed"):
    idx = find_only_one_key(lmp_lines, ["fix", "dpgen_plm"])
    lmp_lines[idx] = "fix             dpgen_plm all plumed plumedfile %s outfile %s" % (
        in_plm,
        out_plm,
    )
    return lmp_lines


def revise_by_keys(lmp_lines, keys, values):
    """Replace complete revision tokens without matching identifier prefixes."""
    for kk, vv in zip(keys, values):  # type: ignore
        replacement = str(vv)
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(kk)}(?![A-Za-z0-9_])")
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = pattern.sub(lambda _match: replacement, lmp_lines[ii])
    return lmp_lines


# DPGEN and DPGEN2 templates conventionally use standalone V_* tokens for
# revisions. Native LAMMPS or PLUMED identifiers may use the same spelling;
# strict_revisions controls whether unexpected tokens are errors or warnings.
_REVISION_VARIABLE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])V_[A-Z][A-Z0-9_]*(?![A-Za-z0-9_])"
)


def _strip_lammps_comments(content: str) -> str:
    """Remove unquoted LAMMPS comments while preserving quoted hash characters.

    This prevents V_* patterns in comments (e.g., "# set V_PRESS later")
    from being flagged as unreplaced variables without hiding executable
    placeholders inside single- or double-quoted strings.
    """
    stripped: List[str] = []
    for line in content.split("\n"):
        quote: Optional[str] = None
        escaped = False
        kept: List[str] = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
            elif char == "\\" and quote is not None:
                kept.append(char)
                escaped = True
            elif char in ("'", '"'):
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
                kept.append(char)
            elif char == "#" and quote is None:
                break
            else:
                kept.append(char)
        stripped.append("".join(kept))
    return "\n".join(stripped)


def find_unreplaced_variables(content: str) -> Set[str]:
    """Scan text for standalone V_* tokens that may be unreplaced revisions.

    Strips LAMMPS comments before scanning to avoid false positives from
    commented-out variable references.

    Parameters
    ----------
    content : str
        The LAMMPS input content after revision substitution.

    Returns
    -------
    Set[str]
        Set of variable names (e.g. {"V_PRESS", "V_UNDEFINED"}) still present.
    """
    stripped = _strip_lammps_comments(content)
    return set(_REVISION_VARIABLE_PATTERN.findall(stripped))


def report_undefined_revision_variables(
    variables: Set[str],
    revision_keys: List[str],
    strict: bool,
) -> None:
    """Raise for undefined tokens in strict mode, otherwise emit a warning."""
    if not variables:
        return
    message = (
        f"LAMMPS template contains undefined revision variable(s): "
        f"{sorted(variables)}. Defined revisions: {sorted(revision_keys)}. "
        f"Please add missing variables to 'revisions' in your exploration config, "
        f"or remove them from the template."
    )
    if strict:
        raise ValueError(message)
    warnings.warn(
        message + " Continuing because strict revision validation is disabled.",
        stacklevel=3,
    )


def check_revisions_completeness(
    templates_content: List[str],
    revision_keys: List[str],
    template_raw: str = "",
    strict: bool = True,
) -> None:
    """Validate that all V_* placeholders in the template have been substituted.

    This function performs three checks:
    1. **Raw-template definition check**: Compare complete placeholder tokens with
       the revision keys before applying substitutions.
    2. **Post-substitution residual check**: After applying revisions, scan the output
       for any remaining V_* variables that were not replaced. This catches typos in
       template variables or missing keys in revisions.
    3. **Unused key warning**: If a revision key is defined but never appears in the
       raw template, emit a warning (possible typo in the key name).

    Parameters
    ----------
    templates_content : List[str]
        List of template strings after revision substitution (one per revision combo).
    revision_keys : List[str]
        The keys defined in the revisions dict.
    template_raw : str
        The raw template content before substitution (for unused key detection).
    strict : bool
        If true, undefined V_* tokens are errors. If false, warn and continue
        so templates may use V_* as native LAMMPS or PLUMED identifiers.

    Raises
    ------
    ValueError
        If undefined V_* tokens are found and strict is true.

    Warns
    -----
    UserWarning
        If undefined V_* tokens are found and strict is false, or if a
        revision key is unused.
    """
    revision_key_set = set(revision_keys)

    # Check 1: Compare raw tokens before substitution so a shorter defined key
    # cannot erase the prefix of a longer undefined placeholder.
    raw_variables = find_unreplaced_variables(template_raw) if template_raw else set()
    undefined_raw = raw_variables - revision_key_set
    report_undefined_revision_variables(
        undefined_raw,
        revision_keys,
        strict=strict,
    )

    # Check 2: Residual unreplaced variables
    all_unreplaced: Set[str] = set()
    for content in templates_content:
        all_unreplaced.update(find_unreplaced_variables(content))

    report_undefined_revision_variables(
        all_unreplaced - undefined_raw,
        revision_keys,
        strict=strict,
    )

    # Check 3: Unused revision keys (warning only)
    if template_raw and revision_keys:
        for key in revision_keys:
            if key not in raw_variables:
                warnings.warn(
                    f"Revision key '{key}' is defined but does not appear in the "
                    f"LAMMPS/PLUMED template. Possible typo?",
                    stacklevel=3,
                )
