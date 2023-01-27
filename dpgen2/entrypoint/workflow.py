import argparse, os, json, logging
from dflow import (
    Workflow,
)
from typing import (
    Optional,
)
from dpgen2.entrypoint.args import (
    normalize as normalize_args,
)
from dpgen2.entrypoint.common import (
    global_config_workflow,
)

workflow_subcommands = [
    "terminate",
    "stop",
    "suspend",
    "delete",
    "retry",
    "resume",
]


def add_subparser_workflow_subcommand(subparsers, command: str):
    parser_cmd = subparsers.add_parser(
        command,
        help=f"{command.capitalize()} a DPGEN2 workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_cmd.add_argument("CONFIG", help="the config file in json format.")
    parser_cmd.add_argument("ID", help="the ID of the workflow.")


def execute_workflow_subcommand(
    command: str,
    wfid: str,
    wf_config: Optional[dict] = {},
):
    wf_config = normalize_args(wf_config)
    global_config_workflow(wf_config)
    wf = Workflow(id=wfid)
    getattr(wf, command)()
