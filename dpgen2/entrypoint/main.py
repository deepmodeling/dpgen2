import argparse
import json
import logging
import os
import textwrap
from typing import (
    List,
    Optional,
)

from dflow import (
    Step,
    Steps,
    Workflow,
    download_artifact,
    upload_artifact,
)

from dpgen2 import (
    __version__,
)
from dpgen2.utils.download_dpgen2_artifacts import (
    print_op_download_setting,
)

from .common import (
    expand_idx,
)
from .download import (
    download,
    download_by_def,
)
from .showkey import (
    showkey,
)
from .status import (
    status,
)
from .submit import (
    make_concurrent_learning_op,
    make_naive_exploration_scheduler,
    resubmit_concurrent_learning,
    submit_concurrent_learning,
    workflow_concurrent_learning,
)
from .watch import (
    default_watching_keys,
    watch,
)
from .workflow import (
    add_subparser_workflow_subcommand,
    execute_workflow_subcommand,
    workflow_subcommands,
)


def main_parser() -> argparse.ArgumentParser:
    """DPGEN2 commandline options argument parser.

    Notes
    -----
    This function is used by documentation.

    Returns
    -------
    argparse.ArgumentParser
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description="DPGEN2: concurrent learning workflow generating the "
        "machine learning potential energy models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    ##########################################
    # submit
    parser_run = subparsers.add_parser(
        "submit",
        help="Submit DPGEN2 workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_run.add_argument(
        "CONFIG", help="the config file in json format defining the workflow."
    )
    parser_run.add_argument(
        "-o",
        "--old-compatible",
        action="store_true",
        help="compatible with old-style input script used in dpgen2 < 0.0.6.",
    )

    ##########################################
    # resubmit
    parser_resubmit = subparsers.add_parser(
        "resubmit",
        help="Submit DPGEN2 workflow resuing steps from an existing workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_resubmit.add_argument(
        "CONFIG", help="the config file in json format defining the workflow."
    )
    parser_resubmit.add_argument("ID", help="the ID of the existing workflow.")
    parser_resubmit.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list the Steps of the existing workflow.",
    )
    parser_resubmit.add_argument(
        "-u",
        "--reuse",
        type=str,
        nargs="+",
        default=None,
        help="specify which Steps to reuse.",
    )
    parser_resubmit.add_argument(
        "-k",
        "--keep-schedule",
        action="store_true",
        help="if set then keep schedule of the old workflow. otherwise use the schedule defined in the input file",
    )
    parser_resubmit.add_argument(
        "-o",
        "--old-compatible",
        action="store_true",
        help="compatible with old-style input script used in dpgen2 < 0.0.6.",
    )

    ##########################################
    # show key
    parser_showkey = subparsers.add_parser(
        "showkey",
        help="Print the keys of the successful DPGEN2 steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_showkey.add_argument("CONFIG", help="the config file in json format.")
    parser_showkey.add_argument("ID", help="the ID of the existing workflow.")

    ##########################################
    # status
    parser_status = subparsers.add_parser(
        "status",
        help="Print the status (stage, iteration, convergence) of the  DPGEN2 workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_status.add_argument("CONFIG", help="the config file in json format.")
    parser_status.add_argument("ID", help="the ID of the existing workflow.")

    ##########################################
    # download
    parser_download = subparsers.add_parser(
        "download",
        help=("Download the artifacts of DPGEN2 steps.\n"),
        description=(
            textwrap.dedent(
                """
            Typically there are three ways of using the command

            1. list all supported steps and their input/output artifacts
            $ dpgen2 download CONFIG ID -l

            2. donwload all the input/output of all the steps.
            $ dpgen2 download CONFIG ID

            3. donwload specified input/output artifacts of certain steps. For example
            $ dpgen2 download CONFIG ID -i 0-8 8 9 -d prep-run-train/input/init_data prep-run-lmp/output/trajs

            The command will download the init_data of prep-run-train's input and trajs of the prep-run-lmp's output from iterations 0 to 9 (by -i 0-8 8 9).
            The supported step and the names of input/output can be checked by the -l flag.
            """
            )
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser_download.add_argument("CONFIG", help="the config file in json format.")
    parser_download.add_argument("ID", help="the ID of the existing workflow.")
    parser_download.add_argument(
        "-l",
        "--list-supported",
        action="store_true",
        help="list all supported steps artifacts",
    )
    parser_download.add_argument(
        "-k",
        "--keys",
        type=str,
        nargs="+",
        help="the keys of the downloaded steps. If not provided download all artifacts",
    )
    parser_download.add_argument(
        "-i",
        "--iterations",
        type=str,
        nargs="+",
        help="the iterations to be downloaded, support ranging expression as 0-10.",
    )
    parser_download.add_argument(
        "-d",
        "--step-definitions",
        type=str,
        nargs="+",
        help="the definition for downloading step artifacts",
    )
    parser_download.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="the prefix of the path storing the download artifacts",
    )
    parser_download.add_argument(
        "-n",
        "--no-check-point",
        action="store_false",
        help="if specified, download regardless whether check points exist.",
    )

    ##########################################
    # watch
    parser_watch = subparsers.add_parser(
        "watch",
        help="Watch a DPGEN2 workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_watch.add_argument("CONFIG", help="the config file in json format.")
    parser_watch.add_argument("ID", help="the ID of the existing workflow.")
    parser_watch.add_argument(
        "-k",
        "--keys",
        type=str,
        nargs="+",
        default=default_watching_keys,
        help="the subkey to watch. For example, 'prep-run-train' 'prep-run-lmp'",
    )
    parser_watch.add_argument(
        "-f",
        "--frequency",
        type=float,
        default=600.0,
        help="the frequency of workflow status query. In unit of second",
    )
    parser_watch.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="whether to download artifacts of a step when it finishes",
    )
    parser_watch.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="the prefix of the path storing the download artifacts",
    )
    parser_watch.add_argument(
        "-n",
        "--no-check-point",
        action="store_false",
        help="if specified, download regardless whether check points exist.",
    )

    # workflow subcommands
    for cmd in workflow_subcommands:
        add_subparser_workflow_subcommand(subparsers, cmd)

    # --version
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="DPGEN v%s" % __version__,
    )

    return parser


def parse_args(args: Optional[List[str]] = None):
    """DPGEN2 commandline options argument parsing.

    Parameters
    ----------
    args : List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv
    """
    parser = main_parser()

    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()

    return parsed_args


def main():
    #####################################
    # logging
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    dict_args = vars(args)

    if args.command == "submit":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        submit_concurrent_learning(
            config,
            old_style=args.old_compatible,
        )
    elif args.command == "resubmit":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        resubmit_concurrent_learning(
            config,
            wfid,
            list_steps=args.list,
            reuse=args.reuse,
            old_style=args.old_compatible,
            replace_scheduler=(not args.keep_schedule),
        )
    elif args.command == "status":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        status(
            wfid,
            config,
        )
    elif args.command == "showkey":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        showkey(
            wfid,
            config,
        )
    elif args.command == "download":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        if args.list_supported is not None and args.list_supported:
            print(print_op_download_setting())
        elif args.keys is not None:
            download(
                wfid,
                config,
                wf_keys=args.keys,
                prefix=args.prefix,
                chk_pnt=args.no_check_point,
            )
        else:
            download_by_def(
                wfid,
                config,
                iterations=(
                    expand_idx(args.iterations) if args.iterations is not None else None
                ),
                step_defs=args.step_definitions,
                prefix=args.prefix,
                chk_pnt=args.no_check_point,
            )
    elif args.command == "watch":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        watch(
            wfid,
            config,
            watching_keys=args.keys,
            frequency=args.frequency,
            download=args.download,
            prefix=args.prefix,
            chk_pnt=args.no_check_point,
        )
    elif args.command in workflow_subcommands:
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        execute_workflow_subcommand(args.command, wfid, config)
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
