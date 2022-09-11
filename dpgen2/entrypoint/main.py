import argparse, os, json

from dflow import (
    Workflow,
    Step,
    Steps,
    upload_artifact,
    download_artifact,
)
from typing import (
    Optional,
    List,
)
from .submit import (
    make_concurrent_learning_op,
    make_naive_exploration_scheduler,
    workflow_concurrent_learning,
    submit_concurrent_learning,
    resubmit_concurrent_learning,
)
from .status import (
    status,
)
from dpgen2 import (
    __version__
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
    
    parser_run = subparsers.add_parser(
        "submit",
        help="Submit DPGEN2 workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_run.add_argument(
        "CONFIG", help="the config file in json format defining the workflow."
    )

    parser_resubmit = subparsers.add_parser(
        "resubmit",
        help="Submit DPGEN2 workflow resuing steps from an existing workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_resubmit.add_argument(
        "CONFIG", help="the config file in json format defining the workflow."
    )
    parser_resubmit.add_argument(
        "ID", help="the ID of the existing workflow."
    )
    parser_resubmit.add_argument(
        "--list", action='store_true', help="list the Steps of the existing workflow."
    )
    parser_resubmit.add_argument(
        "--reuse", type=str, nargs='+', default=None, help="specify which Steps to reuse."
    )

    parser_status = subparsers.add_parser(
        "status",
        help="Print the status (stage, iteration, convergence) of the  DPGEN2 workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_status.add_argument(
        "CONFIG", help="the config file in json format."
    )
    parser_status.add_argument(
        "ID", help="the ID of the existing workflow."
    )

    # --version
    parser.add_argument(
        '--version', 
        action='version', 
        version='DPGEN v%s' % __version__,
    )

    return parser


def parse_args(args: Optional[List[str]] = None):
    """DPGEN2 commandline options argument parsing.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv
    """
    parser = main_parser()

    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()

    return parsed_args
    

def main():
    args = parse_args()
    dict_args = vars(args)

    if args.command == "submit":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        submit_concurrent_learning(config)
    elif args.command == "resubmit":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        resubmit_concurrent_learning(
            config, wfid, list_steps=args.list, reuse=args.reuse,
        )
    elif args.command == "status":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        status(
            wfid, config,
        )        
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
        
