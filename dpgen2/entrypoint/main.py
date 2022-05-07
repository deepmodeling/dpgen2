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
)
from dpgen2 import (
    __version__
)

def parse_args(args: Optional[List[str]] = None):
    """DPGEN2 commandline options argument parser.

    Parameters
    ----------
    args: List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv
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
        "INPUT", help="the input file in json format defining the workflow."
    )

    # --version
    parser.add_argument(
        '--version', 
        action='version', 
        version='DPGEN v%s' % __version__,
    )

    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()

    return parsed_args
    

def main():
    args = parse_args()
    dict_args = vars(args)

    if args.command == "submit":
        with open(args.INPUT) as fp:
            config = json.load(fp)
        submit_concurrent_learning(config)
    else:
        raise RuntimeError(f"unknown command {args.command}")
        
