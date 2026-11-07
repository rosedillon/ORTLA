"""Command-line interface for orta.

Each sub-command is driven by a YAML config file describing the run; see
the ``configs/`` directory for examples. Run
``orta <command>`` to see the config argument for that
command.
"""

from __future__ import annotations

import argparse
import sys

import caf.toolkit as ctk

import orta
from orta import config


def create_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser, one sub-command per model."""
    parser = argparse.ArgumentParser(
        prog="orta",
        description=__doc__,
        formatter_class=ctk.arguments.TidyUsageArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    for name, (model, help_text) in config.COMMANDS.items():
        model_args = ctk.arguments.ModelArguments(model)
        sub = subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
            formatter_class=ctk.arguments.TidyUsageArgumentDefaultsHelpFormatter,
        )
        model_args.add_config_arguments(sub)
    return parser


def parse_args(argv: list[str] | None = None) -> config.BaseRunConfig:
    """Parse command-line arguments into a runnable config.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list (defaults to ``sys.argv``). Shows help when empty.

    Returns
    -------
    config.BaseRunConfig
        The loaded, validated config for the chosen sub-command.
    """
    parser = create_arg_parser()
    if argv is None:
        # No arguments given -> show the help text instead of erroring
        # out.
        argv = sys.argv[1:] if len(sys.argv[1:]) > 0 else ["-h"]
    args = parser.parse_args(argv)
    # caf.toolkit attaches this function; it loads the YAML file named
    # on the command line and validates it into the config object for
    # that command.
    parameters = args.dataclass_parse_func(args)
    if not isinstance(parameters, config.BaseRunConfig):
        raise TypeError("Parsed config is not a BaseRunConfig.")
    return parameters


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, set up logging and run the chosen command."""
    parameters = parse_args(argv)
    details = ctk.ToolDetails(name="orta", version=orta.__version__)
    # LogHelper sets up logging to both the screen and a log file for
    # the run; everything inside the `with` block is captured. run()
    # does the real work.
    with ctk.LogHelper("orta", details, log_file=parameters.logging_path):
        parameters.run()


if __name__ == "__main__":
    main()
