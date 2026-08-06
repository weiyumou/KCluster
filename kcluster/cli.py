"""Command-line entry point.

Subcommands are ported from the legacy repositories in phases; until they
land, this stub only reports the version and prints help.
"""

import argparse

from kcluster import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kcluster",
        description="LLM-based knowledge component discovery and question generation",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
