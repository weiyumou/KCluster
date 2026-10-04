import argparse
import ast
import importlib
import inspect

import pytest

from kcluster.cli import COMMANDS, main


def test_help_lists_all_commands(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    for command in ("concept", "pmi", "build-kc", "build-datashop-kc", "refine-datashop-kc"):
        assert command in out


def test_no_args_prints_help(capsys):
    main([])
    assert "usage: kcluster" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "kcluster" in capsys.readouterr().out


def test_unknown_command_exits_with_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert excinfo.value.code == 2
    assert "invalid command" in capsys.readouterr().err


def test_subcommand_help_via_lazy_dispatch(capsys):
    # build-kc has no torch dependency at module level, so this exercises
    # the lazy import path end to end.
    with pytest.raises(SystemExit) as excinfo:
        main(["build-kc", "--help"])
    assert excinfo.value.code == 0
    assert "--result_dir" in capsys.readouterr().out


@pytest.mark.parametrize("name, module_path", [(name, module_path) for name, module_path, _ in COMMANDS])
def test_a_command_reads_only_the_options_its_parser_defines(name, module_path):
    """Commands read ``args.<option>`` directly, so an option the parser does
    not define is an AttributeError at run time — and for the commands that
    need a GPU, no offline test would reach it."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        pytest.skip(f"{name} needs an optional extra: {e}")

    parser = argparse.ArgumentParser()
    module.add_arguments(parser)
    defined = {action.dest for action in parser._actions}
    read = {node.attr for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args"}
    assert read <= defined
