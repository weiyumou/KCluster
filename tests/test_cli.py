import pytest

from kcluster.cli import main


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
    assert "--concept_dir" in capsys.readouterr().out
