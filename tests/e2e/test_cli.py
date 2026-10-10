from __future__ import annotations

from textwrap import dedent
from unittest.mock import patch

import pytest
from exo import compile_procs
from exo.main import load_user_code

from exojit import cli


def test_cli_deduplicates_exported_proc_names(tmp_path, capsys):
    source = tmp_path / "dup_kernel.py"
    source.write_text(
        dedent("""
            from exo import *
            from exo.stdlib.scheduling import fission

            @proc
            def kernel(x: f32[4] @ DRAM):
                for i in seq(0, 4):
                    x[i] = 0.0
                    for j in seq(0, 4):
                        x[i] += 1.0

            opt = fission(kernel, kernel.find("for j in _: _").before(), n_lifts=1)
            """)
    )

    compile_procs([load_user_code(source).opt], tmp_path, "o.c", "o.h")
    expected = (tmp_path / "o.c").read_text() + "\n"
    with patch("tempfile.mkdtemp", side_effect=AssertionError("C output must not create a temporary directory")):
        cli([str(source), "--c"])

    output = capsys.readouterr().out
    assert output == expected
    assert "multiple procs named" not in output
    assert output.count("void kernel") == 1
    assert output.count("for (int_fast32_t i = 0; i < 4; i++)") == 2
    assert output.count("for (int_fast32_t j = 0; j < 4; j++)") == 1


@pytest.mark.parametrize("flags", [[], ["--c", "--mlir"]])
def test_cli_requires_one_output_format(tmp_path, flags):
    with patch("exojit.load_user_code") as load, pytest.raises(SystemExit, match="choose exactly one of --c or --mlir"):
        cli([str(tmp_path / "kernel.py"), *flags])
    load.assert_not_called()


@pytest.mark.parametrize("directory", [False, True])
def test_cli_rejects_invalid_source(tmp_path, directory):
    source = tmp_path if directory else tmp_path / "missing.py"
    with patch("exojit.load_user_code") as load, pytest.raises(SystemExit, match="source must be an existing file"):
        cli([str(source), "--c"])
    load.assert_not_called()


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "exojit" in output
    assert "--c" in output
    assert "--mlir" in output
    assert "Output C source" in output
    assert "Output MLIR" in output
