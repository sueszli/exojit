from __future__ import annotations

from textwrap import dedent
from unittest.mock import patch

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
