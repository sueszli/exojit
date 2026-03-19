from __future__ import annotations

from textwrap import dedent

from click.testing import CliRunner

from exojit.main import cli


def test_cli_deduplicates_exported_proc_names(tmp_path):
    source = tmp_path / "dup_kernel.py"
    source.write_text(dedent("""
            from exo import *
            from exo.stdlib.scheduling import fission

            @proc
            def kernel(x: f32[4] @ DRAM):
                for i in seq(0, 4):
                    x[i] = 0.0
                    for j in seq(0, 4):
                        x[i] += 1.0

            opt = fission(kernel, kernel.find("for j in _: _").before(), n_lifts=1)
            """))

    result = CliRunner().invoke(cli, [str(source), "--c"])

    assert result.exit_code == 0, result.output
    assert "multiple procs named" not in result.output
    assert result.output.count("void kernel") == 1
    assert result.output.count("for (int_fast32_t i = 0; i < 4; i++)") == 2
    assert result.output.count("for (int_fast32_t j = 0; j < 4; j++)") == 1


def test_cli_asm_after_module_level_jit(tmp_path):
    source = tmp_path / "jit_then_asm.py"
    source.write_text(dedent("""
            from exo import *

            from exojit.main import jit

            @proc
            def fill(x: f32[4] @ DRAM):
                for i in seq(0, 4):
                    x[i] = 1.0

            buf = [0.0, 0.0, 0.0, 0.0]
            jit(fill)(buf)
            """))

    result = CliRunner().invoke(cli, [str(source), "--asm"])

    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    assert ".globl" in result.output
