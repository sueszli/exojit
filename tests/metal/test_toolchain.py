import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import backend_metal

XCODE = "/Applications/Xcode.app/Contents/Developer"
XCODE_BETA = "/Applications/Xcode-beta.app/Contents/Developer"
CUSTOM = "/opt/xcode/Xcode.app/Contents/Developer"
LLVM_IR = "define void @copy(float* %a, float* %out, i32 %id) {\nentry:\n  ret void\n}\n"


def discover(environ: dict[str, str], usable: set[str | None], apps: tuple[str, ...] = ("/Applications/Xcode-beta.app", "/Applications/Xcode.app")) -> tuple[dict[str, str], list[str | None], dict[str, str]]:
    # runs the uncached discovery against a fake xcrun that finds metal only for the developer dirs in `usable` (None: the xcode-select default)
    probes: list[str | None] = []

    def run(cmd, **kwargs):
        assert cmd == ["xcrun", "-sdk", "macosx", "metal", "--version"], cmd
        probes.append(kwargs["env"].get("DEVELOPER_DIR"))
        return SimpleNamespace(returncode=0 if probes[-1] in usable else 1)

    with patch.dict(os.environ, environ, clear=True), patch("backend_metal.subprocess.run", run), patch("backend_metal.glob", return_value=list(apps)):
        return backend_metal.metal_toolchain_env.__wrapped__(), probes, dict(os.environ)


def test_xcode_select_default_with_metal_is_used_as_is():
    env, probes, environ = discover({"PATH": "/usr/bin"}, usable={None})
    assert probes == [None] and "DEVELOPER_DIR" not in env and env["PATH"] == "/usr/bin"
    assert env["MACOSX_DEPLOYMENT_TARGET"] == "15.0" and environ == {"PATH": "/usr/bin"}  # air 2.7 target, process env untouched


def test_command_line_tools_fall_back_to_xcode_app_first():
    env, probes, environ = discover({"PATH": "/usr/bin"}, usable={XCODE, XCODE_BETA})
    assert env["DEVELOPER_DIR"] == XCODE and probes == [None, XCODE]
    assert "DEVELOPER_DIR" not in environ and "MACOSX_DEPLOYMENT_TARGET" not in environ


def test_command_line_tools_fall_back_to_other_xcode_apps():
    env, probes, _ = discover({}, usable={XCODE_BETA})
    assert env["DEVELOPER_DIR"] == XCODE_BETA and probes == [None, XCODE, XCODE_BETA]


def test_explicit_developer_dir_and_deployment_target_are_respected():
    env, probes, _ = discover({"DEVELOPER_DIR": CUSTOM, "MACOSX_DEPLOYMENT_TARGET": "26.0"}, usable={CUSTOM, XCODE})
    assert env["DEVELOPER_DIR"] == CUSTOM and env["MACOSX_DEPLOYMENT_TARGET"] == "26.0" and probes == [CUSTOM]


def test_unusable_explicit_developer_dir_fails_without_fallback():
    with pytest.raises(AssertionError, match=f"no usable metal compiler in DEVELOPER_DIR={CUSTOM}"):
        discover({"DEVELOPER_DIR": CUSTOM}, usable={XCODE})


def test_missing_full_xcode_fails_clearly():
    with pytest.raises(AssertionError, match="no full Xcode with the Metal Toolchain found in /Applications"):
        discover({}, usable=set(), apps=())


@pytest.fixture
def fresh_datalayout():
    # get_target_datalayout is cached, a hit would hide the subprocess call (or skip a patched probe) under test
    backend_metal.get_target_datalayout.cache_clear()
    yield
    backend_metal.get_target_datalayout.cache_clear()


SELECTED_ENV = {"DEVELOPER_DIR": XCODE, "MACOSX_DEPLOYMENT_TARGET": "15.0"}


def probe(stdout: str = "Apple metal version 99.1 (metalfe-99.1)\nTarget: air64-apple-darwin\n") -> tuple[str, str, list[tuple[list[str], dict]]]:
    # runs both metadata probes on a mocked mac with metal on PATH and a selected full xcode
    calls: list[tuple[list[str], dict]] = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        Path(cmd[cmd.index("-o") + 1]).write_text('target datalayout = "e-selected"\n')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("backend_metal.platform.system", return_value="Darwin"), patch("backend_metal.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"), patch("backend_metal.metal_toolchain_env", return_value=SELECTED_ENV), patch("backend_metal.subprocess.run", run):
        return backend_metal.get_metal_version(), backend_metal.get_target_datalayout(), calls


def test_metadata_probes_use_selected_toolchain_env(fresh_datalayout):
    environ = dict(os.environ)
    version, layout, calls = probe()
    assert version == "Apple metal version 99.1 (metalfe-99.1)" and layout == "e-selected"
    assert [cmd[:4] for cmd, _ in calls] == [["xcrun", "-sdk", "macosx", "metal"]] * 2
    assert calls[0][0][-1] == "--version" and "-emit-llvm" in calls[1][0]
    assert all(kwargs["env"] is SELECTED_ENV and not kwargs.get("shell", False) for _, kwargs in calls)
    assert dict(os.environ) == environ  # process env untouched


@pytest.mark.parametrize("stdout", ["", "\n", "clang version 1\n"])
def test_unusable_metal_version_output_falls_back(stdout, fresh_datalayout):
    version, _, calls = probe(stdout)
    assert version == "Apple metal version 32023.830 (metalfe-32023.830.2)" and calls[0][1]["env"] is SELECTED_ENV


def test_off_mac_probes_ignore_metal_on_path(fresh_datalayout):
    # an unrelated `metal` executable on a non-darwin PATH must not be probed or select a toolchain
    with patch("backend_metal.platform.system", return_value="Linux"), patch("backend_metal.shutil.which", return_value="/usr/local/bin/metal"), patch("backend_metal.metal_toolchain_env", side_effect=AssertionError("unexpected toolchain lookup")), patch("backend_metal.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
        assert backend_metal.get_metal_version() == "Apple metal version 32023.830 (metalfe-32023.830.2)"
        assert backend_metal.get_target_datalayout().startswith("e-p:64:64:64-") and backend_metal.get_target_datalayout().endswith("-n8:16:32")
        air = backend_metal.to_air(LLVM_IR)
    assert '!"Apple metal version 32023.830 (metalfe-32023.830.2)"' in air


def test_off_mac_translation_never_invokes_xcrun(fresh_datalayout):
    # no xcrun or metal on PATH, and any subprocess fails the test
    with patch("backend_metal.shutil.which", return_value=None), patch("backend_metal.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
        air = backend_metal.to_air(LLVM_IR)
        assert 'target triple = "air64_v27-apple-macosx' in air and '!"air.thread_position_in_grid"' in air and '!"air.read_write"' in air
        with pytest.raises(AssertionError, match="xcrun not found"):
            backend_metal.compile_to_metallib(air)
