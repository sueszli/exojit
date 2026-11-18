import importlib
import platform
import sys

import pytest


def import_gpu_utils(monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    # re-imports the shared gpu test helper as if running on `system` with pyobjc Metal absent
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setitem(sys.modules, "Metal", None)  # a None entry makes `import Metal` fail like a missing package
    monkeypatch.delitem(sys.modules, f"{__package__}.utils", raising=False)
    importlib.import_module(".utils", __package__)


def test_simulated_mac_fails_on_missing_metal(monkeypatch):
    # BaseException also catches pytest's Skipped, so a regression to skipping on macos fails here instead of being reported as a skip
    with pytest.raises(BaseException) as excinfo:
        import_gpu_utils(monkeypatch, "Darwin")
    assert excinfo.type is ModuleNotFoundError, f"a mac without pyobjc must fail, got {excinfo.type.__name__}: {excinfo.value}"


@pytest.mark.parametrize("system", ["Linux", "Windows", "FreeBSD"])
def test_non_mac_skips_gpu_tests_before_importing_metal(monkeypatch, system):
    with pytest.raises(pytest.skip.Exception, match="need macOS"):
        import_gpu_utils(monkeypatch, system)
