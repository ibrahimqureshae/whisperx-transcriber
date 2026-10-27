"""Tests for torch index selection in setup_wizard.py."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make sure the project root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard

CU128 = "https://download.pytorch.org/whl/cu128"
CU126 = "https://download.pytorch.org/whl/cu126"


# -- torch_index_for_cuda ------------------------------------------------------

@pytest.mark.parametrize("major,minor", [(12, 8), (13, 0)])
def test_cuda_12_8_and_newer_maps_to_cu128(major, minor):
    assert setup_wizard.torch_index_for_cuda(major, minor) == CU128


@pytest.mark.parametrize("major,minor", [(12, 6), (12, 7)])
def test_cuda_12_6_and_12_7_map_to_cu126(major, minor):
    assert setup_wizard.torch_index_for_cuda(major, minor) == CU126


@pytest.mark.parametrize("major,minor", [(12, 4), (12, 1), (11, 8)])
def test_cuda_below_12_6_returns_none(major, minor):
    assert setup_wizard.torch_index_for_cuda(major, minor) is None


# -- _cuda_torch_index ---------------------------------------------------------

def _patch_run(monkeypatch, returncode=0, stdout=""):
    def fake(*args, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(setup_wizard.subprocess, "run", fake)


def test_nvidia_smi_cuda_12_4_returns_none(monkeypatch):
    _patch_run(monkeypatch, stdout="| NVIDIA-SMI 550  CUDA Version: 12.4  |")
    assert setup_wizard._cuda_torch_index() is None


def test_nvidia_smi_cuda_13_0_returns_cu128(monkeypatch):
    _patch_run(monkeypatch, stdout="| NVIDIA-SMI 580  CUDA Version: 13.0 |")
    assert setup_wizard._cuda_torch_index() == CU128


def test_nvidia_smi_610_umd_header_returns_cu128(monkeypatch):
    # driver 610 header, as seen on an RTX 4060 laptop on 26 Oct 2026
    _patch_run(monkeypatch, stdout="| NVIDIA-SMI 610.62                 KMD Version: 610.62        "
                                   "CUDA UMD Version: 13.3     |")
    assert setup_wizard._cuda_torch_index() == CU128


def test_nvidia_smi_umd_header_old_cuda_returns_none(monkeypatch):
    _patch_run(monkeypatch, stdout="| NVIDIA-SMI 610.62  KMD Version: 610.62  CUDA UMD Version: 12.4 |")
    assert setup_wizard._cuda_torch_index() is None


def test_nvidia_smi_nonzero_exit_returns_none(monkeypatch):
    _patch_run(monkeypatch, returncode=1, stdout="CUDA Version: 13.0")
    assert setup_wizard._cuda_torch_index() is None


def test_nvidia_smi_missing_returns_none(monkeypatch):
    def fake(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(setup_wizard.subprocess, "run", fake)
    assert setup_wizard._cuda_torch_index() is None


# -- right-to-left packages in the install --------------------------------------

RTL_SPECS = ("python-bidi>=0.6", "arabic-reshaper>=3.0")


def _run_install_harness(monkeypatch, tmp_path, installed, cuda_index=None, pip_rc=0):
    """Drive SetupWizard._run_install with pip faked; returns (pip_calls, logs, wiz)."""
    (tmp_path / "Scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Scripts" / "python.exe").write_bytes(b"")
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    wiz.python_path = "python"
    wiz.runtime_dir = str(tmp_path)
    wiz.win = SimpleNamespace(after=lambda *a, **k: None)
    wiz._show_done = lambda: None
    wiz._show_error = lambda msg: None
    pip_calls, logs = [], []
    wiz._log = lambda msg: logs.append(msg)
    wiz._set_status = lambda msg: None
    wiz._set_progress = lambda *a, **k: None
    wiz._check_gpu = lambda venv_py: None
    wiz._pkg_installed = lambda venv_pip, pkg: pkg in installed

    def fake_pip(cmd):
        pip_calls.append(list(cmd))
        if "show" in cmd:
            return 0
        return pip_rc
    wiz._run_pip = fake_pip
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: cuda_index)
    monkeypatch.setattr(setup_wizard.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="12\n", stderr=""))
    wiz._run_install()
    return pip_calls, logs, wiz


def _main_install_cmd(pip_calls):
    return next(c for c in pip_calls if "whisperx" in c and "install" in c)


@pytest.mark.parametrize("cuda_index", [None, CU128])
def test_fresh_install_command_includes_rtl_packages(monkeypatch, tmp_path, cuda_index):
    pip_calls, _, _ = _run_install_harness(
        monkeypatch, tmp_path, installed={"imageio-ffmpeg"}, cuda_index=cuda_index)
    cmd = _main_install_cmd(pip_calls)
    for spec in RTL_SPECS:
        assert spec in cmd


def test_existing_install_without_bidi_adds_both_in_one_pip_call(monkeypatch, tmp_path):
    installed = {"whisperx", "tkinterdnd2", "sherpa-onnx", "imageio-ffmpeg"}
    pip_calls, logs, _ = _run_install_harness(monkeypatch, tmp_path, installed)
    rtl = [c for c in pip_calls if any(s in c for s in RTL_SPECS)]
    assert len(rtl) == 1
    assert all(s in rtl[0] for s in RTL_SPECS) and "install" in rtl[0]
    assert any("right-to-left" in m.lower() or "bidi" in m.lower() for m in logs)


def test_existing_install_with_both_rtl_packages_skips_rtl_pip(monkeypatch, tmp_path):
    installed = {"whisperx", "tkinterdnd2", "sherpa-onnx", "imageio-ffmpeg", "python-bidi", "arabic-reshaper"}
    pip_calls, _, _ = _run_install_harness(monkeypatch, tmp_path, installed)
    assert not [c for c in pip_calls if any(s in c for s in RTL_SPECS)]


def test_existing_install_missing_only_reshaper_adds_both(monkeypatch, tmp_path):
    installed = {"whisperx", "tkinterdnd2", "sherpa-onnx", "imageio-ffmpeg", "python-bidi"}
    pip_calls, _, _ = _run_install_harness(monkeypatch, tmp_path, installed)
    rtl = [c for c in pip_calls if any(s in c for s in RTL_SPECS)]
    assert len(rtl) == 1 and all(s in rtl[0] for s in RTL_SPECS)


def test_rtl_pip_failure_is_a_warning_not_an_error(monkeypatch, tmp_path):
    installed = {"whisperx", "tkinterdnd2", "sherpa-onnx", "imageio-ffmpeg"}
    pip_calls, logs, wiz = _run_install_harness(monkeypatch, tmp_path, installed, pip_rc=1)
    assert not any("ERROR" in m for m in logs)
    assert any("warning" in m.lower() and ("right-to-left" in m.lower() or "bidi" in m.lower())
               for m in logs)
    assert (tmp_path / ".setup_complete").exists()

def test_install_failure_reaches_the_error_page(monkeypatch, tmp_path):
    """Tk runs after() callbacks once the except block has ended; the message must survive that."""
    from types import SimpleNamespace
    (tmp_path / "Scripts").mkdir()
    (tmp_path / "Scripts" / "python.exe").write_bytes(b"")
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    wiz.python_path, wiz.runtime_dir = "python", str(tmp_path)
    later, shown = [], []
    wiz.win = SimpleNamespace(after=lambda ms, fn=None, *a: later.append(fn))
    wiz._log = wiz._set_status = lambda *a, **k: None
    wiz._set_progress = lambda *a, **k: None
    wiz._show_error = shown.append
    wiz._pkg_installed = lambda venv_pip, pkg: False

    def boom(cmd):
        raise OSError("no network")
    wiz._run_pip = boom
    monkeypatch.setattr(setup_wizard, "_cuda_torch_index", lambda: None)
    monkeypatch.setattr(setup_wizard.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    wiz._run_install()
    for fn in later:
        if fn is not None:
            fn()
    assert shown and "no network" in shown[0]
