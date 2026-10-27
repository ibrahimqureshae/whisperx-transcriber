"""Tests for parsing pip progress lines into bytes and speed in setup_wizard.py."""
import queue
import re
import sys
from pathlib import Path

import pytest

# Make sure the project root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard


# -- parse_pip_progress: sizes and speed ---------------------------------------

def test_gb_line_with_speed_gives_decimal_bytes_and_speed():
    done, total, speed = setup_wizard.parse_pip_progress(
        "━━━━ 1.2/2.5 GB 8.1 MB/s eta 0:02:41"
    )
    assert int(round(done)) == 1_200_000_000
    assert int(round(total)) == 2_500_000_000
    assert speed == pytest.approx(8_100_000.0)


def test_kb_line_scales_bytes_and_speed():
    done, total, speed = setup_wizard.parse_pip_progress(
        "412.3/980.0 kB 1.1 MB/s eta 0:00:01"
    )
    assert int(round(done)) == 412_300
    assert int(round(total)) == 980_000
    assert speed == pytest.approx(1_100_000.0)


def test_the_unit_after_the_second_number_applies_to_both_numbers():
    done, total, speed = setup_wizard.parse_pip_progress(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.0/100.0 MB 512.0 kB/s eta 0:01:34"
    )
    assert int(round(done)) == 52_000_000
    assert int(round(total)) == 100_000_000
    assert speed == pytest.approx(512_000.0)


def test_line_with_sizes_but_no_speed_gives_none_speed():
    done, total, speed = setup_wizard.parse_pip_progress("━━━━ 0.5/1.0 MB eta 0:00:03")
    assert int(round(done)) == 500_000
    assert int(round(total)) == 1_000_000
    assert speed is None


def test_binary_units_use_1024_multiples():
    result = setup_wizard.parse_pip_progress("━━━━ 1.5/3.0 GiB")
    assert result is not None
    done, total, speed = result
    assert int(round(done)) == 1_610_612_736
    assert int(round(total)) == 3_221_225_472
    assert speed is None


def test_mib_and_kib_use_1024_multiples():
    done, total, _ = setup_wizard.parse_pip_progress("━━ 2.0/8.0 MiB")
    assert int(round(done)) == 2 * 1024 * 1024
    assert int(round(total)) == 8 * 1024 * 1024
    done, total, _ = setup_wizard.parse_pip_progress("━━ 1.0/4.0 KiB")
    assert int(round(done)) == 1024
    assert int(round(total)) == 4 * 1024


# -- parse_pip_progress: lines that are not progress ---------------------------

@pytest.mark.parametrize("line", [
    "Downloading torch-2.8.0+cu128-cp311-cp311-win_amd64.whl (2.5 GB)",
    "",
    "   ",
    "Collecting numpy",
    "Installing collected packages: numpy, torch",
    "Successfully installed numpy-2.0.0",
])
def test_non_progress_lines_give_none(line):
    assert setup_wizard.parse_pip_progress(line) is None


# -- human_bytes / human_speed -------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (950, "950 B"),
    (1_200_000, "1.2 MB"),
    (2_500_000_000, "2.5 GB"),
])
def test_human_bytes_formats_decimal_units(n, expected):
    assert setup_wizard.human_bytes(n) == expected


def test_human_speed_formats_megabytes_per_second():
    assert setup_wizard.human_speed(8_100_000) == "8.1 MB/s"


# -- _maybe_emit_progress ------------------------------------------------------

def _wizard_with_queue():
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    wiz._q = queue.Queue()
    return wiz


def _drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_maybe_emit_progress_queues_bytes_and_speed_tuple():
    wiz = _wizard_with_queue()
    wiz._maybe_emit_progress("━━━━ 1.2/2.5 GB 8.1 MB/s eta 0:02:41")
    items = _drain(wiz._q)
    assert len(items) == 1
    kind, payload = items[0]
    assert kind == "p"
    done, total, speed = payload
    assert int(round(done)) == 1_200_000_000
    assert int(round(total)) == 2_500_000_000
    assert speed == pytest.approx(8_100_000.0)


def test_maybe_emit_progress_queues_none_speed_when_line_has_no_speed():
    wiz = _wizard_with_queue()
    wiz._maybe_emit_progress("━━━━ 1.5/3.0 GiB")
    items = _drain(wiz._q)
    assert len(items) == 1
    kind, (done, total, speed) = items[0]
    assert kind == "p"
    assert int(round(done)) == 1_610_612_736
    assert int(round(total)) == 3_221_225_472
    assert speed is None


@pytest.mark.parametrize("line", [
    "Downloading torch-2.8.0+cu128-cp311-cp311-win_amd64.whl (2.5 GB)",
    "Collecting numpy",
    "",
])
def test_maybe_emit_progress_queues_nothing_for_non_progress_line(line):
    wiz = _wizard_with_queue()
    wiz._maybe_emit_progress(line)
    assert _drain(wiz._q) == []


# -- _set_install_progress -----------------------------------------------------

class _FakeBar:
    def __init__(self):
        self.set_values = []
        self.configure_calls = []

    def set(self, value):
        self.set_values.append(value)

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)


class _FakeLabel:
    def __init__(self):
        self.configure_calls = []

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)

    @property
    def text(self):
        texts = [c["text"] for c in self.configure_calls if "text" in c]
        return texts[-1] if texts else None


@pytest.mark.parametrize("done,total,speed,pct", [
    (3_000_000, 4_000_000, 8_100_000.0, 75),
    (1_000_000, 4_000_000, None, 25),
])
def test_set_install_progress_accepts_tuple_and_sets_bar_and_percent(done, total, speed, pct):
    wiz = setup_wizard.SetupWizard.__new__(setup_wizard.SetupWizard)
    wiz._prog = _FakeBar()
    wiz._pct_lbl = _FakeLabel()

    wiz._set_install_progress((done, total, speed))

    assert wiz._prog.set_values, "progress bar was never set"
    assert wiz._prog.set_values[-1] == pytest.approx(done / total)
    assert wiz._pct_lbl.text is not None
    assert re.search(rf"(?<!\d){pct}\s*%", wiz._pct_lbl.text)
