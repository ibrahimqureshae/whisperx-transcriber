import inspect
import sys
import tkinter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rrect_draws_antialiased_circles_rectangles_with_edge_first(monkeypatch):
    import customtkinter as ctk
    from ui import theme

    assert str(inspect.signature(theme.rrect)) == "(cv, x0, y0, x1, y1, r, fill, edge=None, b=1, tags=())"

    try:
        root = ctk.CTk()
    except tkinter.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    try:
        cv = ctk.CTkCanvas(root)
        calls = []

        def record(kind):
            def inner(*args, **kwargs):
                calls.append((kind, args, kwargs))
                return len(calls)
            return inner

        monkeypatch.setattr(cv, "create_aa_circle", record("aa_circle"))
        monkeypatch.setattr(cv, "create_rectangle", record("rectangle"))
        monkeypatch.setattr(cv, "create_polygon", record("polygon"))

        theme.rrect(cv, 10, 20, 80, 60, 9, fill="#111111", edge="#eeeeee", b=2, tags=("box",))

        assert calls
        assert {kind for kind, _args, _kwargs in calls} <= {"aa_circle", "rectangle"}
        fills = [kwargs["fill"] for _kind, _args, kwargs in calls]
        assert fills[:6] == ["#eeeeee"] * 6
        assert fills[6:] == ["#111111"] * 6
        assert all(kwargs.get("tags") == ("box",) for _kind, _args, kwargs in calls)
    finally:
        root.destroy()


def test_no_smooth_true_polygons_remain_under_ui():
    root = Path(__file__).parent.parent
    offenders = []
    for path in (root / "ui").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "smooth=True" in text:
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


def test_wave_returns_requested_output_size():
    from ui.waveform import _wave

    on, off = _wave([0.0, 0.5, 1.0, 0.25], 300, 60, 4, "#ffffff", "#777777", "#111111", "#000000")

    assert on.size == (300, 60)
    assert off.size == (300, 60)
