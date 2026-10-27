"""SessionLane draws its corner timecodes and playhead tag with the scaled tk font tuple.

A plain canvas ignores CustomTkinter's widget scale, so a CTkFont renders at design size."""
import tkinter.font as tkf

import customtkinter as ctk
import pytest

from ui import theme
from ui.theme import C
from ui.waveform import SessionLane


@pytest.fixture(params=[1.0, 1.75], ids=["scale1.0", "scale1.75"])
def lane(request):
    ctk.set_widget_scaling(request.param)
    root = ctk.CTk()
    try:
        root.geometry("400x200")
        ln = SessionLane(root)
        ln.set_bins([0.2, 0.8, 0.5, 1.0] * 20)
        ln.set_duration(120)
        ln.pack(fill="both", expand=True)
        root.update()
        yield root, ln
    finally:
        root.destroy()
        ctk.set_widget_scaling(1.0)


def _actual(root, spec):
    return tkf.Font(root=root, font=spec).actual()


def _same(root, a, b):
    fa, fb = _actual(root, a), _actual(root, b)
    return fa["family"] == fb["family"] and fa["size"] == fb["size"]


def _timecode_items(ln):
    return [i for i in ln.find_all()
            if ln.type(i) == "text" and ln.itemcget(i, "text") in ("0:00", "2:00")]


def test_lane_has_real_geometry(lane):
    _, ln = lane
    assert ln.winfo_width() >= 300 and ln.winfo_height() >= 96


def test_corner_timecodes_use_scaled_tk_font(lane):
    root, ln = lane
    items = _timecode_items(ln)
    assert sorted(ln.itemcget(i, "text") for i in items) == ["0:00", "2:00"]
    want = theme._tkfont("mono-sm", ln)
    for i in items:
        assert _same(root, ln.itemcget(i, "font"), want)


def test_playhead_tag_uses_scaled_tk_font(lane):
    root, ln = lane
    assert _same(root, ln.itemcget(ln._tag, "font"), theme._tkfont("mono-sm", ln))


def test_timecode_font_is_not_the_ctk_font(lane):
    root, ln = lane
    if theme._k(ln) == 1.0:
        pytest.skip("scaled and unscaled fonts coincide at 1.0")
    unscaled = theme.font("mono-sm")
    for i in _timecode_items(ln) + [ln._tag]:
        assert not _same(root, ln.itemcget(i, "font"), unscaled)


# ── R5: corner timecodes sit above the wave and yield to the playhead tag ─────
@pytest.fixture(params=[0.85, 1.0, 1.5], ids=["scale0.85", "scale1.0", "scale1.5"])
def wide(request):
    ctk.set_widget_scaling(request.param)
    root = ctk.CTk()
    try:
        root.geometry("1100x260")
        ln = SessionLane(root)
        ln.set_bins([0.2, 0.8, 0.5, 1.0] * 20)
        ln.set_duration(120)
        ln.pack(fill="both", expand=True)
        root.update()
        yield root, ln
    finally:
        root.destroy()
        ctk.set_widget_scaling(1.0)


def _corners(ln):
    """(left, right) corner timecode items, found by their text."""
    def pick(s):    # the playhead tag can read the same text at 0 % or 100 %
        return [i for i in ln.find_all() if i != ln._tag and ln.type(i) == "text" and ln.itemcget(i, "text") == s]
    left, right = pick("0:00"), pick("2:00")
    return left, right


def _hidden(ln, item):
    return ln.itemcget(item, "state") == "hidden"


def _extent(ln, item):
    """Horizontal extent of a text item, measured even while it is hidden."""
    was = ln.itemcget(item, "state")
    ln.itemconfigure(item, state="normal")
    b = ln.bbox(item)
    ln.itemconfigure(item, state=was)
    return b[0], b[2]


def _above_wave(ln, item):
    order = list(ln.find_all())
    return order.index(item) > order.index(ln._img)


def _check_stacking(ln):
    left, right = _corners(ln)
    assert len(left) == 1 and len(right) == 1
    assert _above_wave(ln, left[0])
    assert _above_wave(ln, right[0])


def test_corner_timecodes_above_wave_after_render(wide):
    _, ln = wide
    _check_stacking(ln)


@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_corner_timecodes_above_wave_after_paint(wide, frac):
    root, ln = wide
    ln.set_progress(frac, snap=True)
    root.update()
    _check_stacking(ln)
    ln._paint(frac)
    _check_stacking(ln)


def test_corner_timecodes_above_wave_after_progress_sweep(wide):
    root, ln = wide
    for frac in (0.1, 0.4, 0.7, 1.0, 0.3, 0.0):
        ln._paint(frac)
        _check_stacking(ln)


def test_both_corners_shown_at_zero(wide):
    root, ln = wide
    ln.set_progress(0.0, snap=True)
    root.update()
    left, right = _corners(ln)
    assert not _hidden(ln, left[0])
    assert not _hidden(ln, right[0])


def test_both_corners_shown_at_half_on_wide_lane(wide):
    root, ln = wide
    ln.set_progress(0.5, snap=True)
    root.update()
    left, right = _corners(ln)
    assert not _hidden(ln, left[0])
    assert not _hidden(ln, right[0])


def test_right_corner_hidden_and_left_shown_at_full(wide):
    root, ln = wide
    ln.set_progress(1.0, snap=True)
    root.update()
    left, right = _corners(ln)
    assert _hidden(ln, right[0])
    assert not _hidden(ln, left[0])


def test_right_corner_returns_when_playhead_backs_off(wide):
    root, ln = wide
    ln.set_progress(1.0, snap=True)
    root.update()
    ln.set_progress(0.5, snap=True)
    root.update()
    left, right = _corners(ln)
    assert not _hidden(ln, right[0])
    assert not _hidden(ln, left[0])


def test_left_corner_hidden_when_tag_covers_it(wide):
    """Playhead just right of the start: its tag sits over the left corner time."""
    root, ln = wide
    ln.set_progress(0.002, snap=True)
    root.update()
    left, right = _corners(ln)
    assert _hidden(ln, left[0])
    assert not _hidden(ln, right[0])


def test_corner_visibility_follows_tag_clearance_rule(wide):
    """Hidden iff the tag overlaps the corner time or is within 8 scaled px of it.
    Positions within 3 px of the threshold are skipped so rounding choices don't matter."""
    root, ln = wide
    gap = 8 * theme._k(ln)
    checked_hidden = checked_shown = 0
    for n in range(1, 100):
        frac = n / 100
        ln.set_progress(frac, snap=True)
        root.update()
        left, right = _corners(ln)
        t0, t1 = ln.bbox(ln._tag)[0], ln.bbox(ln._tag)[2]
        for item in (left[0], right[0]):
            c0, c1 = _extent(ln, item)
            clearance = max(c0 - t1, t0 - c1)      # negative while they overlap
            if abs(clearance - gap) < 3:
                continue
            if clearance < gap:
                assert _hidden(ln, item), (frac, clearance, gap)
                checked_hidden += 1
            else:
                assert not _hidden(ln, item), (frac, clearance, gap)
                checked_shown += 1
    assert checked_hidden and checked_shown


def test_corner_times_keep_their_text_and_position(wide):
    """Hiding must not move or retype the corner times."""
    root, ln = wide
    left, right = _corners(ln)
    before = [ln.coords(left[0]), ln.coords(right[0])]
    for frac in (1.0, 0.002, 0.5):
        ln.set_progress(frac, snap=True)
        root.update()
    left, right = _corners(ln)
    assert [ln.coords(left[0]), ln.coords(right[0])] == before
    assert ln.itemcget(left[0], "text") == "0:00"
    assert ln.itemcget(right[0], "text") == "2:00"


def test_corner_times_have_well_backing_between_wave_and_text(wide):
    root, ln = wide
    pad = 4 * theme._k(ln)
    for frac in (0.0, 0.5, 1.0, 0.002):
        ln.set_progress(frac, snap=True)
        root.update()
        for back, txt, _, _ in ln._corners:
            order = list(ln.find_all())
            assert order.index(ln._img) < order.index(back) < order.index(txt)
            assert ln.itemcget(back, "fill") == C["well"]
            assert ln.itemcget(back, "state") == ln.itemcget(txt, "state")
            c0, c1 = _extent(ln, txt)
            b = ln.coords(back)
            assert b[0] <= c0 - pad + 1 and b[2] >= c1 + pad - 1
