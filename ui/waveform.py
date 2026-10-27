"""Session lane and file-row thumbnail (DESIGN.md, Session block and lanes)."""
import math
import time

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from ui import theme
from ui.theme import C, Tooltip, _mix


def timecode(seconds):
    """m:ss, or h:mm:ss from one hour up."""
    s = max(0, int(seconds))
    h, m = divmod(s // 60, 60)
    return f"{h}:{m:02d}:{s % 60:02d}" if h else f"{m}:{s % 60:02d}"


# ── Smooth motion ─────────────────────────────────────────────────────────────
class Tween:
    """Eases a value toward its latest target on a 16 ms frame loop (ease-out, about 220 ms to settle;
    the follow stays smooth while new targets keep arriving). The loop stops once the value arrives,
    so nothing runs while idle. A lower target (a new file) snaps."""
    TAU = 0.07

    def __init__(self, widget, apply, value=0.0):
        self.w, self.apply, self.v, self.target = widget, apply, value, value
        self._job, self._t = None, 0.0

    def to(self, target, snap=False):
        self.target = target
        if snap or target < self.v:
            if self._job is not None:
                self.w.after_cancel(self._job)
                self._job = None
            self.v = target
            self.apply(target)
        elif self._job is None and target != self.v:
            self._t = time.perf_counter()
            self._job = self.w.after(16, self._step)

    def _step(self):
        self._job = None
        now = time.perf_counter()
        self.v += (self.target - self.v) * (1 - math.exp(-(now - self._t) / self.TAU))
        self._t = now
        if abs(self.target - self.v) < 5e-4:
            self.v = self.target
        try:
            self.apply(self.v)
            if self.v != self.target:
                self._job = self.w.after(16, self._step)
        except Exception:         # widget destroyed mid-animation
            pass


def _wave(bins, w, h, pad, on, off, bg_on, bg_off):
    """Smooth anti-aliased waveform as two w x h RGB images: heat on a tinted bed, inert on the plain bed
    (drawn at 3x and downsampled once; per frame the lane only pastes the two together)."""
    k = 3
    mask = Image.new("L", (w * k, h * k), 0)
    n = len(bins)
    if n and w > 8:
        if n > w // 2:                                    # fold so every sample is at least 2 px wide
            g = n / (w // 2)
            bins = [max(bins[int(i * g):max(int(i * g) + 1, int((i + 1) * g))]) for i in range(w // 2)]
            n = len(bins)
        half, mid = (h - 2 * pad) * k / 2, h * k / 2
        top, bot = [], []
        for x in range(w * k):
            f = x / (w * k - 1) * (n - 1)
            i = min(int(f), n - 2) if n > 1 else 0
            t = (1 - math.cos(math.pi * (f - i))) / 2 if n > 1 else 0
            v = bins[i] + ((bins[i + 1] if n > 1 else bins[i]) - bins[i]) * t
            a = max(0.8 * k, half * max(0.0, min(1.0, v)))
            top.append((x, mid - a)); bot.append((x, mid + a))
        ImageDraw.Draw(mask).polygon(top + bot[::-1], fill=255)
    mask = mask.resize((w, h), Image.Resampling.LANCZOS)
    solid = lambda col: Image.new("RGB", (w, h), col)
    return Image.composite(solid(on), solid(bg_on), mask), Image.composite(solid(off), solid(bg_off), mask)


class _Wave(ctk.CTkCanvas):
    """Canvas that shows a waveform image whose heat part follows an eased progress."""

    def _wave_set(self, bins, x0, y0, x1, y1, pad, tint):
        self._box = (x0, y0, x1, y1)
        w, h = x1 - x0, y1 - y0
        self._on, self._off = _wave(bins, w, h, pad, C["heat"], C["inert-fill"],
                                    _mix(C["heat"], C["well"], 0.14) if tint else C["well"], C["well"])
        self._photo = ImageTk.PhotoImage(self._off, master=self)
        self._img = self.create_image(x0, y0, anchor="nw", image=self._photo)

    def _wave_paint(self, px):
        x0, y0, x1, y1 = self._box
        k = max(0, min(x1 - x0, int(px - x0)))
        frame = self._off.copy()
        if k:
            frame.paste(self._on.crop((0, 0, k, y1 - y0)), (0, 0))
        self._photo.paste(frame)
# ── Fixed-time glide and cut labels ───────────────────────────────────────────
class Glide:
    """Glides a value to its target in either direction (ease-out, ~200 ms; unlike Tween, a lower target glides too) on a Tk frame loop that runs only while moving.
    `apply(value)` is called every frame and once with the exact target."""

    def __init__(self, widget, apply, value=0.0, ms=200, step=20):
        self.w, self.apply, self.value, self.ms, self.step = widget, apply, float(value), ms, step
        self._job = self._from = self._t0 = None
        self.target = self.value

    @property
    def running(self):
        return self._job is not None

    def to(self, target, snap=False):
        target = float(target)
        if self.running and target == self.target:
            return
        self.stop()
        self.target = target
        if snap or abs(target - self.value) < 1e-4:
            self.value = target
            self.apply(target)
            return
        self._from, self._t0 = self.value, time.perf_counter()
        self._tick()

    def stop(self):
        if self._job is not None:
            try:
                self.w.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _tick(self):
        k = min(1.0, (time.perf_counter() - self._t0) * 1000 / self.ms)
        self.value = self.target if k >= 1 else self._from + (self.target - self._from) * (1 - (1 - k) ** 3)
        self.apply(self.value)
        self._job = None if k >= 1 else self.w.after(self.step, self._tick)


class ClipTip(Tooltip):
    """Tooltip that shows only while its text is set (the full text of a cut label); `probe(event)`
    gives the text under the pointer for a canvas."""

    def __init__(self, widget, text="", probe=None):
        super().__init__(widget, text)
        if probe:
            widget.bind("<Motion>", lambda e: self.probe(probe(e)), add="+")

    def probe(self, text):
        if text != self.text:
            self._hide()
            self.text = text
            self._enter()

    def _enter(self, _e=None):
        if self.text:
            super()._enter(_e)


def clip_px(text, fnt, px):
    """`text` cut with an ellipsis so it measures at most px wide."""
    if fnt.measure(text) <= px:
        return text
    while text and fnt.measure(text + "…") > px:
        text = text[:-1]
    return text + "…"


# ── Session lane ──────────────────────────────────────────────────────────────
class SessionLane(_Wave):
    """Lane (96px by default): smooth waveform in heat up to the playhead and inert beyond, a 14% heat tint
    behind the done part, a 1px mark every 30 s of audio, a 2px ink playhead with a timecode tag.
    Progress eases toward each new value instead of jumping."""
    CHUNK_S = 30

    def __init__(self, parent, height=96, **kw):
        super().__init__(parent, height=height, bg=C["well"], highlightthickness=1,
                         highlightbackground=C["rule"], **kw)
        self._bins = []; self._progress = 0.0; self._duration = 0.0; self._size = None
        self._tw = Tween(self, self._paint)
        self.bind("<Configure>", lambda e: self._render())

    def set_bins(self, bins):
        self._bins = list(bins); self._rebuild()

    def set_progress(self, fraction, snap=False):
        self._progress = max(0.0, min(1.0, fraction)); self._tw.to(self._progress, snap)

    def set_duration(self, seconds):
        self._duration = float(seconds or 0); self._rebuild()

    def _rebuild(self):
        self._size = None; self._render()

    def _render(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or (w, h) == self._size:
            return
        self._size = (w, h)
        self.delete("all")
        x0, x1 = 2, w - 2
        self._wave_set(self._bins, x0, 1, x1, h - 1, 11, True)
        mark = _mix(C["ink"], C["well"], 0.16)
        for t in range(self.CHUNK_S, int(self._duration), self.CHUNK_S):
            x = x0 + (x1 - x0) * t / self._duration
            self.create_line(x, 1, x, h - 1, fill=mark)
        self._corners = []       # (backing, text, x0, x1) per corner timecode
        if self._duration:       # start and end timecodes in the lower corners, on a well-colour backing
            pad = round(4 * theme._k(self))
            for x, anchor, s in ((6, "sw", 0), (w - 6, "se", self._duration)):
                txt = self.create_text(x, h - 4, text=timecode(s), anchor=anchor, fill=C["ink-3"],
                                       font=theme._tkfont("mono-sm", self))
                b = self.bbox(txt)
                back = self.create_rectangle(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad,
                                             width=0, fill=C["well"])
                self.tag_lower(back, txt)
                self._corners.append((back, txt, b[0], b[2]))
        self._edge = self.create_line(0, 1, 0, h - 1, fill=_mix(C["ember"], C["well"], 0.5))
        self._head = self.create_rectangle(0, 0, 2, h, width=0, fill=C["ink"])
        self._tag_bg = self.create_rectangle(0, 0, 1, 1, width=0, fill=C["ink"])
        self._tag = self.create_text(0, 6, text="", fill=C["ground"], font=theme._tkfont("mono-sm", self))
        self._paint(self._tw.v)

    def _paint(self, frac):
        if self._size is None:
            return
        w, h = self._size
        x0, x1 = 2, w - 2
        px = x0 + (x1 - x0) * frac
        self._wave_paint(px)
        vis = "normal" if frac > 0 else "hidden"
        for it in (self._edge, self._head, self._tag_bg, self._tag):
            self.itemconfigure(it, state=vis)
        if frac <= 0:
            self._clear_corners(None)
            return
        self.coords(self._edge, px, 1, px, h - 1)
        self.coords(self._head, px - 1, 0, px + 1, h)
        tag = timecode(frac * self._duration) if self._duration else f"{round(frac * 100)} %"
        left = px + 6 + 60 > w
        self.itemconfigure(self._tag, text=tag, anchor="ne" if left else "nw")
        self.coords(self._tag, px - 6 if left else px + 6, 6)
        bx = self.bbox(self._tag)
        self.coords(self._tag_bg, bx[0] - 6, bx[1] - 1, bx[2] + 6, bx[3] + 1)
        self._clear_corners(bx)

    def _clear_corners(self, tag_box):
        """Hide a corner timecode (and its backing) while the tag overlaps it or is within 8 scaled px."""
        gap = 8 * theme._k(self)
        for back, txt, c0, c1 in self._corners:
            hit = tag_box is not None and max(c0 - tag_box[2], tag_box[0] - c1) < gap
            for it in (back, txt):
                self.itemconfigure(it, state="hidden" if hit else "normal")


# ── File-row thumbnail ────────────────────────────────────────────────────────
class MiniLane(_Wave):
    """22px thumbnail: smooth inert waveform, heat up to `progress` (eased) for the running job."""
    _STUB = [0.25 + 0.5 * abs(math.sin(i * 0.7) * math.cos(i * 0.19)) for i in range(40)]

    def __init__(self, parent, width=150, **kw):
        super().__init__(parent, width=width, height=22, bg=C["well"], highlightthickness=0, **kw)
        self._bins = []; self._progress = 0.0; self._size = None
        self._tw = Tween(self, self._paint)
        self.bind("<Configure>", lambda e: self._render())

    def set(self, bins=None, progress=0.0, snap=False):
        bins = list(bins or [])
        self._progress = progress
        if type(self)._render is not MiniLane._render:     # a subclass draws itself (ui/queue.py)
            self._bins = bins; self._render(); return
        if bins != self._bins:
            self._bins = bins; self._size = None; self._render()
        self._tw.to(progress, snap)

    def _render(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 8 or (w, h) == self._size:
            return
        self._size = (w, h)
        self.delete("all")
        self._wave_set(self._bins or self._STUB, 2, 0, w - 2, h, 3, False)
        self._paint(self._tw.v)

    def _paint(self, frac):
        if self._size is not None:
            self._wave_paint(2 + (self._size[0] - 4) * frac)
