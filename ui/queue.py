"""Queue screen — tabs, search, total line and one row per job (DESIGN.md, Lists and tables).

Rows are virtualized: a small pool of row frames is repointed at the visible slice of the
filtered job list, so cost does not grow with the queue length."""
import os
import tkinter as tk

import customtkinter as ctk

from ui import theme
from ui.models import LightScrollbar, fmt_size, gutter
from ui.theme import C, R_INNER, _bars, _mix, font
from ui.transcribe import PILLS, _label, _pill, lang_label
from ui.waveform import ClipTip, Glide, MiniLane, clip_px, timecode as tc

TABS = (("All", None), ("Waiting", ("waiting", "running")), ("Finished", ("done",)),
        ("Failed", ("failed", "cancelled")))
# No · File · Lane (+ percent) · Language · Length · Status · Outputs · Action
COLS = ((66, 0), (226, 1), (214, 0), (130, 0), (70, 0), (116, 0), (96, 0), (150, 0))
HEADS = ("No", "File", "Lane", "Language", "Length", "Status", "Outputs", "")
PILL_W = 92                     # one status pill width, left-aligned in its column
ROW_H = 64
POOL = 10                       # row widgets built at most (the viewport shows ~8 rows)
WHEEL_ROWS = 2


def _grid(frame):
    k = theme._k(frame)                     # plain tk minsize is raw px: apply DPI x UI scale
    for i, (w, weight) in enumerate(COLS):
        frame.columnconfigure(i, minsize=round(w * k), weight=weight)


class _Rows:
    """Lets LightScrollbar drive the virtualized row pool (it expects a scrollable widget)."""

    def __init__(self, yview):
        self.yview_moveto = lambda f: yview("moveto", f)

    def configure(self, **_):
        pass


class _Lane(MiniLane):
    """MiniLane whose bars before the split take `tint` (heat when running, grey-green when done)."""

    def __init__(self, parent, tint, **kw):
        super().__init__(parent, width=150, **kw)
        self._tint = tint

    def _render(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 8:
            return
        bins = self._bins
        if bins:
            k = max(1, len(bins) // 50)
            bins = [max(bins[i:i + k]) for i in range(0, len(bins), k)]
        _bars(self, bins or self._STUB, 2, w - 2, 3, h - 3, 2 + (w - 4) * self._progress,
              self._tint, C["inert-fill"])


def _speed_ratio(job, stages=None):
    """How fast a finished job ran, preferring the recorded transcribe stage."""
    if job.status == "done" and job.duration:
        st = (stages or {}).get(job.id, {}).get("transcribe")
        if st and st.get("t0") is not None and st.get("t1") is not None and st["t1"] > st["t0"]:
            return job.duration / (st["t1"] - st["t0"])
        if not (job.started and job.finished and job.finished > job.started):
            return None
        elapsed = job.finished - job.started
        return job.duration / elapsed
    return None


def _speed(job, stages=None):
    """The length cell's tooltip: how fast a finished job ran."""
    r = _speed_ratio(job, stages)
    if r is not None:
        return f"{r:.0f}× realtime" if r >= 10 else f"{r:.1f}× realtime"
    return ""


def _outputs(job):
    names = []
    for p in job.outputs:
        n = os.path.splitext(p)[1].lstrip(".").upper()
        if n and n not in names:
            names.append(n)
    return " · ".join(names) or "—"


def _first_line(text):
    lines = (text or "").strip().splitlines()
    return lines[0] if lines else ""


VERBS = {"running": "", "waiting": "Remove", "done": "Open", "failed": "Retry",
         "cancelled": "Retry"}


def _tl(parent, role, colour):
    """Plain tk label: cheap to re-text (a CTkLabel redraws its canvas on every configure)."""
    return tk.Label(parent, text="", font=theme._tkfont(role, parent), fg=C[colour], bg=C["ground"], bd=0,
                    padx=0, pady=0, anchor="w")


# ── One pooled row ────────────────────────────────────────────────────────────
class _Row:
    """A row frame that bind() repoints at a job, touching only the values that changed."""

    def __init__(self, scr, parent):
        self.scr, self.job, self.vals, self._fnts = scr, None, {}, {}
        self.frame = f = ctk.CTkFrame(parent, height=ROW_H, corner_radius=10, fg_color="transparent")
        f.grid_propagate(False)
        _grid(f)
        f.rowconfigure(0, weight=1)
        self.rule = tk.Frame(f, height=1, bg=C["rule-soft"], bd=0)
        self.no = _tl(f, "mono", "ink-2")
        self.k = theme._k(parent)
        self.gap = round(12 * self.k)               # cell paddings are physical px, scaled here
        self.no.grid(column=0, row=0, sticky="w", padx=(round(18 * self.k), self.gap))
        self.cw = [round(w * self.k) - self.gap for w, _ in COLS]          # content width of each column
        name = tk.Frame(f, bg=C["ground"], bd=0, width=self.cw[1], height=round(40 * self.k))     # raw px, like minsize
        name.grid(column=1, row=0, sticky="w", padx=(0, self.gap))
        name.pack_propagate(False)
        top = tk.Frame(name, bg=C["ground"], bd=0)
        top.pack(anchor="w", expand=True)           # centred in the cell; the error line sits below
        self.name = _tl(top, "body", "ink")
        self.name.pack(side="left")
        self.size = _tl(top, "hint", "ink-3")
        self.size.pack(side="left", padx=(8, 0))
        self.err = _tl(name, "hint", "bad")
        self.tips = {"name": ClipTip(self.name), "err": ClipTip(self.err)}
        lanebox = tk.Frame(f, bg=C["ground"], bd=0)
        lanebox.grid(column=2, row=0, sticky="ew", padx=(0, self.gap))
        self.pct = _tl(lanebox, "mono-sm", "ink")
        self.pct.pack(side="right")
        self.lane = _Lane(lanebox, C["heat"])
        self.lane.pack(side="left")
        self.lang = _tl(f, "body", "ink")
        self.lang.grid(column=3, row=0, sticky="w", padx=(0, self.gap))
        self.tips["lang"] = ClipTip(self.lang)
        self.length = _tl(f, "mono-sm", "ink")
        self.length.grid(column=4, row=0, sticky="e", padx=(0, self.gap))
        self.tips["length"] = ClipTip(self.length)
        self.glide = Glide(f, self._glide)
        self.pill = _pill(f, "waiting", PILL_W)
        self.pill.grid(column=5, row=0, sticky="w", padx=(0, self.gap / self.k))   # CTk widget: it rescales padx by k
        self.outs = _tl(f, "hint", "ink-2")
        self.outs.grid(column=6, row=0, sticky="w", padx=(0, self.gap))
        acts = tk.Frame(f, bg=C["ground"], bd=0)
        acts.grid(column=7, row=0, sticky="e", padx=(0, round(10 * self.k)))
        self.main = self._button(acts, lambda: self.scr.act(self.job, self.verb))
        self.main.pack(side="right")
        self.move = self._button(acts, lambda: self.scr.act(self.job, "Move up"), "Move up", 70)
        self.verb = ""
        self.cells = (name, top, self.no, self.name, self.size, self.err, self.lang, self.length,
                      lanebox, self.pct, self.outs, acts)

    @staticmethod
    def _button(parent, cmd, text="", width=64):
        return ctk.CTkButton(parent, text=text, width=width, height=28, fg_color="transparent",
                             hover_color=C["raised"], text_color=C["ink-2"], font=font("button"),
                             command=cmd)

    def _put(self, key, widget, **cfg):
        """configure() only when the value differs from what the widget already shows."""
        if self.vals.get(key) != cfg:
            self.vals[key] = cfg
            if isinstance(widget, tk.Label) and "text_color" in cfg:
                cfg = {("fg" if k == "text_color" else k): v for k, v in cfg.items()}
            widget.configure(**cfg)

    def fnt(self, role):
        """Measuring font matching the scaled tk labels."""
        if role not in self._fnts:
            self._fnts[role] = theme.measure_font(theme._tkfont(role, self.frame), self.frame)
        return self._fnts[role]

    def _fit(self, key, widget, full, role, px):
        """Cut `full` to px by measured width; the tooltip carries the whole text when it was cut."""
        text = clip_px(full, self.fnt(role), px)
        self._put(key, widget, text=text)
        self.tips[key].text = full if text != full else ""

    def bind(self, i, j, first_waiting):
        self.job = j
        run = j.status == "running"
        if self.vals.get("bg") != run:
            self.vals["bg"] = run
            self.frame.configure(fg_color=C["panel"] if run else "transparent")
            for w in self.cells:
                w.configure(bg=C["panel"] if run else C["ground"])
        if i and not run:
            self.rule.place(x=0, y=0, relwidth=1)
        else:
            self.rule.place_forget()
        self._put("no", self.no, text=f"{i + 1:02d}")
        size = self.scr.size_of(j)
        self._fit("name", self.name, os.path.basename(j.audio), "body",
                  self.cw[1] - self.fnt("hint").measure(size) - round(8 * self.k))
        self._put("size", self.size, text=size)
        top = self.cells[1]
        full_err = (j.error or "").strip() if j.status == "failed" else ""
        err = _first_line(full_err)
        if err:
            self._fit("err", self.err, err, "hint", self.cw[1])
            if "\n" in full_err or self.tips["err"].text:
                self.tips["err"].text = full_err
            top.pack_configure(expand=False)
            self.err.pack(anchor="w")
        else:
            self.tips["err"].text = ""
            top.pack_configure(expand=True)
            self.err.pack_forget()
        self.live(j)
        self._put("outs", self.outs, text=_outputs(j))
        text, tint, ink = PILLS[j.status]
        self._put("pill", self.pill, text=f"  {text.upper()}  ", text_color=C[ink],
                  fg_color=_mix(C[tint], C["panel"], 0.14) if tint else "transparent")
        self.verb = VERBS.get(j.status, "")
        self._put("main", self.main, text=self.verb)
        if self.verb:
            self.main.pack(side="right")
        else:
            self.main.pack_forget()
        if j.status == "waiting" and j is not first_waiting:
            self.move.pack(side="right", padx=(0, 4))
        else:
            self.move.pack_forget()

    def live(self, j):
        """The fields that change while a job runs; lane and percent glide to the new progress."""
        run = j.status == "running"
        self.lane._tint = C["heat"] if run else _mix(C["ok"], C["inert-fill"], 0.25)
        self.bins = j.bins
        fresh = self.vals.get("jid") != j.id
        self.vals["jid"] = j.id
        self.glide.to(j.progress, snap=fresh or not run)
        self._glide(self.glide.value)
        text, doubt = lang_label(j)
        self._fit("lang", self.lang, text, "body", self.cw[3])
        self._put("lang_ink", self.lang, text_color=C["warn" if doubt else "ink"])
        if doubt:
            self.tips["lang"].text = doubt
        self._put("length", self.length, text=tc(j.duration) if j.duration else "—")
        self.tips["length"].text = _speed(j, self.scr.state.stages)

    def _glide(self, v):
        j = self.job
        run = j is not None and j.status == "running"
        prog = v if run else 1.0 if j is not None and j.status == "done" else 0.0
        key = (self.lane._tint, id(self.bins), round(prog, 3))
        if self.vals.get("lane") != key:
            self.vals["lane"] = key
            self.lane.set(self.bins, prog)
        self._put("pct", self.pct, text=f"{round(v * 100)} %" if run else "")


# ── Screen ────────────────────────────────────────────────────────────────────
class Screen:
    def __init__(self, parent, state):
        self.state = state
        self.page = ctk.CTkFrame(parent, fg_color=C["ground"], corner_radius=0)
        self.page.screen = self
        self.tab = 0
        self.pool = []              # row widgets, built on demand and reused while scrolling
        self.bound = {}             # job id -> the pooled row currently showing it
        self.shown = []             # jobs passing the tab and search filters
        self.top = 0                # index into shown of the first visible row
        self._sizes = {}
        self._labels = []
        self._dirty = False
        self.page.columnconfigure(0, weight=1)
        self.page.rowconfigure(2, weight=1)
        self._build_toolbar()
        head = ctk.CTkFrame(self.page, fg_color="transparent", height=30)
        head.grid(row=1, column=0, sticky="ew", padx=(28, 28 + gutter(self.page) / theme._k(self.page)))   # CTk scales it; the gutter is physical px
        _grid(head)
        k = theme._k(head)
        for i, text in enumerate(HEADS):       # row cells pad in raw px, so the labels pad the same physical px (CTk rescales by k)
            _label(head, text.upper(), "label").grid(
                row=0, column=i, sticky="e" if i == 4 else "w",
                padx=(round(18 * k) / k, round(12 * k) / k) if i == 0 else (0, round(12 * k) / k))
        wrap = ctk.CTkFrame(self.page, fg_color="transparent")
        wrap.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 22))
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        self.rows = ctk.CTkFrame(wrap, fg_color="transparent")
        self.rows.grid(row=0, column=0, sticky="nsew")
        self.rows.bind("<Configure>", lambda e: self._paint())
        self.bar = LightScrollbar(wrap, _Rows(self._yview))     # ctk.CTkScrollbar redraws took ~0.5 s per set()
        self.bar.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):     # CTk forbids bind_all
            tk.Misc.bind_all(self.page, seq, self._wheel, add="+")
        self._refresh()
        for topic, cb in (("job", self._on_job), ("runner", self._on_runner)):
            state.on(topic, cb)

    # ── Toolbar ───────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        bar = ctk.CTkFrame(self.page, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 14))
        self.tabs = theme.segmented(bar, [t[0] for t in TABS], command=self._pick_tab)
        self.tabs.pack(side="left")
        search = ctk.CTkEntry(bar, width=200, height=36, corner_radius=R_INNER,
                              fg_color=C["well"], border_color=C["rule"], text_color=C["ink"],
                              placeholder_text="Search file names", font=font("body"))
        search.pack(side="left", padx=14)
        search.bind("<KeyRelease>", lambda e: self._search())
        self.search = search
        theme.pill(bar, "Add files", lambda: self.state.emit("browse")).pack(side="right")
        ctk.CTkButton(bar, text="Clear finished", width=100, height=32, fg_color="transparent",
                      hover_color=C["raised"], text_color=C["ink-2"], font=font("button"),
                      command=self._clear).pack(side="right", padx=(0, 8))
        self.total = _label(bar, "", "body", "ink-2")
        self.total.pack(side="right", padx=14)

    def _pick_tab(self, label):
        self.tab = self._labels.index(label)
        self.top = 0
        self._refresh()

    def _search(self):
        self.top = 0
        self._refresh()

    def _clear(self):
        self.state.queue.clear_finished()
        self.state.queue.save(self.state.queue_path)
        self.state.emit("job", job=None)        # other screens drop the cleared rows
        self._refresh()

    def _total_text(self):
        jobs = self.state.queue.jobs
        audio = sum(j.duration or 0 for j in jobs)
        if not audio:
            return ""
        text = f"{tc(audio)} of audio"
        timed = [j for j in jobs if j.status == "done" and j.duration and j.started and j.finished
                 and j.finished > j.started]
        left = sum((j.duration or 0) * (1 - j.progress if j.status == "running" else 1)
                   for j in jobs if j.status in ("waiting", "running"))
        if timed and left:
            last = max(timed, key=lambda j: j.finished)
            rate = last.duration / (last.finished - last.started)
            text += f" · about {max(1, round(left / rate / 60))} min left"
        return text

    # ── Rows ──────────────────────────────────────────────────────────────────
    def _refresh(self):
        """Re-filter, recount and repaint the visible slice; no per-job widgets are made or destroyed."""
        self._dirty = False
        q = self.search.get().strip().lower()
        want = TABS[self.tab][1]
        counts = [0] * len(TABS)
        self.shown = []
        for j in self.state.queue.jobs:
            if q and q not in os.path.basename(j.audio).lower():
                continue
            for i, (_, st) in enumerate(TABS):
                if st is None or j.status in st:
                    counts[i] += 1
            if want is None or j.status in want:
                self.shown.append(j)
        labels = [f"{t[0]}  {n}" for t, n in zip(TABS, counts)]
        if labels != self._labels:
            self._labels = labels
            self.tabs.configure(values=labels)
        self.tabs.set(labels[self.tab])
        self.total.configure(text=self._total_text())
        self._paint()

    def _vis(self):
        pitch = ROW_H * ctk.ScalingTracker.get_widget_scaling(self.page)      # place() scales y
        return max(1, int((self.rows.winfo_height() or 8 * pitch) // pitch))

    def _paint(self):
        n, vis = len(self.shown), self._vis()
        self.top = max(0, min(self.top, n - vis))
        first = self.state.queue.next_waiting()
        self.bound = {}
        for slot in range(POOL):
            i = self.top + slot
            if i >= n:
                if slot < len(self.pool):
                    self.pool[slot].frame.place_forget()
                continue
            if slot == len(self.pool):
                self.pool.append(_Row(self, self.rows))
            row, j = self.pool[slot], self.shown[i]
            row.frame.place(x=0, y=slot * ROW_H, relwidth=1)
            row.bind(i, j, first)
            self.bound[j.id] = row
        self.bar.set(*((0, 1) if n <= vis else (self.top / n, (self.top + vis) / n)))

    def size_of(self, j):
        if j.id not in self._sizes:
            try:
                self._sizes[j.id] = fmt_size(os.path.getsize(j.audio))
            except OSError:
                self._sizes[j.id] = ""
        return self._sizes[j.id]

    def _yview(self, *a):
        if a[0] == "moveto":
            self.top = round(float(a[1]) * len(self.shown))
        elif a[0] == "scroll":
            self.top += int(a[1]) * (self._vis() if a[2] == "pages" else 1)
        self._paint()

    def _wheel(self, e):
        w = str(e.widget)
        if not self.page.winfo_ismapped() or not w.startswith(str(self.page)):
            return None
        up = e.num == 4 or getattr(e, "delta", 0) > 0
        self._yview("scroll", WHEEL_ROWS * (-1 if up else 1), "units")
        return "break"

    # ── Actions ───────────────────────────────────────────────────────────────
    def act(self, j, verb):
        if verb == "Cancel":
            self.state.cancel()
        elif verb == "Move up":
            self._move_up(j)
        elif verb == "Remove":
            self._remove(j)
        elif verb == "Open":
            self._open(j)
        elif verb == "Retry":
            self._retry(j)

    def _save_emit(self, j):
        self.state.queue.save(self.state.queue_path)
        self.state.emit("job", job=j)

    def _move_up(self, j):
        self.state.queue.move_up(j.id)
        self._save_emit(j)

    def _remove(self, j):
        self.state.queue.remove(j.id)
        self._save_emit(j)

    def _open(self, j):
        # switch first: the Transcript screen is built lazily and must exist to hear the open
        self.state.emit("screen", name="transcript")
        self.state.emit("open-transcript", job=j)

    def _retry(self, j):
        j.status, j.error, j.progress = "waiting", None, 0.0
        self._save_emit(j)

    # ── Events ────────────────────────────────────────────────────────────────
    def _on_job(self, job=None, **_):
        if not self._dirty:
            self._dirty = True
            self.page.after_idle(self._refresh)

    def _on_runner(self, job, ev, **_):
        t = ev.get("t")
        row = self.bound.get(job.id)
        if row and t in ("wave", "progress", "done", "stage", "language"):
            row.live(job)


def build(parent, state):
    """Queue screen (fixed layout, virtualized rows); returns its page frame."""
    return Screen(parent, state).page
