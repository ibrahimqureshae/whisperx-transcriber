"""Models screen — whisper models plus cached helper models, with download / delete and disk use."""
import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core import pipeline
from ui import theme
from ui.theme import C, font
from ui.waveform import ClipTip, Glide, clip_px

_GROUPS = {"whisper": ("", 0), "align": ("Word timing", 1), "speaker": ("Speakers", 3)}       # kind -> (divider label, order)
# name | size | share | action  (header label, column width)
_COLS = (("Model", 0), ("Size", 90), ("Share of cache", 150), ("", 90))


_SPEAKER_ROWS = (("segmentation", "Speaker segmentation"), ("embedding", "Speaker voice embedding"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prefetch_punkt(md):
    """Best-effort: fetch the sentence tokenizer with a whisper model so Offline-only
    users still get word timings."""
    try:
        pipeline.ensure_punkt(md, offline=False)
    except Exception:
        pass

def fmt_size(nbytes):
    """KB under 1 MB, one decimal MB under 10 MB, whole MB under 1 GB, else GB; the unit is picked on
    the rounded value shown (999 500 B is '1.0 MB', not '1000 KB')."""
    kb = (nbytes + 500) // 1000
    if kb < 1000:
        return f"{kb} KB"
    tenths = (nbytes + 50_000) // 100_000
    if tenths < 100:
        return f"{tenths // 10}.{tenths % 10} MB"
    mb = (nbytes + 500_000) // 1_000_000
    if mb < 1000:
        return f"{mb} MB"
    tenths = (nbytes + 50_000_000) // 100_000_000
    return f"{tenths // 10}.{tenths % 10} GB"


def _free_bytes(path):
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    try:
        return shutil.disk_usage(path or ".").free
    except OSError:
        return 0


def _collect(md):
    """Rows for every known whisper model plus other cached models: whisper models, then word-timing
    (alignment) models, then the rest; cached first, then name, within each group."""
    hub = {e["name"]: e for e in pipeline.cached_models(md)}
    rows = []
    for name, (_repo, mb) in pipeline.MODEL_INFO.items():
        cached = os.path.isdir(pipeline.model_cache_dir(name, md))
        if cached:
            size = hub[name]["size_bytes"] if name in hub else pipeline._dir_size(
                pipeline.model_cache_dir(name, md))
        else:
            size = mb * 1_000_000
        rows.append({"name": name, "kind": "whisper", "cached": cached, "size": size,
                     "estimate": not cached})
    rows += [{"name": e["name"], "kind": e["kind"], "cached": True, "size": e["size_bytes"],
              "estimate": False} for e in hub.values() if e["name"] not in pipeline.MODEL_INFO]
    for key, label in _SPEAKER_ROWS:
        ready = os.path.isfile(pipeline.speaker_model_path(md, key))
        rows.append({"name": label, "kind": "speaker", "cached": ready, "estimate": not ready,
                     "size": os.path.getsize(pipeline.speaker_model_path(md, key)) if ready
                     else pipeline.SPEAKER_MODELS[key][2]})
    for r in rows:
        r["group"], r["order"] = _GROUPS.get(r["kind"], ("Other", 2))
    rows.sort(key=lambda r: (r["order"], not r["cached"], r["name"]))
    return rows


# ── Screen ────────────────────────────────────────────────────────────────────
_ROW_H = 52                             # row height
_PAD = 18                               # row inner padding


def gutter(w):
    """Physical px the scrollbar lane takes beside a table: the bar (10 design px) and its 4 px gap."""
    return round(10 * theme._k(w)) + 4


def _col_x(width, k=1.0):
    """Left x of every column for a table `width` px wide (fixed columns packed from the right);
    k is the UI scale that plain tk pixels need."""
    xs, x = [0] * len(_COLS), width - _PAD * k
    for i in range(len(_COLS) - 1, 0, -1):
        x -= _COLS[i][1] * k
        xs[i] = x
        x -= 12 * k
    xs[0] = _PAD * k
    return xs


class LightScrollbar(ctk.CTkCanvas):
    """Thin canvas scrollbar (ctk.CTkScrollbar costs ~280 ms to build)."""

    def __init__(self, parent, target):
        super().__init__(parent, width=round(10 * theme._k(parent)), bg=C["ground"],
                         highlightthickness=0, bd=0)
        self._t, self._lo, self._hi = target, 0.0, 1.0
        target.configure(yscrollcommand=self.set)
        self.bind("<Button-1>", self._jump)
        self.bind("<B1-Motion>", self._jump)
        self.bind("<Configure>", lambda e: self._paint())

    def set(self, lo, hi):
        self._lo, self._hi = float(lo), float(hi)
        self._paint()

    def _paint(self):
        self.delete("all")
        h = self.winfo_height()
        if self._hi - self._lo < 0.999 and h > 1:
            k, top = theme._k(self), round(self._lo * h)
            theme.rrect(self, round(2 * k), top, round(8 * k), max(round(self._hi * h), top + round(16 * k)),
                        round(3 * k), C["inert-fill"])

    def _jump(self, e):
        self._t.yview_moveto(e.y / max(1, self.winfo_height()) - (self._hi - self._lo) / 2)


class ModelsScreen(ctk.CTkFrame):
    """The table is one canvas (rows are drawn items, not widgets) so it builds in a few ms."""

    def __init__(self, parent, state):
        super().__init__(parent, fg_color=C["ground"], corner_radius=0)
        self.state = state
        self._dl = None             # {"name", "frac"} while a download runs
        self._stop = False
        self._rows, self._md, self._total = [], "", 0
        self._cells = {}            # name -> ([meter cell item ids], pct item id)
        self._clipped = {}          # slot index -> full name, for names cut to the column
        self._slot = {}             # model name -> slot index (a slot is a row or a group divider)
        self._glide = Glide(self, self._paint_dl)     # download meter and percent
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=28, pady=(16, 8))
        self._path = ctk.CTkLabel(bar, text="", font=font("mono"), text_color=C["ink"])
        self._path.pack(side="left")
        theme.pill(bar, "Change folder", self._change_folder).pack(side="right")

        self._head = ctk.CTkCanvas(self, height=22, bg=C["ground"], highlightthickness=0, bd=0)
        self._head.grid(row=1, column=0, sticky="ew", padx=(28, 28 + gutter(self)))
        box = tk.Frame(self, bg=C["ground"], bd=0)
        box.grid(row=2, column=0, sticky="nsew", padx=28, pady=(0, 6))
        self._cv = ctk.CTkCanvas(box, bg=C["ground"], highlightthickness=0, bd=0)
        sb = LightScrollbar(box, self._cv)
        sb.pack(side="right", fill="y", padx=(4, 0))
        self._cv.pack(side="left", fill="both", expand=True)
        self._cv.bind("<Configure>", lambda e: self._paint_table())
        self._cv.bind("<MouseWheel>", lambda e: self._cv.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        ClipTip(self._cv, probe=self._probe)

        disk = ctk.CTkFrame(self, fg_color="transparent")
        disk.grid(row=3, column=0, sticky="ew", padx=28, pady=(6, 14))
        ctk.CTkLabel(disk, text="DISK", font=font("label"), text_color=C["ink-3"]).pack(side="left")
        self._disk_bar = ctk.CTkProgressBar(disk, height=8, corner_radius=4, border_width=1,
                                            fg_color=C["well"], border_color=C["rule-soft"],
                                            progress_color=C["ember"])
        self._disk_bar.pack(side="left", fill="x", expand=True, padx=14)
        self._disk_bar.set(0)
        self._disk_txt = ctk.CTkLabel(disk, text="", font=font("mono"), text_color=C["ink-2"])
        self._disk_txt.pack(side="left")

        state.on("settings", self._on_settings)
        state.on("finished", lambda **_: self.refresh())
        self.bind("<Destroy>", lambda e: setattr(self, "_stop", True) if e.widget is self else None)
        self.refresh()

    # ── Data -> canvas ────────────────────────────────────────────────────────
    def _on_settings(self, key=None, **_):
        if key in ("model_dir", "*"):
            self.refresh()

    def refresh(self):
        if not self.winfo_exists():
            return
        md = self.state.model_dir()
        self._path.configure(text=md if len(md) <= 56 else "…" + md[-55:])
        self._rows = _collect(md)
        self._md = md
        self._total = sum(r["size"] for r in self._rows if r["cached"])
        free = _free_bytes(md)
        self._disk_bar.set(self._total / (self._total + free) if self._total + free else 0)
        self._disk_txt.configure(text=f"{fmt_size(self._total)} of models · {fmt_size(free)} free")
        self._paint_table()

    def _paint_table(self):
        cv, w = self._cv, self._cv.winfo_width()
        if w < 200:                         # not laid out yet; <Configure> will call again
            return
        k = theme._k(cv)
        rh = round(_ROW_H * k)
        cv.configure(yscrollincrement=rh)
        self._head.configure(height=round(22 * k))
        xs = _col_x(w, k)
        self._paint_head(xs, k)
        cv.delete("all")
        self._cells, self._clipped, self._slot = {}, {}, {}
        slot, group = 0, ""
        for r in self._rows:
            if r["group"] != group:
                group = r["group"]
                if group:
                    self._paint_divider(slot, group, xs, w, rh, k)
                    slot += 1
            self._paint_row(slot, r, xs, w, rh, k)
            self._slot[r["name"]] = slot
            slot += 1
        cv.configure(scrollregion=(0, 0, w, max(slot * rh, 1)))

    def _paint_head(self, xs, k):
        h = self._head
        h.delete("all")
        for i, (text, cw) in enumerate(_COLS):
            if text:
                right = i == 1
                h.create_text(xs[i] + (cw * k if right else 0), 11 * k, text=text.upper(),
                              font=theme._tkfont("label", h), fill=C["ink-3"], anchor="e" if right else "w")

    def _paint_divider(self, slot, label, xs, w, rh, k):
        """Quiet caps label that names the group of rows below it."""
        y = slot * rh
        self._cv.create_text(xs[0], y + rh - 10 * k, text=label.upper(), font=theme._tkfont("label", self._cv),
                             fill=C["ink-3"], anchor="sw")
        self._cv.create_line(0, y + rh - 1, w, y + rh - 1, fill=C["rule-soft"], width=max(1, round(k)))

    def _paint_row(self, slot, r, xs, w, rh, k):
        cv, name, dl = self._cv, r["name"], self._dl
        y = slot * rh
        mid = y + rh / 2
        cv.create_line(0, y + rh - 1, w, y + rh - 1, fill=C["rule-soft"], width=max(1, round(k)))

        def text(col, s, f="body", colour="ink", right=False):
            cv.create_text(xs[col] + (_COLS[col][1] * k if right else 0), mid, text=s,
                           font=theme._tkfont(f, cv), fill=C[colour], anchor="e" if right else "w")

        fnt = theme.measure_font(theme._tkfont("button", cv), cv)
        shown = clip_px(name, fnt, xs[1] - xs[0] - 12 * k)
        if shown != name:
            self._clipped[slot] = name
        text(0, shown, "button")
        text(1, ("≈" if r["estimate"] else "") + fmt_size(r["size"]), "mono-sm",
             "ink" if r["cached"] else "ink-2", True)

        busy = bool(dl and dl["name"] == name)
        if busy or r["cached"]:
            mx, step = xs[2], (theme.Meter.CW + theme.Meter.CG) * k
            cw, ch = theme.Meter.CW * k, 5 * k
            cells = [cv.create_rectangle(mx + n * step, mid - ch, mx + n * step + cw, mid + ch,
                                         width=0, fill=C["inert-fill"]) for n in range(theme.METER_CELLS)]
            pct = cv.create_text(mx + theme.METER_CELLS * step + 8 * k, mid, text="",
                                 font=theme._tkfont("mono-sm", cv), fill=C["ink-3"], anchor="w")
            self._cells[name] = (cells, pct)
            self._paint_meter(name, self._glide.value if busy else r["size"] / self._total if self._total else 0)

        if r["cached"]:
            self._action(slot, "Delete", "bad", xs, mid, k, lambda: self._delete(r, self._md))
        elif r["kind"] in ("whisper", "speaker"):
            self._action(slot, "Download", "ink", xs, mid, k, lambda: self._download(name, self._md),
                         off=bool(dl))

    def _action(self, slot, text, colour, xs, mid, k, command, off=False):
        """Flat text button; the whole ~70x30 pill behind it is the hit area and tints on hover."""
        cv, tag = self._cv, f"act{slot}"
        cx = xs[3] + _COLS[3][1] * k / 2
        fnt = theme._tkfont("button", cv)
        if off:
            cv.create_text(cx, mid, text=text, font=fnt, fill=C["ink-3"])
            return
        pill = f"pill{slot}"            # the box only; the label under `tag` keeps its colour
        theme.rrect(cv, round(cx - 35 * k), round(mid - 15 * k), round(cx + 35 * k), round(mid + 15 * k),
                    round(15 * k), C["ground"], tags=(tag, pill))
        cv.create_text(cx, mid, text=text, font=fnt, tags=(tag,), fill=C[colour])
        cv.tag_bind(tag, "<Enter>", lambda _e: (cv.itemconfigure(pill, fill=C["raised"]),
                                                cv.configure(cursor="hand2")))
        cv.tag_bind(tag, "<Leave>", lambda _e: (cv.itemconfigure(pill, fill=C["ground"]),
                                                cv.configure(cursor="")))
        cv.tag_bind(tag, "<Button-1>", lambda _e: command())

    def _probe(self, e):
        """Full model name when the pointer is over a name that was cut to its column."""
        cv = self._cv
        k = theme._k(cv)
        xs = _col_x(cv.winfo_width(), k)
        if not xs[0] <= cv.canvasx(e.x) < xs[1] - 12 * k:
            return ""
        return self._clipped.get(int(cv.canvasy(e.y) // round(_ROW_H * k)), "")

    def _paint_meter(self, name, frac):
        cells, pct = self._cells[name]
        lit = round(max(0.0, min(1.0, frac)) * theme.METER_CELLS)
        for k, c in enumerate(cells):
            self._cv.itemconfigure(c, fill=C["ink-2"] if k < lit else C["inert-fill"])
        self._cv.itemconfigure(pct, text=f"{round(frac * 100)} %", fill=C["ink"])

    # ── Actions ───────────────────────────────────────────────────────────────
    def _change_folder(self):
        path = filedialog.askdirectory(title="Model folder", initialdir=self.state.model_dir())
        if path:
            self.state.set("model_dir", os.path.normpath(path))
            self.refresh()

    def _delete(self, r, md):
        name = r["name"]
        if not messagebox.askyesno(f"Delete {name} ({fmt_size(r['size'])})?",
                                   f"This removes {name} from {md}. You can download it again later."):
            return
        try:
            if r["kind"] == "speaker":
                shutil.rmtree(os.path.join(md, "speakers"), ignore_errors=True)
            elif name in pipeline.MODEL_INFO:
                cache = pipeline.model_cache_dir(name, md)
                if os.path.exists(cache):
                    shutil.rmtree(cache)
            else:
                pipeline.delete_cached_model(md, name)
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
        self.refresh()

    def _download(self, name, md):
        if self._dl:
            return
        self._dl = {"name": name, "frac": 0.0}
        self._glide.to(0.0, snap=True)
        self.refresh()
        sched = self.state.schedule

        def progress(done, total):
            frac = done / total if total else 0.0

            def paint():
                if self._dl and self._dl["name"] == name:
                    self._dl["frac"] = frac
                    self._paint_progress()
            sched(paint)

        def work():
            err = None
            try:
                if name in [l for _k, l in _SPEAKER_ROWS]:
                    pipeline.ensure_speaker_models(md, False, lambda f: progress(f, 1))
                else:
                    pipeline.download_model(name, md, progress, lambda: self._stop)
                if name in pipeline.MODEL_INFO and not self._stop:
                    _prefetch_punkt(md)
            except Exception as e:
                err = e

            def done():
                self._glide.stop()
                self._dl = None
                if err is not None and not self._stop:
                    messagebox.showerror("Download failed", f"{name}: {err}")
                if not self._stop:
                    self.refresh()
            sched(done)

        threading.Thread(target=work, daemon=True).start()

    def _paint_progress(self):
        d = self._dl
        if d:
            self._glide.to(d["frac"])

    def _paint_dl(self, v):
        d = self._dl
        if d and d["name"] in self._cells:
            self._paint_meter(d["name"], v)


def build(parent, state):
    return ModelsScreen(parent, state)
