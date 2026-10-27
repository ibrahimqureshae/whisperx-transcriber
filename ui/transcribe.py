"""Transcribe screen — session block, files with per-file stage dots, language/model/output setup."""
import os
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from core.pipeline import EXPORT_FORMATS, MODEL_INFO, model_cache_dir, supports_translate
from ui import theme
from ui.state import LANGUAGES, MEDIA_TYPES
from ui.theme import C, PAD_LG, R_INNER, font, _mix
from ui.waveform import ClipTip, MiniLane, SessionLane, Tween, timecode as tc

STAGES = (("load", "Load model"), ("detect", "Detect language"), ("vad", "Find speech"),
          ("transcribe", "Transcribe"), ("align", "Align words"),
          ("diarize", "Identify speakers"), ("export", "Write files"))
FORMAT_LABELS = {"srt": "SRT", "vtt": "VTT", "txt": "TXT", "tsv": "TSV", "json": "JSON",
                 "word_json": "Word JSON", "aud": "Audacity labels"}
SWITCHES = (("align", "Word timestamps", "Gives every word its own start and end time."),
            ("diarize", "Speakers", "Finds who speaks when. The speaker models (about 34 MB) download automatically the first time."),
            ("open_folder", "Open folder", "Opens the output folder when a file finishes (once per batch if set in Settings)."),
            ("highlight", "Karaoke", "Highlights each word as it is spoken (SRT and VTT)."))
# every language whisper can detect (the menu above lists only the common ones)
_WHISPER = ("en English|zh Chinese|de German|es Spanish|ru Russian|ko Korean|fr French|ja Japanese|pt Portuguese|"
            "tr Turkish|pl Polish|ca Catalan|nl Dutch|ar Arabic|sv Swedish|it Italian|id Indonesian|hi Hindi|"
            "fi Finnish|vi Vietnamese|he Hebrew|uk Ukrainian|el Greek|ms Malay|cs Czech|ro Romanian|da Danish|"
            "hu Hungarian|ta Tamil|no Norwegian|th Thai|ur Urdu|hr Croatian|bg Bulgarian|lt Lithuanian|la Latin|"
            "mi Maori|ml Malayalam|cy Welsh|sk Slovak|te Telugu|fa Persian|lv Latvian|bn Bengali|sr Serbian|"
            "az Azerbaijani|sl Slovenian|kn Kannada|et Estonian|mk Macedonian|br Breton|eu Basque|is Icelandic|"
            "hy Armenian|ne Nepali|mn Mongolian|bs Bosnian|kk Kazakh|sq Albanian|sw Swahili|gl Galician|mr Marathi|"
            "pa Punjabi|si Sinhala|km Khmer|sn Shona|yo Yoruba|so Somali|af Afrikaans|oc Occitan|ka Georgian|"
            "be Belarusian|tg Tajik|sd Sindhi|gu Gujarati|am Amharic|yi Yiddish|lo Lao|uz Uzbek|fo Faroese|"
            "ht Haitian Creole|ps Pashto|tk Turkmen|nn Nynorsk|mt Maltese|sa Sanskrit|lb Luxembourgish|"
            "my Burmese|bo Tibetan|tl Tagalog|mg Malagasy|as Assamese|tt Tatar|haw Hawaiian|ln Lingala|ha Hausa|"
            "ba Bashkir|jw Javanese|su Sundanese|yue Cantonese")
LANG_NAMES = {**dict(e.split(" ", 1) for e in _WHISPER.split("|")),
              **{code: name for name, code in LANGUAGES.items()}}
UNCERTAIN = "The language guess is uncertain ({}). Pick the language for a better result."


def lang_label(job):
    """(name, warning) for a job's detected language: '' until detected; the warning (name and percent) is set
    when the guess is weak. The name carries no percent (Say-It-Once: the big percent is progress)."""
    if not job.language or job.language == "auto":
        return "", ""
    text = LANG_NAMES.get(job.language, job.language)
    prob = job.language_prob
    if prob is None:
        return text, ""
    return text, UNCERTAIN.format(f"{text} {round(prob * 100)} %") if prob < 0.5 else ""
AUD_TIP = "A tab-separated list of segments that Audacity imports as a label track."
EXTS = tuple(t[1:] for t in MEDIA_TYPES.split())
ADD_TIP = ("Add audio or video files (Ctrl+O). Copied files can be pasted with Ctrl+V. "
           "Folders add every media file inside.")
TASKS = {"transcribe": "Transcribe", "translate": "Translate to English"}
ROW_H = 30
FINISHED_ROWS = 8      # finished files kept on the Files list (the Queue screen has them all)
COLS = {"idx": 32, "name": 238, "mini": 134, "len": 76, "track": 348, "pill": 90, "open": 58, "x": 38}
PILLS = {"waiting": ("Queued", "ink-3", "ink-3"), "running": ("Running", "heat", "ember"),
         "done": ("Done", "ok", "ok"), "failed": ("Failed", "bad", "bad"),
         "cancelled": ("Cancelled", "warn", "warn")}


VERBS = {"transcribe": "Transcribing", "translate": "Translating", "done": "Done",
         "failed": "Failed", "cancelled": "Cancelled"}
STAGE_VERBS = {"load": "Loading model", "detect": "Detecting language", "vad": "Finding speech",
               "align": "Aligning words", "diarize": "Identifying speakers", "export": "Writing files"}


def _pill(parent, status, width=None):
    text, tint, ink = PILLS[status]
    return ctk.CTkLabel(parent, text=f"  {text.upper()}  ", height=22, corner_radius=11, **({"width": width} if width else {}),
                        font=font("label"), text_color=C[ink],
                        fg_color=_mix(C[tint], C["panel"], 0.14) if tint else "transparent")


def _label(parent, text, role="hint", colour="ink-3", **kw):
    return ctk.CTkLabel(parent, text=text, font=font(role), text_color=C[colour], anchor="w", **kw)


def _mono(parent, colour, bg):
    """Plain tk label (auto-sizes to its text; a CTkLabel clips changing text)."""
    return tk.Label(parent, text="", font=theme._tkfont("mono-sm", parent), fg=C[colour], bg=C[bg], anchor="w", bd=0)


def _clip(text, fnt, px):
    """Trim text with an ellipsis so it measures at most px wide (no trim before layout)."""
    if px < 40 or fnt.measure(text) <= px:
        return text
    while text and fnt.measure(text + "…") > px:
        text = text[:-1]
    return text + "…"


def _short(text, n):
    return text if len(text) <= n else text[:n - 1] + "…"


def _size(path):
    """File size: KB under 1 MB, MB with one decimal under 10 MB, else whole MB ('' if unreadable)."""
    try:
        n = os.path.getsize(path)
    except OSError:
        return ""
    return f"{n / 1e3:.0f} KB" if n < 1e6 else f"{n / 1e6:.1f} MB" if n < 1e7 else f"{n / 1e6:.0f} MB"


class _Scroller(tk.Frame):
    """Scrolling list inside a panel: rows go in this frame, which sits on a canvas with a thin
    LightScrollbar (CTkScrollableFrame's scrollbar took ~1.5 s to redraw at 175 % scaling).
    pack() places the outer frame, as CTkScrollableFrame does."""

    def __init__(self, parent):
        from ui.models import LightScrollbar
        bg = C["panel"]
        self._outer = tk.Frame(parent, bg=bg, height=40)
        self._cv = tk.Canvas(self._outer, bg=bg, highlightthickness=0, bd=0, height=40,
                             yscrollincrement=round(ROW_H * theme._k(parent)))
        self._bar = LightScrollbar(self._outer, self._cv)
        self._bar.configure(bg=bg)
        self._bar.pack(side="right", fill="y")
        self._cv.pack(side="left", fill="both", expand=True)
        super().__init__(self._cv, bg=bg)
        win = self._cv.create_window(0, 0, window=self, anchor="nw")
        self.bind("<Configure>", lambda e: self._cv.configure(scrollregion=(0, 0, e.width, e.height)))
        self._cv.bind("<Configure>", lambda e: self._cv.itemconfigure(win, width=e.width))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            tk.Misc.bind_all(self._cv, seq, self._wheel, add="+")

    def pack(self, **kw):
        self._outer.pack(**kw)

    def _wheel(self, e):
        if not str(e.widget).startswith(str(self._outer)) or not self._cv.winfo_ismapped():
            return
        top, bottom = self._cv.yview()
        if bottom - top < 1:
            step = -1 if (getattr(e, "num", 0) == 4 or e.delta > 0) else 1
            self._cv.yview_scroll(step, "units")


def _scroller(parent):
    return _Scroller(parent)


# ── Stage track (one per file row) ───────────────────────────────────────────
def stage_view(job, rec):
    """Dot state per stage: done / now / pending / skip (hollow) / bad (failed here) / stopped (cancelled here).
    `rec` is AppState.stages[job.id]; the job's cfg and the runner's warnings predict skipped stages."""
    cfg, rec = job.cfg or {}, rec or {}
    predicted = {"align": not cfg.get("align", True) or cfg.get("task") == "translate",
                 "diarize": not cfg.get("diarize")}
    out = []
    for key, _ in STAGES:
        r = rec.get(key)
        cut = "bad" if job.status == "failed" else "stopped"
        if predicted.get(key) or (r and r["skip"]):
            out.append("skip")
        elif r and r["state"] == "failed":
            out.append("bad")
        elif r and r["state"] == "done":
            out.append("done")
        elif r and r["state"] == "now":
            out.append("now" if job.status == "running" else cut)
        elif r and r["state"] == "stopped":
            out.append(cut)
        else:
            out.append("done" if job.status == "done" else "pending")
    return out


def stage_label(job, states):
    """(text, colour token) for the label after the dots: only what the status pill doesn't say."""
    title = lambda *want: next((lbl for (_, lbl), s in zip(STAGES, states) if s in want), None)
    if job.status == "waiting":
        return "", "ink-3"
    if job.status == "running":
        return title("now") or "Starting", "ink"
    if job.status == "done":
        return (f"Took {tc(job.finished - job.started)}" if job.started and job.finished else ""), "ink-2"
    cut = title("bad", "stopped")
    if job.status == "failed":
        err = (job.error or "").strip().splitlines()
        return " · ".join(p for p in (cut, err[0] if err else "") if p), "bad"
    return (f"Stopped at {cut}" if cut else ""), "ink-3"


def stage_tip(job, states, rec):
    """One line per stage: title, runner detail, seconds."""
    lines, now = [], time.time()
    for (key, title), s in zip(STAGES, states):
        r = (rec or {}).get(key)
        if s == "skip":
            lines.append(f"{title} — skipped")
            continue
        el = f"{(r['t1'] or now) - r['t0']:.1f} s" if r and r["t0"] else ""
        if r and re.search(r"\d s$", r["detail"]):     # the runner's detail already ends in seconds
            el = ""
        lines.append(" · ".join(p for p in (f"{title} — {r['detail']}" if r and r["detail"] else title,
                                            el) if p))
    return "\n".join(lines)


class StageTrack(tk.Canvas):
    """Seven dots plus a label on one canvas. A dot that changes state fades its colour over ~180 ms."""
    W, H, STEP, TEXT_X, FADE = 340, 26, 15, 114, 4

    def __init__(self, parent, bg):
        k = self.k = theme._k(parent)          # plain tk canvas: apply the DPI / UI scale ourselves
        super().__init__(parent, width=round(self.W * k), height=round(self.H * k), bg=bg,
                         highlightthickness=0, bd=0)
        self.bgc, self.fnt = bg, font("body")
        self.tkfnt = theme._tkfont("body", parent)
        self.ring, self.dot, self.now, self.goal, self.fade = [], [], [], [], {}
        for i in range(len(STAGES)):
            cx = (8 + self.STEP * i) * k
            self.ring.append(self.create_oval(cx - 7 * k, 6 * k, cx + 7 * k, 20 * k, width=0, fill=bg))
            self.dot.append(self.create_oval(cx - 4 * k, 9 * k, cx + 4 * k, 17 * k, width=max(1, round(k)),
                                             fill=bg, outline=bg))
            self.now.append((bg, bg, bg)); self.goal.append(None)
        self.label = self.create_text(self.TEXT_X * k, 13 * k, anchor="w", font=self.tkfnt, fill=C["ink-3"], text="")
        self._text, self._col, self._fading = "", None, False
        self.tip = theme.Tooltip(self, "")

    def _look(self, state):
        bg = self.bgc
        return {"done": (C["ink-2"], C["ink-2"], bg), "pending": (C["inert-fill"],) * 2 + (bg,),
                "now": (C["heat"], C["heat"], _mix(C["heat"], bg, 0.25)), "skip": (bg, C["ink-3"], bg),
                "bad": (C["bad"], C["bad"], bg), "stopped": (C["ink-3"], C["ink-3"], bg)}[state]

    def set(self, states, text, colour, tip=None, snap=False):
        for i, s in enumerate(states):
            if s != self.goal[i]:
                new = self._look(s)
                if snap or self.goal[i] is None:
                    self.now[i] = new; self._paint(i)
                else:
                    self.fade[i] = (self.now[i], new, 0)
                self.goal[i] = s
        if self.fade and not self._fading:
            self._fading = True
            self.after(45, self._step)
        full = _clip(text, self.fnt, self.W - self.TEXT_X - 4)      # CTkFont measures in unscaled px
        if (full, colour) != (self._text, self._col):
            self._text, self._col = full, colour
            self.itemconfigure(self.label, text=full, fill=C[colour])
        if tip is not None:
            self.tip.text = tip

    def _paint(self, i):
        fill, line, ring = self.now[i]
        self.itemconfigure(self.dot[i], fill=fill, outline=line)
        self.itemconfigure(self.ring[i], fill=ring, state="hidden" if ring == self.bgc else "normal")  # paint less

    def _step(self):
        self._fading = False
        try:
            for i, (a, b, n) in list(self.fade.items()):
                n += 1
                t = n / self.FADE
                self.now[i] = tuple(_mix(y, x, t) for x, y in zip(a, b))
                self._paint(i)
                if n >= self.FADE:
                    del self.fade[i]
                else:
                    self.fade[i] = (a, b, n)
            if self.fade:
                self._fading = True
                self.after(45, self._step)
        except tk.TclError:                  # row was rebuilt mid-fade
            pass


# ── Setup controls ───────────────────────────────────────────────────────────
CTL_H = 34


def _dev_shown(code, devices):
    """Device box value: Auto says what it resolved to."""
    if code == "auto":
        return "Auto · GPU" if "cuda" in devices or "mps" in devices else "Auto · CPU"
    return _dev_label(code)


def _dev_label(code):
    """Device menu text: Auto / CPU / GPU (with the card's name once torch has been probed)."""
    if code == "cuda":
        torch = sys.modules.get("torch")
        try:
            name = torch.cuda.get_device_name(0) if torch else ""
        except Exception:
            name = ""
        return f"GPU · {name}" if name else "GPU"
    return {"auto": "Auto", "cpu": "CPU", "mps": "GPU · Metal"}.get(code, code)


SPEAKER_CHOICES = ["Auto"] + [f"{n} speaker{'s' if n > 1 else ''}" for n in range(1, 7)]
SPK_W = 132


class _Drop(ctk.CTkCanvas):
    """Dropdown on a well box (34 px, 8 px radius, rule border, chevron right) drawn on one canvas, that opens a
    native menu. `values()` lists the choices; `disabled(v)` greys one out; set() changes only the shown text."""
    TEXT_PX = 200

    def __init__(self, parent, values, command, disabled=None, key=None, width=1):
        k = self.k = theme._k(parent)
        super().__init__(parent, width=round(width * k), height=round(CTL_H * k), bg=C["panel"],
                         highlightthickness=0, bd=0, cursor="hand2")
        self._values, self._cb, self._off = values, command, disabled or (lambda v: False)
        self.tip = self.tip_text = None
        self.key, self.text, self.fnt = key, "", theme._tkfont("body", self)
        # muted key at the left; values start at one x in every box
        self.x = round(12 * k) + (round((font("body").measure("Language") + 16) * k) if key else 0)
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Button-1>", self._open)

    def _paint(self):
        self.delete("all")
        w, h, k = self.winfo_width(), self.winfo_height(), self.k
        _rrect(self, 0, 0, w, h, round(R_INNER * k), C["well"], C["rule"], max(1, round(k)))
        for x, text, col, anchor in ((round(12 * k), self.key, "ink-3", "w"), (self.x, self.text, "ink", "w"),
                                     (w - round(12 * k), "▾", "ink-2", "e")):
            if text:
                self.create_text(x, h / 2, text=text, anchor=anchor, font=self.fnt, fill=C[col])

    def set(self, text):
        shown = self.text = _clip(text, font("body"), self.TEXT_PX)
        self._paint()
        if self.tip:                             # a cut value is spelled out in the tooltip
            self.tip.text = self.tip_text + (f" Now: {text}." if shown != text else "")

    def _open(self, _e=None):
        m = tk.Menu(self, tearoff=0, bg=C["panel"], fg=C["ink"], activebackground=C["raised"],
                    activeforeground=C["ink"], disabledforeground=C["ink-3"], bd=0, relief="flat",
                    font=theme._tkfont("body", self))
        for v in self._values():
            m.add_command(label=v, state="disabled" if self._off(v) else "normal",
                          command=lambda v=v: self._cb(v))
        m.tk_popup(self.winfo_rootx(), self.winfo_rooty() + self.winfo_height())


class _Check(ctk.CTkFrame):
    """Format chip with a check box: selected = raised fill, ink-2 border, filled box with a tick."""

    def __init__(self, parent, text, command):
        super().__init__(parent, height=CTL_H, corner_radius=R_INNER, border_width=1)
        self.pack_propagate(False)
        self.configure(width=font("button").measure(text) + 54)
        self._on, self._cb = False, command
        k = self._k = theme._k(self)
        self._tk = theme._tkfont("label", self)
        self.box = tk.Canvas(self, width=round(16 * k), height=round(16 * k), highlightthickness=0, bd=0)
        self.box.pack(side="left", padx=(12, 8))
        self.name = ctk.CTkLabel(self, text=text, font=font("button"))
        self.name.pack(side="left")
        for w in (self, self.box, self.name):
            w.bind("<Button-1>", self._click)
            w.configure(cursor="hand2")
        self._paint()

    def _paint(self):
        on = self._on
        self.configure(fg_color=C["raised"] if on else "transparent",
                       border_color=C["ok"] if on else C["rule"])
        cv, e = self.box, round(16 * self._k)
        cv.configure(bg=C["raised"] if on else C["panel"])
        cv.delete("all")
        cv.create_rectangle(1, 1, e - 1, e - 1, width=1, fill=C["ok"] if on else C["panel"],
                            outline=C["ok"] if on else C["ink-3"])
        if on:
            cv.create_text(e / 2, e / 2, text="✓", fill=C["ground"], font=self._tk)
        self.name.configure(text_color=C["ink"] if on else C["ink-2"])

    def _click(self, _e=None):
        self.set(not self._on)
        self._cb(self._on)

    def get(self):
        return self._on

    def set(self, on):
        if bool(on) != self._on:
            self._on = bool(on)
            self._paint()


def _rrect(cv, x0, y0, x1, y1, r, fill, edge=None, b=1, tags=()):
    """Anti-aliased rounded box on a CTkCanvas: font-shape corners, as CTk's draw engine makes them."""
    for col, d in ((edge, 0), (fill, b)) if edge else ((fill, 0),):
        for x, y in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r)):
            cv.create_aa_circle(x, y, r - d, fill=col, tags=tags)
        cv.create_rectangle(x0 + d, y0 + r, x1 - d, y1 - r, width=0, fill=col, tags=tags)
        cv.create_rectangle(x0 + r, y0 + d, x1 - r, y1 - d, width=0, fill=col, tags=tags)


class _Formats(ctk.CTkCanvas):
    """All format chips on one canvas (was 6 windows per chip), two equal columns over the setup rows.
    Each looks like _Check; click, or Tab to it and move with the arrows, Space ticks."""

    def __init__(self, parent, labels, command, tips):
        k = self.k = theme._k(parent)
        self.h, self.gap = round(CTL_H * k), round(4 * k)
        n = (len(labels) + 1) // 2
        super().__init__(parent, width=1, height=n * self.h + (n - 1) * self.gap, bg=C["panel"],
                         highlightthickness=0, bd=0, cursor="hand2", takefocus=1)
        self.keys, self.labels, self.cb = list(labels), labels, command
        self.on, self.cur, self.focused = set(), self.keys[0], False
        self.fb, self.ft = theme._tkfont("button", self), theme._tkfont("label", self)
        self.tip = ClipTip(self, "", lambda e: tips.get(self._at(e.x, e.y), ""))
        self.bind("<Configure>", lambda e: self._paint())
        self.bind("<Button-1>", lambda e: self._click(self._at(e.x, e.y)), add="+")
        self.bind("<space>", lambda e: self._click(self.cur))
        for seq, on in (("<FocusIn>", True), ("<FocusOut>", False)):
            self.bind(seq, lambda e, on=on: (setattr(self, "focused", on), self._paint()))
        for seq, d in (("<Left>", -1), ("<Right>", 1), ("<Up>", -2), ("<Down>", 2)):
            self.bind(seq, lambda e, d=d: self._move(d))

    def _box(self, i):
        r, c = divmod(i, 2)
        half = (self.winfo_width() - 2 * self.gap) / 2
        x0, y0 = round(c * (half + 2 * self.gap)), r * (self.h + self.gap)
        return x0, y0, round(x0 + half), y0 + self.h

    def _at(self, x, y):
        return next((key for i, key in enumerate(self.keys)
                     if (b := self._box(i))[0] <= x < b[2] and b[1] <= y < b[3]), None)

    def _paint(self):
        self.delete("all")
        k, e = self.k, round(16 * self.k)
        for i, key in enumerate(self.keys):
            x0, y0, x1, y1 = self._box(i)
            on, my, bx = key in self.on, (y0 + y1) / 2, x0 + round(12 * k)
            edge = C["ember"] if self.focused and key == self.cur else C["ok"] if on else C["rule"]
            _rrect(self, x0, y0, x1, y1, round(R_INNER * k), C["raised"] if on else C["panel"], edge, max(1, round(k)))
            self.create_rectangle(bx + 1, my - e / 2 + 1, bx + e - 1, my + e / 2 - 1, width=1,
                                  fill=C["ok"] if on else C["panel"], outline=C["ok"] if on else C["ink-3"])
            if on:
                self.create_text(bx + e / 2, my, text="✓", fill=C["ground"], font=self.ft)
            self.create_text(bx + e + round(8 * k), my, anchor="w", text=self.labels[key], font=self.fb,
                             fill=C["ink"] if on else C["ink-2"])

    def _move(self, d):
        self.cur = self.keys[(self.keys.index(self.cur) + d) % len(self.keys)]
        self._paint()

    def _click(self, key):
        if key is None:
            return
        self.cur, on = key, key not in self.on
        self.on ^= {key}
        self._paint()
        self.cb(key, on)

    def get(self):
        return [f for f in self.keys if f in self.on]

    def set(self, keys):
        if set(keys) != self.on:
            self.on = set(keys)
            self._paint()


class _Row(ctk.CTkCanvas):
    """One file row on one canvas (was ~25 windows): number, name and size, length, pill, Open and × are
    items, hit-tested; the mini lane and the stage track are its only child windows. The spare width
    sits right of the name column: items tagged "r" move with the right edge."""

    def __init__(self, parent, scr, i, job):
        k = self.k = theme._k(parent)
        self._span, self._dx, self._hov = round(sum(COLS.values()) * k), 0, None
        h, bg, mid = round(ROW_H * k), C["panel"], round(ROW_H * k) / 2
        super().__init__(parent, width=self._span, height=h, bg=bg, highlightthickness=0, bd=0)
        x, acc = {}, 0
        for key, w in COLS.items():
            x[key], acc = round(acc * k), acc + w
        X = lambda key, off=0: x[key] + round(off * k)
        f = lambda role: theme._tkfont(role, self)
        full, size = os.path.basename(job.audio), _size(job.audio)
        ClipTip(self, "", lambda e: full if X("name") <= e.x < X("name", COLS["name"]) else "")
        if i:
            self.create_line(0, 0, 10 ** 5, 0, fill=C["rule-soft"])
        self.create_text(0, mid, anchor="w", text=f"{i + 1:02d}", font=f("mono-sm"), fill=C["ink-3"])
        fb, fh = f("body"), f("hint")
        room = (COLS["name"] - 10) * k - (theme.measure_font(fh, self).measure(size) + 8 * k if size else 0)
        name = self.create_text(X("name"), mid, anchor="w",
                                text=_clip(full, theme.measure_font(fb, self), room), font=fb,
                                fill=C["ink-2" if job.status == "done" else "ink"])
        if size:
            self.create_text(self.bbox(name)[2] + round(8 * k), mid, anchor="w", text=size, font=fh,
                             fill=C["ink-3"])
        self.mini = MiniLane(self, width=round((COLS["mini"] - 24) * k))
        self.create_window(X("mini", 12), mid, anchor="w", window=self.mini, tags="r")
        self.len = self.create_text(X("len", COLS["len"] - 12), mid, anchor="e", font=f("mono-sm"), fill=C["ink"],
                                    tags="r", text=tc(job.duration) if job.duration else "—")
        self.track = StageTrack(self, bg)
        self.create_window(X("track"), mid, anchor="w", window=self.track, tags="r")
        text, tint, ink = PILLS[job.status]
        doubt = lang_label(job)[1] if job.status == "done" else ""
        if doubt:
            tint = ink = "warn"
        text = f"  {text.upper()}  "
        pw, ph, cx = theme.measure_font(f("label"), self).measure(text) + 2 * round(11 * k), round(22 * k), X("pill", 45)
        self.create_line(cx - (pw - ph) / 2, mid, cx + (pw - ph) / 2, mid, width=ph, capstyle="round",
                         fill=_mix(C[tint], bg, 0.14), tags="r")
        self.create_text(cx, mid, text=text, font=f("label"), fill=C[ink], tags="r")
        ClipTip(self, "", lambda e: doubt if abs(e.x - self._dx - cx) <= pw / 2 else "")
        self.btns = {}                    # name -> (box, corner radius, action)
        if job.status == "done":
            self.btns["open"] = ((X("open"), mid - 14 * k, X("open", 52), mid + 14 * k), round(6 * k),
                                 lambda: scr._open(job))
            self.create_text(X("open", 26), mid, text="Open", font=f("button"), fill=C["ink-2"], tags="r")
        if job.status != "running":
            self.btns["x"] = ((X("x"), mid - 14 * k, X("x", 28), mid + 14 * k), round(14 * k), lambda: scr._remove(job))
            self.create_text(X("x", 14), mid, text="×", font=f("body"), fill=C["ink-3"], tags="r")
        self.bind("<Configure>", self._fit, add="+")
        self.bind("<Motion>", lambda e: self._hover(self._at(e.x, e.y)), add="+")
        self.bind("<Leave>", lambda e: self._hover(None), add="+")
        self.bind("<Button-1>", lambda e: (b := self._at(e.x, e.y)) and self.btns[b][2](), add="+")

    def _fit(self, e):
        d = max(0, e.width - self._span) - self._dx
        if d:
            self.move("r", d, 0)
            self._dx += d

    def _at(self, x, y):
        return next((n for n, ((a, b, c, d), _, _) in self.btns.items()
                     if a <= x - self._dx < c and b <= y < d), None)

    def _hover(self, name):
        if name == self._hov:
            return
        self._hov = name
        self.delete("hov")
        self.configure(cursor="hand2" if name else "")
        if name:
            (a, b, c, d), r, _ = self.btns[name]
            _rrect(self, a + self._dx, b, c + self._dx, d, r, C["raised"], tags=("hov", "r"))
            self.tag_lower("hov")

    def set_len(self, job):
        self.itemconfigure(self.len, text=tc(job.duration) if job.duration else "—")


class Screen:
    def __init__(self, parent, state):
        self.state = state
        self.page = ctk.CTkFrame(parent, fg_color=C["ground"], corner_radius=0)
        self.cur = None                 # job shown in the session block
        self.run_id = self._shown = None
        self.minis = {}                 # job id -> MiniLane
        self.tracks = {}                # job id -> StageTrack
        self.rows = {}                  # job id -> _Row
        self._ticking = self._dirty = False
        self.page.columnconfigure(0, weight=1)
        self.page.rowconfigure(1, weight=1)
        self._build_session()
        self._build_files(self.page)
        self._build_setup(self.page)
        done = [j for j in state.queue.jobs if j.finished]
        self.cur = max(done, key=lambda j: j.finished) if done else None
        self._sync()
        self._refresh_lists()
        self._paint_conf()
        self._paint_session()
        for topic, cb in (("transcribe", self._run), ("browse", self.browse), ("drop", self.add),
                          ("drag", self._drag), ("job", self._on_job),
                          ("runner", self._on_runner), ("finished", self._on_finished),
                          ("settings", lambda **_: self._sync())):
            state.on(topic, cb)

    def _link(self, parent, text, command):
        """Quiet text link with an arrow (a plain tk button: a CTkButton is 3 windows)."""
        b = tk.Button(parent, text=text, font=theme._tkfont("label", parent), fg=C["ink-2"], bg=C["panel"],
                      activebackground=C["panel"], activeforeground=C["ink"], bd=0, relief="flat",
                      highlightthickness=0, padx=0, pady=0, cursor="hand2", command=command)
        b.bind("<Enter>", lambda e: b.configure(fg=C["ink"]))
        b.bind("<Leave>", lambda e: b.configure(fg=C["ink-2"]))
        return b

    # ── Session block ─────────────────────────────────────────────────────────
    def _build_session(self):
        box = self.session = theme.panel(self.page)
        box.grid(row=0, column=0, sticky="ew", padx=28, pady=(10, 0))
        k = self.k = theme._k(box)                    # plain tk head: a CTkLabel is 3 windows
        head = tk.Frame(box, bg=C["panel"])
        head.pack(fill="x", padx=round(PAD_LG * k), pady=(round(10 * k), round(6 * k)))
        lab = lambda role, col: tk.Label(head, text="", font=theme._tkfont(role, head), fg=C[col], bg=C["panel"],
                                         bd=0, padx=0, pady=0)
        self.s_name = lab("heading", "ink")
        self.s_name.pack(side="left")
        self.s_tip = theme.Tooltip(self.s_name, "")
        self.s_pill = lab("label", "ember")
        self.s_head = head
        self.s_meta = lab("mono-sm", "ink-3")
        self.m_tip = ClipTip(self.s_meta)
        self.s_pct = lab("mono-hero", "ember")
        self.s_pct.pack(side="right", padx=(round(16 * k), 0))
        self._pct = Tween(self.page, lambda v: self.s_pct.configure(text=f"{round(v * 100)} %"))
        self.readouts = theme.readout_strip(head, [("Elapsed", "—"), ("Left", "—")])
        self.readouts.pack(side="right")
        self.lane = SessionLane(box, height=64)       # start/end timecodes sit in its corners
        self.lane.pack(fill="x", padx=PAD_LG, pady=(0, 12))

    def _stage_verb(self, j):
        task = VERBS.get(j.cfg.get("task"), "Transcribing")
        rec = self.state.stages.get(j.id) or {}
        live = next((k for k, _ in STAGES if (rec.get(k) or {}).get("t0") and not (rec.get(k) or {}).get("t1")), None)
        return task if live == "transcribe" else STAGE_VERBS.get(live, task)

    def _paint_session(self):
        j = self.cur
        if j is None:                  # idle: the file rows already name what's queued (Say-It-Once)
            self._pct.to(0, True); self.s_pct.configure(text="")
            self._shown = None
            self.s_pill.pack_forget()
            for w in (self.readouts, self.lane):
                w.pack_forget()
            self.session.grid_remove()
            return
        self.session.grid()
        if not self.lane.winfo_manager():
            self.readouts.pack(side="right")
            self.lane.pack(fill="x", padx=PAD_LG, pady=(0, 12))
        full = os.path.basename(j.audio)
        fh = theme._tkfont("heading", self.s_head)
        self.s_name.configure(text=_clip(full, theme.measure_font(fh, self.s_name), 300 * self.k))
        self.s_tip.text = full
        verb = VERBS.get(j.status) or self._stage_verb(j)
        fg = C[{"done": "ok", "failed": "bad", "cancelled": "warn"}.get(j.status, "ember")]
        if self.s_pill.cget("text") != verb or self.s_pill.cget("fg") != fg:
            self.s_pill.configure(text=verb, fg=fg)
        self.s_pill.pack(side="left", padx=round(14 * self.k), after=self.s_name)
        self.s_meta.pack_forget()
        lang, doubt = lang_label(j)
        parts = [tc(j.duration) if j.duration else "", lang]
        self.s_meta.configure(text=" · ".join(p for p in parts if p), fg=C["warn" if doubt else "ink-3"])
        self.m_tip.text = doubt
        self.s_meta.pack(side="left", after=self.s_pill)
        fresh, self._shown = j.id != self._shown, j.id      # a different file snaps, the same one eases
        self._pct.to(j.progress, fresh)
        self.lane.set_duration(j.duration)
        self.lane.set_bins(j.bins)
        self.lane.set_progress(j.progress, fresh)
        self._paint_readouts()

    def _paint_readouts(self):
        j = self.cur
        if j is None:
            return
        running = self.state.running is j
        now = time.time()
        el = (j.finished or now) - (j.started or now) if j.started else 0
        self.readouts.set("Elapsed", tc(el) if j.started else "—")
        self.readouts.set("Left", tc(el / j.progress - el) if running and j.progress > 0.02 else "—")
        self.readouts.show("Left", j.status == "running")

    # ── Files ─────────────────────────────────────────────────────────────────
    def _build_files(self, col):
        box = theme.panel(col)
        k = theme._k(box)
        head = tk.Frame(box, bg=C["panel"])
        head.pack(fill="x", padx=round(PAD_LG * k), pady=(round(10 * k), round(4 * k)))
        tk.Label(head, text="Queue", font=theme._tkfont("heading", head), fg=C["ink"], bg=C["panel"],
                 bd=0, padx=0, pady=0).pack(side="left")
        self.files_r = link = self._link(head, "Open queue  →", lambda: self.state.emit("screen", name="queue"))
        link.pack(side="right")
        self.files_tip = ClipTip(link)
        box.grid(row=1, column=0, sticky="nsew", padx=28, pady=(8, 8))
        self.drop_col = C["panel"]
        drop = self.drop = ctk.CTkFrame(box, fg_color=self.drop_col, corner_radius=R_INNER, height=34)
        drop.pack_propagate(False)
        self.dash = ctk.CTkCanvas(drop, bg=self.drop_col, highlightthickness=0)
        self.dash.place(x=0, y=0, relwidth=1, relheight=1)
        self.dash.bind("<Configure>", lambda e: self._paint_dash(self._over))
        self.dnd = bool(getattr(self.state, "dnd", False))
        self.title = _label(drop, "", "body", "ink")
        self.title.configure(fg_color=self.drop_col)
        self.drop_labels = [self.title]
        self.add_btn = theme.pill(drop, "Add files", self.browse)
        theme.tooltip(self.add_btn, ADD_TIP)
        self._over = self._empty = None
        self.file_list = _scroller(box)
        self._set_empty(True)

    def _set_empty(self, empty):
        """Empty list: one big drop area with the heat Add files pill (and a secondary Transcribe);
        with files: the slim row above the list. An instant swap, not a tween."""
        if empty == self._empty:
            return
        self._empty = empty
        for w in (self.title, self.add_btn):
            w.pack_forget(); w.place_forget()
        self.file_list.pack_forget()
        self.drop.pack_forget()
        if empty:
            self.drop.pack(fill="both", expand=True, padx=PAD_LG, pady=(4, 12))
            self.title.configure(font=font("heading"), text_color=C["ink"],
                                 text="Drop audio or video" if self.dnd else "Add audio or video")
            self.title.place(relx=0.5, rely=0.5, y=-12, anchor="s")
            theme.restyle_pill(self.add_btn, True)
            self.add_btn.place(relx=0.5, rely=0.5, y=4, anchor="n")
        else:
            self.drop.pack(fill="x", padx=PAD_LG)
            self.title.configure(font=font("body"), text="Drop more here" if self.dnd else "Add more here")
            self.title.configure(text_color=C["ink-2"])
            self.title.pack(side="left", padx=16)
            theme.restyle_pill(self.add_btn, False, height=30)
            self.add_btn.pack(side="right", padx=8)
            self.file_list.pack(fill="both", expand=True, padx=(PAD_LG, PAD_LG - 8), pady=(4, 6))
        self.state.emit("files-empty", empty=empty)
        self.drop.update_idletasks()
        self._paint_dash(self._over)

    def _paint_dash(self, over):
        self._over = over
        col = _mix(C["heat"], C["panel"], 0.14) if over else C["panel"]
        for w in (self.drop, self.dash, *self.drop_labels):
            w.configure(**{"bg" if w is self.dash else "fg_color": col})
        cv = self.dash; cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        cv.create_rectangle(1, 1, w - 2, h - 2, dash=(4, 3),
                            outline=C["ember"] if over else C["inert-fill"])

    def _drag(self, on, **_):
        """The window is being dragged over: tint the drop area."""
        self._paint_dash(on)

    def browse(self, **_):
        last = self.state.settings.get("last_dir") or ""
        paths = filedialog.askopenfilenames(
            title="Add audio or video", initialdir=last if os.path.isdir(last) else None,
            filetypes=[("Audio and video", MEDIA_TYPES), ("All files", "*.*")])
        if paths:
            self.state.set("last_dir", os.path.dirname(os.path.normpath(paths[0])))
            self.add(paths)

    def add(self, paths, **_):
        """Queue files; a folder adds its media files (not its subfolders) and reports the counts."""
        files, folder, skipped = [], False, 0
        for p in paths:
            if os.path.isdir(p):
                folder = True
                names = [n for n in os.listdir(p) if os.path.isfile(os.path.join(p, n))]
                files += sorted(os.path.join(p, n) for n in names if n.lower().endswith(EXTS))
                skipped += sum(not n.lower().endswith(EXTS) for n in names)
            else:
                files.append(os.path.normpath(p))
        added = len(self.state.add_files(files)) if files else 0
        if folder:
            tail = f" · skipped {skipped} not audio/video" if skipped else ""
            self.state.emit("status", text=f"Added {added} file{'' if added == 1 else 's'} from folder{tail}")

    def _run(self, **_):
        if self.state.running is not None:
            return
        job = self.state.queue.next_waiting()
        if job:
            self.state.run_job(job)
        else:
            self.browse()

    def _remove(self, job):
        if job.status == "running" or job is self.state.running:
            return
        self.state.queue.remove(job.id)
        self.state.queue.save(self.state.queue_path)
        self.state.emit("job", job=job)

    def _refresh_lists(self):
        self._dirty = False
        q = self.state.queue.jobs
        for w in self.file_list.winfo_children():
            w.destroy()
        self.minis, self.tracks, self.rows = {}, {}, {}
        if self.cur is None:
            self._paint_session()
        live = [j for j in q if j.status in ("waiting", "running")
                or (j.id in self.state.stages and j.status not in ("done", "failed"))]
        old = sorted((j for j in q if j.status in ("done", "failed")),
                     key=lambda j: j.finished or 0, reverse=True)[:FINISHED_ROWS]
        jobs = [j for j in q if j in live or j in old]
        total = sum(j.duration or 0 for j in jobs)
        self._set_empty(not jobs)
        self.files_tip.text = f"{len(jobs)} file{'' if len(jobs) == 1 else 's'} · {tc(total) if total else '—'}"
        for i, j in enumerate(jobs):
            row = self.rows[j.id] = _Row(self.file_list, self, i, j)
            row.pack(fill="x")
            self.minis[j.id], self.tracks[j.id] = row.mini, row.track
            row.mini.set(j.bins, j.progress if j.status == "running" else 0, snap=True)
            self._paint_track(j, tip=True, snap=True)

    def _paint_track(self, job, tip=False, snap=False):
        """Repaint one file's dots and label (the only repaint a progress event causes)."""
        trk = self.tracks.get(job.id)
        if trk is None:
            return
        rec = self.state.stages.get(job.id)
        states = stage_view(job, rec)
        text, colour = stage_label(job, states)
        body = stage_tip(job, states, rec) if tip else None
        err = (job.error or "").strip() if job.status == "failed" else ""
        if body is not None and err and ("\n" in err or _clip(text, trk.fnt, trk.W - trk.TEXT_X - 4) != text):
            body = "\n".join((body, err))
        trk.set(states, text, colour, body, snap)

    # ── Setup: language, model, device, task, formats, switches ───────────────
    def _build_setup(self, col):
        box = theme.panel(col)
        box.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 10))
        s = self.state
        k = theme._k(box)                             # plain tk frames and labels: scale by hand
        body = tk.Frame(box, bg=C["panel"])
        body.pack(fill="x", padx=round(PAD_LG * k), pady=round(8 * k))
        body.columnconfigure(1, weight=1)
        for i in (0, 2):                              # fixed side columns (grid sizes are raw pixels)
            body.columnconfigure(i, minsize=round(300 * k))
        # one 4-row grid shared by the three columns: row r of each column sits on the same lines
        for i, title in enumerate(("Recognition", "Output", "Options")):
            tk.Label(body, text=title.upper(), font=theme._tkfont("label", body), fg=C["ink-3"], bg=C["panel"],
                     anchor="w", bd=0, padx=0, pady=0).grid(row=0, column=i, sticky="ew", pady=(0, round(4 * k)),
                                                            padx=(0 if i == 0 else round(14 * k), 0))
        self.more = more = self._link(body, "More settings  →",
                                      lambda: self.state.emit("settings-tab", name="Accuracy"))
        more.grid(row=0, column=2, sticky="e", pady=(0, round(4 * k)))
        theme.tooltip(more, "Hotwords, decoding and all other options.")

        def cell(row, col):
            f = tk.Frame(body, bg=C["panel"], height=round(CTL_H * k))
            f.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else round(14 * k), 0),
                   pady=(0, round(4 * k) if row < 4 else 0))
            f.grid_propagate(False)
            f.pack_propagate(False)
            return f
        rec = [cell(r, 0) for r in range(1, 5)]
        opt = [cell(r, 2) for r in range(1, 5)]
        self._model_labels = {}
        devs = lambda: [_dev_label(d) for d in s.devices]
        self.lang = _Drop(rec[0], lambda: list(LANGUAGES), lambda v: s.set("language", v), key="Language")
        self.model = _Drop(rec[1], lambda: list(self._model_labels), self._pick_model, key="Model")
        self.device = _Drop(rec[2], devs, lambda v: s.set("device", next(
            (d for d in s.devices if _dev_label(d) == v), v)), key="Device")
        self.task = _Drop(rec[3], lambda: list(TASKS.values()), self._pick_task,
                          lambda v: v == TASKS["translate"] and not supports_translate(s.settings["model"]), key="Task")
        for k, w in enumerate((self.lang, self.model, self.device, self.task)):
            w.pack(fill="both", expand=True)
        for w, tx in ((self.lang, "The language spoken. Auto-detect listens to the start of each file."),
                      (self.model, "The model that turns speech into text. Larger is more accurate and slower."),
                      (self.device, "Where the work runs. GPU is much faster than CPU.")):
            w.tip_text = tx
            w.tip = theme.Tooltip(w, tx)
        self._task_tip = self.task.tip = theme.Tooltip(self.task, "")
        # output: formats on 2 equal columns over the shared rows (Audacity labels last, row 4 left)
        self.chips = _Formats(body, {f: FORMAT_LABELS.get(f, f.upper()) for f in EXPORT_FORMATS}, self._fmt,
                              {"aud": AUD_TIP})
        self.chips.grid(row=1, column=1, rowspan=4, sticky="new", padx=(round(14 * k), 0))
        # options: one on/off row per grid row
        self.switches = {}
        for n, (key, title, desc) in enumerate(SWITCHES):
            row = opt[n]
            sw = self.switches[key] = theme.light_switch(row, title, command=lambda v, k=key: s.set(k, v))
            sw.pack(fill="both", expand=True)
            sw._cv.grid_configure(column=2)
            sw.rowconfigure(0, weight=1)                       # title centred on the 34 px row, level with the chips
            theme.tooltip(sw, desc)
            if key == "diarize":
                self.spk_box = _Drop(sw, lambda: list(SPEAKER_CHOICES), self._pick_speakers, width=SPK_W)
                self.spk_box.TEXT_PX = 96
                self.spk_box.grid(row=0, column=1, rowspan=2, padx=(round(8 * k), 0))
                theme.tooltip(self.spk_box, "How many people talk in the recording. Auto lets the app work it out.")

    def _pick_model(self, label):
        self.state.set("model", self._model_labels.get(label, label))

    def _pick_task(self, label):
        self.state.set("task", next(k for k, v in TASKS.items() if v == label))

    def _pick_speakers(self, label):
        n = "" if label == "Auto" else label.split()[0]
        self.state.set("min_speakers", n)
        self.state.set("max_speakers", n)

    def _paint_conf(self):
        """Language menu shows the setting only; the detected language lives in the session panel."""
        self.lang.set(self.state.settings["language"])

    def _fmt(self, key, on):
        cur = list(self.state.settings["formats"])
        if on and key not in cur:
            cur.append(key)
        elif not on and key in cur:
            cur.remove(key)
        if not cur:                                    # at least one format stays ticked
            self.chips.set([key])
            return
        self.state.set("formats", [f for f in EXPORT_FORMATS if f in cur])

    # ── Settings -> widgets ───────────────────────────────────────────────────
    def _sync(self):
        s = self.state.settings
        md = self.state.model_dir()
        self._model_labels = {}
        for n, (_, mb) in MODEL_INFO.items():
            size = f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb} MB"
            cached = " · cached" if os.path.isdir(model_cache_dir(n, md)) else ""
            self._model_labels[f"{n} · {size}{cached}"] = n
        self.model.set(next((k for k, v in self._model_labels.items() if v == s["model"]), s["model"]))
        self.device.set(_dev_shown(s["device"], self.state.devices))
        self.task.tip_text = ("Transcribe writes what is said; Translate writes it in English."
                              if supports_translate(s["model"]) else "This model can only transcribe.")
        self.task.set(TASKS.get(s["task"], TASKS["transcribe"]))
        self._paint_conf()
        self.chips.set(s["formats"])
        for k, sw in self.switches.items():
            sw.set(bool(s[k]))
        (self.spk_box.grid if s["diarize"] else self.spk_box.grid_remove)()
        lo, hi = s["min_speakers"], s["max_speakers"]
        n = lo or hi
        self.spk_box.set("Auto" if not n else f"{lo or '1'}–{hi or '?'} speakers" if lo != hi
                         else f"{n} speaker{'' if n == '1' else 's'}")

    def _open(self, job):
        # switch first: the Transcript screen is built lazily and must exist to hear the open
        self.state.emit("screen", name="transcript")
        self.state.emit("open-transcript", job=job)

    # ── Events ────────────────────────────────────────────────────────────────
    def _begin(self, job):
        self.run_id, self.cur = job.id, job
        if not self._ticking:
            self._ticking = True
            self.page.after(1000, self._tick)

    def _tick(self):
        if self.state.running is None:
            self._ticking = False
            return
        self._paint_readouts()
        if self.cur is self.state.running:
            self._paint_track(self.cur, tip=True)       # keeps the running stage's seconds fresh
        self.page.after(1000, self._tick)

    def _on_job(self, job, **_):
        if job is not None and job.status == "running" and job.id != self.run_id:
            self._begin(job)
            self._paint_session()
        if not self._dirty:
            self._dirty = True
            self.page.after_idle(self._refresh_lists)

    def _on_runner(self, job, ev, **_):
        if job.id != self.run_id:
            self._begin(job)
        t = ev.get("t")
        if t == "wave":
            self.lane.set_duration(job.duration)
            self.lane.set_bins(job.bins)
            self._paint_session()
            if job.id in self.minis:
                self.minis[job.id].set(job.bins, job.progress)
                self.rows[job.id].set_len(job)          # the length is known from the wave on
        elif t == "stage":
            self._paint_track(job, tip=True)
            self._paint_session()
        elif t in ("progress", "done"):           # the 30 fps estimate ("est") or a reported value
            self.lane.set_progress(job.progress)
            self._pct.to(job.progress)
            if job.id in self.minis:
                self.minis[job.id].set(job.bins, job.progress)
            if not ev.get("est"):
                self._paint_track(job)
        elif t == "language":
            self._paint_session()
            self._paint_conf()
            self._paint_track(job)
        elif t == "warn":
            self._paint_track(job, tip=True)
        self._paint_readouts()

    def _on_finished(self, job, **_):
        self.cur = job
        self._paint_session(); self._paint_conf()
        self._refresh_lists()


def build(parent, state):
    """Fixed-layout Transcribe screen; the transport bar stays in the shell."""
    return Screen(parent, state).page
