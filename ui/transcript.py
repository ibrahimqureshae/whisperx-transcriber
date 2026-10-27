"""Transcript screen — speaker-lane timeline with playhead, segment list, find, speakers, export, doubtful words, run."""
import bisect
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk

import customtkinter as ctk

from core import pipeline
from core.jobqueue import load_transcript
from core.pipeline import EXPORT_FORMATS
from ui import rtl, theme
from ui.models import LightScrollbar
from ui.queue import _speed_ratio
from ui.theme import C, GAP, LANES, PAD, PAD_LG, font, _mix
from core.settings import validate
from ui.transcribe import AUD_TIP, FORMAT_LABELS, LANG_NAMES, _Check, _clip, _label
from ui.waveform import ClipTip, Glide, clip_px, timecode as tc

WIN = 150.0                 # seconds in the timeline view
NAME_W, RULER_H, LANE_H, LANE_GAP = 150, 24, 46, 4
EMPTY_ROWS = 8      # finished files listed when no job is open
FILE_W, DUR_W, DATE_W = 560, 70, 130     # finished-list columns: file | length | finished | Open
SHOW_LANES = 3              # lanes visible at once; more scroll inside the timeline
LOW = 0.5                   # words under this score are doubtful
_EDGE = " \t.,;:!?\"'()[]{}،؛؟۔…“”‘’"


def _rrect(cv, x0, y0, x1, y1, r, fill, edge=None, b=1):
    """theme.rrect on whole physical pixels: coordinates, radius and edge are rounded, radius fits the box."""
    x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
    r = max(1, min(round(r), (x1 - x0) // 2, (y1 - y0) // 2))
    if edge and min(x1 - x0, y1 - y0) <= 2 * b:         # too small for an inset fill
        fill, edge = edge, None
    theme.rrect(cv, x0, y0, x1, y1, r, fill, edge, b)


def _btn(parent, text, command, w=None, h=30, flat=True):
    return ctk.CTkButton(parent, text=text, height=h, width=w or font("button").measure(text) + 20,
                         corner_radius=h // 2, fg_color="transparent" if flat else C["raised"],
                         hover_color=C["raised"] if flat else C["rule"],
                         text_color=C["ink-2"] if flat else C["ink"], font=font("button"), command=command)


def _entry(parent, w, h, placeholder):
    return ctk.CTkEntry(parent, width=w, height=h, corner_radius=8, border_width=1, fg_color=C["well"],
                        border_color=C["rule"], text_color=C["ink"], placeholder_text=placeholder,
                        placeholder_text_color=C["ink-3"], font=font("body"))


def _digits(s):
    return int(s) if str(s).strip().isdigit() else None


def _load_segments(state, job):
    if state.transcripts.get(job.id):
        return state.transcripts[job.id]
    stored = load_transcript(state.base, job.id)
    if stored:
        return stored
    for path in job.outputs or []:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f) if path.lower().endswith(".json") else None
        except Exception:
            continue
        if isinstance(data, dict) and data.get("segments"):
            return data["segments"]
    return None


def _open_folder(path):
    try:
        os.startfile(path) if sys.platform == "win32" else subprocess.Popen(
            ["open" if sys.platform == "darwin" else "xdg-open", path])
    except Exception:
        pass


TABS = ("Speakers", "Export", "Doubtful", "Re-run")
COL_T, COL_S = 70, 120      # transcript line grid: time | speaker | text (logical px, scaled with the UI)


class Screen:
    def __init__(self, parent, state):
        self.state = state
        self.page = ctk.CTkFrame(parent, fg_color=C["ground"], corner_radius=0)
        self.job = None
        self.segs, self.spk, self.talk = [], [], {}
        self.words, self.starts, self.low = [], [], []
        self.lines, self._rtl, self._wrap_w = [], False, 0     # Text line -> segment index; RTL rows wrap to the width
        self.pos, self.vs, self.total, self.vy = 0.0, 0.0, 1.0, 0
        self._lit = None
        self._sb = False        # dragging the lane scrollbar thumb
        self._goal = 0.0
        self._glide = Glide(self.page, self._glide_pos)
        self._find_at = "1.0"
        self.fmt_chips, self.name_entries, self._empty_err, self._tab = {}, {}, None, None
        self._build_empty()
        self._build_body()
        state.on("open-transcript", lambda job, **_: self.open(job))
        state.on("settings", lambda **_: self._paint_export())
        state.on("finished", lambda **_: self.job is None and self._paint_empty(self._empty_err))
        done = [j for j in state.queue.jobs if j.finished and j.status == "done"]
        self.open(max(done, key=lambda j: j.finished) if done else None)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _kv(self, parent, rows):
        for w in parent.winfo_children():
            w.destroy()
        parent.columnconfigure(0, weight=1)
        for i, (k, v) in enumerate(rows):
            _label(parent, k, "body", "ink-2").grid(row=i, column=0, sticky="w", pady=3)
            _label(parent, v, "mono-sm", "ink").grid(row=i, column=1, sticky="e", pady=3)

    def _build_empty(self):
        self.empty = ctk.CTkFrame(self.page, fg_color="transparent")
        card = theme.panel(self.empty)
        card.pack(fill="both", expand=True, padx=28, pady=22)
        _label(card, "Finished", "heading", "ink").pack(anchor="w", padx=PAD_LG, pady=(PAD, 4))
        self.empty_msg = _label(card, "", "body", "bad")
        self._empty_card, self.empty_list = card, None      # list is (re)built per paint, see _paint_empty

    def _paint_empty(self, err=None):
        """No job open: the screen lists the finished jobs, newest first, each with Open."""
        self._empty_err = err
        done = sorted((j for j in self.state.queue.jobs if j.finished and j.status == "done"),
                      key=lambda j: j.finished, reverse=True)[:EMPTY_ROWS]    # the Queue screen has the rest
        if self.empty_list is not None:
            self.empty_list.destroy()
        self.empty_list = ctk.CTkFrame(self._empty_card, fg_color="transparent")
        self.empty_list.pack(fill="both", expand=True, padx=PAD_LG, pady=(0, 10))
        self.empty_msg.pack_forget()
        if err:
            self.empty_msg.configure(text=err)
            self.empty_msg.pack(anchor="w", padx=PAD_LG, before=self.empty_list)
        for i, j in enumerate(done or [None]):
            row = ctk.CTkFrame(self.empty_list, fg_color="transparent", height=48)
            row.pack(fill="x")
            row.pack_propagate(False)
            row.columnconfigure(0, minsize=FILE_W)
            row.columnconfigure(1, minsize=DUR_W)
            row.columnconfigure(2, minsize=DATE_W, weight=1)
            if j is None:
                _label(row, "None", "body", "ink-3").grid(row=0, column=0, sticky="w")
                break
            if i:
                ctk.CTkFrame(row, height=1, fg_color=C["rule-soft"], corner_radius=0).place(x=0, y=0, relwidth=1)
            name = os.path.basename(j.audio)
            lab = _label(row, clip_px(name, font("body"), FILE_W - 12), "body", "ink")
            lab.grid(row=0, column=0, sticky="w")
            ClipTip(lab, name if clip_px(name, font("body"), FILE_W - 12) != name else "")
            _label(row, tc(j.duration) if j.duration else "", "mono-sm").grid(row=0, column=1, sticky="e")
            _label(row, time.strftime("%d %b %H:%M", time.localtime(j.finished)), "mono-sm").grid(
                row=0, column=2, sticky="e", padx=12)
            _btn(row, "Open", lambda j=j: self.open(j)).grid(row=0, column=3, sticky="e")

    def _build_body(self):
        self.body = ctk.CTkFrame(self.page, fg_color="transparent")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(1, weight=1)
        tl = theme.panel(self.body)
        tl.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, GAP))
        head = ctk.CTkFrame(tl, fg_color="transparent")
        head.pack(fill="x", padx=PAD_LG, pady=(10, 4))
        self.h_name = _label(head, "", "body", "ink")
        self.h_name.pack(side="left")
        self.h_tip = ClipTip(self.h_name)
        self.h_span = _label(head, "", "body", "ink-2")
        self.h_span.pack(side="left", padx=(8, 0))
        self._head, self._arrows, self._arrows_on = head, [], True
        for txt, d in (("→", 1), ("←", -1)):
            b = _btn(head, txt, lambda d=d: self._scroll(d * self._win() / 2), 30, 26, False)
            b.pack(side="right", padx=(4, 0))
            self._arrows.append(b)
        self.cv = ctk.CTkCanvas(tl, height=round(theme._k(tl) * (RULER_H + SHOW_LANES * (LANE_H + LANE_GAP))), bg=C["panel"],
                            highlightthickness=0, bd=0)
        self.cv.pack(fill="x", padx=PAD_LG, pady=(0, 12))
        self.cv.bind("<Configure>", lambda e: self._draw())
        self.cv.bind("<Button-1>", self._down)
        self.cv.bind("<B1-Motion>", self._press)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.cv.bind(seq, self._wheel)
        self.cv.bind("<ButtonRelease-1>", lambda e: setattr(self, "_sb", False))
        ClipTip(self.cv, probe=self._probe)

        left = theme.panel(self.body)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, GAP))
        head = ctk.CTkFrame(left, fg_color="transparent")
        head.pack(fill="x", padx=PAD_LG, pady=(PAD, 10))
        _label(head, "Transcript", "heading", "ink").pack(side="left")
        _btn(head, "Next", self._find_next, 56, 32, False).pack(side="right")
        self.find = _entry(head, 170, 32, "Find")
        self.find.pack(side="right", padx=(0, 8))
        self.find.bind("<Return>", lambda e: self._find_next())
        self.find.bind("<KeyRelease>", lambda e: self._find_mark())
        # plain tk.Text + light scrollbar: CTkTextbox's scrollbar forces a slow layout pass on build
        wrap = tk.Frame(left, bg=C["well"], highlightthickness=1, highlightbackground=C["rule-soft"], bd=0)
        wrap.pack(fill="both", expand=True, padx=PAD_LG, pady=(0, PAD))
        self.tx = self.t = tk.Text(wrap, height=4, wrap="word", bg=C["well"], fg=C["ink"], font=theme._tkfont("body", wrap),
                                   bd=0, highlightthickness=0, padx=10, pady=8, insertwidth=0,
                                   selectbackground=C["raised"], inactiveselectbackground=C["raised"])
        LightScrollbar(wrap, self.t).pack(side="right", fill="y", padx=(0, 3), pady=3)
        self.t.pack(side="left", fill="both", expand=True)
        self.t.bind("<Button-1>", self._click_line, add="+")
        self.t.bind("<Configure>", self._rewrap, add="+")

        # inspector: one panel, a segmented switch picks Speakers / Export / Doubtful / Run
        box = theme.panel(self.body, width=392)
        box.grid(row=1, column=1, sticky="nsew")
        box.grid_propagate(False)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(1, weight=1)
        self._tab_of = {t: t for t in TABS}          # tab label -> tab (Doubtful's label carries its count)
        self._seg = theme.segmented(box, TABS, command=self._show_tab)
        self._seg.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(PAD, 10))
        self._pages = {}
        for name in TABS[1:]:
            self._pages[name] = ctk.CTkFrame(box, fg_color="transparent")
        self.c_spk = self._pages["Speakers"] = ctk.CTkFrame(box, fg_color="transparent")
        self.c_exp, self.c_run = self._pages["Export"], self._pages["Re-run"]
        dbl = self._pages["Doubtful"]
        self.low_n = _label(dbl, "", "mono-sm")
        self.low_n.pack(anchor="w")
        theme.tooltip(self.low_n, "Words the model scored under 50 % confidence. They are underlined in the transcript.")
        self.c_low = ctk.CTkFrame(dbl, fg_color="transparent")
        self.c_low.pack(fill="x", pady=(6, 0))
        self._built, self._dirty = set(), set(TABS)      # tab pages are built / painted on first view
        self._show_tab("Speakers")

    def _retab(self):
        lab = f"Doubtful {len(self.low)}" if self.low else "Doubtful"
        if lab in self._tab_of:
            return
        self._tab_of = {(lab if t == "Doubtful" else t): t for t in TABS}
        self._seg.configure(values=list(self._tab_of))
        self._seg.set(next(l for l, t in self._tab_of.items() if t == self._tab))

    def _show_tab(self, name):
        name = self._tab_of.get(name, name)
        if self._tab:
            self._pages[self._tab].grid_forget()
        self._tab = name
        self._seg.set(next(l for l, t in self._tab_of.items() if t == name))
        self._pages[name].grid(row=1, column=0, sticky="nsew", padx=PAD, pady=(0, PAD))
        if self.job and name in self._dirty:
            self._dirty.discard(name)
            if name not in self._built:
                self._built.add(name)
                {"Export": self._build_export, "Doubtful": self._build_low}.get(name, lambda: None)()
            self._paint_tab(name)

    def _paint_tab(self, name):
        if name == "Speakers":
            self._build_speakers()
        elif name == "Export":
            for f, chip in self.fmt_chips.items():
                chip.set(f in (self.job.cfg.get("formats") or []))
            self.exp_msg.configure(text="")
            self._paint_export()
        elif name == "Doubtful":
            self._paint_low()
        else:
            self._paint_run()

    def _build_export(self):
        grid = ctk.CTkFrame(self.c_exp, fg_color="transparent")
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1, uniform="fmt")
        for i, f in enumerate(EXPORT_FORMATS):         # Transcribe's check chips on 2 equal columns
            c = _Check(grid, FORMAT_LABELS.get(f, f.upper()), lambda on: self._paint_export())
            c.configure(width=1, height=30)
            c.grid(row=i // 2, column=i % 2, sticky="ew", padx=(4, 0) if i % 2 else (0, 4), pady=(0, 6))
            self.fmt_chips[f] = c
            if f == "aud":
                theme.tooltip(c, AUD_TIP)
        self.exp_in = {}
        for key, title in (("max_width", "Max line width"), ("max_count", "Max lines per cue")):
            row = ctk.CTkFrame(self.c_exp, fg_color="transparent")
            row.pack(fill="x", pady=3)
            _label(row, title, "body", "ink-2").pack(side="left")
            e = self.exp_in[key] = theme.LightEntry(row, "no limit")
            e.configure(width=9)
            e.pack(side="right", ipady=e.ipady)
            for ev in ("<Return>", "<FocusOut>"):
                e.bind(ev, lambda _, k=key: self._set_opt(k), add="+")
        self.exp_sw = theme.switch_row(self.c_exp, "Speaker names in cues",
                                       command=lambda v: self.state.set("speaker_names_in_cues", v))
        self.exp_sw.pack(fill="x", pady=(6, 0))
        btns = ctk.CTkFrame(self.c_exp, fg_color="transparent")
        btns.pack(fill="x", pady=(10, 0))
        self.exp_btn = theme.pill(btns, "Export 0 files", self._export)
        self.exp_btn.pack(side="left")
        _btn(btns, "Open folder", self._folder).pack(side="left", padx=(8, 0))
        self.exp_msg = _label(self.c_exp, "", "hint", wraplength=300, justify="left")
        self.exp_msg.pack(fill="x", pady=(8, 0))

    def _set_opt(self, key):
        e, s = self.exp_in[key], self.state.settings
        ok, v, _ = validate(key, e.get(), s)
        if not ok:
            e.set(s.get(key, ""))               # back to the last good value
        elif v != s.get(key, ""):
            self.state.set(key, v)

    def _build_low(self):
        self.low_kv = ctk.CTkFrame(self.c_low, fg_color="transparent")
        self.low_kv.pack(fill="x")
        btns = ctk.CTkFrame(self.c_low, fg_color="transparent")
        btns.pack(fill="x", pady=(8, 0))
        _btn(btns, "Jump to next", self._jump_low).pack(side="left")
        _btn(btns, "Add to hotwords and re-run", self._rerun).pack(side="left")

    # ── Loading a job ─────────────────────────────────────────────────────────
    def open(self, job):
        segs = _load_segments(self.state, job) if job else None
        if not segs:
            self.job = None
            self._paint_empty("No transcript data for this file" if job else None)
            self.body.pack_forget()
            self.empty.pack(fill="both", expand=True)
            return
        self.job, self.segs, self.talk = job, segs, {}
        for s in segs:
            k = s.get("speaker") or "Speech"
            self.talk[k] = self.talk.get(k, 0.0) + max(0.0, s.get("end", 0) - s.get("start", 0))
        self.spk = list(self.talk)
        self.total = max(job.duration or 0, max((s.get("end", 0) for s in segs), default=0), 1.0)
        self._glide.stop()
        self.pos = self.vs = 0.0
        self.empty.pack_forget()
        self.body.pack(fill="both", expand=True, padx=28, pady=22)
        self.vy = 0
        self._render_text(); self._paint_head(); self._retab(); self._draw()
        self._dirty = set(TABS)
        self._show_tab(self._tab)

    def _name(self, k):
        return (self.job.cfg.get("speaker_names") or {}).get(k, "").strip() or k

    def _colour(self, k):
        return LANES[self.spk.index(k) % len(LANES)]

    # ── Segment list ──────────────────────────────────────────────────────────
    def _render_text(self):
        top = self.t.yview()[0]
        tb, t = self.tx, self.t
        tb.configure(state="normal")
        tb.delete("1.0", "end")
        for g in t.tag_names():
            g != "sel" and t.tag_delete(g)
        k = theme._k(t)
        xs, xt = round(COL_T * k), round((COL_T + COL_S) * k)       # fixed columns: time | speaker | text
        spk_px = theme.measure_font(theme._tkfont("body", t), t)
        t.configure(tabs=(xs, xt))
        t.tag_config("row", lmargin1=0, lmargin2=xt)                # wrapped lines hang under the text column
        t.tag_config("ts", foreground=C["ink-3"], font=theme._tkfont("mono-sm", t))
        t.tag_config("low", foreground=C["warn"], underline=True)
        t.tag_config("find", background=C["ink-2"], foreground=C["ground"])
        for i, sp in enumerate(self.spk):
            t.tag_config(f"sp{i}", foreground=self._colour(sp))
            t.tag_config(f"lit{i}", background=self._colour(sp), foreground=C["ground"])
        self._wrap_w = t.winfo_width()
        tw = self._wrap_w - 2 * int(t.cget("padx")) - xt - round(8 * k)     # text column, for wrapping RTL rows
        self.words, self.low, self.lines, self._rtl = [], [], [], False
        for n, s in enumerate(self.segs):
            sp = s.get("speaker") or "Speech"
            si, col = self.spk.index(sp), 0
            ws = s.get("words") or []
            texts = [(w.get("word") or "").replace("\n", " ") for w in ws]
            text = " ".join(texts) if ws else (s.get("text") or "").strip().replace("\n", " ")

            def put(txt, *tags):
                nonlocal col
                tb.insert("end", txt, tags + ("row",))
                col += len(txt)

            def word(j, shown):
                w, txt, c0 = ws[j], texts[j], col
                score = w.get("score")
                dull = score is not None and score < LOW
                put(shown, *(("low",) if dull else ()))
                if "start" in w:
                    self.words.append((w["start"], w.get("end", w["start"]), len(self.lines), c0, col, si))
                if dull:
                    self.low.append((w.get("start", s.get("start", 0)), txt, score))
            put(tc(s.get("start", 0)) + "\t", "ts")
            put(_clip(self._name(sp), spk_px, xt - xs - round(12 * k)) + "\t", f"sp{si}")
            if rtl.is_rtl(text):        # Tk does no bidi: wrap in reading order, each row in display order
                self._rtl = True
                parts, starts, at = rtl.wrap(text, tw, spk_px.measure), [], 0
                for p in parts:
                    starts.append(at)
                    at += len(p) + 1
                offs, at = [], 0
                for x in texts:
                    offs.append(at)
                    at += len(x) + 1
                row_of = [bisect.bisect_right(starts, o) - 1 for o in offs]
                for r, part in enumerate(parts):
                    if r:
                        tb.insert("end", "\n", "row")
                        col = 0
                        put("\t\t")
                    self.lines.append(n)
                    js = [j for j, x in enumerate(row_of) if x == r]
                    for i, v in enumerate(rtl.order([texts[j] for j in js])):
                        if i:
                            put(" ")
                        word(js[v], rtl.display(texts[js[v]]))
                    if not ws:
                        put(rtl.display(part))
            else:
                self.lines.append(n)
                for j in range(len(ws)):
                    if j:
                        put(" ")
                    word(j, texts[j])
                if not ws:
                    put(text)
            tb.insert("end", "\n", "row")
        self.words.sort(); self.low.sort()
        self.starts = [w[0] for w in self.words]
        for i in range(len(self.spk)):
            t.tag_raise(f"lit{i}")
        tb.configure(state="disabled")
        self._lit = None
        t.yview_moveto(top)
        self._find_mark()

    def _click_line(self, e):
        line = int(self.t.index(f"@{e.x},{e.y}").split(".")[0])
        if 1 <= line <= len(self.lines):
            self._set_pos(self.segs[self.lines[line - 1]].get("start", 0), scroll=True, see=False, glide=True)

    def _rewrap(self, e):
        if self._rtl and e.width != self._wrap_w:
            self._render_text(); self._paint_word(False)

    def _paint_word(self, see=True):
        t = self.t
        i = bisect.bisect_right(self.starts, self.pos) - 1
        cur = self.words[i] if i >= 0 and self.pos <= max(self.words[i][1], self.words[i][0] + 0.3) else None
        if cur == self._lit:
            return
        self._lit = cur
        for k in range(len(self.spk)):
            t.tag_remove(f"lit{k}", "1.0", "end")
        if cur:
            _, _, line, c0, c1, si = cur
            t.tag_add(f"lit{si}", f"{line}.{c0}", f"{line}.{c1}")
            if see:
                t.see(f"{line}.{c0}")

    # ── Find ──────────────────────────────────────────────────────────────────
    def _find_mark(self):
        t, term = self.t, rtl.display(self.find.get())      # RTL rows are shown in display order
        t.tag_remove("find", "1.0", "end")
        self._find_at = at = "1.0"
        while term and (at := t.search(term, at, stopindex="end", nocase=True)):
            t.tag_add("find", at, f"{at}+{len(term)}c")
            at = f"{at}+{len(term)}c"

    def _find_next(self):
        term, t = rtl.display(self.find.get()), self.t
        at = term and (t.search(term, self._find_at, stopindex="end", nocase=True)
                       or t.search(term, "1.0", stopindex="end", nocase=True))
        if at:
            t.see(at)
            self._find_at = f"{at}+{len(term)}c"

    # ── Timeline ──────────────────────────────────────────────────────────────
    def _g(self):
        """Timeline geometry in canvas pixels: (k, name column, ruler, lane step, lane height)."""
        k = theme._k(self.cv)
        return k, NAME_W * k, RULER_H * k, (LANE_H + LANE_GAP) * k, LANE_H * k

    def _win(self):
        """Seconds the timeline shows: the whole file when short, else WIN."""
        return max(10.0, min(WIN, self.total))

    def _x(self, t):
        k, nw = self._g()[:2]
        return nw + 10 * k + (t - self.vs) / self._win() * (self.cv.winfo_width() - nw - 14 * k)

    def _wheel(self, e):
        """Wheel over the name column scrolls the lanes (when there are more than fit), else time."""
        up = e.num == 4 or e.delta > 0              # num 4 / 5 are the X11 wheel buttons
        if e.x < self._g()[1] and len(self.spk) > SHOW_LANES:
            self._lanes(-1 if up else 1)
        else:
            self._scroll(-30 if up else 30)
        return "break"

    def _fonts(self):
        """Scaled tk fonts for the canvas text, with a measuring font each (made once)."""
        if not hasattr(self, "_fnt"):
            cv = self.cv
            self._fnt = {r: (theme._tkfont(r, cv), theme.measure_font(theme._tkfont(r, cv), cv))
                         for r in ("body", "mono-sm")}
        return self._fnt

    def _probe(self, e):
        """Full speaker name when the pointer is over a lane name that was cut."""
        k, nw, rh, step, _ = self._g()
        i = self.vy + (e.y - rh) // step
        if e.x >= nw or e.y < rh or not 0 <= i < len(self.spk):
            return ""
        full = self._name(self.spk[int(i)])
        return full if _clip(full, self._fonts()["body"][1], nw - 28 * k) != full else ""

    def _lanes(self, d):
        top = max(0, len(self.spk) - SHOW_LANES)
        self.vy = max(0, min(top, self.vy + d))
        self._draw()

    def _scroll(self, dt):
        self.vs = max(0.0, min(max(0.0, self.total - self._win()), self.vs + dt))
        self._draw()

    def _down(self, e):
        k, _, rh, _, _ = self._g()
        self._sb = len(self.spk) > SHOW_LANES and e.x >= self.cv.winfo_width() - 12 * k and e.y >= rh
        self._press(e)

    def _press(self, e):
        if self._sb:        # thumb drag: the pointer centres the lanes in view
            _, _, rh, step, _ = self._g()
            n, span = len(self.spk), SHOW_LANES * step
            self._lanes(round((e.y - rh) / span * n - SHOW_LANES / 2) - self.vy)
        elif self.job and e.x >= self._g()[1]:
            k, nw = self._g()[:2]
            self._set_pos(self.vs + (e.x - nw - 10 * k) / max(1, self.cv.winfo_width() - nw - 14 * k) * self._win())

    def _set_pos(self, t, scroll=False, see=True, glide=False):
        """Move the playhead; a programmatic jump inside the view glides, anything else is immediate."""
        goal = max(0.0, min(self.total, t))
        moved = False
        if scroll and not (self.vs <= goal < self.vs + self._win()):
            self.vs = max(0.0, min(max(0.0, self.total - self._win()), goal - 10))
            moved = True
        self._goal, self._see = goal, see
        self._glide.value = self.pos
        self._glide.to(goal, snap=not glide or moved)

    def _glide_pos(self, v):
        self.pos = v
        if v == self._goal:             # the word follows once the playhead has landed
            self._paint_word(self._see)
        self._draw()

    def _draw(self):
        cv = self.cv
        cv.delete("all")
        w = cv.winfo_width()
        k, nw, rh, step, lh = self._g()
        if not self.job or w < nw + 60 * k:
            return
        win = self._win()
        on = self.total > win
        if on != self._arrows_on:       # arrows only when the file is longer than the view
            self._arrows_on = on
            for b in self._arrows:
                b.pack_forget()
            if on:
                for b in self._arrows:
                    b.pack(side="right", padx=(4, 0))
        if not on:
            self.vs = 0.0
        x0, x1 = self._x(self.vs), self._x(self.vs + win)
        tick = next((c for c in (1, 2, 5, 10, 30, 60) if win / c <= 16), 60)
        (mono, mono_m), (body, body_m) = self._fonts()["mono-sm"], self._fonts()["body"]
        for t in range(int(-(-self.vs // tick)) * tick, int(self.vs + win) + 1, tick):
            x, big = round(self._x(t)), t % (30 if tick >= 10 else 10 if tick == 5 else 5 * tick) == 0
            cv.create_line(x, round(rh - (10 if big else 5) * k), x, round(rh - 2 * k),
                           fill=C["ink-3"] if big else C["rule"], width=max(1, round(k)))
            if big:
                cv.create_text(x + 3 * k, 8 * k, text=tc(t), anchor="w", fill=C["ink-3"], font=mono)
        cur = next((s.get("speaker") or "Speech" for s in self.segs
                    if s.get("start", 0) <= self.pos <= s.get("end", 0)), None)
        for i, sp in enumerate(self.spk):
            if not self.vy <= i < self.vy + SHOW_LANES:
                continue
            y = rh + (i - self.vy) * step
            col = self._colour(sp) if cur in (None, sp) else _mix(self._colour(sp), C["panel"], 0.42)
            _rrect(cv, 0, y + 18 * k, 10 * k, y + 28 * k, 3 * k, col)
            cv.create_text(20 * k, y + lh / 2, text=_clip(self._name(sp), body_m, nw - 28 * k), anchor="w",
                           fill=C["ink"], font=body)
            _rrect(cv, nw + 10 * k, y, w - 4 * k, y + lh, 8 * k, C["well"], C["rule-soft"], max(1, round(k)))
            cl = _mix(col, C["well"], 0.14)
            for s in self.segs:
                if (s.get("speaker") or "Speech") != sp:
                    continue
                a, b = s.get("start", 0), s.get("end", 0)
                if b < self.vs or a > self.vs + win:
                    continue
                xa, xb = max(self._x(a), x0), min(self._x(b), x1)
                if xb - xa < 2:
                    continue
                _rrect(cv, xa, y + 5 * k, xb, y + lh - 5 * k, 6 * k, cl, col, max(1, round(1.5 * k)))
                if xb - xa > 70 * k:
                    cv.create_text(xa + 8 * k, y + lh / 2, text=rtl.display(_clip((s.get("text") or "").strip(), mono_m, xb - xa - 16 * k)),
                                   anchor="w", fill=col, font=mono)
        n = len(self.spk)
        if n > SHOW_LANES:          # thumb showing which lanes are in view
            span = SHOW_LANES * step
            _rrect(cv, w - 6 * k, rh + self.vy / n * span, w - 2 * k, rh + (self.vy + SHOW_LANES) / n * span,
                   k, C["inert-fill"])
        px, pw = round(self._x(self.pos)), max(1, round(k))
        if x0 <= px <= x1:
            cv.create_rectangle(px - pw, 0, px + pw, cv.winfo_height(), fill=C["ink"], width=0)
            cv.create_rectangle(round(px + 4 * k), round(2 * k), round(px + 16 * k + mono_m.measure(tc(self.pos))),
                                round(22 * k), fill=C["ink"], width=0)
            cv.create_text(round(px + 10 * k), round(12 * k), text=tc(self.pos), anchor="w", fill=C["ground"], font=mono)

    # ── Speakers ──────────────────────────────────────────────────────────────
    def _build_speakers(self):
        self.name_entries = {}
        for w in self.c_spk.winfo_children():
            w.destroy()
        # up to 6 speakers fit; more get a scroll frame (slow to build, so only when needed)
        box = self.c_spk
        if len(self.spk) > 6:
            box = ctk.CTkScrollableFrame(self.c_spk, fg_color="transparent", corner_radius=0)
            box.grid(row=0, column=0, sticky="nsew")
            self.c_spk.columnconfigure(0, weight=1)
            self.c_spk.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        names = self.job.cfg.get("speaker_names") or {}
        for i, k in enumerate(self.spk):
            cell = ctk.CTkFrame(box, fg_color="transparent")
            cell.grid(row=i, column=0, sticky="w", pady=4)
            ctk.CTkFrame(cell, width=10, height=10, corner_radius=3, fg_color=self._colour(k)).pack(side="left")
            _label(cell, k, "body", "ink").pack(side="left", padx=(8, 6))
            _label(cell, tc(self.talk[k]), "mono-sm").pack(side="left")
            if not k.startswith("SPEAKER_"):        # a run without speakers has one "Speech" lane, nothing to rename
                continue
            e = _entry(box, 130, 32, "Name")
            e.grid(row=i, column=1, padx=(10, 0), pady=4)
            if names.get(k):
                e.insert(0, names[k])
            for ev in ("<Return>", "<FocusOut>"):
                e.bind(ev, lambda _, k=k: self._rename(k))
            self.name_entries[k] = e

    def _rename(self, k):
        job = self.job
        if job is None or k not in self.name_entries:
            return
        new, names = self.name_entries[k].get().strip(), job.cfg.setdefault("speaker_names", {})
        if names.get(k, "") == new:
            return
        if new:
            names[k] = new
        else:
            names.pop(k, None)
        self.state.queue.save(self.state.queue_path)
        self._render_text(); self._paint_word(False); self._draw()

    # ── Export ────────────────────────────────────────────────────────────────
    def _paint_export(self):
        if not self.job or "Export" not in self._built:
            return
        s = self.state.settings
        for k, e in self.exp_in.items():
            if e is not e.focus_get() and e.get() != s.get(k, ""):
                e.set(s.get(k, ""))
        if self.exp_sw.get() != bool(s.get("speaker_names_in_cues", True)):
            self.exp_sw.set(s.get("speaker_names_in_cues", True))
        n = sum(c.get() for c in self.fmt_chips.values())
        self.exp_btn.configure(text=f"Export {n} file{'s' if n != 1 else ''}")

    def _out_dir(self):
        return self.job.cfg.get("out_dir") or os.path.dirname(os.path.abspath(self.job.audio))

    def _folder(self):
        if self.job and os.path.isdir(self._out_dir()):
            _open_folder(self._out_dir())

    def _export(self):
        job = self.job
        fmts = [f for f in EXPORT_FORMATS if self.fmt_chips[f].get()]
        if not job or not fmts:
            self.exp_msg.configure(text="Pick at least one format.", text_color=C["warn"])
            return
        s = self.state.settings
        opts = {"max_width": _digits(s.get("max_width", "")), "max_count": _digits(s.get("max_count", "")),
                "highlight": bool(s.get("highlight")), "segment_resolution": s.get("segment_resolution", "sentence"),
                "speaker_names": dict(job.cfg.get("speaker_names") or {}),
                "speaker_names_in_cues": bool(s.get("speaker_names_in_cues", True))}
        result, out_dir = {"segments": self.segs, "language": job.language or "en"}, self._out_dir()
        self.exp_msg.configure(text="Exporting…", text_color=C["ink-3"])
        self.exp_btn.configure(state="disabled")

        def done(text, colour):
            self.exp_btn.configure(state="normal")
            if self.job is job:
                self.exp_msg.configure(text=text, text_color=colour)

        def work():
            try:
                res = pipeline.export(result, fmts, out_dir, job.audio, opts)
                ok, err = sum(r[2] is None for r in res), [f"{r[0]}: {r[2]}" for r in res if r[2]]
            except Exception as e:
                ok, err = 0, [str(e)]
            self.state.schedule(lambda: done(
                f"Saved {ok} file{'s' if ok != 1 else ''}" + (" · " + "; ".join(err) if err else ""),
                C["bad"] if err else C["ok"]))
        threading.Thread(target=work, daemon=True).start()

    # ── Doubtful words ────────────────────────────────────────────────────────
    def _paint_low(self):
        self.low_n.configure(text=f"{len(self.low)} under {round(LOW * 100)} %")
        self._kv(self.low_kv, [(rtl.display(w), f"{tc(t)} · {round(sc * 100)} %") for t, w, sc in self.low[:8]]
                 or [("None", "")])

    def _jump_low(self):
        if self.low:
            self._set_pos(next((w for w in self.low if w[0] > self.pos + 0.01), self.low[0])[0], scroll=True, glide=True)

    def _rerun(self):
        job, state = self.job, self.state
        if not job or not self.low:
            return
        new = {}
        for _, txt, _ in self.low:
            if txt.strip(_EDGE):
                new.setdefault(txt.strip(_EDGE).lower(), txt.strip(_EDGE))
        have = {h.strip().lower(): h.strip() for h in state.settings.get("hotwords", "").split(",") if h.strip()}
        for k, w in list(new.items())[:20]:
            have.setdefault(k, w)
        state.set("hotwords", ", ".join(have.values()))
        state.add_files([job.audio])
        (getattr(state, "start_queue", None) or (lambda: state.emit("transcribe")))()
        state.emit("screen", name="queue")

    # ── Run ───────────────────────────────────────────────────────────────────
    def _paint_run(self):
        j, cfg = self.job, self.job.cfg
        took = fin = "—"
        if j.started and j.finished:
            took = tc(j.finished - j.started)
            r = _speed_ratio(j, self.state.stages)
            if r is not None:
                took += f" · {r:.1f}×"
        if j.finished:
            lt, day = time.localtime(j.finished), "%Y%m%d"
            fin = time.strftime("%H:%M", lt) + (" today" if time.strftime(day, lt) == time.strftime(day)
                                                 else " · " + time.strftime("%d %b", lt))
        self._kv(self.c_run, (("Model", cfg.get("model", "—")),
                              ("Device", cfg.get("device", "—") + " · " + cfg.get("compute", "")),
                              ("Alignment", (cfg.get("align_model") or "default") if cfg.get("align", True) else "off"),
                              ("Speakers", "on" if cfg.get("diarize") else "off"),
                              ("Took", took), ("Finished", fin)))

    def _paint_head(self):
        j = self.job
        name = os.path.basename(j.audio)
        text = clip_px(name, font("body"), FILE_W)
        self.h_name.configure(text=text)
        self.h_tip.text = name if text != name else ""
        self.h_span.configure(text=" · ".join(["", tc(self.total)] + ([LANG_NAMES.get(j.language, j.language)]
                                                                       if j.language else [])).lstrip(" "))


def build(parent, state):
    return Screen(parent, state).page
