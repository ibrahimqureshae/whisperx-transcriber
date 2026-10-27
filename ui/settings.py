"""Settings screen — areas of controls that save themselves (DESIGN.md / prototype)."""
import copy
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core.pipeline import COMPUTE_TYPES, EXPORT_FORMATS, MODEL_INFO, supports_translate
from core.settings import DEFAULTS, validate
from ui import theme
from ui.state import LANGUAGES, default_model_dir
from ui.theme import C, GAP, PAD, PAD_LG, font

# ── Table: (title, lines); a line is a list of cells, laid out in 4 equal columns ──
# cell = (kind, key, label, opt, span)   kind: text dir menu choice chips switch
# opt: text/dir -> placeholder (the default shown); menu -> values (list, or fn(state)); choice/chips -> [(label, value)]
# Explanations live in TIPS (hover tooltips), never on the screen.
_D = "decode."
_COND = "condition_on_previous_text"    # a true/false/blank text in the file, a switch here
_DEC = {                                  # decode key -> (label, library default shown as placeholder)
    "temperatures": ("Temperature fallback", "0, 0.2, 0.4, 0.6, 0.8, 1.0"),
    "best_of": ("Best of", "5"), "patience": ("Patience", "1"),
    "length_penalty": ("Length penalty", "1"), "repetition_penalty": ("Repetition penalty", "1"),
    "no_repeat_ngram_size": ("No-repeat n-gram size", "0"),
    "compression_ratio_threshold": ("Compression ratio threshold", "2.4"),
    "log_prob_threshold": ("Log-probability threshold", "-1.0"),
    "no_speech_threshold": ("No-speech threshold", "0.6"),
    "prompt_reset_on_temperature": ("Prompt reset on temperature", "0.5"),
    "hallucination_silence_threshold": ("Hallucination silence threshold", "off"),
    "max_new_tokens": ("Max new tokens", "auto"),
}
TIPS = {
    "task": "Transcribe writes what is said. Translate writes it in English.",
    "model": "Bigger models are more accurate but slower and need more memory.",
    "beam": "How many candidate transcriptions are tried at once. Higher is more accurate and slower.",
    "initial_prompt": "Text that primes the model with the style, spelling or vocabulary you expect.",
    "hotwords": "Comma-separated names and terms the model should favour.",
    "suppress_numerals": "Writes numbers as words instead of digits.",
    _D + "temperatures": "Randomness levels tried in turn when a chunk fails the checks. Blank uses the default list.",
    _D + "best_of": "Candidates sampled at each non-zero temperature; the best is kept.",
    _D + "patience": "Beam search patience. Above 1 keeps searching longer before settling.",
    _D + "length_penalty": "Above 1 favours longer transcriptions, below 1 shorter ones.",
    _D + "repetition_penalty": "Above 1 discourages repeating the same words.",
    _D + "no_repeat_ngram_size": "Blocks any run of this many words from repeating. 0 turns it off.",
    _D + "compression_ratio_threshold": "A chunk that compresses more than this is treated as repetitive and retried.",
    _D + "log_prob_threshold": "A chunk with a lower average confidence than this is retried.",
    _D + "no_speech_threshold": "A chunk the model thinks is silence above this probability is dropped.",
    _D + "condition_on_previous_text": "Feeds the previous text in as context. Can cause repeated loops on long audio.",
    _D + "prompt_reset_on_temperature": "Forgets the previous text once the temperature rises past this value.",
    _D + "hallucination_silence_threshold": "Skips silences longer than this many seconds when a hallucination is suspected.",
    _D + "max_new_tokens": "Longest output per chunk, in tokens. Blank means no limit.",
    "vad_method": "The detector that finds speech and splits the audio.",
    "vad_onset": "How readily speech starts. Lower picks up quieter speech; higher ignores noise.",
    "vad_offset": "How readily speech ends. Lower ends it sooner.",
    "chunk": "Longest stretch of speech, in seconds, sent to the model at once.",
    "char_alignments": "Adds a time for every letter to the JSON, for karaoke tools.",
    "align_model": "Any wav2vec2 model id, for languages without a bundled one. Blank uses the default for the language.",
    "interpolate_method": "How a word with no audio evidence gets a time: from the nearest word, a straight line, or none.",
    "diarize": "Finds who speaks when. The speaker models (about 34 MB) download automatically the first time.",
    "min_speakers": "Fewest speakers to look for. Blank detects automatically.",
    "max_speakers": "Most speakers to look for. Blank detects automatically.",
    "speaker_embeddings": "Writes each speaker's voice fingerprint to a JSON next to the transcript, for matching later.",
    "speaker_names_in_cues": "Starts each subtitle cue with the speaker's name.",
    "formats": "Audacity labels are a tab-separated list of segments that Audacity imports as a label track.",
    "max_width": "Characters per subtitle line before it wraps. Blank means no limit.",
    "max_count": "Lines per subtitle cue. Blank means no limit.",
    "segment_resolution": "Cut cues at sentence ends, or wherever the model chunked the audio.",
    "highlight": "Highlights each word as it is spoken (SRT and VTT).",
    "device_index": "Which GPU to use when there is more than one.",
    "compute": "Number format. Lower precision is faster and uses less memory.",
    "batch": "Chunks decoded together. Lower it to 8 or 4 if the GPU runs out of memory.",
    "threads": "CPU threads used for decoding.",
    "offline_only": "Never downloads models; uses only what is already on disk.",
    "open_folder_per_batch": "Opens one folder window after the whole queue, not one per file.",
    "queue_autostart": "Runs the next waiting file automatically when one finishes.",
}

NO_TRANSLATE = "This model can only transcribe. Pick large-v3 or medium to translate."


def _t(key, label, ph="", span=1):
    return ("text", key, label, ph, span)


def _sw(key, label, span=1):
    return ("switch", key, label, None, span)


_DECODE_LINES = [[("text", _D + k, lab, ph, 1) for k, (lab, ph) in list(_DEC.items())[i:i + 4]]
                 for i in range(0, len(_DEC), 4)]

# Fields fill a 4-column grid (rows add up to 4); each area's switches follow in one column at the end.
AREAS = [
    ("Accuracy",
     [[("menu", "language", "Language", list(LANGUAGES), 1), ("menu", "model", "Model", list(MODEL_INFO), 1),
       ("choice", "task", "Task", [("Transcribe", "transcribe"), ("Translate to English", "translate")], 2)],
      [_t("beam", "Beam size", "5"), _t("hotwords", "Hotwords"), _t("initial_prompt", "Initial prompt", span=2)]]
     + _DECODE_LINES
     + [[_sw(_D + _COND, "Condition on previous text", 2)], [_sw("suppress_numerals", "Spell out numbers", 2)]]),
    ("Speech detection",
     [[("choice", "vad_method", "Detector", [("Silero", "silero"), ("Pyannote", "pyannote")], 1),
       _t("vad_onset", "Sensitivity (onset)"), _t("vad_offset", "Release (offset)"),
       _t("chunk", "Chunk length (s)")]]),
    ("Word timing",
     [[_t("align_model", "Alignment model override", span=2),
       ("choice", "interpolate_method", "Words without audio evidence",
        [("Nearest", "nearest"), ("Linear", "linear"), ("Ignore", "ignore")], 2)],
      [_sw("align", "Align words", 2)], [_sw("char_alignments", "Character timings", 2)]]),
    ("Speakers",
     [[_t("min_speakers", "Minimum speakers", "auto", 2), _t("max_speakers", "Maximum speakers", "auto", 2)],
      [_sw("diarize", "Identify speakers", 2)], [_sw("speaker_embeddings", "Save speaker embeddings", 2)],
      [_sw("speaker_names_in_cues", "Speaker names in cues", 2)]]),
    ("Subtitles",
     [[("chips", "formats", "Output formats",
        [("Audacity labels" if f == "aud" else f.upper().replace("_", " "), f) for f in EXPORT_FORMATS], 4)],
      [_t("max_width", "Max line width", "no limit"), _t("max_count", "Max lines per cue", "no limit"),
       ("choice", "segment_resolution", "Cue breaks", [("Sentence", "sentence"), ("Chunk", "chunk")], 2)],
      [_sw("highlight", "Karaoke highlight", 2)]]),
    ("Hardware",
     [[("menu", "device", "Device", lambda s: s.devices, 1), _t("device_index", "Device index"),
       ("menu", "compute", "Compute type", list(COMPUTE_TYPES), 1), _t("batch", "Batch size")],
      [("dir", "model_dir", "Models folder", default_model_dir(), 2),
       ("dir", "out_dir", "Save transcripts to", "Next to each file", 2)],
      [_t("threads", "CPU threads", span=2),
       ("choice", "queue_on_failure", "When a file fails", [("Continue", "continue"), ("Stop", "stop")], 2)],
      [_sw("offline_only", "Offline only", 2)], [_sw("queue_autostart", "Start the queue automatically", 2)],
      [_sw("open_folder", "Open folder when done", 2)], [_sw("open_folder_per_batch", "Open once per batch", 2)]]),
    ("Reset", None),
]

KEYS_WITH_CONTROLS = {"decode" if c[1].startswith(_D) else c[1]
                      for _a in AREAS for line in (_a[1] or []) for c in line}


# ── Screen ────────────────────────────────────────────────────────────────────
class SettingsScreen(ctk.CTkFrame):
    def __init__(self, parent, state):
        super().__init__(parent, fg_color=C["ground"], corner_radius=0)
        self.state = state
        self._refreshers = []          # (key, reload fn, entry or None) per control
        self._stamps = {}              # area index -> "Saved hh:mm:ss" label
        self._areas = {}               # area index -> frame, built on first selection
        self._cur = None
        self._tips = {}                # key -> its tooltips (text swapped to show why input was refused)
        self._flash = {}               # key -> pending `after` that restores a refused field
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        titles = [a[0] for a in AREAS]
        self._tabs = theme.light_pills(self, titles, value=titles[0], bg=C["ground"],
                                       command=lambda t: self._show(titles.index(t)))
        self._tabs.grid(row=0, column=0, sticky="w", padx=28, pady=(22, 12))
        self._body = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self._body.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 22))
        self._body.columnconfigure(0, weight=1)
        self._body.rowconfigure(0, weight=1)
        state.on("settings", self._on_settings)
        self._show(0)

    # layout
    def _show(self, i):
        if i not in self._areas:
            self._areas[i] = self._build_area(i, AREAS[i][1])
        if self._cur is not None and self._cur != i:
            self._areas[self._cur].grid_remove()
        self._areas[i].grid()
        self._cur = i

    def open_tab(self, title):
        titles = [a[0] for a in AREAS]
        if title in titles:
            self._tabs.set(title)
            self._show(titles.index(title))

    def _build_area(self, i, lines):
        area = tk.Frame(self._body, bg=C["ground"])
        area.grid(row=0, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1)
        pan = theme.panel(area)
        pan.grid(row=0, column=0, sticky="new")
        inner = tk.Frame(pan, bg=C["panel"])
        inner.pack(fill="x", padx=PAD_LG, pady=PAD)
        for c in range(4):                     # four equal columns on every line
            inner.columnconfigure(c, weight=1, uniform="c")
        if lines is None:
            theme.pill(inner, "Reset all settings", self._reset, fg_color="transparent", border_width=0,
                       text_color=C["bad"], hover_color=C["raised"]).grid(row=0, column=0, sticky="w")
        for r, line in enumerate(lines or []):
            c = 0
            for cell in line:
                span, px = cell[4], (0 if c == 0 else 14, 0)
                if cell[0] == "text":          # label + entry sit straight in the grid: two widgets, no wrapper
                    lab = self._caption(inner, cell[2])
                    lab.grid(row=2 * r, column=c, columnspan=span, sticky="sew", padx=px, pady=(0, 4))
                    e = self._entry(inner, i, cell[1], cell[3])[0]
                    e.grid(row=2 * r + 1, column=c, columnspan=span, sticky="new", padx=px, pady=(0, 12),
                           ipadx=e.ipadx, ipady=e.ipady)
                    ws = (lab, e)
                else:
                    w = self._cell(inner, i, cell)
                    w.grid(row=2 * r, rowspan=2, column=c, columnspan=span, sticky="new", padx=px, pady=(0, 12))
                    ws = (w,)
                for w in ws:
                    if TIPS.get(cell[1]):
                        self._tips.setdefault(cell[1], []).append(theme.Tooltip(w, TIPS[cell[1]]))
                c += span
        self._stamps[i] = tk.Label(inner, text="", font=theme._tkfont("mono-sm", inner), fg=C["ink-3"],
                                   bg=C["panel"], anchor="e", padx=0, pady=0)
        self._stamps[i].grid(row=2 * len(lines or []) + 1, column=0, columnspan=4, sticky="ew")
        return area

    # state access
    def _get(self, key):
        s = self.state.settings
        return s["decode"].get(key[len(_D):], "") if key.startswith(_D) else s[key]

    def _save(self, area, key, value):
        if key.startswith(_D):
            dec = dict(self.state.settings["decode"])
            dec[key[len(_D):]] = value
            key, value = "decode", dec
        self.state.set(key, value)
        self._stamps[area].configure(text="Saved " + time.strftime("%H:%M:%S"))

    def _on_settings(self, key=None, **_):
        try:
            focused = self.focus_get()
        except Exception:
            focused = None
        for k, refresh, entry in self._refreshers:
            if key == "*" or k == key or (key == "decode" and k.startswith(_D)):
                if key != "*" and entry is not None and focused is entry:
                    continue                   # don't overwrite what is being typed
                refresh()

    # cells
    def _cell(self, parent, area, cell):
        kind, key, label, opt, _span = cell
        if kind == "switch":
            on = lambda: self._get(key) in (True, "true")        # decode switches hold "true" / "false" / blank
            w = theme.light_switch(parent, label, value=on(),
                                   command=lambda v: self._save(area, key, str(v).lower() if key.startswith(_D) else v))
            self._refreshers.append((key, lambda: w.set(on()), None))
            return w
        wrap = tk.Frame(parent, bg=C["panel"])
        getattr(self, "_" + kind)(wrap, area, key, label, opt)
        return wrap

    @staticmethod
    def _caption(parent, label):
        return tk.Label(parent, text=label, font=theme._tkfont("button", parent), fg=C["ink-2"], bg=C["panel"],
                        anchor="w", padx=0, pady=0)

    def _entry(self, parent, area, key, ph, show=None):
        f = theme.LightEntry(parent, ph, show)
        f.set(self._get(key))

        def commit(_e=None):
            ok, v, why = validate(key, f.get(), self.state.settings)
            if not ok:
                f.set(self._get(key))          # back to the last good value
                self._refuse(key, f, why)
                return
            if v != f.get():
                f.set(v)
            if v != self._get(key):
                self._save(area, key, v)
        f.bind("<Return>", commit, add="+")
        f.bind("<FocusOut>", commit, add="+")
        self._refreshers.append((key, lambda: f.set(self._get(key)), f))
        return f, commit

    def _refuse(self, key, entry, why):
        """Bad border for 2 s and the reason in the tooltip; nothing is saved."""
        tips = self._tips.setdefault(key, [])
        if not tips:
            tips.append(theme.Tooltip(entry, why))
        for t in tips:
            t.text = why
        entry.configure(highlightbackground=C["bad"], highlightcolor=C["bad"])
        if key in self._flash:                 # a second refusal restarts the 2 s
            entry.after_cancel(self._flash.pop(key))

        def restore():
            self._flash.pop(key, None)
            try:
                entry.configure(highlightbackground=C["rule"], highlightcolor=C["ember"])
            except tk.TclError:
                return
            if TIPS.get(key):
                for t in tips:
                    t.text = TIPS[key]
        self._flash[key] = entry.after(2000, restore)

    def _field(self, wrap, area, key, label, ph, show=None):
        """Caption + entry stacked inside a wrapper cell (for fields that carry a button below)."""
        self._caption(wrap, label).pack(fill="x", pady=(0, 4))
        f, commit = self._entry(wrap, area, key, ph, show)
        f.pack(fill="x", ipadx=f.ipadx, ipady=f.ipady)
        return f, commit

    def _dir(self, wrap, area, key, label, ph):
        f, commit = self._field(wrap, area, key, label, ph)

        def browse():
            d = filedialog.askdirectory(initialdir=f.get() or self.state.settings.get("last_dir")
                                        or os.path.expanduser("~"), title=label)
            if d:
                self.state.set("last_dir", d)
                f.set(d)
                commit()
        btn = theme.light_pills(wrap, ["Browse"], multi=True, command=lambda _l, _on: (btn.set([]), browse()))
        btn.pack(anchor="w", pady=(6, 0))

    def _menu(self, wrap, area, key, label, opt):
        self._caption(wrap, label).pack(fill="x", pady=(0, 4))
        m = theme.light_menu(wrap, (lambda: list(opt(self.state))) if callable(opt) else list(opt),
                             value=self._get(key), command=lambda v: self._save(area, key, v))
        m.pack(fill="x")
        self._refreshers.append((key, lambda: m.set(self._get(key)), None))

    def _choice(self, wrap, area, key, label, opt):
        """Single choice as a dropdown; Translate is greyed (and the setting reset) for models that can't."""
        self._caption(wrap, label).pack(fill="x", pady=(0, 4))
        to_val, to_lab = dict(opt), {v: lab for lab, v in opt}
        task = key == "task"
        m = theme.light_menu(wrap, [lab for lab, _ in opt], value=to_lab.get(self._get(key)),
                             command=lambda lab: self._save(area, key, to_val[lab]),
                             disabled=(lambda lab: to_val[lab] == "translate" and not self._can_translate())
                             if task else None)
        m.pack(fill="x")
        self._refreshers.append((key, lambda: m.set(to_lab.get(self._get(key), opt[0][0])), None))
        if task:
            def guard():
                ok = self._can_translate()
                if not ok and self._get(key) == "translate":
                    self._save(area, key, "transcribe")
                for t in self._tips.get(key, []):
                    t.text = TIPS[key] if ok else NO_TRANSLATE
            self._refreshers.append(("model", guard, None))
            wrap.after_idle(guard)

    def _can_translate(self):
        return supports_translate(self.state.settings["model"])

    def _chips(self, wrap, area, key, label, opt):
        self._caption(wrap, label).pack(fill="x", pady=(0, 4))
        to_lab = {f: lab for lab, f in opt}
        chips = theme.light_pills(wrap, [lab for lab, _ in opt], multi=True, check=True,
                                  selected=[to_lab[f] for f in self._get(key) if f in to_lab])

        def changed(_lab, _on):
            chosen = [f for lab, f in opt if lab in chips.get()]
            if not chosen:                     # at least one format stays on
                chips.set([_lab])
                return
            self._save(area, key, chosen)
        chips._cb = changed
        chips.pack(anchor="w")
        self._refreshers.append((key, lambda: chips.set([to_lab[f] for f in self._get(key) if f in to_lab]),
                                 None))

    # reset
    def _reset(self):
        if not messagebox.askyesno("Reset settings", "Reset every setting to its default? "
                                   "Downloaded models and transcripts stay.", parent=self):
            return
        self.state.settings = copy.deepcopy(DEFAULTS)
        self.state.save_settings()
        self.state.emit("settings", key="*")
        stamp = self._stamps.get(len(AREAS) - 1)      # the Reset tab is built on first visit
        if stamp:
            stamp.configure(text="Reset " + time.strftime("%H:%M:%S"))


def build(parent, state):
    return SettingsScreen(parent, state)
