"""Theme tokens and reusable customtkinter builders (DESIGN.md / prototype)."""
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

# ── Tokens ────────────────────────────────────────────────────────────────────
C = {
    "ground": "#0c0c0e", "panel": "#151517", "raised": "#1e1e21", "well": "#101012",
    "readout-well": "#060607", "rule": "#26262a", "rule-soft": "#1b1b1e",
    "ink": "#f3efe8", "ink-2": "#b9b3aa", "ink-3": "#8f8981",
    "heat": "#ff6a1a", "heat-hover": "#ff7d38", "ember": "#ffb067",
    "lane-violet": "#8b6cff", "lane-magenta": "#ff3d8a", "lane-teal": "#2fd3c4",
    "ok": "#38e07b", "warn": "#ffb020", "bad": "#ff4d4f",
    "inert-fill": "#3a3a40", "on-heat": "#1a0d05",
}
LANES = (C["heat"], C["lane-violet"], C["lane-magenta"], C["lane-teal"])
R_PANEL = 14; R_INNER = 8; R_LANE = 10; CONTROL_H = 36; GAP = 16; PAD = 18; PAD_LG = 20
METER_CELLS = 12

# ── Fonts ─────────────────────────────────────────────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_ui_ok = False
_mono_ok = False
_loaded = False


def load_fonts():
    """Register the bundled TTFs once; stay silent if anything fails."""
    global _ui_ok, _mono_ok, _loaded
    if _loaded:
        return
    _loaded = True

    def _load(name):
        try:
            return bool(ctk.FontManager.load_font(str(_FONT_DIR / name)))
        except Exception:
            return False

    ui = [_load(n) for n in ("Barlow-Regular.ttf", "Barlow-Medium.ttf",
                             "Barlow-SemiBold.ttf", "Barlow-Bold.ttf")]
    _ui_ok = all(ui)
    _mono_ok = _load("MartianMono.ttf")


# role -> (is_mono, size, weight)
_ROLES = {
    "display": (False, 24, "semibold"), "title": (False, 18, "semibold"),
    "heading": (False, 15, "semibold"), "body": (False, 14, "regular"),
    "button": (False, 14, "medium"), "button-primary": (False, 15, "bold"),
    "label": (False, 12, "semibold"), "hint": (False, 12, "regular"),
    "mono": (True, 15, "normal"), "mono-hero": (True, 18, "bold"),
    "mono-sm": (True, 12, "normal"),
}
_FACE = {"regular": "Barlow", "medium": "Barlow Medium", "semibold": "Barlow SemiBold", "bold": "Barlow"}
_font_cache = {}


def font(role, size=None):
    """CTkFont for a DESIGN.md typography role; Segoe UI / Consolas if bundled fonts are missing."""
    mono, dsize, weight = _ROLES[role]
    size = size or dsize
    key = (role, size, _ui_ok, _mono_ok, id(tk._default_root))     # fonts belong to one Tk root
    cached = _font_cache.get(key)
    if cached is not None:
        try:
            cached.actual("size")          # a font of a destroyed Tk root raises here
        except Exception:
            del _font_cache[key]
    if key not in _font_cache:
        bold = "bold" if weight == "bold" else "normal"
        if mono:
            fam = "Martian Mono" if _mono_ok else "Consolas"
            _font_cache[key] = ctk.CTkFont(family=fam, size=size, weight=bold)
        elif _ui_ok:
            _font_cache[key] = ctk.CTkFont(family=_FACE[weight], size=size, weight=bold)
        else:
            heavy = "bold" if weight in ("semibold", "bold") else "normal"
            _font_cache[key] = ctk.CTkFont(family="Segoe UI", size=size, weight=heavy)
    return _font_cache[key]


def _mix(a, b, t):
    """Blend hex colour a over b by fraction t (stands in for CSS opacity / color-mix)."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x * t + y * (1 - t)) for x, y in zip(ca, cb))


# ── Surfaces and buttons ──────────────────────────────────────────────────────
def panel(parent, **kw):
    kw.setdefault("fg_color", C["panel"]); kw.setdefault("corner_radius", R_PANEL)
    kw.setdefault("border_width", 1); kw.setdefault("border_color", C["rule"])
    return ctk.CTkFrame(parent, **kw)


def _pill_look(primary, text):
    """Style kwargs for the one heat primary pill (44px) or a secondary pill (raised, 36px)."""
    if primary:
        look = dict(height=44, corner_radius=22, fg_color=C["heat"], hover_color=C["heat-hover"],
                    text_color=C["on-heat"], border_width=0, font=font("button-primary"))
    else:
        look = dict(height=CONTROL_H, corner_radius=CONTROL_H // 2, fg_color=C["raised"],
                    hover_color=C["rule"], text_color=C["ink"], border_width=1,
                    border_color=C["rule"], font=font("button"))
    look["width"] = look["font"].measure(text) + (52 if primary else 36)
    return look


def pill(parent, text, command=None, *, primary=False, **kw):
    """Secondary pill (raised, 36px) or the one primary heat pill (44px)."""
    return ctk.CTkButton(parent, text=text, command=command, **{**_pill_look(primary, text), **kw})


def restyle_pill(btn, primary, **kw):
    """Switch an existing pill between primary and secondary (the border colour is kept for primary)."""
    btn.configure(**{**_pill_look(primary, btn.cget("text")), **kw})


class Chip(ctk.CTkButton):
    """Multi-select toggle pill (30px)."""

    def __init__(self, parent, text, selected=False, command=None):
        self._on = bool(selected); self._cb = command
        super().__init__(parent, text=text, height=30, corner_radius=15, border_width=1,
                         font=font("button"), command=self._toggle)
        self._paint()

    def _paint(self):
        if self._on:
            self.configure(fg_color=C["raised"], hover_color=C["raised"],
                           text_color=C["ink"], border_color=C["ink-2"])
        else:
            self.configure(fg_color="transparent", hover_color=C["raised"],
                           text_color=C["ink-2"], border_color=C["rule"])

    def _toggle(self):
        self.set(not self._on)
        if self._cb:
            self._cb(self._on)

    def get(self):
        return self._on

    def set(self, on):
        self._on = bool(on); self._paint()


def chip(parent, text, *, selected=False, command=None):
    return Chip(parent, text, selected, command)


def segmented(parent, values, *, command=None, value=None):
    seg = ctk.CTkSegmentedButton(
        parent, values=list(values), command=command, height=34, corner_radius=17, border_width=3,
        fg_color=C["well"], selected_color=C["raised"], selected_hover_color=C["raised"],
        unselected_color=C["well"], unselected_hover_color=C["rule-soft"],
        text_color=C["ink"], font=font("button"))
    seg.set(value if value is not None else list(values)[0])
    return seg


class SwitchRow(ctk.CTkFrame):
    """Title (+ optional hint) at left, 34x20 switch at right."""

    def __init__(self, parent, title, description=None, value=False, command=None):
        super().__init__(parent, fg_color="transparent")
        self.columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=title, font=font("body"), text_color=C["ink"],
                     anchor="w").grid(row=0, column=0, sticky="w")
        if description:
            ctk.CTkLabel(self, text=description, font=font("hint"), text_color=C["ink-3"],
                         anchor="w", justify="left").grid(row=1, column=0, sticky="w")
        self._sw = ctk.CTkSwitch(self, text="", width=34, height=20, switch_width=34, switch_height=20,
                                 fg_color=C["inert-fill"], progress_color=C["ok"],
                                 button_color=C["ink"], button_hover_color=C["ink"],
                                 command=(lambda: command(self.get())) if command else None)
        self._sw.grid(row=0, column=1, rowspan=2, padx=(GAP, 0))
        self.set(value)

    def get(self):
        return bool(self._sw.get())

    def set(self, on):
        if on:
            self._sw.select()
        else:
            self._sw.deselect()


def switch_row(parent, title, description=None, *, value=False, command=None):
    return SwitchRow(parent, title, description, value, command)


class Field(ctk.CTkFrame):
    """Label above a 38px well entry; focus border turns ember."""

    def __init__(self, parent, label=None, placeholder="", width=None, show=None):
        super().__init__(parent, fg_color="transparent")
        if label:
            ctk.CTkLabel(self, text=label, font=font("button"), text_color=C["ink-2"],
                         anchor="w").pack(fill="x", pady=(0, 6))
        kw = {"width": width} if width else {}
        self.entry = ctk.CTkEntry(self, height=38, corner_radius=R_INNER, border_width=1,
                                  fg_color=C["well"], border_color=C["rule"], text_color=C["ink"],
                                  placeholder_text=placeholder, placeholder_text_color=C["ink-3"],
                                  font=font("body"), show=show, **kw)
        self.entry.pack(fill="x")
        self.entry.bind("<FocusIn>", lambda e: self.entry.configure(border_color=C["ember"]), add="+")
        self.entry.bind("<FocusOut>", lambda e: self.entry.configure(border_color=C["rule"]), add="+")

    def get(self):
        return self.entry.get()

    def set(self, text):
        self.entry.delete(0, "end")
        if text:
            self.entry.insert(0, text)
        else:
            # CTkEntry starts "focused", so delete() alone never re-shows the placeholder
            self.entry._activate_placeholder()


def field(parent, label=None, *, placeholder="", width=None, show=None):
    return Field(parent, label, placeholder, width, show)


# ── Light controls (plain tk, a fraction of the cost of the CTk ones; for dense screens) ──
def _k(w):
    """Widget scale factor (DPI x UI scale) that plain tk widgets must apply themselves."""
    return ctk.ScalingTracker.get_widget_scaling(w)


def _tkfont(role, w):
    return font(role).create_scaled_tuple(_k(w))


def rrect(cv, x0, y0, x1, y1, r, fill, edge=None, b=1, tags=()):
    """Anti-aliased rounded box on a CTkCanvas: font-shape corners, as CTk's draw engine makes them."""
    r = max(1, r)
    b = min(b, r - 1)
    if edge and b < 1:                  # no room for an inset fill: edge colour only
        fill, edge = edge, None
    for col, d in ((edge, 0), (fill, b)) if edge else ((fill, 0),):
        for x, y in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r)):
            cv.create_aa_circle(x, y, r - d, fill=col, tags=tags)
        cv.create_rectangle(x0 + d, y0 + r, x1 - d, y1 - r, width=0, fill=col, tags=tags)
        cv.create_rectangle(x0 + r, y0 + d, x1 - r, y1 - d, width=0, fill=col, tags=tags)


def measure_font(spec, w):
    """tkinter Font for measuring a tk font tuple. Font(font=tuple) goes through `font actual`,
    which rounds a pixel size to whole points (14 px -> 10 pt = 13.3 px at 133 % scaling), so
    measured text came out narrower than the label draws it; keep the exact pixel size instead."""
    import tkinter.font as tkf
    style = spec[2].split() if len(spec) > 2 else []
    return tkf.Font(root=w, family=spec[0], size=spec[1],
                    weight="bold" if "bold" in style else "normal",
                    slant="italic" if "italic" in style else "roman")


_linespace = {}


def _lh(tkfont, w):
    """Line height in pixels of a tk font tuple (measured once per font)."""
    if tkfont not in _linespace:
        _linespace[tkfont] = measure_font(tkfont, w).metrics("linespace")
    return _linespace[tkfont]


class LightEntry(tk.Entry):
    """Well entry whose own highlight ring is the border (ember on focus). The placeholder is drawn by hand.
    Grid it with ipady=entry.ipady to get the requested height."""

    def __init__(self, parent, placeholder="", show=None, height=30):
        k = _k(parent)
        f = _tkfont("body", parent)
        super().__init__(parent, font=f, bg=C["well"], fg=C["ink"], insertbackground=C["ink"], relief="flat",
                         bd=0, highlightthickness=1, highlightbackground=C["rule"], highlightcolor=C["ember"],
                         selectbackground=C["raised"], selectforeground=C["ink"])
        self.ipady = max(1, round((height * k - 2 - _lh(f, parent)) / 2))
        self.ipadx = round(7 * k)
        self._mask, self._hidden, self._ph_on = show, True, False
        self._ph = "" if show else placeholder        # a secret field never shows a placeholder
        self._apply_mask()
        self.bind("<FocusIn>", lambda e: self._clear_ph(), add="+")
        self.bind("<FocusOut>", lambda e: self._draw_ph(), add="+")
        self._draw_ph()

    def _apply_mask(self):
        """Secret fields show the mask char unless the user chose Show."""
        if self._mask and not self._ph_on:
            self.configure(show=self._mask if self._hidden else "")

    def set_hidden(self, hidden):
        self._hidden = bool(hidden)
        self._apply_mask()

    def is_hidden(self):
        return self._hidden

    def _draw_ph(self):
        if self._ph and not self._ph_on and not tk.Entry.get(self):
            self._ph_on = True
            self.configure(show="", fg=C["ink-3"])
            self.insert(0, self._ph)

    def _clear_ph(self):
        if self._ph_on:
            self._ph_on = False
            self.delete(0, "end")
            self.configure(fg=C["ink"], show="")
            self._apply_mask()

    def get(self):
        return "" if self._ph_on else tk.Entry.get(self)

    def set(self, text):
        self._clear_ph()
        self.delete(0, "end")
        if text:
            self.insert(0, text)
        elif self.focus_get() is not self:
            self._draw_ph()
        self._apply_mask()


class LightSwitch(tk.Frame):
    """Title left, 34x20 switch at the right edge of its row; get/set like SwitchRow."""

    def __init__(self, parent, title, value=False, command=None):
        super().__init__(parent, bg=C["panel"])
        k = _k(self)
        self._cb, self._on, self._k = command, bool(value), k
        self.columnconfigure(0, weight=1)
        tk.Label(self, text=title, font=_tkfont("body", self), fg=C["ink"], bg=C["panel"], anchor="w",
                 padx=0, pady=0).grid(row=0, column=0, sticky="w")
        w, h = round(34 * k), round(20 * k)
        self._cv = ctk.CTkCanvas(self, width=w, height=h, bg=C["panel"], highlightthickness=0, bd=0, cursor="hand2")
        self._cv.grid(row=0, column=1, sticky="e", padx=(round(12 * k), 0))
        self._cv.bind("<Button-1>", self._toggle)
        self._paint()

    def _paint(self):
        cv, k = self._cv, self._k
        cv.delete("all")
        w, h = round(34 * k), round(20 * k)
        col = C["ok"] if self._on else C["inert-fill"]
        for x in (h / 2, w - h / 2):                  # anti-aliased ends and knob, as CTkSwitch draws them
            cv.create_aa_circle(x, h / 2, h // 2, fill=col)
        cv.create_rectangle(h // 2, 0, w - h // 2, h, fill=col, width=0)
        x = w - h / 2 if self._on else h / 2
        cv.create_aa_circle(x, h / 2, round(h / 2 - 3 * k), fill=C["ink"])

    def _toggle(self, _e=None):
        self.set(not self._on)
        if self._cb:
            self._cb(self._on)

    def get(self):
        return self._on

    def set(self, on):
        self._on = bool(on)
        self._paint()


def light_switch(parent, title, *, value=False, command=None):
    return LightSwitch(parent, title, value, command)


class LightMenu(tk.Frame):
    """Drop-down on a well box that opens a native menu; stands in for CTkOptionMenu (set/get/values)."""

    def __init__(self, parent, values, value=None, command=None, height=30, disabled=None):
        super().__init__(parent, bg=C["rule"], height=round(height * _k(parent)))
        k = _k(self)
        self.pack_propagate(False)
        self._values, self._cb, self._var = values, command, tk.StringVar(value=value or "")
        self._off = disabled or (lambda v: False)
        well = tk.Frame(self, bg=C["well"], cursor="hand2")
        well.pack(fill="both", expand=True, padx=1, pady=1)
        self._lab = tk.Label(well, textvariable=self._var, font=_tkfont("body", self), fg=C["ink"],
                             bg=C["well"], anchor="w", padx=0, pady=0)
        self._lab.pack(side="left", fill="x", expand=True, padx=(round(8 * k), 0))
        self._arrow = tk.Label(well, text="▾", font=_tkfont("body", self), fg=C["ink-2"], bg=C["well"],
                               padx=0, pady=0)
        self._arrow.pack(side="right", padx=round(8 * k))
        for w in (well, self._lab, self._arrow):
            w.bind("<Button-1>", self._open)

    def _open(self, _e=None):
        m = tk.Menu(self, tearoff=0, bg=C["panel"], fg=C["ink"], activebackground=C["raised"],
                    activeforeground=C["ink"], disabledforeground=C["ink-3"], bd=0, relief="flat",
                    font=_tkfont("body", self))
        for v in (self._values() if callable(self._values) else self._values):
            m.add_command(label=v, state="disabled" if self._off(v) else "normal", command=lambda v=v: self._pick(v))
        m.tk_popup(self.winfo_rootx(), self.winfo_rooty() + self.winfo_height())

    def _pick(self, v):
        self._var.set(v)
        if self._cb:
            self._cb(v)

    def get(self):
        return self._var.get()

    def set(self, v):
        self._var.set(v)


def light_menu(parent, values, *, value=None, command=None, height=30, disabled=None):
    return LightMenu(parent, values, value, command, height, disabled)


class LightPills(tk.Canvas):
    """One canvas drawing a row of pills: a segmented control (single choice, `value`) or chips
    (multi choice, `selected`). set/get take a label, or a list of labels for chips."""

    def __init__(self, parent, labels, *, multi=False, value=None, selected=(), command=None, bg=None, check=False):
        self._bg = bg or C["panel"]
        k = self._k = _k(parent)
        self._font = _tkfont("button", parent)
        f = measure_font(self._font, parent)
        self._labels, self._multi, self._cb, self._hover = list(labels), multi, command, None
        self._box = round(16 * k) if check else 0       # check box at the left of each chip
        self._sel = set(selected) if multi else value
        pad, self._h = round((16 if multi else 18) * k), round((30 if multi else 34) * k)
        self._gap = round(8 * k) if multi else 0
        self._edge = 0 if multi else round(3 * k)
        self._x = []                             # (x0, x1) per label
        x = self._edge
        for lab in self._labels:
            w = f.measure(lab) + 2 * pad + (self._box + round(8 * k) if self._box else 0)
            self._x.append((x, x + w))
            x += w + self._gap
        self._span = x - self._gap + self._edge
        super().__init__(parent, width=self._span, height=self._h, bg=self._bg, highlightthickness=0, bd=0,
                         cursor="hand2")
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._move)
        self.bind("<Leave>", lambda e: self._set_hover(None))
        self._paint()

    def _at(self, x):
        for i, (a, b) in enumerate(self._x):
            if a <= x < b:
                return i
        return None

    def _pill(self, x0, x1, y0, y1, fill):
        r = (y1 - y0) / 2
        self.create_line(x0 + r, (y0 + y1) / 2, x1 - r, (y0 + y1) / 2, width=y1 - y0, capstyle="round", fill=fill)

    def _paint(self):
        self.delete("all")
        h = self._h
        if not self._multi:
            self._pill(0, self._span, 0, h, C["well"])
        for i, lab in enumerate(self._labels):
            x0, x1 = self._x[i]
            on = lab in self._sel if self._multi else lab == self._sel
            if self._multi:
                self._pill(x0, x1, 0, h, (C["ok"] if self._box else C["ink-2"]) if on else C["rule"])
                self._pill(x0 + 1, x1 - 1, 1, h - 1, C["raised"] if on or i == self._hover else self._bg)
                col = C["ink"] if on else C["ink-2"]
            else:
                if on:
                    self._pill(x0, x1, self._edge, h - self._edge, C["raised"])
                elif i == self._hover:
                    self._pill(x0, x1, self._edge, h - self._edge, C["rule-soft"])
                col = C["ink"]
            tx = (x0 + x1) / 2
            if self._box:
                e, k = self._box, self._k
                bx = x0 + round(12 * k)
                self.create_rectangle(bx, (h - e) / 2, bx + e, (h + e) / 2, width=1,
                                      fill=C["ok"] if on else self._bg, outline=C["ok"] if on else C["ink-3"])
                if on:
                    self.create_text(bx + e / 2, h / 2, text="✓", fill=C["ground"], font=self._font)
                tx = (bx + e + round(8 * k) + x1 - round(12 * k)) / 2
            self.create_text(tx, h / 2, text=lab, font=self._font, fill=col)

    def _move(self, e):
        self._set_hover(self._at(e.x))

    def _set_hover(self, i):
        if i != self._hover:
            self._hover = i
            self._paint()

    def _click(self, e):
        i = self._at(e.x)
        if i is None:
            return
        lab = self._labels[i]
        if self._multi:
            on = lab not in self._sel
            self._sel.symmetric_difference_update({lab})
            self._paint()
            if self._cb:
                self._cb(lab, on)
        else:
            self._sel = lab
            self._paint()
            if self._cb:
                self._cb(lab)

    def get(self):
        return sorted(self._sel, key=self._labels.index) if self._multi else self._sel

    def set(self, value):
        self._sel = set(value) if self._multi else value
        self._paint()


def light_pills(parent, labels, *, multi=False, value=None, selected=(), command=None, bg=None, check=False):
    return LightPills(parent, labels, multi=multi, value=value, selected=selected, command=command, bg=bg,
                      check=check)


# ── Tooltip ───────────────────────────────────────────────────────────────────
class Tooltip:
    """Raised panel near the pointer after ~450 ms of hover; hides on leave or click. No shadow."""
    DELAY, WRAP = 450, 300

    def __init__(self, widget, text):
        self.w, self.text, self._job, self._tip, self._kids = widget, text, None, None, set()
        self._bind(widget)

    def _bind(self, w):
        self._kids.add(w)
        for ev, fn in (("<Enter>", self._enter), ("<Leave>", self._leave), ("<ButtonPress>", self._hide)):
            try:
                w.bind(ev, fn, add="+")
            except Exception:
                pass
        for c in w.winfo_children():
            self._bind(c)

    def _enter(self, _e=None):
        if self._job is None and self._tip is None:
            self._job = self.w.after(self.DELAY, self._show)

    def _leave(self, _e=None):
        self.w.after(40, self._check)

    def _check(self):
        """Moving between children also fires <Leave>; hide only once the pointer left the whole widget."""
        try:
            under = self.w.winfo_containing(*self.w.winfo_pointerxy())
        except Exception:
            under = None
        if under not in self._kids:
            self._hide()

    def _hide(self, _e=None):
        if self._job is not None:
            self.w.after_cancel(self._job)
            self._job = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _show(self):
        self._job = None
        try:
            x, y = self.w.winfo_pointerxy()
            tip = self._tip = tk.Toplevel(self.w)
            tip.wm_overrideredirect(True)
            key = "#010203"                       # transparent key, so the panel keeps its 8px corners
            tip.configure(bg=key)
            try:
                tip.wm_attributes("-transparentcolor", key)
            except Exception:
                tip.configure(bg=C["rule"])
            box = ctk.CTkFrame(tip, fg_color=C["raised"], corner_radius=R_INNER, border_width=1,
                               border_color=C["rule"], bg_color=key)
            box.pack()
            ctk.CTkLabel(box, text=self.text, font=font("hint"), text_color=C["ink-2"], anchor="w",
                         justify="left", wraplength=self.WRAP).pack(padx=12, pady=8)
            tip.wm_geometry(f"+{x + 14}+{y + 18}")
        except Exception:
            self._tip = None


def tooltip(widget, text):
    """Attach a hover tooltip to a widget and its children."""
    if text:
        Tooltip(widget, text)
    return widget


# ── Canvases: lane, meter, waveform, readouts ─────────────────────────────────
def _bars(cv, bins, x0, x1, y0, y1, split, on, off):
    """Draw centred bars for bins across [x0,x1]; bars left of x=split use `on`."""
    n = len(bins)
    if not n or x1 <= x0:
        return
    step = (x1 - x0) / n
    mid = (y0 + y1) / 2
    half = (y1 - y0) / 2
    for i, b in enumerate(bins):
        h = max(1.5, half * max(0.0, min(1.0, b)))
        x = x0 + i * step
        cv.create_rectangle(x + 0.5, mid - h, x + max(1.0, step - 1.5), mid + h, width=0,
                            fill=on if x + step / 2 <= split else off)


class LaneRow(ctk.CTkFrame):
    """Speaker lane: name column on the panel, then an 8px well bed holding a 6px clip."""

    def __init__(self, parent, label, colour):
        super().__init__(parent, fg_color=C["panel"], corner_radius=0, height=64)
        self.colour = colour; self._bins = []; self._active = True
        self.grid_propagate(False); self.columnconfigure(1, weight=1); self.rowconfigure(0, weight=1)
        name = ctk.CTkFrame(self, fg_color="transparent", width=150)
        name.grid(row=0, column=0, sticky="ns"); name.pack_propagate(False)
        self._dot = ctk.CTkFrame(name, width=10, height=10, corner_radius=3, fg_color=colour)
        self._dot.pack(side="left")
        ctk.CTkLabel(name, text=label, font=font("body"), text_color=C["ink"],
                     anchor="w").pack(side="left", padx=(8, 0))
        bed = ctk.CTkFrame(self, fg_color=C["well"], corner_radius=R_INNER, border_width=1,
                           border_color=C["rule-soft"])
        bed.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.canvas = ctk.CTkCanvas(bed, bg=C["well"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=3, pady=3)
        self.canvas.bind("<Configure>", lambda e: self._render())

    def set_bins(self, bins):
        self._bins = list(bins); self._render()

    def set_active(self, active):
        """The active lane is full colour; the rest sit at 42%."""
        self._active = bool(active)
        self._dot.configure(fg_color=self.colour if active else _mix(self.colour, C["panel"], 0.42))
        self._render()

    def _render(self):
        cv = self.canvas; cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 20:
            return
        col = self.colour if self._active else _mix(self.colour, C["well"], 0.42)
        k = _k(cv)
        x0, y0, x1, y1, r = round(4 * k), round(3 * k), w - round(4 * k), h - round(3 * k), round(6 * k)
        rrect(cv, x0, y0, x1, y1, r, _mix(col, C["well"], 0.14), edge=col, b=max(1, round(1.5 * k)))
        _bars(cv, self._bins, x0 + 6, x1 - 6, y0 + 6, y1 - 6, w, col, col)


def lane_row(parent, label, colour):
    return LaneRow(parent, label, colour)


class Meter(ctk.CTkCanvas):
    """12 segmented cells (6x10, 2px gap); .set(fraction 0..1)."""
    CW, CH, CG = 6, 10, 2

    def __init__(self, parent, colour=None, **kw):
        w = METER_CELLS * self.CW + (METER_CELLS - 1) * self.CG
        super().__init__(parent, width=w, height=self.CH, bg=kw.pop("bg", C["panel"]),
                         highlightthickness=0, **kw)
        self.colour = colour or C["ember"]; self._cells = []
        for i in range(METER_CELLS):
            x = i * (self.CW + self.CG)
            self._cells.append(self.create_rectangle(x, 0, x + self.CW, self.CH, width=0,
                                                     fill=C["inert-fill"]))

    def set(self, fraction, colour=None):
        if colour:
            self.colour = colour
        lit = round(max(0.0, min(1.0, fraction)) * METER_CELLS)
        for i, cell in enumerate(self._cells):
            self.itemconfigure(cell, fill=self.colour if i < lit else C["inert-fill"])


class ReadoutStrip(ctk.CTkFrame):
    """Readout-well strip: caps label then ember mono value on one baseline, pairs split by 1px rules."""

    def __init__(self, parent, items):
        super().__init__(parent, fg_color=C["readout-well"], corner_radius=R_INNER, border_width=1,
                         border_color=C["rule-soft"], width=1, height=1)
        self._vals = {}; self._cells = {}
        for i, (label, value) in enumerate(items):
            rule = None
            if i:
                rule = ctk.CTkFrame(self, width=1, height=1, fg_color=C["rule"], corner_radius=0)
                rule.pack(side="left", fill="y", padx=(0, 18), pady=9)
            cell = ctk.CTkFrame(self, fg_color="transparent", width=1, height=1)
            cell.pack(side="left", padx=(14 if i == 0 else 0, 18 if i < len(items) - 1 else 14), pady=9)
            self._cells[label] = (rule, cell)
            ctk.CTkLabel(cell, text=label.upper(), font=font("label"), text_color=C["ink-3"],
                         anchor="w").pack(side="left", padx=(0, 8))
            self._vals[label] = ctk.CTkLabel(cell, text=value, font=font("mono"),
                                             text_color=C["ember"], anchor="w")
            self._vals[label].pack(side="left")

    def set(self, label, value):
        self._vals[label].configure(text=value)

    def show(self, label, on):
        """Show/hide one pair (the last one; its rule goes with it)."""
        rule, cell = self._cells[label]
        if on and not cell.winfo_manager():
            if rule:
                rule.pack(side="left", fill="y", padx=(0, 18), pady=9)
            cell.pack(side="left", padx=(0, 14), pady=9)
        elif not on and cell.winfo_manager():
            cell.pack_forget()
            if rule:
                rule.pack_forget()


def readout_strip(parent, items):
    return ReadoutStrip(parent, items)


class Waveform(ctk.CTkCanvas):
    """Bars centred on the midline: heat up to the playhead, inert beyond; 2px ink playhead."""

    def __init__(self, parent, height=96, **kw):
        super().__init__(parent, height=height, bg=kw.pop("bg", C["well"]), highlightthickness=0, **kw)
        self._bins = []; self._progress = 0.0
        self.bind("<Configure>", lambda e: self._render())

    def set_bins(self, bins):
        self._bins = list(bins); self._render()

    def set_progress(self, fraction):
        self._progress = max(0.0, min(1.0, fraction)); self._render()

    def _render(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 8:
            return
        x = 8 + (w - 16) * self._progress
        _bars(self, self._bins, 8, w - 8, 8, h - 8, x, C["heat"], C["inert-fill"])
        if self._progress > 0:
            self.create_rectangle(x - 1, 0, x + 1, h, width=0, fill=C["ink"])


# ── Demo: python -m ui.theme ──────────────────────────────────────────────────
def _demo():
    import math
    ctk.set_appearance_mode("dark")
    app = ctk.CTk(fg_color=C["ground"])
    app.title("WhisperX theme demo"); app.geometry("1280x800")
    load_fonts()
    bins = [0.15 + 0.8 * abs(math.sin(i * 0.37) * math.cos(i * 0.11)) for i in range(120)]
    app.columnconfigure((0, 1), weight=1, uniform="c"); app.rowconfigure(0, weight=1)
    left = panel(app); left.grid(row=0, column=0, sticky="nsew", padx=(GAP, GAP // 2), pady=GAP)
    right = panel(app); right.grid(row=0, column=1, sticky="nsew", padx=(GAP // 2, GAP), pady=GAP)

    def head(p, t):
        ctk.CTkLabel(p, text=t, font=font("heading"), text_color=C["ink"], anchor="w").pack(
            anchor="w", padx=PAD_LG, pady=(PAD, 8))

    head(left, "Controls")
    row = ctk.CTkFrame(left, fg_color="transparent"); row.pack(anchor="w", padx=PAD_LG, pady=4)
    pill(row, "Cancel").pack(side="left", padx=(0, 12))
    pill(row, "Transcribe", primary=True).pack(side="left")
    row = ctk.CTkFrame(left, fg_color="transparent"); row.pack(anchor="w", padx=PAD_LG, pady=8)
    for i, f in enumerate(("SRT", "VTT", "TXT")):
        chip(row, f, selected=i == 0).pack(side="left", padx=(0, 8))
    segmented(left, ["Transcribe", "Translate"]).pack(anchor="w", padx=PAD_LG, pady=8)
    switch_row(left, "Diarize speakers", value=True).pack(
        fill="x", padx=PAD_LG, pady=8)
    field(left, "Save to", placeholder="Choose a folder").pack(fill="x", padx=PAD_LG, pady=8)
    head(right, "Session")
    readout_strip(right, [("Device", "RTX 4070"), ("VRAM", "6.2 GB"), ("RTF", "0.12x")]).pack(
        anchor="w", padx=PAD_LG, pady=4)
    m = Meter(right); m.set(0.6); m.pack(anchor="w", padx=PAD_LG, pady=12)
    wf = Waveform(right); wf.pack(fill="x", padx=PAD_LG, pady=4)
    wf.set_bins(bins); wf.set_progress(0.4)
    for i, name in enumerate(("Speaker 1", "Speaker 2", "Speaker 3")):
        r = lane_row(right, name, LANES[i + 1]); r.pack(fill="x", padx=PAD_LG, pady=4)
        r.set_bins(bins[i * 10:i * 10 + 60]); r.set_active(i == 0)
    app.mainloop()


if __name__ == "__main__":
    _demo()
