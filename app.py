import os
import sys
import json
import queue
import threading
import subprocess
import webbrowser
import customtkinter as ctk
from tkinter import filedialog, messagebox

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

if IS_WIN:
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Muqaddimah.WhisperXTranscriber")
    except Exception:
        pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Color palette — (light, dark) ────────────────────────────────────────────
_SB      = ("#f5f5f5", "#111111")
_CARD    = ("#ffffff", "#1e1e1e")
_CONT    = ("#f0f0f0", "#161616")
_ENTRY   = ("#f8f8f8", "#141414")
_BAR     = ("#e8e8e8", "#0d0d0d")
_SEP     = ("#d8d8d8", "#1e1e1e")
_FMT     = ("#ececec", "#161616")
_LOGHDR  = ("#e0e0e0", "#111111")
_NAVACT  = ("#e5e5e5", "#1e1e1e")
_NAVHOV  = ("#ebebeb", "#1a1a1a")
_MID     = ("#d8d8d8", "#2a2a2a")
_MIDHOV  = ("#c8c8c8", "#3a3a3a")
_BORDER  = ("#d8d8d8", "#2a2a2a")
_SCRL_B  = ("#c8c8c8", "#2a2a2a")
_SCRL_H  = ("#b8b8b8", "#3a3a3a")
_PROG_BG = ("#d0d0d0", "#1a1a1a")
_MAC_BG  = ("#e8f5e8", "#162016")
_DROP    = ("#f0f0f0", "#1e1e1e")

_PRI     = ("#111111", "#ffffff")
_LOGO_S  = ("#888888", "#888888")
_NAV_OF  = ("#777777", "#666666")
_LABEL   = ("#3a3a3a", "#b0b0b0")
_HINT    = ("#6e6e6e", "#8c8c8c")
_SECHI   = ("#5a5a5a", "#9a9a94")
_STAT    = ("#3a3a3a", "#b0b0b0")
_LOG_H   = ("#555555", "#8c8c8c")
_LOG_B   = ("#111111", "#d4d4d4")
_FMT_N   = ("#1a1a1a", "#e4e4e4")
_FMT_D   = ("#666666", "#8c8c8c")
_TOGG_L  = ("#1a1a1a", "#e0e0e0")
_DIM_TXT = ("#999999", "#888888")
_MAC_TXT = ("#2a7a2a", "#3a7a3a")


# ── Fonts ─────────────────────────────────────────────────────────────────────
def F(size=13, weight="normal"):
    family = "SF Pro Display" if IS_MAC else "Segoe UI"
    return ctk.CTkFont(family=family, size=size, weight=weight)

def FM(size=10):
    family = "Menlo" if IS_MAC else "Consolas"
    return ctk.CTkFont(family=family, size=size)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _default_model_dir():
    if getattr(sys, "frozen", False):
        # PyInstaller onedir: store Models next to the .exe, not inside _internal/
        return os.path.join(os.path.dirname(sys.executable), "Models")
    if IS_MAC:
        return os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support", "WhisperX", "Models")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")


def _available_devices():
    """Returns device list. Safe to call from any thread; imports torch lazily."""
    try:
        import torch
        devs = ["auto", "cpu"]
        if torch.cuda.is_available():
            devs.insert(1, "cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devs.insert(1, "mps")
        return devs
    except Exception:
        return ["auto", "cpu"]


# ── Reusable UI builders ──────────────────────────────────────────────────────
def combo(parent, var, values, width=None):
    kw = dict(variable=var, values=values, height=38, corner_radius=8,
              fg_color=_ENTRY, button_color=_MID, button_hover_color=_MIDHOV,
              dropdown_fg_color=_DROP, dropdown_hover_color=_NAVHOV,
              dropdown_text_color=_PRI, text_color=_PRI,
              font=F(13), dropdown_font=F(12),
              dynamic_resizing=False, anchor="w")
    if width:
        kw["width"] = width
    return ctk.CTkOptionMenu(parent, **kw)


def entry(parent, var, width=None, placeholder=""):
    kw = dict(textvariable=var, font=F(13), height=38, corner_radius=8,
              fg_color=_ENTRY, border_color=_BORDER, border_width=1,
              text_color=_PRI, placeholder_text=placeholder)
    if width:
        kw["width"] = width
    return ctk.CTkEntry(parent, **kw)


# ── Main App ──────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("WhisperX")
        self.geometry("720x640")
        self.resizable(False, False)

        self._stop       = threading.Event()
        self._q          = queue.Queue()
        self._panels     = {}
        self._nav_key    = None
        self._device_combo = None  # populated by background probe
        self._proc       = None    # running transcription subprocess

        # All StringVars / BooleanVars — created here so any panel can use them
        self.v_file       = ctk.StringVar()
        self.v_lang       = ctk.StringVar(value="Auto-detect")
        self.v_model      = ctk.StringVar(value="large-v2")
        self.v_device     = ctk.StringVar(value="auto")
        self.v_compute    = ctk.StringVar(value="int8" if IS_MAC else "float16")
        self.v_batch      = ctk.StringVar(value="8" if IS_MAC else "16")
        self.v_beam       = ctk.StringVar(value="5")
        self.v_chunk      = ctk.StringVar(value="30")
        self.v_out_dir    = ctk.StringVar()
        self.v_fmts       = {
            "srt":       ctk.BooleanVar(value=True),
            "vtt":       ctk.BooleanVar(value=False),
            "txt":       ctk.BooleanVar(value=False),
            "tsv":       ctk.BooleanVar(value=False),
            "json":      ctk.BooleanVar(value=False),
            "word_json": ctk.BooleanVar(value=False),
        }
        self.v_align      = ctk.BooleanVar(value=True)
        self.v_vad        = ctk.BooleanVar(value=True)
        self.v_vad_onset  = ctk.StringVar(value="0.500")
        self.v_vad_offset = ctk.StringVar(value="0.363")
        self.v_highlight  = ctk.BooleanVar(value=False)
        self.v_max_width  = ctk.StringVar(value="")
        self.v_max_count  = ctk.StringVar(value="")
        self.v_model_dir  = ctk.StringVar(value=_default_model_dir())
        self._lang_map    = {
            "Auto-detect": "auto", "English": "en", "Arabic": "ar",
            "Chinese": "zh", "Dutch": "nl", "French": "fr",
            "Georgian": "ka", "German": "de", "Hindi": "hi",
            "Italian": "it", "Japanese": "ja", "Korean": "ko",
            "Pashto": "ps", "Persian": "fa", "Polish": "pl",
            "Portuguese": "pt", "Russian": "ru", "Spanish": "es",
            "Turkish": "tr", "Urdu": "ur",
        }

        self._build()
        self._setup_icon()
        self._poll_log()
        self._nav("Transcribe")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if IS_MAC:
            self.createcommand("tk::mac::Quit", self._on_close)

        # Probe for GPU devices in a background thread so torch import
        # does not block the window from appearing on first launch.
        threading.Thread(target=self._probe_devices, daemon=True).start()

    # ── Device probe (background) ─────────────────────────────────────────────

    def _probe_devices(self):
        devs = _available_devices()
        if self._device_combo is not None:
            self.after(0, lambda: self._device_combo.configure(values=devs))

    # ── Icon ──────────────────────────────────────────────────────────────────

    def _setup_icon(self):
        # Try a pre-built icon file first (present in packaged builds).
        ico_candidates = []
        if getattr(sys, "frozen", False):
            ico_candidates.append(
                os.path.join(os.path.dirname(sys.executable), "assets", "icon.ico"))
        ico_candidates += [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico"),
        ]

        for path in ico_candidates:
            if os.path.isfile(path) and IS_WIN:
                try:
                    self.after(50, lambda p=path: self.iconbitmap(p))
                    return
                except Exception:
                    pass

        # Fall back to generating the icon at runtime using PIL.
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageTk
            import tempfile

            size = 128
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([0, 0, size - 1, size - 1],
                                radius=size // 6,
                                fill=(74, 158, 255, 255))

            font = None
            for path in [
                "C:/Windows/Fonts/arialbd.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]:
                try:
                    font = ImageFont.truetype(path, int(size * 0.70))
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()

            bb = d.textbbox((0, 0), "W", font=font)
            x = (size - (bb[2] - bb[0])) // 2 - bb[0]
            y = (size - (bb[3] - bb[1])) // 2 - bb[1] - int(size * 0.04)
            d.text((x, y), "W", fill=(255, 255, 255, 255), font=font)

            if IS_WIN:
                ico_path = os.path.join(tempfile.gettempdir(), "whisperx.ico")
                img.save(ico_path, format="ICO",
                         sizes=[(128, 128), (64, 64), (32, 32), (16, 16)])
                self.after(50, lambda: self.iconbitmap(ico_path))

            self._icon_img = ImageTk.PhotoImage(
                img.resize((64, 64), Image.Resampling.LANCZOS))
            self.iconphoto(True, self._icon_img)

        except Exception:
            pass

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._sb = ctk.CTkFrame(self, width=64, corner_radius=0,
                                fg_color=_SB)
        self._sb.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self._sb.grid_propagate(False)
        self._sb.grid_rowconfigure(10, weight=1)
        self._build_sidebar()

        self._content_wrap = ctk.CTkFrame(self, corner_radius=0,
                                          fg_color=_CONT)
        self._content_wrap.grid(row=0, column=1, sticky="nsew")
        self._content_wrap.grid_rowconfigure(0, weight=1)
        self._content_wrap.grid_columnconfigure(0, weight=1)
        self._build_content()

        self._bar = ctk.CTkFrame(self, height=52, corner_radius=0,
                                 fg_color=_BAR)
        self._bar.grid(row=1, column=1, sticky="ew")
        self._build_statusbar()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = self._sb
        sb.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sb, text="W", font=F(20, "bold"),
                     text_color="#4a9eff",
                     width=64, height=52).grid(row=0, column=0)

        ctk.CTkFrame(sb, height=1, fg_color=_SEP).grid(
            row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        self._nav_btns = {}
        for i, (key, icon) in enumerate([
            ("Transcribe", "▶"),
            ("Settings",   "⚙"),
        ], start=2):
            btn = ctk.CTkButton(
                sb, text=icon, width=44, height=44,
                corner_radius=10, fg_color="transparent",
                hover_color=_NAVHOV, text_color=_NAV_OF,
                font=F(17), command=lambda k=key: self._nav(k),
            )
            btn.grid(row=i, column=0, pady=2)
            self._nav_btns[key] = btn

        ctk.CTkFrame(sb, fg_color="transparent").grid(row=10, column=0,
                                                      sticky="nsew")

        ctk.CTkFrame(sb, height=1, fg_color=_SEP).grid(
            row=11, column=0, sticky="ew", padx=10, pady=(0, 8))

        ctk.CTkButton(
            sb, text="♥", width=44, height=44,
            corner_radius=10, fg_color="transparent",
            hover_color=_NAVHOV, text_color=("#c44569", "#e05c8a"),
            font=F(17), command=self._show_support,
        ).grid(row=12, column=0, pady=2)

        self._theme_sw = ctk.CTkSwitch(sb, text="", width=44,
                                       button_color="#4a9eff",
                                       progress_color="#4a9eff",
                                       command=self._toggle_theme)
        self._theme_sw.grid(row=13, column=0, pady=(4, 18))
        self._theme_sw.select()

    # ── Content panels ────────────────────────────────────────────────────────

    def _build_content(self):
        body = ctk.CTkFrame(self._content_wrap, fg_color="transparent",
                            corner_radius=0)
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self._body = body

        self._panels["Transcribe"] = self._panel_transcribe(body)
        self._panels["Settings"]   = self._panel_settings(body)

        # Activity log — overlays the bottom of the content area on demand.
        # Lives in the SAME grid cell as `body` so showing it never resizes
        # the panel; it's just lifted on top and anchored to the bottom.
        self._log_wrap = ctk.CTkFrame(self._content_wrap, fg_color=_BAR,
                                      corner_radius=0, height=200,
                                      border_width=1, border_color=_SEP)
        self._log_wrap.grid_propagate(False)
        self._log_wrap.grid_columnconfigure(0, weight=1)
        self._log_wrap.grid_rowconfigure(1, weight=1)

        lh = ctk.CTkFrame(self._log_wrap, fg_color=_BAR,
                          corner_radius=0, height=32)
        lh.grid(row=0, column=0, sticky="ew")
        lh.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(lh, text="   ACTIVITY", font=F(10, "bold"),
                     text_color=_LOG_H).grid(row=0, column=0, sticky="w",
                                              pady=6)
        btn_frame = ctk.CTkFrame(lh, fg_color="transparent")
        btn_frame.grid(row=0, column=1, padx=8, pady=4)
        ctk.CTkButton(btn_frame, text="Clear", width=50, height=24, font=F(11),
                      fg_color="transparent", hover_color=_NAVACT,
                      text_color=_NAV_OF,
                      command=self._clear_log).pack(side="left")
        ctk.CTkButton(btn_frame, text="✕", width=30, height=24, font=F(12),
                      fg_color="transparent", hover_color=_NAVACT,
                      text_color=_NAV_OF,
                      command=self._hide_log).pack(side="left", padx=(2, 0))

        self.log_area = ctk.CTkTextbox(
            self._log_wrap, font=FM(11), fg_color=_BAR,
            text_color=_LOG_B, corner_radius=0, state="disabled",
            scrollbar_button_color=_SCRL_B)
        self.log_area.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 2))

        self._log_visible = False

    # ── Transcribe panel — everything needed to run a job ─────────────────────

    def _panel_transcribe(self, parent):
        p = ctk.CTkFrame(parent, fg_color="transparent")
        p.grid_columnconfigure(0, weight=1)

        def glabel(text, row, top=14):
            ctk.CTkLabel(p, text=text, font=F(10, "bold"),
                         text_color=_SECHI).grid(
                row=row, column=0, sticky="w", pady=(top, 6))

        # ── Header ─────────────────────────────────────────────────────────────
        ctk.CTkLabel(p, text="Transcribe", font=F(20, "bold"),
                     text_color=_PRI).grid(row=0, column=0, sticky="w",
                                            pady=(0, 12))

        # ── Media file box ─────────────────────────────────────────────────────
        fb = ctk.CTkFrame(p, fg_color=_CARD, corner_radius=10,
                          border_width=1, border_color=_BORDER)
        fb.grid(row=1, column=0, sticky="ew")
        fb.grid_columnconfigure(0, weight=1)
        self._file_name_lbl = ctk.CTkLabel(
            fb, text="No file selected", font=F(13),
            text_color=_HINT, anchor="w", wraplength=480, justify="left")
        self._file_name_lbl.grid(row=0, column=0, sticky="w",
                                  padx=14, pady=(11, 0))
        self._file_meta_lbl = ctk.CTkLabel(
            fb, text="Choose an audio or video file to get started",
            font=F(10), text_color=_HINT, anchor="w")
        self._file_meta_lbl.grid(row=1, column=0, sticky="w",
                                  padx=14, pady=(0, 11))
        ctk.CTkButton(
            fb, text="Browse", width=88, height=34, corner_radius=8,
            fg_color="#4a9eff", hover_color="#6ab0ff", text_color="#ffffff",
            font=F(12, "bold"), command=self._browse_file,
        ).grid(row=0, column=1, rowspan=2, padx=(8, 11))

        # ── Language / Model / Device ──────────────────────────────────────────
        glabel("LANGUAGE & MODEL", 2)
        row3 = ctk.CTkFrame(p, fg_color="transparent")
        row3.grid(row=3, column=0, sticky="ew")
        row3.grid_columnconfigure((0, 1, 2), weight=1)
        for col, (var, vals, setter) in enumerate([
            (self.v_lang, list(self._lang_map.keys()), None),
            (self.v_model,
             ["tiny", "tiny.en", "base", "base.en", "small", "small.en",
              "medium", "medium.en", "large-v1", "large-v2", "large-v3"], None),
            (self.v_device, ["auto", "cpu"], "device"),
        ]):
            cb = combo(row3, var, vals)
            cb.grid(row=0, column=col, sticky="ew",
                    padx=(0 if col == 0 else 5, 5 if col == 0 else 0))
            if setter == "device":
                self._device_combo = cb

        # ── Output formats ─────────────────────────────────────────────────────
        glabel("OUTPUT FORMATS", 4)
        fmt_row = ctk.CTkFrame(p, fg_color="transparent")
        fmt_row.grid(row=5, column=0, sticky="w")
        self._fmt_chips = {}
        for col, (key, label) in enumerate([
            ("srt", "SRT"), ("vtt", "VTT"), ("txt", "TXT"),
            ("tsv", "TSV"), ("json", "JSON"), ("word_json", "Word JSON"),
        ]):
            btn = ctk.CTkButton(
                fmt_row, text=label,
                width=82 if key == "word_json" else 60,
                height=32, corner_radius=8, font=F(11, "bold"),
                command=lambda k=key: self._toggle_fmt(k))
            btn.grid(row=0, column=col, padx=(0, 7))
            self._fmt_chips[key] = btn
            self._update_fmt_chip(key)

        # ── Save location ──────────────────────────────────────────────────────
        glabel("SAVE TO", 6)
        save_row = ctk.CTkFrame(p, fg_color="transparent")
        save_row.grid(row=7, column=0, sticky="ew")
        save_row.grid_columnconfigure(0, weight=1)
        entry(save_row, self.v_out_dir).grid(row=0, column=0, sticky="ew",
                                             padx=(0, 6))
        ctk.CTkButton(save_row, text="...", width=42, height=36,
                      corner_radius=8, fg_color=_MID, hover_color=_MIDHOV,
                      font=F(14), command=self._browse_out_dir).grid(
            row=0, column=1)

        # ── Options (compact toggles) ──────────────────────────────────────────
        glabel("OPTIONS", 8)
        opt = ctk.CTkFrame(p, fg_color="transparent")
        opt.grid(row=9, column=0, sticky="ew")
        opt.grid_columnconfigure((0, 1), weight=1)
        for col, (lbl, var) in enumerate([
            ("Word timestamps", self.v_align),
            ("Remove silence",  self.v_vad),
        ]):
            cell = ctk.CTkFrame(opt, fg_color=_CARD, corner_radius=8)
            cell.grid(row=0, column=col, sticky="ew",
                      padx=(0, 5) if col == 0 else (5, 0))
            cell.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(cell, text=lbl, font=F(12),
                         text_color=_TOGG_L).grid(row=0, column=0,
                                                   sticky="w", padx=12, pady=11)
            ctk.CTkSwitch(cell, text="", variable=var, width=42,
                          button_color="#4a9eff",
                          progress_color="#4a9eff").grid(
                row=0, column=1, padx=(0, 12))

        # Spacer pushes the action button to the bottom of the panel
        p.grid_rowconfigure(10, weight=1)

        # ── Primary action button ──────────────────────────────────────────────
        self._run_btn = ctk.CTkButton(
            p, text="▶   Transcribe", height=46, corner_radius=12,
            fg_color="#4a9eff", hover_color="#6ab0ff", text_color="#ffffff",
            font=F(15, "bold"), command=self._on_run)
        self._run_btn.grid(row=11, column=0, sticky="ew", pady=(16, 2))

        return p

    def _toggle_fmt(self, key):
        self.v_fmts[key].set(not self.v_fmts[key].get())
        self._update_fmt_chip(key)

    def _update_fmt_chip(self, key):
        btn = self._fmt_chips.get(key)
        if btn is None:
            return
        active = self.v_fmts[key].get()
        btn.configure(
            fg_color="#4a9eff" if active else _FMT,
            border_color=("#4a9eff", "#4d8ef7") if active else _BORDER,
            border_width=0 if active else 1,
            hover_color="#6ab0ff" if active else _NAVHOV,
            text_color="#ffffff" if active else _FMT_N,
        )

    # ── Settings panel — advanced / rarely-touched config ─────────────────────

    def _panel_settings(self, parent):
        p = ctk.CTkFrame(parent, fg_color="transparent")
        p.grid_columnconfigure(0, weight=1)

        def glabel(text, row, top=14):
            ctk.CTkLabel(p, text=text, font=F(10, "bold"),
                         text_color=_SECHI).grid(
                row=row, column=0, sticky="w", pady=(top, 6))

        # ── Header ─────────────────────────────────────────────────────────────
        ctk.CTkLabel(p, text="Settings", font=F(20, "bold"),
                     text_color=_PRI).grid(row=0, column=0, sticky="w",
                                            pady=(0, 2))
        ctk.CTkLabel(p, text="Defaults work for most files.", font=F(11),
                     text_color=_HINT).grid(row=1, column=0, sticky="w",
                                             pady=(0, 6))

        # ── Performance ────────────────────────────────────────────────────────
        glabel("PERFORMANCE", 2, top=8)
        perf = ctk.CTkFrame(p, fg_color="transparent")
        perf.grid(row=3, column=0, sticky="ew")
        perf.grid_columnconfigure((0, 1, 2, 3), weight=1)
        # Precision (wide) + Speed / Accuracy / Segment
        ctk.CTkLabel(perf, text="Precision", font=F(10),
                     text_color=_LABEL).grid(row=0, column=0, sticky="w",
                                              padx=(0, 5), pady=(0, 3))
        for col, lbl in enumerate(["Speed", "Accuracy", "Segment"], start=1):
            ctk.CTkLabel(perf, text=lbl, font=F(10), text_color=_LABEL).grid(
                row=0, column=col, sticky="w", padx=5, pady=(0, 3))
        combo(perf, self.v_compute,
              ["int8", "float16", "float32", "int8_float16"]).grid(
            row=1, column=0, sticky="ew", padx=(0, 5))
        for col, var in enumerate([self.v_batch, self.v_beam, self.v_chunk],
                                  start=1):
            entry(perf, var).grid(row=1, column=col, sticky="ew", padx=5)

        if IS_MAC:
            note = ctk.CTkFrame(p, fg_color=_MAC_BG, corner_radius=8)
            note.grid(row=4, column=0, sticky="ew", pady=(8, 0))
            ctk.CTkLabel(note,
                         text="  Mac: Transcription on CPU · Alignment on MPS",
                         font=F(11), text_color=_MAC_TXT).grid(
                padx=12, pady=9, sticky="w")

        # ── Silence detection ──────────────────────────────────────────────────
        glabel("SILENCE DETECTION", 5)
        vf = ctk.CTkFrame(p, fg_color="transparent")
        vf.grid(row=6, column=0, sticky="w")
        for i, (lbl, var) in enumerate([("Sensitivity", self.v_vad_onset),
                                        ("Release",     self.v_vad_offset)]):
            ctk.CTkLabel(vf, text=lbl, font=F(11), text_color=_LABEL).grid(
                row=0, column=i * 2, sticky="w", padx=(0 if i == 0 else 20, 8))
            entry(vf, var, width=100).grid(row=0, column=i * 2 + 1, sticky="w")

        # ── Subtitle formatting ────────────────────────────────────────────────
        glabel("SUBTITLE FORMATTING", 7)
        cell = ctk.CTkFrame(p, fg_color=_CARD, corner_radius=8)
        cell.grid(row=8, column=0, sticky="ew")
        cell.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(cell, text="Karaoke highlighting", font=F(12),
                     text_color=_TOGG_L).grid(row=0, column=0, sticky="w",
                                               padx=12, pady=11)
        ctk.CTkSwitch(cell, text="", variable=self.v_highlight, width=42,
                      button_color="#4a9eff",
                      progress_color="#4a9eff").grid(row=0, column=1,
                                                      padx=(0, 12))
        sf = ctk.CTkFrame(p, fg_color="transparent")
        sf.grid(row=9, column=0, sticky="w", pady=(8, 0))
        for i, (lbl, var, ph) in enumerate([
            ("Max line width", self.v_max_width, "default"),
            ("Max line count", self.v_max_count, "default"),
        ]):
            ctk.CTkLabel(sf, text=lbl, font=F(11), text_color=_LABEL).grid(
                row=0, column=i * 2, sticky="w", padx=(0 if i == 0 else 20, 8))
            entry(sf, var, width=100, placeholder=ph).grid(
                row=0, column=i * 2 + 1, sticky="w")

        # ── Model storage ──────────────────────────────────────────────────────
        glabel("AI MODELS", 10)
        mr = ctk.CTkFrame(p, fg_color="transparent")
        mr.grid(row=11, column=0, sticky="ew")
        mr.grid_columnconfigure(0, weight=1)
        entry(mr, self.v_model_dir).grid(row=0, column=0, sticky="ew",
                                         padx=(0, 6))
        ctk.CTkButton(mr, text="...", width=42, height=36, corner_radius=8,
                      fg_color=_MID, hover_color=_MIDHOV, font=F(14),
                      command=self._browse_model_dir).grid(row=0, column=1)

        return p

    # ── Support dialog ────────────────────────────────────────────────────────

    def _show_support(self):
        w = ctk.CTkToplevel(self)
        w.title("Support the Project")
        w.geometry("420x470")
        w.resizable(False, False)
        w.grab_set()
        w.focus_set()
        w.grid_columnconfigure(0, weight=1)
        w.grid_rowconfigure(0, weight=1)

        root = ctk.CTkFrame(w, fg_color=_CONT, corner_radius=0)
        root.grid(row=0, column=0, sticky="nsew")
        root.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(root, text="♥", font=F(30),
                     text_color=("#c44569", "#e05c8a")).grid(
            row=0, column=0, pady=(30, 2))
        ctk.CTkLabel(root, text="Support this project", font=F(18, "bold"),
                     text_color=_PRI).grid(row=1, column=0, pady=(0, 20))

        msg = ctk.CTkFrame(root, fg_color=_CARD, corner_radius=12)
        msg.grid(row=2, column=0, padx=28, sticky="ew", pady=(0, 22))
        msg.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            msg,
            text=(
                "I'm a working student engineer and part-time researcher\n"
                "building this entirely in my spare time — free, forever.\n\n"
                "If this app has saved you hours of work, a small contribution\n"
                "goes a long way and keeps development moving forward."
            ),
            font=F(12), text_color=_LABEL,
            justify="center", wraplength=340,
        ).grid(row=0, column=0, padx=20, pady=16)

        def _open(url):
            return lambda: webbrowser.open(url)

        ctk.CTkButton(
            root, text="♥   Sponsor on GitHub", height=42, corner_radius=10,
            fg_color=("#c44569", "#d63384"), hover_color=("#a83058", "#bf2070"),
            text_color="#ffffff", font=F(13, "bold"),
            command=_open("https://github.com/sponsors/ibrahimqureshae"),
        ).grid(row=3, column=0, padx=28, sticky="ew", pady=(0, 10))

        ctk.CTkButton(
            root, text="   Donate via PayPal", height=42, corner_radius=10,
            fg_color=("#0070ba", "#0085cc"), hover_color=("#005ea0", "#006eb0"),
            text_color="#ffffff", font=F(13, "bold"),
            command=_open("https://www.paypal.me/mibrahimqr"),
        ).grid(row=4, column=0, padx=28, sticky="ew", pady=(0, 10))

        ctk.CTkButton(
            root, text="★   Star on GitHub  —  it's free!", height=38,
            corner_radius=10, fg_color="transparent", hover_color=_NAVHOV,
            text_color=_LOGO_S, border_width=1, border_color=_BORDER,
            font=F(12),
            command=_open("https://github.com/ibrahimqureshae/whisperx-transcriber"),
        ).grid(row=5, column=0, padx=28, sticky="ew", pady=(0, 28))

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bar = self._bar
        bar.grid_columnconfigure(2, weight=1)

        self._dot = ctk.CTkLabel(bar, text="●", font=F(10),
                                 text_color="#34c759", width=16)
        self._dot.grid(row=0, column=0, padx=(18, 4))

        self._status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(bar, textvariable=self._status_var,
                     font=F(12), text_color=_STAT).grid(row=0, column=1,
                                                          sticky="w")

        self._progress = ctk.CTkProgressBar(bar, width=180, height=3,
                                            corner_radius=2,
                                            mode="indeterminate",
                                            fg_color=_PROG_BG,
                                            progress_color="#4a9eff")
        self._progress.grid(row=0, column=2, padx=20)
        self._progress.set(0)

        self._stop_btn = ctk.CTkButton(
            bar, text="Cancel", width=80, height=34, corner_radius=8,
            fg_color="transparent", hover_color=_BAR,
            text_color=_BAR, text_color_disabled=_BAR,
            border_width=0, state="disabled",
            font=F(12), command=self._on_stop)
        self._stop_btn.grid(row=0, column=3, padx=(0, 18))

    # ── Navigation ────────────────────────────────────────────────────────────

    def _nav(self, key):
        for k, btn in self._nav_btns.items():
            if k == key:
                btn.configure(fg_color=_NAVACT, text_color=_PRI)
            else:
                btn.configure(fg_color="transparent", text_color=_NAV_OF)
        self._nav_key = key
        for k, panel in self._panels.items():
            if k == key:
                panel.grid(row=0, column=0, sticky="nsew", padx=22, pady=(18, 14))
            else:
                panel.grid_remove()

    # ── Activity log — auto-shown while working, dismissable when idle ─────────

    def _show_log(self):
        if not self._log_visible:
            self._log_wrap.grid(row=0, column=0, sticky="sew")
            self._log_wrap.lift()
            self._log_visible = True

    def _hide_log(self):
        if self._log_visible:
            self._log_wrap.grid_remove()
            self._log_visible = False

    def _toggle_theme(self):
        ctk.set_appearance_mode(
            "dark" if self._theme_sw.get() else "light")

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, msg):
        self._q.put(str(msg))
        self.after(0, self._show_log)

    def _clear_log(self):
        self.log_area.configure(state="normal")
        self.log_area.delete("1.0", "end")
        self.log_area.configure(state="disabled")

    def _poll_log(self):
        while True:
            try:
                msg = self._q.get_nowait()
                self.log_area.configure(state="normal")
                self.log_area.insert("end", msg + "\n")
                self.log_area.see("end")
                self.log_area.configure(state="disabled")
            except queue.Empty:
                break
        self.after(100, self._poll_log)

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _browse_file(self):
        p = filedialog.askopenfilename(
            title="Select Audio / Video",
            filetypes=[("Media",
                        "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav "
                        "*.m4a *.flac *.ogg *.webm"),
                       ("All", "*.*")])
        if p:
            self.v_file.set(p)
            if not self.v_out_dir.get():
                self.v_out_dir.set(os.path.dirname(p))
            # Update display labels
            name = os.path.basename(p)
            ext  = os.path.splitext(name)[1].upper().lstrip(".")
            try:
                mb = os.path.getsize(p) / (1024 * 1024)
                meta = f"{ext}  ·  {mb:.0f} MB"
            except Exception:
                meta = ext
            self._file_name_lbl.configure(text=name, text_color=_PRI)
            self._file_meta_lbl.configure(text=meta, text_color=_HINT)

    def _browse_model_dir(self):
        p = filedialog.askdirectory(title="AI Models Folder")
        if p:
            self.v_model_dir.set(p)

    def _browse_out_dir(self):
        p = filedialog.askdirectory(title="Save to Folder")
        if p:
            self.v_out_dir.set(p)

    # ── Run / Stop ────────────────────────────────────────────────────────────

    def _on_run(self):
        if not self.v_file.get().strip():
            messagebox.showerror("Missing input",
                                 "Please select an audio/video file.")
            return
        if not self.v_out_dir.get().strip():
            messagebox.showerror("Missing output",
                                 "Please select an output directory.")
            return
        if not any(v.get() for v in self.v_fmts.values()):
            messagebox.showerror("No format",
                                 "Select at least one output format.")
            return

        def safe_int(s, default):
            try: return int(s)
            except Exception: return default

        def safe_float(s, default):
            try: return float(s)
            except Exception: return default

        cfg = {
            "audio":      self.v_file.get().strip(),
            "model_dir":  self.v_model_dir.get().strip(),
            "model":      self.v_model.get(),
            "device":     self.v_device.get(),
            "compute":    self.v_compute.get(),
            "batch":      safe_int(self.v_batch.get(), 16),
            "beam":       safe_int(self.v_beam.get(), 5),
            "chunk":      safe_int(self.v_chunk.get(), 30),
            "out_dir":    self.v_out_dir.get().strip(),
            "formats":    [k for k, v in self.v_fmts.items() if v.get()],
            "align":      bool(self.v_align.get()),
            "language":   self._lang_map.get(self.v_lang.get(), "auto"),
            "vad":        bool(self.v_vad.get()),
            "vad_onset":  safe_float(self.v_vad_onset.get(), 0.5),
            "vad_offset": safe_float(self.v_vad_offset.get(), 0.363),
            "highlight":  bool(self.v_highlight.get()),
            "max_width":  safe_int(self.v_max_width.get(), None)
                          if self.v_max_width.get().strip().isdigit() else None,
            "max_count":  safe_int(self.v_max_count.get(), None)
                          if self.v_max_count.get().strip().isdigit() else None,
        }

        self._stop.clear()
        self._run_btn.configure(state="disabled", text="Transcribing…",
                                fg_color="#2a5a8a")
        self._stop_btn.configure(state="normal", text_color=("#cc3333", "#ff453a"),
                                  hover_color=_NAVHOV)
        self._progress.configure(mode="indeterminate")
        self._progress.start()
        self._status_var.set("Running...")
        self._dot.configure(text_color="#ff9f0a")
        self._show_log()
        self._start_proc(cfg)

    def _on_close(self):
        self._stop.set()
        self._kill_proc()
        try:
            self.destroy()
        finally:
            os._exit(0)

    def _on_stop(self):
        # Hard-cancel: kill the worker process immediately.
        self._stop.set()
        self._log("Stopping…")
        self._status_var.set("Stopping...")
        self._stop_btn.configure(state="disabled", text_color=_BAR,
                                  hover_color=_BAR)
        self._kill_proc()

    def _kill_proc(self):
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    # ── Subprocess worker — runs core/runner.py so Cancel can kill it ─────────

    def _start_proc(self, cfg):
        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "core", "runner.py")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        kwargs = dict(
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        if IS_WIN:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self._proc = subprocess.Popen([sys.executable, runner], **kwargs)
        except Exception as e:
            self._log(f"ERROR: could not start worker: {e}")
            self._status_var.set("Error")
            self._dot.configure(text_color="#ff3b30")
            self._reset_btns()
            return
        try:
            self._proc.stdin.write(json.dumps(cfg) + "\n")
            self._proc.stdin.flush()
            self._proc.stdin.close()
        except Exception:
            pass
        threading.Thread(target=self._read_proc, args=(self._proc,),
                         daemon=True).start()

    def _read_proc(self, proc):
        last_path, err = None, None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                self._log(line)            # stray non-JSON output (e.g. traceback)
                continue
            t = ev.get("t")
            if t == "log":
                self._log(ev.get("m", ""))
            elif t == "status":
                self.after(0, lambda m=ev.get("m", ""): self._status_var.set(m))
            elif t == "pmode":
                if ev.get("mode") == "determinate":
                    self.after(0, self._progress_determinate)
                else:
                    self.after(0, self._progress_indeterminate)
            elif t == "progress":
                self.after(0, lambda v=ev.get("v", 0): self._progress.set(
                    min(1.0, max(0.0, v))))
            elif t == "done":
                last_path = ev.get("path")
            elif t == "error":
                err = ev.get("m", "Unknown error")
        rc = proc.wait()
        self.after(0, lambda: self._finish_proc(rc, last_path, err))

    def _finish_proc(self, rc, last_path, err):
        if self._stop.is_set():
            self._status_var.set("Stopped")
            self._dot.configure(text_color="#888888")
        elif err is not None or rc != 0:
            if err:
                self._log(f"ERROR: {err}")
            self._status_var.set("Error")
            self._dot.configure(text_color="#ff3b30")
        else:
            self._status_var.set("Done")
            self._dot.configure(text_color="#34c759")
            if last_path:
                folder = os.path.dirname(os.path.abspath(last_path))
                try:
                    subprocess.Popen(["explorer", folder],
                                     creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception:
                    pass
            self.after(4000, self._hide_log)
        self._proc = None
        self._reset_btns()

    def _reset_btns(self):
        self._run_btn.configure(state="normal", text="▶   Transcribe",
                                fg_color="#4a9eff")
        self._stop_btn.configure(state="disabled", text_color=_BAR,
                                  hover_color=_BAR)
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.set(0)

    # ── Progress helpers ──────────────────────────────────────────────────────

    def _progress_determinate(self):
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(0)

    def _progress_indeterminate(self):
        self._progress.stop()
        self._progress.configure(mode="indeterminate")
        self._progress.set(0)
        self._progress.start()


if __name__ == "__main__":
    App().mainloop()
