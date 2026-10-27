"""Shell — rail, top strip, screen frames and transport bar (DESIGN.md / prototype)."""
import importlib
import math
import os
import queue
import sys
import traceback
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from ui import theme
from ui.theme import C, font
from ui.state import AppState, available_devices, base_dir

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCREENS = ("transcribe", "queue", "transcript", "models", "settings")
WIN_W, WIN_H, TITLEBAR = 1366, 768, 32      # fixed logical window; title bar allowance
TITLES = {"transcribe": "Transcribe", "queue": "Queue", "transcript": "Transcript",
          "models": "Models", "settings": "Settings", "log": "Activity log"}
SUPPORT_LINKS = (       # glyph, label, page: opened in the browser; the app itself sends nothing
    ("★", "Star on GitHub", "https://github.com/ibrahimqureshae/whisperx-transcriber"),
    ("♥", "Sponsor", "https://github.com/sponsors/ibrahimqureshae"),
    ("☕", "Buy me a coffee", "https://buymeacoffee.com/mibrahim99"),
    ("$", "PayPal", "https://www.paypal.me/mibrahimqr"),
)
SUPPORT_ROW_H, SUPPORT_PAD = 34, 6


def _hdrop_paths():
    """Files copied in Explorer (CF_HDROP on the Windows clipboard); [] if none."""
    import ctypes
    from ctypes import wintypes
    user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.GetClipboardData.argtypes, user32.GetClipboardData.restype = [wintypes.UINT], wintypes.HANDLE
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    if not user32.OpenClipboard(None):
        return []
    try:
        h = user32.GetClipboardData(15)
        if not h:
            return []
        out = []
        for i in range(shell32.DragQueryFileW(h, 0xFFFFFFFF, None, 0)):
            buf = ctypes.create_unicode_buffer(shell32.DragQueryFileW(h, i, None, 0) + 1)
            shell32.DragQueryFileW(h, i, buf, len(buf))
            out.append(buf.value)
        return out
    finally:
        user32.CloseClipboard()


# ── Taskbar progress (Windows ITaskbarList3 through ctypes COM; best-effort) ──
TBPF_NOPROGRESS, TBPF_NORMAL, TBPF_ERROR = 0, 2, 4


class Taskbar:
    """Progress on the window's taskbar button. Every COM failure is swallowed; no-op elsewhere."""
    _VT = {"HrInit": 3, "SetProgressValue": 9, "SetProgressState": 10}

    def __init__(self, hwnd):
        self._hwnd = hwnd           # callable returning the frame HWND
        self._obj = None
        self._dead = not IS_WIN

    def _make(self):
        import ctypes
        import uuid
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD),
                        ("d", ctypes.c_ubyte * 8)]

        def guid(text):
            u = uuid.UUID(text)
            return GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
        ole32 = ctypes.windll.ole32
        ole32.CoInitialize(None)
        obj = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(ctypes.byref(guid("56FDF344-FD6D-11d0-958A-006097C9A090")), None, 1,
                                    ctypes.byref(guid("EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF")),
                                    ctypes.byref(obj))
        if hr != 0 or not obj.value:
            raise OSError(f"CoCreateInstance {hr:#x}")
        self._obj = obj
        self._call("HrInit")

    def _call(self, name, *args):
        import ctypes
        vt = ctypes.cast(self._obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        sig = {"HrInit": (), "SetProgressValue": (ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong),
               "SetProgressState": (ctypes.c_void_p, ctypes.c_int)}[name]
        fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *sig)(vt[self._VT[name]])
        hr = fn(self._obj, *args)
        if hr != 0:
            raise OSError(f"{name} {hr:#x}")

    def _do(self, *calls):
        if self._dead:
            return
        try:
            if self._obj is None:
                self._make()
            hwnd = self._hwnd()
            for name, *rest in calls:
                self._call(name, hwnd, *rest)
        except Exception:
            self._dead = True

    def progress(self, fraction):
        self._do(("SetProgressState", TBPF_NORMAL), ("SetProgressValue", round(fraction * 1000), 1000))

    def error(self):
        self._do(("SetProgressValue", 1000, 1000), ("SetProgressState", TBPF_ERROR))

    def clear(self):
        self._do(("SetProgressState", TBPF_NOPROGRESS))


# ── Rail ──────────────────────────────────────────────────────────────────────
class NavItem(ctk.CTkFrame):
    """38px rail row: label at left, mono count at right; current = raised fill + ink text.
    A busy item also shows a pulsing ember dot before its count."""

    def __init__(self, parent, text, command):
        super().__init__(parent, fg_color="transparent", corner_radius=8, height=38)
        self.pack_propagate(False)
        self._current = False
        self.label = ctk.CTkLabel(self, text=text, font=font("button"), text_color=C["ink-2"],
                                  anchor="w")
        self.label.pack(side="left", padx=(12, 0))
        self.count = ctk.CTkLabel(self, text="", font=font("mono-sm"), text_color=C["ink-3"])
        self.count.pack(side="right", padx=(0, 12))
        self._dot = tk.Canvas(self, width=10, height=10, highlightthickness=0, bd=0, bg=C["panel"])
        self._dot.create_oval(1, 1, 9, 9, fill=C["heat"], outline="", tags="d")
        self._phase = None
        for w in (self, self.label, self.count, self._dot):
            w.bind("<Button-1>", lambda e: command())
            w.bind("<Enter>", lambda e: self._hover(True))
            w.bind("<Leave>", lambda e: self._hover(False))

    def _hover(self, on):
        if not self._current:
            self.label.configure(text_color=C["ink"] if on else C["ink-2"])

    def set_current(self, on):
        self._current = on
        self.configure(fg_color=C["raised"] if on else "transparent")
        self.label.configure(text_color=C["ink"] if on else C["ink-2"])
        self._dot.configure(bg=C["raised"] if on else C["panel"])

    def set_count(self, text):
        self.count.configure(text=text)

    def set_busy(self, text):
        """text = the running percent (ember, with the dot); None = idle, nothing shown."""
        if text is None:
            self._phase = None
            self._dot.pack_forget()
            self.count.configure(text="", text_color=C["ink-3"])
            return
        self.count.configure(text=text, text_color=C["ember"])
        if self._phase is None:
            self._dot.pack(side="right", padx=(0, 6), after=self.count)
            self._phase = 0.0
            self._pulse()

    def _pulse(self):
        if self._phase is None:
            return
        self._phase += 0.07
        glow = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(self._phase))
        self._dot.itemconfigure("d", fill=theme._mix(C["heat"], self._dot.cget("bg"), glow))
        self.after(33, self._pulse)


class SupportRow(ctk.CTkFrame):
    """34px link row in the Support menu: glyph and label; hover = rule fill + ink text."""

    def __init__(self, parent, icon, text, command):
        super().__init__(parent, fg_color="transparent", corner_radius=6, height=SUPPORT_ROW_H)
        self.pack_propagate(False)
        self.command = command
        self.icon = ctk.CTkLabel(self, text=icon, width=18, font=font("button"), text_color=C["ink-3"],
                                 cursor="hand2")
        self.icon.pack(side="left", padx=(10, 0))
        self.label = ctk.CTkLabel(self, text=text, font=font("button"), text_color=C["ink-2"],
                                  anchor="w", cursor="hand2")
        self.label.pack(side="left", padx=(8, 0))
        for w in (self, self.icon, self.label):
            w.bind("<Button-1>", lambda e: self.invoke())
            w.bind("<Enter>", lambda e: self._hover(True))
            w.bind("<Leave>", lambda e: self._hover(False))

    def _hover(self, on):
        self.configure(fg_color=C["rule"] if on else "transparent")
        self.label.configure(text_color=C["ink"] if on else C["ink-2"])

    def invoke(self):
        self.command()


class Shell(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=C["ground"])
        dnd = False
        try:        # optional: file drops need tkinterdnd2 (not a hard dependency)
            from tkinterdnd2 import TkinterDnD
            self.TkdndVersion = TkinterDnD._require(self)
            dnd = True
        except Exception:
            pass
        # Worker threads never touch Tk: they post callables here; _drain runs them on the Tk thread.
        self._inbox = queue.SimpleQueue()
        self.app = AppState(schedule=self._inbox.put)
        self.after(30, self._drain)
        self.app.dnd = dnd
        theme.load_fonts()
        ctk.set_appearance_mode("dark")
        self.version = self._read_version()
        self.title(f"WhisperX Transcriber — {self.version}" if self.version else "WhisperX Transcriber")
        self._place_window()
        self._frames = {}
        self.current = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.grid_propagate(False)
        rail = self._build_rail()
        canvas = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        canvas.grid(row=0, column=1, sticky="nsew")
        canvas.columnconfigure(0, weight=1)
        canvas.rowconfigure(1, weight=1)
        self._build_strip(canvas)
        self._host = ctk.CTkFrame(canvas, fg_color=C["ground"], corner_radius=0)
        self._host.grid(row=1, column=0, sticky="nsew")
        self._host.columnconfigure(0, weight=1)
        self._host.rowconfigure(0, weight=1)
        self._build_log()
        self._build_transport(canvas)

        s = self.app
        s.on("screen", lambda name, **_: self.show(name))
        s.on("settings-tab", self._settings_tab)
        s.on("job", lambda **_: self._refresh_jobs())
        s.on("runner", self._on_runner)
        s.on("finished", self._on_finished)
        s.on("settings", lambda **_: self._refresh_settings())
        s.on("notice", lambda text, **_: self._status_lbl.configure(text=text))
        s.on("status", lambda text, **_: self._status_lbl.configure(text=text))
        for i, name in enumerate(SCREENS, 1):
            self.bind(f"<Control-Key-{i}>", lambda e, n=name: self.show(n))
        s.on("files-empty", lambda empty, **_: theme.restyle_pill(self._go, not empty, height=44))
        self.bind("<Control-o>", lambda e: s.emit("browse"))
        self.bind("<Control-v>", self._paste)
        self._edges = []
        self._pct, self._tb_err = None, None
        self.taskbar = Taskbar(lambda: int(self.wm_frame(), 16))
        if dnd:                     # the whole window takes drops; the border and tint show it is live
            for w in (rail, canvas):      # the Tk root itself cannot be a target; these two cover it
                w.drop_target_register("DND_Files")
                w.dnd_bind("<<Drop>>", self._on_drop)
                w.dnd_bind("<<DropEnter>>", lambda e: self._drag(True) or e.action)
                w.dnd_bind("<<DropLeave>>", lambda e: self._drag(False) or e.action)
        self.bind("<Control-Return>", self._start)
        self.bind("<Escape>", self._escape)
        self.bind("<Button-1>", self._click_away, add="+")

        self._setup_icon()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if IS_MAC:
            self.createcommand("tk::mac::Quit", self._on_close)
        self._refresh_jobs()
        self._refresh_settings()
        self.show("transcribe")
        threading.Thread(target=self._probe_devices, daemon=True).start()

    # ── Window geometry ───────────────────────────────────────────────────────
    def _work_area(self, scale):
        """Work area (x, y, w, h) in physical pixels of the monitor the window opens on."""
        if IS_WIN:
            try:
                import ctypes
                from ctypes import wintypes
                user32 = ctypes.windll.user32
                user32.GetParent.restype = wintypes.HWND
                user32.MonitorFromWindow.restype = wintypes.HANDLE
                user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
                user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
                hwnd = user32.GetParent(self.winfo_id()) or self.winfo_id()

                class MonitorInfo(ctypes.Structure):
                    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]
                mi = MonitorInfo(cbSize=ctypes.sizeof(MonitorInfo))
                if user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 2), ctypes.byref(mi)):
                    r = mi.rcWork
                    return r.left, r.top, r.right - r.left, r.bottom - r.top
            except Exception:
                pass
        return 0, 0, self.winfo_screenwidth(), self.winfo_screenheight()

    # CTk truncates with int(); a fractional scale then loses a pixel (652.8 -> 652 reads back as 767)
    def _apply_window_scaling(self, value):
        return round(value * ctk.ScalingTracker.get_window_scaling(self))

    def _reverse_window_scaling(self, scaled_value):
        return round(scaled_value / ctk.ScalingTracker.get_window_scaling(self))

    @staticmethod
    def _pick_scale(avail_w, avail_h, pin="auto"):
        """UI scale for a work area in logical px: a pinned number (setting ui_scale), else
        scaled down (0.05 steps) until the window fits, or up (0.25 steps, max 2) on large displays."""
        try:
            return max(0.5, min(3.0, float(pin)))
        except (TypeError, ValueError):
            pass
        f = min(1.0, avail_w / WIN_W, avail_h / (WIN_H + TITLEBAR))
        if f < 1.0:
            return math.floor(f / 0.05 + 1e-9) * 0.05
        ups = [u for u in (1.25, 1.5, 1.75, 2.0)
               if WIN_W * u <= 0.85 * avail_w and (WIN_H + TITLEBAR) * u <= 0.9 * avail_h]
        return ups[-1] if ups else 1.0

    def _place_window(self):
        """Fixed 1366x768 logical window, centred; drawn at the UI scale that suits the display."""
        self.update_idletasks()
        dpi = ctk.ScalingTracker.get_window_scaling(self) or 1.0
        x, y, pw, ph = self._work_area(dpi)
        f = self._pick_scale(pw / dpi, ph / dpi, self.app.settings.get("ui_scale", "auto"))
        if f != 1.0:
            # CTk pins the window to its tracked size when scaling changes: track the target size
            self._current_width, self._current_height = WIN_W, WIN_H
            ctk.set_widget_scaling(f)
            ctk.set_window_scaling(f)
            print(f"[ui] UI scale {f:.2f} for a {pw}x{ph} work area", file=sys.stderr)
        k = dpi * f                                   # logical -> physical pixels
        px = x + max(0, int((pw - WIN_W * k) / 2))
        py = y + max(0, int((ph - (WIN_H + TITLEBAR) * k) / 2))
        self.resizable(False, False)
        self.geometry(f"{WIN_W}x{WIN_H}+{px}+{py}")

    @staticmethod
    def _read_version():
        for d in (base_dir(), _ROOT):
            try:
                with open(os.path.join(d, "version.txt"), encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_rail(self):
        rail = ctk.CTkFrame(self, width=220, fg_color=C["panel"], corner_radius=0)
        rail.grid(row=0, column=0, sticky="nsew")
        rail.grid_propagate(False)
        rail.columnconfigure(0, weight=1)
        rail.rowconfigure(2, weight=1)
        ctk.CTkFrame(self, width=1, fg_color=C["rule"], corner_radius=0).grid(
            row=0, column=0, sticky="nse")
        ctk.CTkLabel(rail, text="WhisperX", font=font("heading", 17), text_color=C["ink"],
                     anchor="w").grid(row=0, column=0, sticky="w", padx=20, pady=(20, 22))
        nav = ctk.CTkFrame(rail, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", padx=10)
        self._nav = {}
        for name in SCREENS + ("log",):
            if name == "log":
                ctk.CTkFrame(nav, height=14, fg_color="transparent").pack()
            item = NavItem(nav, TITLES[name], lambda n=name: self.show(n))
            item.pack(fill="x", pady=1)
            self._nav[name] = item
        self._build_support(rail)
        return rail

    # ── Support: star the repo or donate, from the foot of the rail ──────────
    def _build_support(self, rail):
        self._support_open, self._support_anim = False, None
        menu = ctk.CTkFrame(rail, fg_color=C["raised"], corner_radius=10, border_width=1,
                            border_color=C["rule"], height=1)
        menu.pack_propagate(False)
        self._support_rows = []
        for i, (icon, label, url) in enumerate(SUPPORT_LINKS):
            row = SupportRow(menu, icon, label, lambda u=url: self._open_link(u))
            row.pack(fill="x", padx=SUPPORT_PAD,
                     pady=(SUPPORT_PAD if i == 0 else 0, SUPPORT_PAD if i == len(SUPPORT_LINKS) - 1 else 0))
            self._support_rows.append(row)
        self._support_menu = menu
        btn = ctk.CTkFrame(rail, fg_color="transparent", corner_radius=8, height=38)
        btn.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 14))
        btn.pack_propagate(False)
        heart = ctk.CTkLabel(btn, text="♥", font=font("button"), text_color=C["lane-magenta"], cursor="hand2")
        heart.pack(side="left", padx=(12, 0))
        label = ctk.CTkLabel(btn, text="Support", font=font("button"), text_color=C["ink-2"],
                             anchor="w", cursor="hand2")
        label.pack(side="left", padx=(8, 0))
        for w in (btn, heart, label):
            w.bind("<Button-1>", lambda e: self._toggle_support())
            w.bind("<Enter>", lambda e: label.configure(text_color=C["ink"]))
            w.bind("<Leave>", lambda e: label.configure(
                text_color=C["ink"] if self._support_open else C["ink-2"]))
        self._support_btn, self._support_label = btn, label

    def _toggle_support(self):
        """Open or close the menu; it grows up from the Support row over ~120 ms."""
        self._support_open = not self._support_open
        self._support_btn.configure(fg_color=C["raised"] if self._support_open else "transparent")
        self._support_label.configure(text_color=C["ink"] if self._support_open else C["ink-2"])
        full = SUPPORT_ROW_H * len(SUPPORT_LINKS) + 2 * SUPPORT_PAD
        menu = self._support_menu
        if self._support_open:
            menu.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 2))
        if self._support_anim:
            self.after_cancel(self._support_anim)
        start = max(1, menu.cget("height"))
        end = full if self._support_open else 1

        def step(i, n=7):
            u = 1 - (1 - i / n) ** 3
            menu.configure(height=round(start + (end - start) * u))
            if i < n:
                self._support_anim = self.after(16, step, i + 1)
                return
            self._support_anim = None
            if not self._support_open:
                menu.grid_remove()
        step(1)

    def _open_link(self, url):
        if self._support_open:
            self._toggle_support()
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    def _escape(self, _event):
        """Esc closes the Support menu first; with it closed, Esc cancels the run as before."""
        if self._support_open:
            self._toggle_support()
            return "break"
        self.app.cancel()

    def _click_away(self, event):
        if not self._support_open:
            return
        w = str(event.widget)
        for inside in (self._support_menu, self._support_btn):
            p = str(inside)
            if w == p or w.startswith(p + "."):
                return
        self._toggle_support()

    def _build_strip(self, canvas):
        strip = ctk.CTkFrame(canvas, height=56, fg_color="transparent", corner_radius=0)
        strip.grid(row=0, column=0, sticky="ew")
        strip.pack_propagate(False)
        ctk.CTkFrame(canvas, height=1, fg_color=C["rule"], corner_radius=0).grid(
            row=0, column=0, sticky="sew")
        self._title = ctk.CTkLabel(strip, text="", font=font("title"), text_color=C["ink"])
        self._title.pack(side="left", padx=(28, 0))
        self._save = ctk.CTkFrame(strip, fg_color="transparent")     # Transcribe only
        self._out_btn = ctk.CTkButton(
            self._save, text="", height=34, corner_radius=8, border_width=1, border_color=C["rule"],
            fg_color=C["raised"], hover_color=C["rule"], text_color=C["ink"],
            font=font("mono-sm"), command=self._pick_out_dir)
        self._out_btn.pack(side="right", padx=(10, 28))
        ctk.CTkLabel(self._save, text="SAVE TO", font=font("label"), text_color=C["ink-3"]).pack(
            side="right")

    def _build_log(self):
        box = ctk.CTkFrame(self._host, fg_color=C["ground"], corner_radius=0)   # opaque: screens stack
        log = ctk.CTkTextbox(box, fg_color=C["well"], text_color=C["ink-2"],
                             font=font("mono-sm"), corner_radius=8, border_width=1,
                             border_color=C["rule-soft"], wrap="word")
        log.pack(fill="both", expand=True, padx=28, pady=22)
        log.configure(state="disabled")
        self._log, self._frames["log"] = log, box

    def _build_transport(self, canvas):
        bar = ctk.CTkFrame(canvas, height=60, fg_color=C["panel"], corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.pack_propagate(False)
        ctk.CTkFrame(canvas, height=1, fg_color=C["rule"], corner_radius=0).grid(
            row=2, column=0, sticky="new")
        self._status_lbl = ctk.CTkLabel(bar, text="Ready", font=font("mono-sm"),
                                        text_color=C["ink"], anchor="w")
        self._status_lbl.pack(side="left", padx=(28, 0))
        self._go = theme.pill(bar, "Transcribe", self._start, primary=True)
        self._go.pack(side="right", padx=(12, 28))
        self._cancel = theme.pill(bar, "Cancel", self.app.cancel,
                                  text_color_disabled=C["ink-3"])
        self._cancel.pack(side="right")

    def _start(self, _e=None):
        if self.current not in ("transcribe", "queue"):
            self.show("transcribe")
        self.app.emit("transcribe")

    def _drain(self):
        while True:
            try:
                fn = self._inbox.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                traceback.print_exc()
        self.after(30, self._drain)

    # ── Adding files from anywhere ────────────────────────────────────────────
    def _drag(self, on):
        """2 px heat border round the content area, and the Transcribe drop area tinted, while dragging."""
        if on and not self._edges:
            for kw in (dict(relx=0, rely=0, relwidth=1, height=2), dict(relx=0, rely=1, relwidth=1, height=2, anchor="sw"),
                       dict(relx=0, rely=0, relheight=1, width=2), dict(relx=1, rely=0, relheight=1, width=2, anchor="ne")):
                e = tk.Frame(self._host, bg=C["heat"], bd=0)
                e.place(**kw)
                self._edges.append(e)
        elif not on:
            for e in self._edges:
                e.destroy()
            self._edges = []
        self.app.emit("drag", on=on)

    def _add_paths(self, paths):
        if self.current not in ("transcribe", "queue"):
            self.show("transcribe")
        self.app.emit("drop", paths=list(paths))

    def _on_drop(self, event):
        self._drag(False)
        self._add_paths(self.tk.splitlist(event.data))
        return event.action

    def _clip_files(self):
        """Paths on the clipboard: copied files (Windows), else text lines naming existing paths."""
        paths = _hdrop_paths() if IS_WIN else []
        if not paths:
            try:
                lines = (ln.strip().strip("\"'") for ln in self.clipboard_get().splitlines())
            except tk.TclError:
                lines = ()
            paths = [p for p in lines if p and os.path.exists(p)]
        return paths

    def _paste(self, _e=None):
        if isinstance(self.focus_get(), (tk.Entry, tk.Text)):      # normal paste in Hotwords and the like
            return None
        paths = self._clip_files()
        if paths:
            self._add_paths(paths)
        return "break"

    # ── Screens ───────────────────────────────────────────────────────────────
    def _make_screen(self, name):
        msg = f"{TITLES[name]} — coming in this build"
        try:
            mod = importlib.import_module(f"ui.{name}")
        except ModuleNotFoundError as e:
            if e.name != f"ui.{name}":      # missing module is expected; a broken import is not
                traceback.print_exc()
                msg = f"{TITLES[name]} failed to load: {e}"
            mod = None
        build = getattr(mod, "build", None)
        if build is not None:
            try:
                return build(self._host, self.app)
            except Exception as e:
                traceback.print_exc()
                msg = f"{TITLES[name]} failed to load: {e}"
        box = ctk.CTkFrame(self._host, fg_color=C["ground"], corner_radius=0)
        card = theme.panel(box)
        card.place(relx=0.5, rely=0.4, anchor="center")
        ctk.CTkLabel(card, text=msg,
                     font=font("heading"), text_color=C["ink-2"]).pack(padx=40, pady=30)
        return box

    def _settings_tab(self, name, **_):
        self.show("settings")
        getattr(self._frames["settings"], "open_tab", lambda _n: None)(name)

    def show(self, name):
        if name not in TITLES:
            return
        if name not in self._frames:
            frame = self._make_screen(name)
            # a CTkScrollableFrame's outermost widget is its container, not the inner frame
            self._frames[name] = getattr(frame, "_parent_frame", frame)
        frame = self._frames[name]
        if not frame.winfo_manager():
            # every built screen stays gridded in the one cell; a switch only restacks (no geometry
            # pass, no CTk redraw). Screens are opaque, so the top one hides the rest (F3).
            frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()
        getattr(frame, "on_show", lambda: None)()
        self.current = name
        for n, item in self._nav.items():
            item.set_current(n == name)
        self._title.configure(text=TITLES[name])
        if name == "transcribe":
            self._save.pack(side="right")
        else:
            self._save.pack_forget()

    # ── State -> widgets ──────────────────────────────────────────────────────
    def _refresh_jobs(self):
        n = sum(1 for j in self.app.queue.jobs if j.status in ("waiting", "running"))
        self._nav["queue"].set_count(str(n) if n else "")
        self._cancel.configure(state="normal" if self.app.running else "disabled")
        job = self.app.running
        idle_text = self._status_lbl.cget("text")
        if job is not None and idle_text == "Ready":     # the sidebar and panel show the run; don't claim "Ready"
            self._status_lbl.configure(text="")
        elif job is None and not idle_text:
            self._status_lbl.configure(text="Ready")
        if job is None:
            self._pct = None
            self._nav["transcribe"].set_busy(None)
            if self._tb_err is None:        # a failure keeps its red until the timer or the next job
                self.taskbar.clear()
        else:
            self._tb_clear_error()
            self._show_progress(job)

    def _show_progress(self, job):
        pct = round(job.progress * 100)
        if pct != self._pct:
            self._pct = pct
            self._nav["transcribe"].set_busy(f"{pct} %")
            self.taskbar.progress(job.progress)

    def _tb_clear_error(self):
        if self._tb_err is not None:
            self.after_cancel(self._tb_err)
            self._tb_err = None

    def _tb_error_done(self):
        self._tb_err = None
        if self.app.running is None:
            self.taskbar.clear()

    def _refresh_settings(self):
        s = self.app.settings
        out = s.get("out_dir", "").strip()
        self._out_btn.configure(
            text="Next to each file" if not out else out if len(out) <= 38 else "\u2026" + out[-37:])

    def _pick_out_dir(self):
        path = filedialog.askdirectory(title="Save transcripts to")
        if path:
            self.app.set("out_dir", os.path.normpath(path))

    def _on_runner(self, job, ev, **_):
        t = ev.get("t")
        if t in ("log", "warn", "error"):
            prefix = {"log": "", "warn": "WARN  ", "error": "ERROR "}[t]
            name = os.path.basename(job.audio) if job is not None else ""
            self._log.configure(state="normal")
            self._log.insert("end", f"{time.strftime('%H:%M:%S')}  {name}  {prefix}{ev.get('m', '')}\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        elif t in ("progress", "done"):     # "done" sets 1.0 before the worker exits
            self._show_progress(job)

    def _on_finished(self, job, **_):
        if job.status == "failed":
            self.taskbar.error()
            self._tb_clear_error()
            self._tb_err = self.after(5000, self._tb_error_done)
        # done, failed and cancelled show in the panel and the file row (Say-It-Once)

    def _probe_devices(self):
        devices = available_devices()
        # screens show what "Auto" resolved to ("Auto · GPU"), so tell them the device list changed
        self.app.schedule(lambda: (setattr(self.app, "devices", devices), self.app.emit("settings", key="devices")))

    # ── Window ────────────────────────────────────────────────────────────────
    def _setup_icon(self):
        # Try a pre-built icon file first (present in packaged builds).
        candidates = []
        if getattr(sys, "frozen", False):
            candidates.append(os.path.join(os.path.dirname(sys.executable), "assets", "icon.ico"))
        candidates.append(os.path.join(_ROOT, "assets", "icon.ico"))
        for path in candidates:
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
            d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 6,
                                fill=(74, 158, 255, 255))
            fnt = None
            for p in ["C:/Windows/Fonts/arialbd.ttf",
                      "/System/Library/Fonts/Helvetica.ttc",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
                try:
                    fnt = ImageFont.truetype(p, int(size * 0.70))
                    break
                except Exception:
                    continue
            if fnt is None:
                fnt = ImageFont.load_default()
            bb = d.textbbox((0, 0), "W", font=fnt)
            x = (size - (bb[2] - bb[0])) // 2 - bb[0]
            y = (size - (bb[3] - bb[1])) // 2 - bb[1] - int(size * 0.04)
            d.text((x, y), "W", fill=(255, 255, 255, 255), font=fnt)
            if IS_WIN:
                ico_path = os.path.join(tempfile.gettempdir(), "whisperx.ico")
                img.save(ico_path, format="ICO", sizes=[(128, 128), (64, 64), (32, 32), (16, 16)])
                self.after(50, lambda: self.iconbitmap(ico_path))
            self._icon_img = ImageTk.PhotoImage(img)
            self.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def _on_close(self):
        self.app.shutdown()
        try:
            self.destroy()
        finally:
            os._exit(0)
