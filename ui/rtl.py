"""Right-to-left display: Tk 8.6 on Windows does no bidi reordering, so RTL text is put in visual order for display only."""
import unicodedata
from functools import lru_cache


def is_rtl(text: str) -> bool:
    """True when the first strong-direction character is right-to-left (R or AL)."""
    for c in text:
        d = unicodedata.bidirectional(c)
        if d in ("L", "R", "AL"):
            return d != "L"
    return False


@lru_cache(maxsize=8192)
def display(text: str) -> str:
    """Visual-order string for one display line; identity for LTR text or when the bidi packages are missing."""
    if not is_rtl(text):
        return text
    try:
        from bidi.algorithm import get_display
        import arabic_reshaper
    except ImportError:
        return text
    if any(unicodedata.name(c, "").startswith("ARABIC") for c in text):
        text = arabic_reshaper.reshape(text)
    return get_display(text)


def order(words: list[str]) -> list[int]:
    """Indices of an RTL line's words in visual order: run order reversed, LTR runs (Latin, digits) keep their order."""
    def ltr(w):
        return not is_rtl(w) and any(c.isdigit() or unicodedata.bidirectional(c) == "L" for c in w)

    def strong(w):
        return is_rtl(w) or ltr(w)
    keys = [w for w in words if strong(w)]
    dirs, k = [], 0         # a neutral word is LTR only between two LTR words
    for w in words:
        if strong(w):
            dirs.append(ltr(w))
            k += 1
        else:
            dirs.append(0 < k < len(keys) and ltr(keys[k - 1]) and ltr(keys[k]))
    out = list(range(len(words)))[::-1]
    i = 0
    while i < len(out):
        e = i
        while e < len(out) and dirs[out[e]]:
            e += 1
        out[i:e] = out[i:e][::-1]
        i = e + 1
    return out


def wrap(text: str, width: int, measure) -> list[str]:
    """Split logical-order text at spaces into lines whose displayed width fits width px, in reading order.
    A word wider than width is its own line. Returns logical-order lines; callers display() each."""
    lines = []
    for w in text.split(" "):
        if lines and measure(display(lines[-1] + " " + w)) <= width:
            lines[-1] += " " + w
        else:
            lines.append(w)
    return lines
