import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _expected_display(text):
    from bidi.algorithm import get_display
    import arabic_reshaper

    return get_display(arabic_reshaper.reshape(text))


def test_display_reorders_arabic_script_and_leaves_ltr_text():
    from ui import rtl

    urdu = "یہ اردو جملہ ہے"
    assert rtl.display(urdu) == _expected_display(urdu)
    assert rtl.display("hello world") == "hello world"


def test_display_keeps_mixed_ltr_words_in_bidi_order():
    from ui import rtl

    mixed = "یہ hello world ہے"
    numeric = "سال 2024 میں"
    assert rtl.display(mixed) == _expected_display(mixed)
    assert rtl.display(numeric) == _expected_display(numeric)


def test_is_rtl_uses_first_strong_direction_character():
    from ui import rtl

    assert rtl.is_rtl("یہ اردو جملہ ہے")
    assert rtl.is_rtl("مرحبا بالعالم")
    assert rtl.is_rtl("שלום עולם")
    assert not rtl.is_rtl("hello")
    assert not rtl.is_rtl("2024 report")


def test_wrap_keeps_logical_order_and_fits_displayed_lines():
    from ui import rtl

    long_urdu = (
        "یہ اردو جملہ ہے اور دائیں سے بائیں لکھا جاتا ہے۔ "
        "یہ دوسرا اردو جملہ ہے اور اسے پڑھنے کی ترتیب برقرار رہنی چاہیے۔ "
        "یہ تیسرا جملہ بھی اسی متن کا حصہ ہے۔"
    )

    lines = rtl.wrap(long_urdu, 120, len)

    assert len(lines) > 1
    assert lines[0].startswith(long_urdu.split()[0])
    assert " ".join(lines) == long_urdu
    for line in lines:
        assert len(rtl.display(line)) <= 120 or " " not in line


def test_core_does_not_import_ui_rtl():
    root = Path(__file__).parent.parent
    paths = list((root / "core").glob("*.py")) + [root / "transcribe.py"]
    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if re.search(r"(^|\s)(from\s+ui\s+import\s+rtl|import\s+ui\.rtl|from\s+ui\.rtl\s+import)", text):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []
