"""Rounding at unit boundaries in human_bytes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import setup_wizard


def test_value_just_under_a_megabyte_rolls_up_to_mb():
    assert setup_wizard.human_bytes(999_960) == "1.0 MB"


def test_value_just_under_a_gigabyte_rolls_up_to_gb():
    assert setup_wizard.human_bytes(999_960_000) == "1.0 GB"


def test_values_inside_a_unit_are_unchanged():
    assert setup_wizard.human_bytes(512_300) == "512.3 kB"
