"""Keyboard-layout aliases for the TUI's ASCII command keys."""

from __future__ import annotations

# Only keys with menu meanings live here.  Case is intentionally explicit:
# g/G are different navigation commands.
_LAYOUTS = {
    "ru": {"a": "ф", "e": "у", "d": "в", "t": "е", "c": "с", "s": "ы", "k": "л", "q": "й", "j": "о", "g": "п", "G": "П"},
    "uk": {"a": "ф", "e": "у", "d": "в", "t": "е", "c": "с", "s": "і", "k": "л", "q": "й", "j": "о", "g": "п", "G": "П"},
    "th": {"a": "ฟ", "e": "ำ", "d": "ก", "t": "ะ", "c": "แ", "s": "ห", "k": "า", "q": "ๆ", "j": "่", "g": "เ", "G": "ฺ"},
}

_KEYS: dict[str, str] = {}
for _layout in _LAYOUTS.values():
    for _ascii, _local in _layout.items():
        if _local in _KEYS and _KEYS[_local] != _ascii:
            raise ValueError(f"keyboard layout collision for {_local!r}")
        _KEYS[_local] = _ascii


def translate_key(key: str) -> str:
    """Return the corresponding command key, or ``key`` unchanged."""
    return _KEYS.get(key, key)
