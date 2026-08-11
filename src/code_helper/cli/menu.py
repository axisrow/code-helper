"""Minimal arrow-key terminal menu (stdlib only — no Rich/Typer/curses).

A single-purpose replacement for a "pick one of N wrappers" prompt. The
raw-keypress reading is layered in two pieces:

- :func:`_translate` — a PURE key-sequence parser. Given the first byte and an
  injectable ``read_more`` (a non-blocking "is there another byte, and what is
  it" callable), it returns a translated key name. No TTY, no ``termios`` —
  fully unit-testable (see ``tests/test_keys.py``).
- :func:`_read_key_raw` — the thin ``termios``/``tty`` wrapper (Unix-only)
  that owns the real TTY and non-blocking peek via ``select``, and feeds
  :func:`_translate`.

:func:`select_from_menu` takes ``read_key`` as an injectable callable so tests
can drive the menu with a fake key sequence instead of a real terminal —
mirroring how :func:`code_helper.services.secrets.resolve_token` injects
``getpass_fn``.
"""

from __future__ import annotations

import getpass
import os
import shutil
import sys
from collections.abc import Callable, Sequence

__all__ = [
    "select_from_menu",
    "read_line",
    "MenuCancelled",
    "press_any_key",
    "Section",
    "MAX_DIGIT_ITEMS",
]

#: How many items can get a digit shortcut. There are only nine single-key
#: digits (``0`` is not used — it would read as "tenth"), so a longer menu
#: necessarily has unnumbered rows. Exported because a caller rendering its own
#: key hint must not promise more digits than this — see ``cli/tui.py``'s
#: ``_hint``, where claiming "1-21" on a 21-item menu was a real, shipped bug.
MAX_DIGIT_ITEMS = 9

#: How long to wait for a follow-up byte after ESC before treating it as a
#: lone Escape keypress. Real escape sequences arrive as one burst from the
#: terminal driver; a human pressing Esc alone never sends a second byte
#: within this window.
_ESC_TIMEOUT = 0.05

#: How long to wait for the paired byte of a CRLF/LFCR Enter. Much shorter
#: than :data:`_ESC_TIMEOUT`: the pair arrives in the SAME burst from the
#: terminal driver (microseconds apart), so there is nothing to wait 50ms
#: for — that wait would just add latency to every single Enter keypress.
_PAIR_TIMEOUT = 0.002

# CSI parameter bytes (0x30-0x3F) and intermediate bytes (0x20-0x2F) per
# ECMA-48; a CSI sequence ends at the first byte in the final-byte range
# (0x40-0x7E).
_CSI_PARAM_OR_INTERMEDIATE = range(0x20, 0x40)
_CSI_FINAL = range(0x40, 0x7F)

# Letter-form final byte -> key name, shared by both escape-sequence shapes
# that end in a single letter: CSI (`\x1b[A`) and SS3 (`\x1bOA`, sent instead
# of CSI in "application cursor mode").
_ARROW_FINAL = {"A": "UP", "B": "DOWN", "H": "HOME", "F": "END"}

# CSI `~`-form final byte: `\x1b[<params>~` — Home/End/PageUp/PageDown as
# sent by terminals that don't use the letter form above.
_TILDE_FINAL = {"1": "HOME", "4": "END", "5": "PAGE_UP", "6": "PAGE_DOWN"}


class MenuCancelled(Exception):
    """Raised when the user cancels the menu.

    ``hard`` distinguishes *why*: ``True`` for Ctrl-C (immediate, unconditional
    exit — the user wants OUT, not "back"); ``False`` for Esc/``q`` (the
    default "go back one level" gesture). Callers that only care about
    cancellation (e.g. the plain CLI's ``edit-token`` picker) can ignore the
    field entirely — it defaults to ``True`` so an un-inspected catch keeps
    today's "cancel = stop" behavior.
    """

    def __init__(self, hard: bool = True) -> None:
        super().__init__()
        self.hard = hard


class Section:
    """A non-selectable section header row inside ``items``.

    A ``Section`` is placed in the ``items`` sequence between regular
    ``(value, label)`` entries to render a bare header row (e.g. ``"claude"``
    or ``"codex"``) that groups the items below it. Unlike an ``unnumbered``
    entry (which is selectable but gets no digit), a ``Section`` is never
    selectable: the cursor never lands on it, Up/Down/Home/End navigation
    skips it, and it gets neither a digit nor a ``>`` marker. It DOES occupy
    one rendered row, so :func:`select_from_menu`'s ``frame_lines`` count
    accounts for it automatically (it counts ``len(pairs)``, and a
    ``Section`` is one entry there) — that is what keeps the in-place redraw
    honest when a section header is added.

    The menu only reads ``.text``; the object is a marker ("render a header
    here"), not a value that can be returned or compared, so it intentionally
    defines no ``__eq__``/``__hash__``.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        if "\n" in text or "\r" in text:
            raise ValueError("Section text must be a single line")
        self.text = text


def _translate(
    first: str,
    read_more: Callable[[float], str | None],
    push_back: Callable[[str], None] = lambda _b: None,
) -> str:
    """Translate one keypress starting with ``first`` into a key name.

    ``read_more(timeout)`` returns the next raw byte if one arrives within
    ``timeout`` seconds, else ``None`` — a non-blocking peek. ``push_back(b)``
    returns a byte this call read but did not consume, so the NEXT
    :func:`_read_key_raw` sees it instead of it being dropped on the floor.
    Together these are the only seams this function needs to be fully
    unit-testable without a real TTY: every escape-sequence edge case (lone
    Esc, CSI, SS3, stray trailing bytes) is a pure function of ``first`` plus
    a fake ``read_more``/``push_back``.

    ``push_back`` defaults to a no-op discard, which keeps every caller that
    only cares about key *names* (and every pre-existing test) working
    unchanged — pushback only matters to a caller reading a continuous byte
    stream, i.e. :func:`_read_key_raw`.

    Returns one of: ``"UP"``, ``"DOWN"``, ``"HOME"``, ``"END"``,
    ``"PAGE_UP"``, ``"PAGE_DOWN"``, ``"ENTER"``, ``"CANCEL"`` (Esc/``q`` — a
    soft "go back" cancel), ``"HARD_CANCEL"`` (Ctrl-C — raw ``\\x03``; in raw
    terminal mode ``ISIG`` is off, so this never raises ``KeyboardInterrupt``
    on its own and MUST be handled as a distinct key, not folded into
    ``CANCEL``), ``"TAB"`` (the Tab key — a caller-side hook key, see
    :func:`select_from_menu`'s ``on_tab``; without one it is ignored just like
    ``"OTHER"``), ``"TOKEN"`` (the ``t``/``T`` key — a caller-side hook key,
    see :func:`select_from_menu`'s ``on_token``; without one it is ignored
    just like ``"OTHER"``), ``"DIGIT_1"``..``"DIGIT_9"``, ``"OTHER"``.
    """
    if first == "\x1b":
        nxt = read_more(_ESC_TIMEOUT)
        if nxt is None:
            return "CANCEL"  # lone Esc — no follow-up byte arrived in time
        if nxt == "O":  # SS3 — exactly one final byte follows
            final = read_more(_ESC_TIMEOUT) or ""
            return _ARROW_FINAL.get(final, "OTHER")
        if nxt == "[":  # CSI — consume params/intermediates up to the final byte
            params = ""
            while True:
                b = read_more(_ESC_TIMEOUT)
                if b is None:
                    return "OTHER"  # sequence cut off — nothing sane to do
                code = ord(b)
                if code in _CSI_FINAL:
                    final = b
                    break
                if code in _CSI_PARAM_OR_INTERMEDIATE:
                    params += b
                    continue
                return "OTHER"  # not a well-formed CSI sequence
            if final in _ARROW_FINAL:
                return _ARROW_FINAL[final]
            if final == "~":
                return _TILDE_FINAL.get(params, "OTHER")
            return "OTHER"
        return "OTHER"

    if first in ("\r", "\n"):
        # CRLF/LFCR: swallow ONLY the actual paired byte, so it doesn't fire a
        # second ENTER on the next read. Anything else that shows up is a
        # genuine next keypress the user made within the window (e.g. Enter
        # then immediately Down) — it is pushed back rather than dropped, or
        # the navigation move would silently vanish.
        pair = "\n" if first == "\r" else "\r"
        nxt = read_more(_PAIR_TIMEOUT)
        if nxt is not None and nxt != pair:
            push_back(nxt)
        return "ENTER"
    if first == "\x03":
        return "HARD_CANCEL"
    if first == "\t":
        return "TAB"
    if first in ("t", "T"):
        return "TOKEN"
    if first in ("q", "Q"):
        return "CANCEL"
    if first in ("k", "K"):
        return "UP"
    if first in ("j", "J"):
        return "DOWN"
    if first == "g":
        return "HOME"
    if first == "G":
        return "END"
    if first in "123456789":
        return f"DIGIT_{first}"
    return "OTHER"


#: One-byte pushback slot, spanning :func:`_read_key_raw` calls.
#:
#: :func:`_translate` peeks one byte past a keypress to detect the LF half of
#: a CRLF Enter. When that byte turns out to be a real next keypress instead,
#: it cannot be un-read from the fd — so it is parked here and the NEXT call
#: consumes it before touching the fd again. Module-level (not a local) for
#: exactly that reason: the byte has to outlive the call that read it.
#:
#: Single-byte is sufficient because only the Enter branch ever pushes back,
#: and it pushes at most one byte per keypress — which the next call drains
#: before it can read (and therefore push back) anything else.
_pending_byte: str | None = None


def _read_key_raw(stream=sys.stdin) -> str:
    """Read one raw keypress from a real TTY, fully parsed via :func:`_translate`.

    Reads via ``os.read(fd, 1)``, NOT ``stream.read(1)``. ``stream`` is a
    buffered ``TextIOWrapper``: a single ``.read(1)`` call is free to pull
    more than one byte off the underlying fd into its own internal buffer
    before decoding and returning the first character. ``select.select``
    only ever sees the fd itself — it has no visibility into that internal
    buffer, so a byte already sitting in the wrapper's buffer never shows up
    as "ready" again. Reading raw bytes straight from the fd sidesteps this:
    every byte ``select`` reports ready is read immediately, never parked in
    a layer ``select`` can't see.

    A byte :func:`_translate` peeked but did not consume is parked in
    :data:`_pending_byte` and drained by the next call — see there.

    .. note::
       Bytes are decoded ONE AT A TIME with ``errors="replace"``, so a
       multi-byte UTF-8 character (e.g. a Cyrillic letter) decodes to U+FFFD
       per byte and translates to several ``"OTHER"`` keys. That is harmless
       here — the menu only acts on ASCII keys and ignores ``"OTHER"`` — but
       this function is NOT usable as-is for reading text; that would need
       accumulating continuation bytes into a full character first.
    """
    global _pending_byte

    import select
    import termios
    import tty

    fd = stream.fileno()
    old = termios.tcgetattr(fd)

    def _read_more(timeout: float) -> str | None:
        global _pending_byte
        if _pending_byte is not None:
            b, _pending_byte = _pending_byte, None
            return b
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        return os.read(fd, 1).decode("utf-8", errors="replace")

    def _push_back(b: str) -> None:
        global _pending_byte
        _pending_byte = b

    try:
        tty.setraw(fd)
        if _pending_byte is not None:
            first, _pending_byte = _pending_byte, None
        else:
            first = os.read(fd, 1).decode("utf-8", errors="replace")
        return _translate(first, _read_more, _push_back)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _normalize(
    items: Sequence[str | tuple[str, str | Callable[[], str]] | Section],
) -> list[tuple[str, str | Callable[[], str]] | Section]:
    """Normalize ``items`` to ``(value, label)`` pairs (``Section`` pass-through).

    A plain ``str`` item is both its own value and label — this is what keeps
    every pre-existing ``select_from_menu(["a", "b"])`` call (and all of
    ``tests/test_menu.py``) working unchanged. A ``Section`` is passed through
    untouched: it is not an item, so wrapping it in a pair would be wrong.
    """
    return [
        item if isinstance(item, (tuple, Section)) else (item, item) for item in items
    ]


def _fit(line: str, width: int) -> str:
    """Truncate ``line`` to ``width`` columns, marking the cut with ``…``.

    A line longer than the terminal is wrapped by the terminal driver into
    two or more PHYSICAL rows, which the in-place redraw's ``frame_lines``
    counter (a count of LOGICAL rows) does not know about — the cursor-up
    escape then moves the wrong number of rows and the menu visibly creeps
    down the screen. Truncating every printed row to the terminal's actual
    width is what keeps one call to ``print_fn`` equal to one physical row —
    a real bug, not a cosmetic one; see ``menu.py`` module docs / CLAUDE.md.
    """
    if width <= 0 or len(line) <= width:
        return line
    if width == 1:
        return "…"
    return line[: width - 1] + "…"


class _MenuState:
    """Mutable state of a ``select_from_menu`` session, separated from its IO.

    Pure construction (no IO) lives in :func:`_build_menu_state`; the redraw
    loop in :func:`select_from_menu` reads and mutates ``index`` / ``first``
    while delegating every frame's rendering to :func:`_render_frame` and
    every keypress's effect to :func:`_dispatch_key`. Splitting the state out
    is what lets the 200-line monolithic body become three readable parts.

    Kept as a plain class (not ``@dataclass``) because ``index`` and ``first``
    are mutated in place by the dispatch return path — a frozen dataclass
    would force rebuilding the whole state on every keypress.
    """

    __slots__ = (
        "pairs",
        "selectable",
        "digit_of",
        "by_digit",
        "index",
        "first",
        "redraw",
        "width",
        "hint_text",
        "frame_lines",
    )

    def __init__(
        self,
        *,
        pairs,
        selectable,
        digit_of,
        by_digit,
        redraw,
        width,
        hint_text,
        frame_lines,
    ) -> None:
        self.pairs = pairs
        self.selectable = selectable
        self.digit_of = digit_of
        self.by_digit = by_digit
        self.redraw = redraw
        self.width = width
        self.hint_text = hint_text
        self.frame_lines = frame_lines
        self.index = 0
        self.first = True


def _build_menu_state(
    items: Sequence[str | tuple[str, str | Callable[[], str]] | Section],
    *,
    hint: str | None | Callable[[], str],
    unnumbered: frozenset[str],
) -> _MenuState:
    """Construct a :class:`_MenuState` from raw ``items`` — pure, no IO.

    Owns the three invariants the redraw loop relies on: which entries are
    selectable (skipping :class:`Section` headers), the value→digit and
    digit→value mappings (kept as one derivation so the rendered digit and
    the dispatch digit can never disagree), and the frame-line count that
    drives the in-place redraw's cursor-up escape.
    """
    pairs = _normalize(items)

    selectable = [i for i, entry in enumerate(pairs) if not isinstance(entry, Section)]
    if not selectable:
        raise ValueError("items must be non-empty")

    digit_of: dict[str, int] = {}
    for entry in pairs:
        if isinstance(entry, Section):
            continue
        value = entry[0]
        if value not in unnumbered and len(digit_of) < MAX_DIGIT_ITEMS:
            digit_of[value] = len(digit_of) + 1
    by_digit = {n: value for value, n in digit_of.items()}

    redraw = sys.stdout.isatty()
    width = shutil.get_terminal_size((80, 24)).columns if redraw else 0
    hint_text = hint() if callable(hint) else hint
    frame_lines = len(pairs) + 2 + (2 if hint_text else 0)

    return _MenuState(
        pairs=pairs,
        selectable=selectable,
        digit_of=digit_of,
        by_digit=by_digit,
        redraw=redraw,
        width=width,
        hint_text=hint_text,
        frame_lines=frame_lines,
    )


def _render_frame(
    state: _MenuState,
    *,
    prompt: str | Callable[[], str],
    clear: bool,
    clear_seq: str,
    clear_screen: str,
    print_fn: Callable[[str], None],
) -> None:
    """Render one menu frame to ``print_fn`` — pure output, no key reading.

    Evaluates a callable ``prompt`` fresh each frame so a header that
    reflects state an ``on_tab`` handler mutated stays current. Every
    rendered row is truncated to the terminal width via :func:`_fit` — a row
    longer than the terminal wraps into extra physical rows the in-place
    redraw does not know about, and the menu creeps down the screen.
    """
    width = state.width
    redraw = state.redraw
    prompt_text = prompt() if callable(prompt) else prompt
    heading = _fit(prompt_text, width) if redraw else prompt_text
    if redraw:
        lead = clear_seq if not state.first else (clear_screen if clear else "")
        print_fn(f"{lead}{heading}\n")
    else:
        print_fn(f"\n{heading}")
    cursor_pair = state.selectable[state.index]
    for i, entry in enumerate(state.pairs):
        if isinstance(entry, Section):
            header = entry.text
            print_fn(f" {_fit(header, width - 1) if redraw else header}")
            continue
        value, label = entry
        label_text = label() if callable(label) else label
        marker = ">" if i == cursor_pair else " "
        digit = str(state.digit_of[value]) if value in state.digit_of else "·"
        row = f"{digit} {marker} {label_text}"
        print_fn(f" {_fit(row, width - 1) if redraw else row}")
    if state.hint_text:
        fitted_hint = _fit(state.hint_text, width) if redraw else state.hint_text
        print_fn(f"\n {fitted_hint}" if redraw else f" {fitted_hint}")


def _dispatch_key(
    key: str,
    state: _MenuState,
    *,
    on_tab: Callable[[], None] | None,
    on_token: Callable[[str], None] | None,
):
    """Act on one translated ``key`` — returns a sentinel or a selected value.

    Returns:
        The selected value when the key selects one (Enter / a digit);
        ``None`` when the key only mutated state (navigation / on_tab /
        on_token) and the loop should redraw; never returns for CANCEL /
        HARD_CANCEL (raises :class:`MenuCancelled`).
    """
    if key == "TAB" and on_tab is not None:
        on_tab()
        return None
    if key == "TOKEN" and on_token is not None:
        on_token(state.pairs[state.selectable[state.index]][0])
        return None
    if key == "UP":
        state.index = (state.index - 1) % len(state.selectable)
        return None
    if key == "DOWN":
        state.index = (state.index + 1) % len(state.selectable)
        return None
    if key in ("HOME", "PAGE_UP"):
        state.index = 0
        return None
    if key in ("END", "PAGE_DOWN"):
        state.index = len(state.selectable) - 1
        return None
    if key == "ENTER":
        return state.pairs[state.selectable[state.index]][0]
    if key.startswith("DIGIT_"):
        n = int(key[len("DIGIT_") :])
        if n in state.by_digit:
            return state.by_digit[n]
        return None
    if key == "CANCEL":
        raise MenuCancelled(hard=False)
    if key == "HARD_CANCEL":
        raise MenuCancelled(hard=True)
    return None  # OTHER / TAB-without-on_tab / TOKEN-without-on_token


def select_from_menu(
    items: Sequence[str | tuple[str, str | Callable[[], str]] | Section],
    *,
    prompt: str | Callable[[], str] = "select:",
    hint: str | None | Callable[[], str] = None,
    on_tab: Callable[[], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    unnumbered: frozenset[str] = frozenset(),
    read_key: Callable[[], str] = _read_key_raw,
    print_fn: Callable[[str], None] = print,
    clear: bool = False,
) -> str:
    """Interactively select one of ``items`` with Up/Down + Enter.

    Args:
        items: The choices to present, in order. Must be non-empty. Each
            entry is either a plain ``str`` (value == label) or a
            ``(value, label)`` pair — the label is rendered, the value is
            returned and is what callers compare against. This lets a menu
            show a human description without control flow depending on
            display text (formerly a ``str.startswith`` on the rendered
            label — see ``cli/tui.py`` history). The ``label`` in a pair may
            itself be a callable — it is evaluated FRESH each frame, exactly
            like a callable ``prompt``, so a row that must reflect state an
            ``on_tab`` handler mutated (e.g. the active-profile row) tracks
            the change instead of freezing at the value it had when the menu
            opened. A callable label must return a single logical line with
            no ``\\n`` (same ``frame_lines`` constraint as ``prompt``).
            An entry may also be a :class:`Section` — a non-selectable
            header row that groups the items below it (e.g. ``Section("claude")``
            above the claude wrappers). A ``Section`` is rendered as a bare
            label with no digit and no ``>`` marker, is skipped by
            Up/Down/Home/End/Page navigation, and cannot be selected by Enter.
            It DOES occupy one rendered row, so ``frame_lines`` accounts for
            it — the in-place redraw stays honest.
        prompt: Heading line printed above the list. May be a callable
            evaluated FRESH each frame — a menu whose header must reflect state
            an ``on_tab`` handler mutated (e.g. the active profile) can pass
            ``lambda: _header()`` and the header tracks the change. A callable
            must return a single logical line with no ``\\n`` — the frame's line
            count must not depend on it (see ``frame_lines`` below).
        hint: Optional key-hint line printed BELOW the list (e.g. "↑/↓ ·
            Enter · Esc back"). Adds two lines to the erasable frame — see
            ``frame_lines`` below. May be a callable; it is evaluated ONCE when
            the menu starts (before ``frame_lines`` is computed), not per frame.
        on_tab: Optional handler invoked on a ``"TAB"`` keypress, after which
            the menu simply redraws. The handler mutates external state; the
            frame is NOT rebuilt from it (only a callable ``prompt`` re-evaluates
            per frame). Without ``on_tab``, Tab is silently ignored — the key
            resolves to ``"TAB"`` but the loop treats it like ``"OTHER"``, so it
            never starts doing anything in a menu that did not opt in.
        on_token: Optional handler invoked on a ``"TOKEN"`` keypress (the
            ``t``/``T`` key), UNLIKE ``on_tab`` receiving the VALUE under the
            cursor as its single argument — a token action is per-row (rotate
            the token of THIS wrapper), so the handler must know which row it
            acts on, while Tab cycles a single shared resource (the active
            profile) and needs no row identity. Without ``on_token``, ``t`` is
            silently ignored, exactly like Tab without ``on_tab``.
        unnumbered: Values that should NOT get a digit shortcut (e.g. a
            trailing "← назад"/"quit" entry). Digits are assigned by the
            menu itself — the same place that handles ``DIGIT_<n>`` below —
            so the number shown next to an item and the number that selects
            it can never drift apart (the failure mode a caller-side
            ``str.startswith`` on the label used to cause — see
            ``cli/tui.py`` history).
        read_key: Source of translated keypresses. Defaults to reading a real
            TTY; tests inject a fake sequence instead.
        print_fn: Injectable output sink for the rendered menu.
        clear: On a TTY, clear the *whole* screen before the first frame
            instead of leaving prior content above it. Ignored when
            ``stdout`` is not a TTY.

    On a real terminal the menu is redrawn *in place* on every keypress; the
    cursor is hidden for the duration (restored in a ``finally``, so it comes
    back even if the caller raises out of ``read_key``). ``HOME``/``END``/
    ``PAGE_UP``/``PAGE_DOWN`` jump to the first/last item (there is nothing to
    page through in a short menu); ``DIGIT_<n>`` selects the n-th NUMBERED
    item immediately if it exists, else is ignored. Every printed row is
    truncated to the real terminal width (see :func:`_fit`) — a row longer
    than the terminal wraps into extra physical rows the in-place redraw
    doesn't know about, and the menu creeps down the screen.

    Returns:
        The selected item's value.

    Raises:
        ValueError: ``items`` is empty.
        MenuCancelled: the user cancelled — ``hard=True`` for Ctrl-C,
            ``hard=False`` for Esc/``q``.
    """
    state = _build_menu_state(items, hint=hint, unnumbered=unnumbered)

    clear_seq = f"\033[{state.frame_lines}A\033[J"
    clear_screen = "\033[2J\033[H"
    hide_cursor = "\033[?25l"
    show_cursor = "\033[?25h"
    try:
        if state.redraw:
            print_fn(hide_cursor)
        while True:
            _render_frame(
                state,
                prompt=prompt,
                clear=clear,
                clear_seq=clear_seq,
                clear_screen=clear_screen,
                print_fn=print_fn,
            )
            state.first = False

            selected = _dispatch_key(
                read_key(), state, on_tab=on_tab, on_token=on_token
            )
            if selected is not None:
                return selected
    except KeyboardInterrupt:
        # Belt-and-suspenders: a real terminal in raw mode has ISIG off, so
        # Ctrl-C arrives as the "\x03" byte above (-> HARD_CANCEL), never as
        # this exception. This branch exists for injected `read_key` fakes
        # (tests, or any future non-TTY caller) that raise KeyboardInterrupt
        # directly instead of returning a translated key name.
        raise MenuCancelled(hard=True) from None
    finally:
        if state.redraw:
            print_fn(show_cursor)


def _read_line_raw(
    prompt: str,
    *,
    secret: bool,
    stream,
    output,
) -> str:
    """Read one editable line while preserving the TUI cancellation contract."""
    import select
    import termios
    import tty

    fd = stream.fileno()
    output.write(prompt)
    output.flush()
    old = termios.tcgetattr(fd)
    chars: list[str] = []
    invalid_secret_char = False

    def read_more(timeout: float | None = None) -> str | None:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
        return os.read(fd, 1).decode("utf-8", errors="replace")

    try:
        tty.setraw(fd)
        while True:
            first = read_more()
            if first is None:
                continue
            if first == "\x03":
                raise MenuCancelled(hard=True)
            if first == "\x1b":
                # A lone Escape is Back. Consume a cursor/function-key
                # sequence instead of accidentally treating its bytes as text.
                nxt = read_more(_ESC_TIMEOUT)
                if nxt is None:
                    raise MenuCancelled(hard=False)
                if nxt in ("[", "O"):
                    while True:
                        tail = read_more(_ESC_TIMEOUT)
                        if tail is None or ord(tail) in _CSI_FINAL:
                            break
                continue
            if first in ("\r", "\n"):
                output.write("\n")
                output.flush()
                if secret and invalid_secret_char:
                    raise ValueError("secret input must contain ASCII characters")
                return "".join(chars).strip()
            if first in ("\x08", "\x7f"):
                if chars:
                    chars.pop()
                    output.write("\b \b")
                    output.flush()
                continue
            if secret and (first == "\ufffd" or ord(first) > 127):
                invalid_secret_char = True
                continue
            chars.append(first)
            output.write("•" if secret else first)
            output.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_line(
    prompt: str = "",
    *,
    secret: bool = False,
    stream=None,
    output=None,
) -> str:
    """Read text with uniform TUI semantics: Esc goes back, Ctrl-C quits.

    Non-TTY callers retain the injectable ``input``/``getpass`` behavior used
    by the CLI and tests. Real TTY callers use the raw reader so text fields
    obey exactly the same cancellation rules as selection menus.
    """
    stream = sys.stdin if stream is None else stream
    output = sys.stdout if output is None else output
    if not stream.isatty() or not output.isatty():
        try:
            value = getpass.getpass(prompt) if secret else input(prompt)
        except KeyboardInterrupt:
            raise MenuCancelled(hard=True) from None
        value = value.strip()
        if secret and not value.isascii():
            raise ValueError("secret input must contain ASCII characters")
        return value
    return _read_line_raw(prompt, secret=secret, stream=stream, output=output)


def press_any_key(prompt: str = "") -> None:
    """Block for one keypress on a real TTY; a no-op off a TTY.

    If ``prompt`` is given it is printed without a trailing newline before the
    read, and a newline is printed after, so a caller can show a "press any key
    to continue" hint on the same line as the cursor. When ``stdout`` is not a
    TTY (piped output, tests) it returns immediately so non-interactive runs
    never block waiting on stdin.

    Reads (and fully consumes) one key via :func:`_read_key_raw` — any key at
    all continues, including one that happens to start an escape sequence
    (arrow, function key, ...): the sequence is parsed to completion rather
    than treated as several separate "any key" presses.
    """
    if not sys.stdout.isatty():
        return
    if prompt:
        print(prompt, end="", flush=True)
    try:
        _read_key_raw()
    except KeyboardInterrupt:
        pass
    if prompt:
        print()
