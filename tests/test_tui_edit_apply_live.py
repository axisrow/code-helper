"""The edit screen's apply must run the handler LIVE, never silently.

An edit that changes the model set re-enters the ctx resolver inside
``wrappers.edit_wrapper``; an unknown-and-unrecorded model opens its
interactive menu there. Under ``silent=True`` the menu sees a captured
non-TTY stdout and refuses (the #132 guard) — the reported "apply raises
interactive menu needs a TTY on stdout" bug. ``live=True`` keeps a ``_Tee``
whose ``isatty`` answers for the real terminal, exactly the add-path
precedent.
"""

import pytest
from test_tui import _tui_args

from codehelper.services.wrappers import UNSET


@pytest.mark.unit
def test_edit_apply_dispatches_live(monkeypatch):
    """``_apply_edit`` passes live=True (and never silent) — a prompting
    handler must render its menu on the real terminal."""
    from codehelper.cli.tui import TuiSession

    session = TuiSession(_tui_args())
    seen = {}

    def fake_run(self, _handler, _req, *, live=False, silent=False):
        seen["live"] = live
        seen["silent"] = silent
        return True

    monkeypatch.setattr(TuiSession, "_run", fake_run)
    monkeypatch.setattr(TuiSession, "_edit_was_applied", lambda self, _alias: None)

    assert (
        session._apply_edit(
            "glm-axisrow",
            {
                "provider": UNSET,
                "auth": UNSET,
                "base_url": UNSET,
                "model": "glm-5.3-flash",
                "tiers": {},
                "subagent": None,
                "effort": None,
                "ctx": UNSET,
            },
        )
        is True
    )
    assert seen == {"live": True, "silent": False}
