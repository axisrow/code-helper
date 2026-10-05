"""The edit screen's interactive pickers, pinned behaviorally.

These are the flows a mutant run showed as uncovered (the rows/labels tests
never ENTER a picker): each test drives one picker with a scripted
interactive source (``_pick`` / ``_choose_add_*`` / ``_read_text``) and pins
the DRAFT effect — the picker's whole job — plus its Back/empty-input
no-ops.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_tui import _tui_args


def _session():
    from codehelper.cli.tui import TuiSession

    return TuiSession(_tui_args())


def _install_claude_zai(alias: str = "glm-x", **spec_kwargs):
    from codehelper.services.paths import Paths
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper, spec_from_installed

    install_wrapper(
        Paths.default(),
        build_spec(agent="claude", provider="zai", model="glm-5.3", alias=alias),
        token="sk-x",
    )
    spec = spec_from_installed(Paths.default(), alias)
    assert spec is not None
    return spec


def _draft(**overrides):
    """The full draft dict exactly as ``_run_edit_screen`` builds it — every
    picker may read any axis, so tests must not hand it a partial one."""
    from codehelper.services.wrappers import UNSET

    draft: dict = {
        "provider": UNSET,
        "auth": UNSET,
        "base_url": UNSET,
        "model": UNSET,
        "tiers": UNSET,
        "subagent": UNSET,
        "effort": UNSET,
        "ctx": UNSET,
    }
    draft.update(overrides)
    return draft


@pytest.mark.integration
def test_ctx_pick_preset_records_explicit_window(monkeypatch):
    """A preset choice becomes the parsed draft value; '0' is the explicit
    suppression answer, not unset."""
    from codehelper.cli.tui import _BACK, TuiSession

    spec = _install_claude_zai()
    draft = _draft()
    prompts: list[str] = []
    wanted = ["1000000"]

    def _pick_record(self, items, prompt, **kw):
        prompts.append(prompt)
        return next((value for value, _label in items if value == wanted[0]), _BACK)

    monkeypatch.setattr(TuiSession, "_pick", _pick_record)
    _session()._pick_edit_ctx(spec, draft)
    assert draft["ctx"] == 1_000_000

    # The suppression preset rides the menu as ("0", "no declaration — …"):
    # the picker must translate it itself — routing menu data through the
    # --context-window parser made the menu's own first choice raise.
    wanted[0] = "0"
    _session()._pick_edit_ctx(spec, draft)
    assert draft["ctx"] == 0
    assert prompts == ["Context window:", "Context window:"]


@pytest.mark.integration
def test_effort_pick_records_the_level_and_clears(monkeypatch):
    """The menu content per shape is pinned elsewhere; this pins the EFFECT:
    a level rides the draft, __clear__ empties it, Back keeps it."""
    from codehelper.cli.tui import _BACK, TuiSession

    spec = _install_claude_zai()
    draft = _draft()
    session = _session()

    monkeypatch.setattr(TuiSession, "_pick", lambda self, items, prompt, **kw: "xhigh")
    session._pick_edit_effort(spec, draft)
    assert draft["effort"] == "xhigh"

    monkeypatch.setattr(
        TuiSession, "_pick", lambda self, items, prompt, **kw: "__clear__"
    )
    session._pick_edit_effort(spec, draft)
    assert draft["effort"] is None

    draft["effort"] = "low"
    monkeypatch.setattr(TuiSession, "_pick", lambda self, items, prompt, **kw: _BACK)
    session._pick_edit_effort(spec, draft)
    assert draft["effort"] == "low"


@pytest.mark.integration
def test_ctx_pick_back_leaves_the_draft_untouched(monkeypatch):
    from codehelper.cli.tui import _BACK, TuiSession
    from codehelper.services.wrappers import UNSET

    spec = _install_claude_zai()
    draft = _draft()
    monkeypatch.setattr(TuiSession, "_pick", lambda self, items, prompt, **kw: _BACK)
    _session()._pick_edit_ctx(spec, draft)
    assert draft["ctx"] is UNSET


@pytest.mark.integration
def test_ctx_pick_custom_accepts_tokens_and_none(monkeypatch):
    from codehelper.cli.tui import TuiSession

    spec = _install_claude_zai()
    draft: dict = {}
    monkeypatch.setattr(
        TuiSession, "_pick", lambda self, items, prompt, **kw: "__custom__"
    )
    monkeypatch.setattr(TuiSession, "_read_text", lambda self, prompt: "4096")
    _session()._pick_edit_ctx(spec, draft)
    assert draft["ctx"] == 4096

    monkeypatch.setattr(TuiSession, "_read_text", lambda self, prompt: "none")
    _session()._pick_edit_ctx(spec, draft)
    assert draft["ctx"] == 0


@pytest.mark.integration
def test_ctx_pick_custom_garbage_notifies_and_keeps_draft(monkeypatch):
    from codehelper.cli.tui import TuiSession

    spec = _install_claude_zai()
    draft: dict = {}
    notified: list[str] = []
    monkeypatch.setattr(
        TuiSession, "_pick", lambda self, items, prompt, **kw: "__custom__"
    )
    monkeypatch.setattr(TuiSession, "_read_text", lambda self, prompt: "lots")
    monkeypatch.setattr(TuiSession, "_notify", lambda self, text: notified.append(text))
    _session()._pick_edit_ctx(spec, draft)
    assert "error:" in notified[0]
    assert "ctx" not in draft

    # Empty input just backs out — no error, no draft write.
    monkeypatch.setattr(TuiSession, "_read_text", lambda self, prompt: "")
    _session()._pick_edit_ctx(spec, draft)
    assert "ctx" not in draft
    assert len(notified) == 1


@pytest.mark.integration
def test_subagent_pick_records_model_unset_and_custom(monkeypatch):
    from codehelper.cli.tui import _BACK, TuiSession

    spec = _install_claude_zai()
    discovered = []

    def _discover(self, provider, profile, token):
        discovered.append(provider.name)
        return ["glm-5.3", "glm-5.3-flash"], False, True

    monkeypatch.setattr(TuiSession, "_discover_models", _discover)
    session = _session()

    # A discovered model rides the draft verbatim.
    draft = _draft()
    monkeypatch.setattr(
        TuiSession, "_pick", lambda self, items, prompt, **kw: "glm-5.3-flash"
    )
    session._pick_edit_subagent(spec, draft)
    assert draft["subagent"] == "glm-5.3-flash"
    assert discovered == ["zai"]

    # __unset__ clears; a typed custom wins; empty custom backs out; Back
    # leaves the draft alone.
    monkeypatch.setattr(
        TuiSession, "_pick", lambda self, items, prompt, **kw: "__unset__"
    )
    session._pick_edit_subagent(spec, draft)
    assert draft["subagent"] is None

    monkeypatch.setattr(
        TuiSession, "_pick", lambda self, items, prompt, **kw: "__custom__"
    )
    monkeypatch.setattr(TuiSession, "_read_text", lambda self, prompt: "my-model")
    session._pick_edit_subagent(spec, draft)
    assert draft["subagent"] == "my-model"

    monkeypatch.setattr(TuiSession, "_read_text", lambda self, prompt: "")
    session._pick_edit_subagent(spec, draft)
    assert draft["subagent"] == "my-model"

    monkeypatch.setattr(TuiSession, "_pick", lambda self, items, prompt, **kw: _BACK)
    session._pick_edit_subagent(spec, draft)
    assert draft["subagent"] == "my-model"


@pytest.mark.integration
def test_subagent_pick_prompt_names_the_discovery_source(monkeypatch):
    """Same labeling rule as add: a stale known-models list must never read
    as something the endpoint just confirmed."""
    from codehelper.cli.tui import _BACK, TuiSession

    spec = _install_claude_zai()
    prompts: list[str] = []

    def _pick_record(self, items, prompt, **kw):
        prompts.append(prompt)
        return _BACK

    monkeypatch.setattr(
        TuiSession,
        "_discover_models",
        lambda self, provider, profile, token: (["m1"], True, False),
    )
    monkeypatch.setattr(TuiSession, "_pick", _pick_record)
    _session()._pick_edit_subagent(spec, _draft())
    assert "known models — discovery unavailable" in prompts[0]

    monkeypatch.setattr(
        TuiSession,
        "_discover_models",
        lambda self, provider, profile, token: (["m1"], False, False),
    )
    _session()._pick_edit_subagent(spec, _draft())
    assert "discovery unavailable" in prompts[1]
    assert "known models" not in prompts[1]


@pytest.mark.integration
def test_tier_pick_starts_and_merges_the_override_dict(monkeypatch):
    from codehelper.cli.tui import _BACK, TuiSession

    spec = _install_claude_zai()
    monkeypatch.setattr(
        TuiSession,
        "_choose_add_model",
        lambda self, provider, profile, token, agent_name=None, discovery=None: (
            f"m-{agent_name.split()[-1]}" if agent_name else "m-x"
        ),
    )
    session = _session()

    draft = _draft()
    session._pick_edit_haiku(spec, draft)
    assert draft["tiers"] == {"haiku": "m-haiku"}

    session._pick_edit_sonnet(spec, draft)
    assert draft["tiers"] == {"haiku": "m-haiku", "sonnet": "m-sonnet"}

    session._pick_edit_opus(spec, draft)
    assert draft["tiers"] == {
        "haiku": "m-haiku",
        "sonnet": "m-sonnet",
        "opus": "m-opus",
    }

    # Back on any tier keeps the already-drafted overrides.
    monkeypatch.setattr(TuiSession, "_choose_add_model", lambda self, *a, **kw: _BACK)
    session._pick_edit_tier(spec, draft, "haiku")
    assert draft["tiers"]["haiku"] == "m-haiku"


@pytest.mark.integration
def test_model_pick_sets_the_draft_only_on_a_choice(monkeypatch):
    from codehelper.cli.tui import _BACK, TuiSession

    spec = _install_claude_zai()
    seen = {}
    draft = _draft()

    def _choose(self, provider, profile, token, agent_name=None, discovery=None):
        seen["agent"] = agent_name
        return "glm-5.2:cloud"

    monkeypatch.setattr(TuiSession, "_choose_add_model", _choose)
    _session()._pick_edit_model(spec, draft)
    assert draft["model"] == "glm-5.2:cloud"
    assert seen["agent"] == "claude"

    monkeypatch.setattr(TuiSession, "_choose_add_model", lambda self, *a, **kw: _BACK)
    _session()._pick_edit_model(spec, draft)
    assert draft["model"] == "glm-5.2:cloud"


@pytest.mark.integration
def test_provider_pick_records_auth_and_url(monkeypatch):
    from codehelper.cli.tui import _BACK, TuiSession
    from codehelper.services.model import get_provider

    spec = _install_claude_zai()
    session = _session()

    draft = _draft()
    monkeypatch.setattr(
        TuiSession,
        "_choose_add_provider",
        lambda self, agent: (get_provider("litellm"), True, "http://127.0.0.1:4000"),
    )
    session._pick_edit_provider(spec, draft)
    assert draft["provider"] == "litellm"
    assert draft["auth"] == "secret"
    assert draft["base_url"] == "http://127.0.0.1:4000"

    monkeypatch.setattr(
        TuiSession,
        "_choose_add_provider",
        lambda self, agent: (get_provider("litellm"), False, ""),
    )
    session._pick_edit_provider(spec, draft)
    assert draft["auth"] == "literal"
    assert draft["base_url"] is None

    monkeypatch.setattr(TuiSession, "_choose_add_provider", lambda self, agent: _BACK)
    session._pick_edit_provider(spec, draft)
    assert draft["provider"] == "litellm"


@pytest.mark.integration
def test_edit_provider_view_serves_recorded_then_drafted_provider(monkeypatch):
    """Until a provider is drafted the pickers see the RECORDED spec; after,
    the drafted one with auth/URL applied — a model pick after a provider
    change must discover against the NEW endpoint."""

    spec = _install_claude_zai()
    session = _session()

    obj, profile = session._edit_provider_view(spec, _draft())
    assert obj is spec.provider
    assert profile == spec.profile_name

    obj, profile = session._edit_provider_view(
        spec, _draft(provider="litellm", auth="secret", base_url=None)
    )
    assert obj.name == "litellm"
    assert profile is None


@pytest.mark.integration
def test_apply_set_default_wrapper_sends_the_spec_axes(monkeypatch):
    from codehelper.cli.requests import SetDefaultRequest
    from codehelper.cli.tui import TuiSession
    from codehelper.services.paths import Paths
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper, spec_from_installed

    install_wrapper(
        Paths.default(),
        build_spec(
            agent="codex",
            provider="ollama-direct",
            model="glm-5.2:cloud",
            alias="codex-x",
        ),
        token="",
    )
    spec = spec_from_installed(Paths.default(), "codex-x")
    assert spec is not None

    seen = {}

    def _run(self, handler, request, live=False, silent=False):
        seen["request"] = request
        seen["live"] = live
        return True

    monkeypatch.setattr(TuiSession, "_run", _run)
    _session()._apply_set_default_wrapper(spec)

    req = seen["request"]
    assert isinstance(req, SetDefaultRequest)
    assert (req.agent, req.provider, req.model) == (
        "codex",
        "ollama-direct",
        "glm-5.2:cloud",
    )
    # ollama-direct is FIXED: its own registry address is never forwarded
    # (issue #30 follow-up — a FIXED provider rejects --base-url even when
    # equal to its default), so the request carries None.
    assert req.base_url is None
    assert seen["live"] is True


@pytest.mark.unit
def test_show_help_names_every_binding_group(monkeypatch):
    from codehelper.cli.tui import TuiSession

    shown: list[str] = []
    monkeypatch.setattr(TuiSession, "_notify", lambda self, text: shown.append(text))
    _session()._show_help()
    text = shown[0]
    for fragment in ("a add", "t token", "e edit", "s settings", "Esc quit"):
        assert fragment in text, f"help missing {fragment!r}"


@pytest.mark.integration
def test_edit_was_applied_reports_agent_or_none_without_a_backend_row():
    from codehelper.cli.tui import TuiSession
    from codehelper.services.paths import Paths
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import install_wrapper

    install_wrapper(
        Paths.default(),
        build_spec(agent="claude", provider="zai", model="glm-5.3", alias="glm-w"),
        token="sk-x",
    )
    install_wrapper(
        Paths.default(),
        build_spec(
            agent="agy", provider="antigravity", model="gemini-3-pro", alias="agy-w"
        ),
        token="",
    )

    session = TuiSession(_tui_args())
    assert session._edit_was_applied("glm-w") == ("claude", False)

    # agy has no chipset row — no backend, no applied claim at all.
    assert session._edit_was_applied("agy-w") is None


@pytest.mark.integration
def test_run_edit_refuses_an_unmanaged_wrapper(monkeypatch):
    from codehelper.cli.tui import TuiSession

    shown: list[str] = []
    monkeypatch.setattr(TuiSession, "_notify", lambda self, text: shown.append(text))
    screens: list[str] = []
    monkeypatch.setattr(
        TuiSession,
        "_run_edit_screen",
        lambda self, alias: screens.append(alias),
    )
    _session()._run_edit("not-installed")
    assert "not installed" in shown[0]
    assert screens == []


@pytest.mark.unit
def test_edit_current_value_covers_every_axis_key():
    """The formatter both labels the screen rows and feeds the -> delta; a
    wrong spelling here mislabels a real recorded value."""
    from codehelper.cli.tui import TuiSession
    from codehelper.services.spec import TierModels, build_spec

    spec = build_spec(
        agent="claude",
        provider="zai",
        model="glm-5.3",
        alias="glm-x",
        tier_models=TierModels(haiku="h", sonnet="s", opus="o"),
        subagent_model="sub-m",
        effort="high",
        context_window=200000,
    )
    value = TuiSession._edit_current_value
    assert value("provider", spec) == "zai"
    assert value("model", spec) == "glm-5.3"
    assert value("haiku", spec) == "h"
    assert value("sonnet", spec) == "s"
    assert value("opus", spec) == "o"
    assert value("uniform", spec) == "per-tier"
    assert value("subagent", spec) == "sub-m"
    assert value("effort", spec) == "high"
    assert value("ctx", spec) == "200000"

    bare = build_spec(agent="claude", provider="zai", model="glm-5.3", alias="glm-y")
    # build_spec normalises a preset's single model to UNIFORM tiers, so the
    # bare wrapper records glm-5.3 on every tier and the uniform row reads
    # "per-tier"; a spec with no tier models at all is the "uniform" spelling.
    assert value("haiku", bare) == "glm-5.3"
    assert value("uniform", bare) == "per-tier"
    tierless = replace(bare, tier_models=None)
    assert value("uniform", tierless) == "uniform"
    assert value("subagent", bare) == "—"
    assert value("effort", bare) == "—"
    assert value("ctx", bare) == "catalog / unset"
    assert value("ctx", replace(bare, context_window=0)) == "none (suppressed)"


@pytest.mark.unit
def test_edit_axis_label_formats_current_and_delta():
    from codehelper.cli.tui import TuiSession
    from codehelper.services.spec import build_spec
    from codehelper.services.wrappers import UNSET

    spec = build_spec(agent="claude", provider="zai", model="glm-5.3", alias="glm-x")

    def axis(key, label=None):
        return SimpleNamespace(key=key, label=label or key)

    # Unset: current only. Set: current -> new. ctx 0 is the suppression
    # spelling; a tier row shows its own override, uniform row says mixed.
    assert (
        TuiSession._edit_axis_label(axis("model", "model"), spec, {"model": UNSET})
        == "model: glm-5.3"
    )
    assert (
        TuiSession._edit_axis_label(
            axis("model", "model"), spec, {"model": "glm-5.2:cloud"}
        )
        == "model: glm-5.3 -> glm-5.2:cloud"
    )
    assert (
        TuiSession._edit_axis_label(axis("ctx", "ctx"), spec, {"ctx": 0})
        == "ctx: catalog / unset -> none (suppressed)"
    )
    assert (
        TuiSession._edit_axis_label(
            axis("haiku", "haiku"), spec, {"tiers": {"haiku": "h1"}}
        )
        == "haiku: glm-5.3 -> h1"
    )
    assert (
        TuiSession._edit_axis_label(
            axis("uniform", "tiers"), spec, {"tiers": {"haiku": "h1"}}
        )
        == "tiers: per-tier -> mixed (per-tier)"
    )
    assert (
        TuiSession._edit_axis_label(axis("uniform", "tiers"), spec, {"tiers": None})
        == "tiers: per-tier -> uniform"
    )
    assert (
        TuiSession._edit_axis_label(axis("rename", "rename"), spec, {})
        == "rename — move the wrapper to a new alias"
    )


@pytest.mark.integration
def test_on_slot_applies_the_numbered_profile(monkeypatch):
    """Digit slots pick the pre-selection: the recorded slot pair becomes the
    active selection; an out-of-range digit does nothing."""
    from codehelper.services.paths import Paths
    from codehelper.services.profiles import profile_slots
    from codehelper.services.state import active_selection

    paths = Paths.default()
    paths.credentials_file().parent.mkdir(parents=True, exist_ok=True)
    paths.credentials_file().write_text(
        '{"zai": {"work": "t2", "default": "t1"}}', encoding="utf-8"
    )
    slots = profile_slots(paths)
    assert ("zai", "work") in slots

    session = _session()
    session._on_slot(slots.index(("zai", "work")))
    assert active_selection(paths) == ("zai", "work")

    before = active_selection(paths)
    session._on_slot(len(slots) + 5)
    assert active_selection(paths) == before


@pytest.mark.integration
def test_uniform_and_rename_picks_close_their_drafts(monkeypatch):
    """Uniform resets the tier overrides to None; rename closes the screen
    when the alias actually moved, and leaves the draft alone on cancel."""
    from codehelper.cli.tui import TuiSession

    spec = _install_claude_zai()
    session = _session()

    draft = _draft(tiers={"haiku": "h1"})
    session._pick_edit_uniform(spec, draft)
    assert draft["tiers"] is None

    draft = _draft()
    monkeypatch.setattr(TuiSession, "_on_rename", lambda self, alias: True)
    session._pick_edit_rename(spec, draft)
    assert draft["closed"] is True

    draft = _draft()
    monkeypatch.setattr(TuiSession, "_on_rename", lambda self, alias: False)
    session._pick_edit_rename(spec, draft)
    assert "closed" not in draft


@pytest.mark.unit
def test_tee_streams_answer_for_the_real_terminal():
    """``_Tee`` must behave like the stream it replaces: isatty answers for
    the real terminal (the menu's TTY guard and read_line both call it), and
    fileno delegates so anything fd-based keeps working."""
    import io
    import sys

    from codehelper.cli.tui import _Tee

    real = sys.stdout
    tee = _Tee(real, io.StringIO())
    assert tee.isatty() == real.isatty()
    assert tee.fileno() == real.fileno()
    assert tee.write("x") == 1
    tee.flush()


@pytest.mark.unit
def test_profile_row_label_states_the_active_profile():

    session = _session()
    session._tab_label = "zai/work"
    assert session._profile_row_label() == "Profile: zai/work"
    session._tab_label = ""
    assert session._profile_row_label() == "Profile"
