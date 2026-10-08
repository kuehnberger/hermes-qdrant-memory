"""Tests for the ``md_docs_roots`` config key.

The key was documented in ``mdsearch.py`` before it existed anywhere in the
code: the module comment claimed roots were "overridable via the
``md_docs_roots`` config key" while ``iter_markdown`` read the hardcoded
:data:`DEFAULT_ROOTS`. These tests pin the behaviour so the comment cannot
drift from reality again.

The rules under test:
  * an override REPLACES that label's path but KEEPS its skip set
  * a NEW label is accepted and gets an empty skip set (dot-dirs are
    skipped by ``iter_markdown`` itself, independently)
  * a malformed value degrades to the defaults with a warning, never raises
  * the FTS5 fast path still does not import the provider or an embedder
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:  # the package-relative import the CI matrix uses
    from plugins.memory.qdrant import mdsearch
except ImportError:  # bare-top-level load mode (running from the plugin dir)
    import mdsearch


def _write_config(monkeypatch, tmp_path: Path, payload) -> None:
    """Point HERMES_HOME at tmp_path and write a qdrant.json payload there."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "qdrant.json").write_text(json.dumps(payload), encoding="utf-8")


def test_no_config_key_returns_the_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert mdsearch.configured_roots() == mdsearch.DEFAULT_ROOTS


def test_missing_config_file_returns_the_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no qdrant.json at all
    assert mdsearch.configured_roots() == mdsearch.DEFAULT_ROOTS


def test_override_replaces_path_and_keeps_skip_set(monkeypatch, tmp_path):
    _write_config(
        monkeypatch, tmp_path, {"md_docs_roots": {"skills": "/tmp/other-skills"}}
    )
    roots = mdsearch.configured_roots()
    assert roots["skills"][0] == "/tmp/other-skills"
    # the skip set is part of the DEFAULT for that label, not restated by the override
    assert roots["skills"][1] == mdsearch.DEFAULT_ROOTS["skills"][1]
    # labels the key does not mention keep their defaults
    assert roots["vault"] == mdsearch.DEFAULT_ROOTS["vault"]


def test_new_label_is_accepted_with_an_empty_skip_set(monkeypatch, tmp_path):
    _write_config(
        monkeypatch, tmp_path, {"md_docs_roots": {"sessions": "/tmp/qd-sessions"}}
    )
    roots = mdsearch.configured_roots()
    assert roots["sessions"] == ("/tmp/qd-sessions", set())
    # and it does not displace any default
    assert set(mdsearch.DEFAULT_ROOTS) <= set(roots)


def test_home_relative_path_is_expanded(monkeypatch, tmp_path):
    _write_config(monkeypatch, tmp_path, {"md_docs_roots": {"sessions": "~/somewhere"}})
    expanded = mdsearch.configured_roots()["sessions"][0]
    assert "~" not in expanded
    assert expanded.endswith("somewhere")


@pytest.mark.parametrize(
    "payload",
    [
        {"md_docs_roots": "not-a-dict"},
        {"md_docs_roots": ["skills"]},
        {"md_docs_roots": {"skills": 42}},
        {"md_docs_roots": {"skills": None}},
        {"md_docs_roots": {"": "/tmp/whatever"}},
        {"md_docs_roots": {"skills": "   "}},
    ],
)
def test_malformed_values_fall_back_instead_of_raising(monkeypatch, tmp_path, payload):
    _write_config(monkeypatch, tmp_path, payload)
    roots = mdsearch.configured_roots()  # must not raise
    # either the default survived, or the bad entry was dropped -- never a crash
    assert "skills" in roots
    assert roots["skills"][1] == mdsearch.DEFAULT_ROOTS["skills"][1]


def test_corrupt_config_file_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "qdrant.json").write_text("{not json", encoding="utf-8")
    assert mdsearch.configured_roots() == mdsearch.DEFAULT_ROOTS


def test_non_object_config_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "qdrant.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert mdsearch.configured_roots() == mdsearch.DEFAULT_ROOTS


def test_key_is_not_reported_unknown_by_the_flat_config_sweep(
    monkeypatch, tmp_path, caplog
):
    """``md_docs_roots`` is a documented key of qdrant.json, not a typo.

    md_search reads it straight from the file, so the flat-provider sweep in
    ``_load_plugin_config`` must neither warn about it (calling a working
    setting "unknown" invites deletion) nor carry it as a provider knob.
    Same rule as the ``display`` exemption in test_display_levels.
    """
    _write_config(
        monkeypatch, tmp_path, {"md_docs_roots": {"skills": "/tmp/x"}}
    )
    # Config-file test: clear the env layer first — env outranks the file,
    # and a CI host may export the vars this test must not see.
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    import hermes_cli.config as _cfg
    monkeypatch.setattr(_cfg, "load_config_readonly", lambda: {})
    import logging as _logging

    import plugins.memory.qdrant as mod
    with caplog.at_level(_logging.WARNING,
                         logger="hermes.plugins.memory.qdrant"):
        merged = mod._load_plugin_config()
    assert "md_docs_roots" not in merged, "it is not a provider knob"
    assert not any(
        "unknown config key" in r.getMessage() for r in caplog.records
    ), "a documented key was reported as unknown"


def test_configured_root_is_actually_walked(monkeypatch, tmp_path):
    """The override must reach ``iter_markdown``, not just the config view."""
    corpus = tmp_path / "exported-sessions"
    (corpus / "sub").mkdir(parents=True)
    (corpus / "a.md").write_text("# A\n\nalpha", encoding="utf-8")
    (corpus / "sub" / "b.md").write_text("# B\n\nbeta", encoding="utf-8")
    (corpus / "sub" / "ignored.txt").write_text("not markdown", encoding="utf-8")
    _write_config(
        monkeypatch, tmp_path, {"md_docs_roots": {"sessions": str(corpus)}}
    )
    labels = {cf.label for cf in mdsearch.iter_markdown()}
    # the key MERGES over the defaults, so the defaults are still walked --
    # what this test pins is that the new root is among them, fully descended,
    # and that non-markdown files stay out.
    assert {"sessions/a.md", "sessions/sub/b.md"} <= labels
    assert not any(
        lbl.startswith("sessions/") and lbl.endswith(".txt") for lbl in labels
    )


def test_explicit_roots_argument_still_wins(monkeypatch, tmp_path):
    """An explicitly passed mapping must not be replaced by the config."""
    _write_config(
        monkeypatch, tmp_path, {"md_docs_roots": {"sessions": "/tmp/ignored"}}
    )
    explicit = {"only": ("/tmp/only", set())}
    labels = {cf.label for cf in mdsearch.iter_markdown(explicit)}
    assert labels <= {"only/whatever.md"}
    assert "sessions/ignored.md" not in labels


def test_fast_path_still_imports_no_embedding_library():
    """The FTS5 promise: configuring roots must not drag in the model."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, ".")
        import mdsearch
        mdsearch.configured_roots()
        mdsearch.fts_query("hello")
        heavy = [m for m in sys.modules
                 if m.split(".")[0] in ("fastembed", "sentence_transformers",
                                        "torch", "onnxruntime", "qdrant_client")]
        print("HEAVY:" + ",".join(sorted(heavy)))
        """
    )
    plugin_dir = str(Path(mdsearch.__file__).resolve().parent)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=plugin_dir,
    )
    assert out.returncode == 0, out.stderr
    assert "HEAVY:\n" in out.stdout or out.stdout.strip().endswith("HEAVY:")
