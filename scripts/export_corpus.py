#!/usr/bin/env python
"""Export a Hermes profile's session DB and agent log as markdown for md_search.

``md_search`` ingests markdown only (``mdsearch.iter_markdown`` walks ``*.md``
under the configured roots). A profile's history lives in two files that are
not markdown — ``state.db`` (sqlite: ``sessions`` + ``messages``) and
``logs/agent.log`` (jsonl) — so this script renders them into markdown that the
existing heading-aware chunker can index without any special-casing.

Output shape (chosen for the chunker, not for human reading):

* one ``.md`` per session, ``sessions/<id>.md``; a ``# <title>`` H1 carrying the
  session id, start time and source in the body, then one ``## user`` /
  ``## assistant`` H2 per turn. The heading-aware chunker turns each of those
  into a breadcrumb like ``"<title> > assistant"``, so a hit reports who said
  it and where it came from.
* one ``.md`` per agent-log day, ``agentlog/YYYY-MM-DD.md``.

Design notes that are not obvious from the code:

* **Exports are derived data.** Everything written here is re-derivable from
  ``state.db`` / ``agent.log``, so the output lives under ``<base>/state/`` and
  never inside a plugin member dir — a file written next to a plugin module
  becomes a hermes build input and re-syncs dependencies on every launch.
* **The DB is read with an immutable-ish read-only URI**, and the WAL is taken
  into account, so exporting a profile that is LIVE (this script is meant to be
  run against the running profile) cannot corrupt or block the gateway's own
  writes.
* **Tool messages are dropped, tool CALLS are kept as a one-line breadcrumb.**
  A ``tool`` message is a multi-kilobyte JSON blob of stdout that would dominate
  the corpus with repetitive command output; the assistant turn that decided to
  call it is the part worth recalling. Reasoning columns are likewise skipped.
* **Redaction is the caller's job, not this script's** — this exports whatever
  the profile already stored, including anything sensitive in it. Output goes
  to a directory only the user chooses.

Usage:
    python scripts/export_corpus.py --profile qd --out ~/state/knowledge/qd-sessions
    python scripts/export_corpus.py --profile default --out /tmp/x --no-agent-log
    python scripts/export_corpus.py --profile qd --out /tmp/x --stats-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

#: Rows whose ``content`` is this long are truncated in the export. A single
#: pasted log or base64 blob should not become a 5 MB "turn".
MAX_TURN_CHARS = 20_000

#: Sessions with fewer messages than this carry no signal worth indexing.
MIN_MESSAGES = 2

_SESSIONS_SQL = """
    SELECT s.id, s.source, s.title, s.display_name, s.started_at, s.ended_at,
           s.message_count, s.end_reason, s.model,
           (SELECT count(*) FROM messages m WHERE m.session_id = s.id) AS n
      FROM sessions s
     ORDER BY COALESCE(s.started_at, 0)
"""

_MESSAGES_SQL = """
    SELECT id, role, content, tool_calls, reasoning, reasoning_content
      FROM messages
     WHERE session_id = ?
     ORDER BY id
"""

#: Roles whose text is the conversation. Everything else is machinery.
_TEXT_ROLES = ("user", "assistant")

#: Hermes' own ``agent.log`` is NOT jsonl — it is plain lines shaped
#: ``2026-09-27 17:31:13,363 INFO hermes_cli.plugins: message``. jsonl is also
#: accepted, because that is what a hand-rolled or redirected logger emits and
#: guessing wrong on a 34k-line file is worse than parsing both.
_PLAIN_LOG_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d+)?)"
    r"\s+(?P<level>[A-Z]+)\s+"
    r"(?P<logger>[\w.]+)\s*:\s*(?P<msg>.*)$"
)


def profile_dir(profile: str) -> Path:
    """The profile's home directory, honouring HERMES_HOME when it is a profile."""
    base = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    candidate = base / "profiles" / profile
    if candidate.is_dir():
        return candidate
    if base.name == profile:  # HERMES_HOME already points INTO the profile
        return base
    raise SystemExit(
        f"no profile directory for {profile!r} (looked at {candidate}); "
        f"pass --profile-dir to override"
    )


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open a LIVE profile database read-only, WAL included.

    ``mode=ro`` still reads the WAL, so a concurrent gateway write is visible
    and cannot be corrupted by us; ``timeout`` keeps a busy writer from turning
    a read into a hard error.
    """
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=15.0)


def _stamp(ts: float | None) -> str:
    if not ts:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return "-"


def _tool_breadcrumb(tool_calls: str | None) -> str:
    """One line naming the tools an assistant turn invoked.

    Not the results — the results live in ``tool`` messages, which are dropped.
    The names alone are usually the recall-worthy part ("which turn ran the
    eval harness?").
    """
    if not tool_calls:
        return ""
    try:
        calls = json.loads(tool_calls)
    except (TypeError, ValueError):
        return ""
    if not isinstance(calls, list):
        return ""
    names: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        name = fn.get("name") if isinstance(fn, dict) else call.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    if not names:
        return ""
    shown = names[:8]
    more = f" (+{len(names) - len(shown)} more)" if len(names) > len(shown) else ""
    return f"*tools: {', '.join(shown)}{more}*"


def render_session(row: tuple, con: sqlite3.Connection, profile: str) -> str | None:
    """One session as markdown, or ``None`` when it carries nothing to index."""
    (sid, source, title, display_name, started_at, ended_at,
     _msg_count, end_reason, model, n_messages) = row
    if int(n_messages or 0) < MIN_MESSAGES:
        return None

    heading = (title or display_name or "").strip() or f"session {sid}"
    lines = [
        f"# {heading}",
        "",
        f"- profile: {profile}",
        f"- session: {sid}",
        f"- started: {_stamp(started_at)}",
    ]
    if ended_at:
        lines.append(f"- ended: {_stamp(ended_at)}")
    if source:
        lines.append(f"- source: {source}")
    if model:
        lines.append(f"- model: {model}")
    if end_reason:
        lines.append(f"- end reason: {end_reason}")
    lines.append("")

    kept = 0
    # reasoning / reasoning_content are selected but unused on purpose: the
    # columns exist in the schema and naming them documents that the export
    # deliberately does not carry a model's private reasoning into the corpus.
    for msg_id, role, content, tool_calls, reasoning, reasoning_content in con.execute(
        _MESSAGES_SQL, (sid,)
    ):
        del reasoning, reasoning_content
        if role not in _TEXT_ROLES:
            continue
        text = (content or "").strip()
        crumb = _tool_breadcrumb(tool_calls) if role == "assistant" else ""
        if not text and not crumb:
            continue
        truncated = False
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS] + "\n…[truncated by export_corpus.py]"
            truncated = True
        lines.append(f"## {role} (msg {msg_id})")
        lines.append("")
        if text:
            lines.append(text)
            lines.append("")
        if crumb:
            lines.append(crumb)
            lines.append("")
        if truncated:
            lines.append(f"_turn {msg_id} truncated_")
            lines.append("")
        kept += 1

    if kept < MIN_MESSAGES:
        return None
    lines.append(f"<!-- {kept} turns exported from session {sid} -->")
    return "\n".join(lines)


def export_sessions(db_path: Path, out_dir: Path, profile: str,
                    stats_only: bool = False) -> tuple[int, int]:
    """Write one ``.md`` per session. Returns ``(files, turns)``."""
    con = open_db(db_path)
    written = turns = 0
    try:
        sessions = con.execute(_SESSIONS_SQL).fetchall()
        target = out_dir / "sessions"
        if not stats_only:
            target.mkdir(parents=True, exist_ok=True)
        for row in sessions:
            doc = render_session(row, con, profile)
            if doc is None:
                continue
            written += 1
            turns += doc.count("\n## ")
            if not stats_only:
                sid = str(row[0]).replace("/", "_")
                (target / f"{sid}.md").write_text(doc, encoding="utf-8")
    finally:
        con.close()
    return written, turns


def export_agent_log(log_path: Path, out_dir: Path,
                     stats_only: bool = False) -> tuple[int, int]:
    """Write one ``.md`` per day of the agent log. Returns ``(files, entries)``.

    The log is jsonl, one object per line, and it rotates (``agent.log.1`` …), so
    every rotated sibling is read too — a rotation boundary is exactly where the
    interesting history would otherwise be cut in half.
    """
    if not log_path.is_file():
        return 0, 0

    # Oldest first: agent.log is newest, .1 is older, and so on.
    siblings: list[Path] = []
    if log_path.with_name(log_path.name + ".1").exists():
        rotated = sorted(
            log_path.parent.glob(f"{log_path.name}.*"),
            key=lambda p: int(p.name.rsplit(".", 1)[-1]),
            reverse=True,
        )
        siblings = list(rotated)
    siblings.append(log_path)

    by_day: dict[str, list[str]] = {}
    plain_lines = json_lines = 0
    for path in siblings:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec: dict | None = None
                    if line.startswith("{"):
                        try:
                            candidate = json.loads(line)
                            rec = candidate if isinstance(candidate, dict) else None
                            json_lines += 1 if rec else 0
                        except ValueError:
                            rec = None
                    if rec is None:
                        m = _PLAIN_LOG_RE.match(line)
                        if m:
                            plain_lines += 1
                            rec = m.groupdict()
                    if rec is None:
                        continue
                    stamp = rec.get("timestamp") or rec.get("time") or rec.get("ts")
                    day = str(stamp)[:10] if stamp else "undated"
                    body = _log_entry(rec)
                    if body:
                        by_day.setdefault(day, []).append(body)
        except OSError as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)

    print(
        f"agent log parsed: {plain_lines} plain lines, "
        f"{json_lines} json lines, {len(siblings)} file(s)",
        file=sys.stderr,
    )

    if not stats_only:
        target = out_dir / "agentlog"
        target.mkdir(parents=True, exist_ok=True)
        for day, entries in sorted(by_day.items()):
            safe_day = day if day != "undated" else "undated"
            body = "\n".join([f"# agent log {day}", ""] + entries)
            (target / f"{safe_day}.md").write_text(body, encoding="utf-8")
    return len(by_day), sum(len(v) for v in by_day.values())


def _log_entry(rec: dict) -> str | None:
    """One log record as a markdown line, or ``None`` when it has no content."""
    level = str(rec.get("level") or rec.get("levelname") or "").upper()
    name = str(rec.get("name") or rec.get("logger") or "")
    msg = rec.get("message") or rec.get("msg") or ""
    if isinstance(msg, dict):  # some formatters pre-render the message
        msg = msg.get("message") or json.dumps(msg, ensure_ascii=False)
    msg = str(msg).strip()
    if not msg:
        return None
    if len(msg) > 4000:
        msg = msg[:4000] + " …[truncated]"
    head = " | ".join(part for part in (level, name) if part)
    return f"- **{head or 'log'}** — {msg}" if head else f"- {msg}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a Hermes profile's sessions + agent log as markdown"
    )
    ap.add_argument("--profile", default="qd", help="profile name (default: qd)")
    ap.add_argument("--profile-dir", default=None,
                    help="explicit profile home, overrides --profile lookup")
    ap.add_argument("--out", default=None,
                    help="output directory; default "
                         "<base hermes home>/state/knowledge/<profile>")
    ap.add_argument("--no-agent-log", action="store_true",
                    help="export sessions only")
    ap.add_argument("--stats-only", action="store_true",
                    help="report what WOULD be written, write nothing")
    args = ap.parse_args(argv)

    pdir = Path(args.profile_dir) if args.profile_dir else profile_dir(args.profile)

    if args.out:
        out = Path(args.out).expanduser()
    else:
        try:
            from hermes_constants import get_hermes_home

            base = Path(get_hermes_home())
            if base.parent.name == "profiles":
                base = base.parent.parent
        except Exception:
            base = Path.home() / ".hermes"
        out = base / "state" / "knowledge" / args.profile

    db = pdir / "state.db"
    if not db.is_file():
        print(f"no session db at {db}", file=sys.stderr)
        return 2

    started = time.time()
    sess_files, turns = export_sessions(db, out, args.profile, args.stats_only)
    log_days = log_entries = 0
    if not args.no_agent_log:
        log_days, log_entries = export_agent_log(
            pdir / "logs" / "agent.log", out, args.stats_only
        )

    verb = "would export" if args.stats_only else "exported"
    print(f"{verb} sessions: {sess_files} files / {turns} turns")
    if not args.no_agent_log:
        print(f"{verb} agent log: {log_days} days / {log_entries} entries")
    if not args.stats_only:
        print(f"output: {out}")
    print(f"in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
