#!/usr/bin/env python3
"""Generate a fleet usage report from the hermes-telemetry SQLite store.

Read-only only: opens the DB in ``mode=ro`` and never writes to it. Emits a
Markdown report to stdout (or ``--out path.md``). It is a GENERATOR, not a
deliverer — routing the report to a notification channel is owned by
default (card t_fa73b6b5); this script does not mail/toast/post.

Usage:
  telemetry_usage_report.py [--window-hours 24] [--out path.md] [--db PATH]

The DB resolves like the plugin's own paths.py: ``$HERMES_TELEMETRY_HOME``
(default ``~/.hermes``) -> ``<home>/telemetry/telemetry.db``.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone


def resolve_db(cli_db: str | None) -> str:
    if cli_db:
        return cli_db
    home = os.environ.get("HERMES_TELEMETRY_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(home, "telemetry", "telemetry.db")


def tok(n) -> str:
    if n is None:
        return "-"
    if n < 1_000_000:
        return f"{n/1024:.1f}K"
    return f"{n/1_000_000:.2f}M"


def money(c) -> str:
    if not c:
        return "$0.00"
    return f"${c:.2f}"


def pct(ok, total) -> str:
    if not total:
        return "-"
    return f"{100*ok/total:.0f}%"


def window_str(hours: float) -> str:
    if hours.is_integer():
        return f"-{int(hours)} hours"
    return f"-{hours} hours"


def run(args: argparse.Namespace) -> int:
    db = resolve_db(args.db)
    if not os.path.exists(db):
        print(f"telemetry store not found at {db} — nothing to report", file=sys.stderr)
        return 1

    win = window_str(args.window_hours)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    def rows(sql, a=()):
        return [dict(r) for r in con.execute(sql, a)]

    runs = rows(
        "SELECT platform, provider, model, status, tokens_in, tokens_out, cost_usd,"
        " duration_ms, api_calls, tool_calls, started_at, ended_at"
        " FROM runs WHERE started_at >= datetime('now', ?) ORDER BY started_at DESC",
        (win,),
    )
    llm = rows(
        "SELECT coalesce(model,'?') AS model, coalesce(provider,'?') AS provider,"
        " SUM(tokens_in+tokens_out) AS tok, SUM(cost_usd) AS cost,"
        " COUNT(*) AS calls, AVG(latency_ms) AS lat_ms"
        " FROM llm_calls WHERE ts >= datetime('now', ?)"
        " GROUP BY model, provider ORDER BY tok DESC LIMIT 8",
        (win,),
    )
    tools = rows(
        "SELECT tool_name, SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok, COUNT(*) AS total"
        " FROM tool_calls WHERE ts >= datetime('now', ?)"
        " GROUP BY tool_name ORDER BY total DESC LIMIT 12",
        (win,),
    )
    by_prof = rows(
        "SELECT coalesce(provider,'unknown') AS provider, SUM(tokens_in+tokens_out) AS tok,"
        " SUM(cost_usd) AS cost, COUNT(*) AS calls"
        " FROM llm_calls WHERE ts >= datetime('now', ?)"
        " GROUP BY provider ORDER BY tok DESC",
        (win,),
    )
    err = rows(
        "SELECT COUNT(*) AS n FROM llm_call_errors WHERE ts >= datetime('now', ?)",
        (win,),
    )[0]["n"]
    con.close()

    ti = sum(r["tokens_in"] or 0 for r in runs)
    to = sum(r["tokens_out"] or 0 for r in runs)
    cost = sum(r["cost_usd"] or 0 for r in runs)
    n_runs = len(runs)
    running = sum(1 for r in runs if r["status"] == "running")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    L = []
    L.append(f"# Hermes fleet usage — last {args.window_hours:g}h (to {now})")
    L.append("")
    L.append(
        f"Sessions: **{n_runs}** ({running} running) · "
        f"Tokens: **{tok(ti+to)}** (in {tok(ti)} / out {tok(to)}) · "
        f"Cost: **{money(cost)}** · LLM errors: {err}"
    )
    L.append("")
    L.append("## Top models (by tokens)")
    L.append("| model | provider | tokens | cost | calls |")
    L.append("|---|---|---|---|---|")
    for r in llm:
        L.append(
            f"| `{r['model']}` | {r['provider'] or '-'} | {tok(r['tok'])} | "
            f"{money(r['cost'])} | {r['calls']} |"
        )
    L.append("")
    L.append("## Top tools")
    L.append("| tool | ok | total | success |")
    L.append("|---|---|---|---|")
    critique = []
    for r in tools:
        rate = (100 * r["ok"] / r["total"]) if r["total"] else 0
        L.append(
            f"| {r['tool_name']} | {r['ok']} | {r['total']} | {pct(r['ok'], r['total'])} |"
        )
        if r["total"] and rate < 85:
            critique.append((r["tool_name"], r["ok"], r["total"], rate))
    L.append("")
    L.append("## By provider")
    L.append("| provider | tokens | cost | calls |")
    L.append("|---|---|---|---|")
    for r in by_prof:
        L.append(
            f"| {r['provider']} | {tok(r['tok'])} | {money(r['cost'])} | {r['calls']} |"
        )
    L.append("")
    L.append("## Critique (>85% success required)")
    if critique:
        for name, ok, total, rate in sorted(critique, key=lambda x: x[3]):
            L.append(
                f"- `{name}` at {rate:.0f}% ({ok}/{total}) — below the 85% success bar; "
                f"review failure modes (e.g. consent-gated long calls, transient errors)."
            )
    else:
        L.append("- All tools at or above 85% success in this window.")
    L.append("")

    md = "\n".join(L)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    print(md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--out", default=None, help="write the report to this path instead of stdout")
    ap.add_argument("--db", default=None, help="override the telemetry db path")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
