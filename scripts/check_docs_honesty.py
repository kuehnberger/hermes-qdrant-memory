#!/usr/bin/env python
"""Docs honesty gate: every command / tool name / count claimed in prose must exist.

Catalog rule 6 treats a declared-vs-actual capability mismatch as a security
issue. Prose is not exempt: a README that says "4 agent tools" when the manifest
declares 5 is the same class of defect, and it is the kind a reviewer finds in
30 seconds and then distrusts everything else.

This script checks the claims that are cheap to state and expensive to verify.
It reads the real source of truth (hermes_cli/subcommands/memory.py for the CLI
surface, tool_schemas.py for the tools) rather than re-reading our own docs.
CLI verification needs a hermes core checkout: $HERMES_CORE (CI fetches the
single file; locally the default path is used, and its absence is a PRINTED
skip — the tool/manifest checks still run and still fail the build).

Run:  python3 scripts/check_docs_honesty.py
Exit: 0 clean, 1 on any violation.
"""

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# HerMES core checkout used to verify claims about the `hermes memory` CLI.
# HERMES_CORE overrides (CI fetches just that file and points this at it);
# the legacy default keeps local runs working without setup.
CORE = Path(os.environ.get("HERMES_CORE") or "/home/gk/.hermes/hermes-agent")

# Files whose prose is rendered to reviewers (the catalog renders README.md at
# the pinned SHA; the others are read by contributors and agents).
PROSE_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "docs/README.md",
    "docs/screenshots/README.md",
    "docs/competitor-analysis-entropicmem.md",
]

problems: list[str] = []


def read(rel: str) -> str:
    p = REPO / rel
    return p.read_text(encoding="utf-8") if p.is_file() else ""


# --- 1. Tool names and count ------------------------------------------------
schemas_src = read("tool_schemas.py")
manifest = read("plugin.yaml")

m = re.search(r"ALL_TOOL_SCHEMAS\s*=\s*\[(.*?)\]", schemas_src, re.S)
if not m:
    problems.append("could not parse ALL_TOOL_SCHEMAS from tool_schemas.py")
else:
    # The list holds SCHEMA VARIABLE names, not string literals, so a regex for
    # "qdrant_*" finds nothing. Resolve each entry against its definition and
    # take the "name": field the schema actually declares.
    entries = re.findall(r"^\s*([A-Z][A-Z0-9_]*_SCHEMA)\s*,?\s*$", m.group(1), re.M)
    tools: list[str] = []
    for var in entries:
        dm = re.search(
            rf"{var}\s*=\s*\{{(.*?)\n\}}", schemas_src, re.S
        )
        if not dm:
            problems.append(
                f"could not resolve schema variable {var} in tool_schemas.py"
            )
            continue
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', dm.group(1))
        if not nm:
            problems.append(f"{var} has no \"name\" field in tool_schemas.py")
            continue
        tools.append(nm.group(1))
    tools = sorted(set(tools))
    print(f"tools declared in tool_schemas.py ({len(tools)}): {', '.join(tools)}")

    if not tools:
        problems.append(
            "resolved zero tools from tool_schemas.py — "
            "parser is broken, not the repo"
        )

    # every tool in the list must appear in the manifest
    for t in tools:
        if f"- {t}" not in manifest:
            problems.append(
                f"{t} in tool_schemas.py but not in "
                "plugin.yaml provides_tools"
            )

    # every prose claim of a COUNT must equal len(tools)
    count_pat = re.compile(
        r"\b(\w+|\d+|five|four|six|seven|eight|nine)\s+agent tools\b", re.I
    )
    words = {
        "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    }
    for rel in PROSE_FILES:
        text = read(rel)
        # A tool-count claim describes the CURRENT surface only inside the
        # changelog's open section. A claim under a released heading (e.g. the
        # 0.1.x entries saying "Five agent tools") is a historical record and
        # was true when written — rewriting it would be falsifying the
        # changelog, and flagging it would force that. Scope the check to the
        # text above the first released version heading.
        scope = text
        if rel == "CHANGELOG.md":
            released = re.search(r"^## \[\d+\.\d+\.\d+\]", text, re.M)
            if released:
                scope = text[: released.start()]
            else:
                scope = ""
        if not scope:
            continue
        for mt in count_pat.finditer(scope):
            tok = mt.group(1).lower()
            claimed = words.get(tok)
            if claimed is None and tok.isdigit():
                claimed = int(tok)
            if claimed is None:
                problems.append(
                    f"{rel}: unparseable tool count {mt.group(0)!r} "
                    f"(should be {len(tools)})"
                )
            elif claimed != len(tools):
                problems.append(
                    f"{rel}: claims {claimed} agent tools, "
                    f"tool_schemas.py has {len(tools)}"
                )

    # every tool named anywhere in prose must be real
    for rel in PROSE_FILES:
        for t in set(re.findall(r"`(qdrant_\w+)`", read(rel))):
            if t not in tools:
                problems.append(f"{rel}: references unknown tool `{t}`")


# --- 2. hermes memory subcommands -------------------------------------------
sub = CORE / "hermes_cli/subcommands/memory.py"
if not sub.is_file():
    # Core genuinely absent (no hermes checkout): say so and verify the rest.
    # CI always provides the file, so this branch never runs there — and it is
    # a *printed* skip, never a silent one.
    print(
        f"SKIP — hermes core not found at {sub}; CLI-surface claims were NOT "
        f"verified (set HERMES_CORE to a hermes-agent checkout)"
    )
else:
    src = sub.read_text(encoding="utf-8")
    body = src[src.index("memory_sub = memory_sub.add_subparsers") :] if False else src
    known = set(re.findall(r'memory_sub\.add_parser\(\s*"(\w+)"', src))
    if not known:
        known = set(re.findall(r'add_parser\(\s*"(setup|status|off|reset)"', src))
    print(f"hermes memory subcommands found: {', '.join(sorted(known))}")

    # `hermes memory <word>` usages, in prose AND in source. teknium1's review
    # found `_setup.py` printing `hermes memory provider qdrant` to the user's
    # terminal — a subcommand that does not exist. The old gate only scanned
    # PROSE_FILES, so a lie our own code prints passed. User-facing strings live
    # in .py, so scan those too.
    #
    # A CHANGELOG entry quoting the bad string while reporting the fix is not a
    # false claim, so scan a window before AND after for an explicit denial
    # ("which is\n  not a subcommand", "used to tell the user", "Now `…`").
    for rel in [*PROSE_FILES, "_setup.py", "__init__.py"]:
        text = read(rel)
        for mt in re.finditer(r"hermes memory ([a-z][a-z-]*)", text):
            cmd = mt.group(1)
            if cmd in ("setup", "status", "off", "reset"):
                continue
            if cmd in known:
                continue
            start = max(0, mt.start() - 200)
            seg = text[start : mt.start() + 200].replace("\n", " ")
            if re.search(
                r"\bnot a subcommand\b|\bno such\b|\bdoes not exist\b|"
                r"\bused to\b|\bpreviously\b|\bwas wrong\b|\bfixed\b",
                seg,
                re.I,
            ):
                continue
            line_no = text[: mt.start()].count("\n") + 1
            problems.append(
                f"{rel}:{line_no}: prints `hermes memory {cmd}`, "
                f"which is not a subcommand (real: {', '.join(sorted(known))})"
            )
        # the --provider flag does not exist on setup
        for _ in re.finditer(r"hermes memory setup\s+--provider", text):
            problems.append(
                f"{rel}: `hermes memory setup --provider` — "
                "provider is POSITIONAL, not a flag"
            )


# --- 3. CLI claims ----------------------------------------------------------
init = read("__init__.py")
registers_cli = bool(re.search(r"register_cli|cli_command", init))
print(f"plugin registers a CLI: {registers_cli}")
for rel in PROSE_FILES:
    text = read(rel)
    if not registers_cli:
        for pat in (r"hermes qdrant [a-z]", r"`hermes qdrant`"):
            for mt in re.finditer(pat, text):
                # Allow an explicit denial or a clearly hypothetical mention:
                # "there is no `hermes qdrant` CLI", "if we ever want a handful
                # of `hermes qdrant` subcommands — not 34". Scan a wide window
                # back to the start of the sentence.
                start = max(0, mt.start() - 160)
                seg = text[start : mt.start() + 40].replace("\n", " ")
                if re.search(
                    r"\bno\b|\bnot\b|\bnever\b|does not|registers none|if we ever|"
                    r"hypothetical|should be|would be|rather than",
                    seg,
                    re.I,
                ):
                    continue
                problems.append(
                    f"{rel}: claims a `hermes qdrant` CLI "
                    "that does not exist"
                )


# --- 5. Dependency floors quoted anywhere must match pyproject --------------
# teknium1's review found the standalone _setup.py still telling users to
# `pip install qdrant-client>=1.14.0`, which contradicts our own pyproject
# floor. A stale pin shown to a user at install time is worse than none.
pyproject_src = read("pyproject.toml")
floors: dict[str, str] = {}
for dep in re.findall(r'^\s*"([A-Za-z0-9_.-]+)([^"]*)"', pyproject_src, re.M):
    floors[dep[0].lower()] = dep[1].strip().strip(",")
for rel in ("_setup.py", "README.md"):
    text = read(rel)
    for mt in re.finditer(r"([a-z0-9_-]+)([><=!~0-9.,]*)", text):
        pkg = mt.group(1).lower()
        spec = mt.group(2).strip().rstrip(",")
        stale = (
            pkg in floors and spec and spec != floors[pkg]
            and spec.startswith((">=", "<"))
        )
        if stale:
            line_no = text[: mt.start()].count("\n") + 1
            problems.append(
                f"{rel}:{line_no}: quotes {pkg}{spec} but pyproject declares "
                f"{pkg}{floors[pkg]}"
            )


# --- report -----------------------------------------------------------------
print()
if problems:
    print(f"FAIL — {len(problems)} docs-honesty problem(s):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("docs honesty gate: clean")
sys.exit(0)
