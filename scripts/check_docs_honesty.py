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
    #
    # The word "agent" in this pattern is what let README:465 ("We ship 6
    # tools") through: the phrasing that drifted had no "agent", so the
    # pattern never matched it and the gate stayed green on a README that
    # contradicted itself two hundred lines from its own correct claim. A
    # count claim is a count claim — match the bare "<n> tools" form too, with
    # an optional intervening qualifier, and let the existing prose carve-outs
    # (changelog history, "five tools" spelled out) do the rest.
    # The alternation must be QUANTITY TOKENS ONLY. An earlier version ended it
    # with `|\w+`, and that silently broke the whole check: on "through the
    # seven agent tools" it matched group(1)="the" and let the qualifier group
    # swallow "seven agent", so a correct README produced three bogus
    # "unparseable tool count" alarms while the real defect it was written for
    # still slipped past. The lesson is the shape of this section, not the
    # count: a pattern loose enough to match English prose is a pattern whose
    # capture group you no longer control. Match a quantity, an optional
    # qualifier, then "tools" — and let anything that is not a quantity simply
    # not match.
    count_pat = re.compile(
        r"\b(\d+|zero|no|none|one|two|three|four|five|six|seven|eight|nine|ten"
        r"|eleven|twelve)\s+((?:[a-z][a-z-]*\s+){0,3}?)tools\b",
        re.I,
    )
    words = {
        "zero": 0, "no": 0, "none": 0,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12,
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

    # --- 1b. The peer-analysis table's "Ours" column is re-measured -----------
    #
    # `docs/competitor-analysis-entropicmem.md` carries a scale table whose
    # "Ours" column is a pile of counts that were correct on 2026-10-02 and
    # silently stopped being true: Tools said 5, `__init__.py` said 1,449, the
    # test count said 134. Nothing flagged it, because the §1 count pattern
    # only reads "<n> agent tools" in prose and this table is pipe-delimited.
    #
    # So each row is re-measured from the tree and the cell must contain the
    # real figure. Measured, never frozen: the numbers are recomputed on every
    # run, so the table cannot drift without the gate noticing.
    peer_doc = "docs/competitor-analysis-entropicmem.md"
    peer_text = read(peer_doc)

    def _py_lines(exclude_prefixes: tuple[str, ...]) -> int:
        """Sum physical lines of plugin .py files outside `exclude_prefixes`."""
        total = 0
        for py in sorted(REPO.rglob("*.py")):
            relp = py.relative_to(REPO).as_posix()
            if "__pycache__" in relp or relp.startswith(exclude_prefixes):
                continue
            total += len(py.read_text(encoding="utf-8").splitlines())
        return total

    init_lines = len((REPO / "__init__.py").read_text(encoding="utf-8").splitlines())

    # Parse `provides_hooks:` as a YAML-ish block list, stopping at the first
    # line that is not a list item. Slicing to end-of-file (the obvious first
    # cut) would let a stray "- x" further down the manifest be counted as a
    # hook, so the terminator matters.
    manifest_text = read("plugin.yaml")
    hooks_declared = 0
    if "provides_hooks:" in manifest_text:
        block = manifest_text.split("provides_hooks:", 1)[1].splitlines()
        for ln in block:
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if not ln.lstrip().startswith("-"):
                break
            hooks_declared += 1
    # A hook the loader cannot see is the rule-6 mismatch in reverse, so count
    # what the CODE implements, not just what the manifest declares.
    hooks_impl = len(
        re.findall(
            r"^\s*def (?:pre|post)_setup|^\s*def on_\w+\(",
            read("__init__.py"),
            re.M,
        )
    )
    cli_commands = len(re.findall(r"add_parser\(", read("__init__.py")))
    # "Screenshots" means evidence of the tool WORKING, not any image in the
    # repo. `docs/banner.jpg` is a README banner asset and was the first thing
    # this check flagged: counting it would have made the gate demand that the
    # peer table admit to screenshots the project does not ship. Exclude
    # banner/OG assets by name, and say so in the table cell rather than
    # quietly widening the definition later.
    banner_names = {"banner.jpg", "banner.png", "og.png", "og.jpg"}
    screenshots = len(
        [
            p
            for p in REPO.glob("docs/**/*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
            and p.name.lower() not in banner_names
        ]
    )
    test_funcs = sum(
        len(re.findall(r"^\s*def test_", p.read_text(encoding="utf-8"), re.M))
        for p in sorted(REPO.glob("tests/*.py"))
    )

    measured: dict[str, list[str]] = {
        "python lines": [f"{_py_lines(('tests/', 'scripts/')):,}"],
        "`__init__.py`": [f"{init_lines:,}"],
        "tools": [str(len(tools))],
        "hooks": [str(hooks_declared)],
        "cli commands": [str(cli_commands)],
        "screenshots": [str(screenshots)],
        "tests": [f"{test_funcs:,}"],
    }
    print(
        "peer-table 'Ours' column re-measured: "
        + ", ".join(f"{k}={v[0]}" for k, v in measured.items())
        + f" (hooks also implemented in code: {hooks_impl})"
    )
    if hooks_declared != hooks_impl:
        problems.append(
            f"{peer_doc}: Hooks row says {hooks_declared} but __init__.py "
            f"implements {hooks_impl} — an implemented-but-undeclared hook is "
            "the same rule-6 mismatch as a declared-but-absent one"
        )

    if peer_text:
        for line in peer_text.splitlines():
            if not line.strip().startswith("|") or "---" in line:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 3:
                continue
            key, ours = cells[0].lower(), cells[2]
            if key in {"", "entropicmem", "ours"} or ours in {"", "—", "-", "n/a"}:
                continue
            for expected in measured.get(key, []):
                if expected not in ours:
                    problems.append(
                        f"{peer_doc}: '{cells[0]}' row claims {ours!r} for "
                        f"'Ours', measured {expected}"
                    )


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
