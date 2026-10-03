"""KNOWLEDGE INDEX: heading-aware markdown chunker + FTS5 lexical index.

This module is the *fast path* of doc search and is deliberately independent of
both the embedding model and Qdrant. A ``md_search`` query that the lexical
index answers must never load fastembed: the whole point of FTS5-first is a
millisecond answer at ~0 RAM, and a 240 MB model load on the gateway's hot path
would defeat it. The model is touched only by the semantic fallback (see
``mdsearch.semantic`` usage in the provider) and by the ingest CLI.

Design notes that are not obvious from the code:

* **State lives outside the plugin dir.** ``<base home>/state/md-search/``.
  Hermes hashes every file inside a plugin member dir into the workspace
  dependency stamp (``pm.workspace.members_stamp()`` — its exclude list names
  only ``.git``/``.venv``/``venv``/``node_modules``/``__pycache__``), so an
  index file written next to ``__file__`` re-syncs dependencies on every launch.
  Same rule as ``status.json`` / ``qdrant.json``.

* **The FTS5 state is a cache, not a database.** Everything in it is derived
  from the ``.md`` files, which are the real source of truth. Losing the file
  costs a full re-ingest, never data.

* **``bm25`` + ``unicode61 remove_diacritics 2``** is the tokenizer choice, not
  a default: remove_diacritics 2 makes ``Grüße`` match a ``grusse`` query, and
  bm25 is the ranking function SQLite ships with.

* **Tokenizer note for the CJK languages in the vault**: ``unicode61`` treats
  Han/Kana runs as one token, so a single ``\u30c6\u30b9\u30c8`` term matches a
  whole run but not an arbitrary substring. This is a lexical-index property,
  not a bug; the semantic fallback covers those queries.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger("hermes.plugins.memory.qdrant.mdsearch")

# ---------------------------------------------------------------------------
# Corpus configuration
# ---------------------------------------------------------------------------

#: Default corpus roots. Overridable via the ``md_docs_roots`` config key (see
#: :func:`configured_roots`) — each entry is ``(root path, set of directory
#: names never descended into)``.
DEFAULT_ROOTS: dict[str, tuple[str, set[str]]] = {
    "skills": (
        str(Path.home() / ".hermes" / "skills"),
        {
            ".archive", ".hub", "__pycache__", ".pytest_cache", ".ruff_cache",
            ".curator_backups", "node_modules", ".git", ".idea", ".vscode",
        },
    ),
    "vault": (
        str(Path.home() / "Documents" / "hermes_default"),
        {".git", ".obsidian", ".trash", ".templates"},
    ),
    "docs": (
        str(Path.home() / ".hermes" / "hermes-agent" / "website"),
        {"node_modules", ".git", ".docusaurus", "build", "dist"},
    ),
}

#: Split a section whose body exceeds this many characters at blank lines.
MAX_CHUNK_CHARS = 1400

#: Skip absurd files rather than chunking a build artifact that ends in .md.
MAX_FILE_BYTES = 5 * 1024 * 1024

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)")

#: The plugin member directory. Named so the test suite can assert that no index
#: state lands inside it, which is the venv-stamp rule this module must obey.
REPO_DIR = Path(__file__).resolve().parent


class CorpusFile(NamedTuple):
    """One indexed file: its stored label, its real path, and its size."""

    label: str    # "<root-label>/<relative path>" — the stable key
    path: str     # absolute path on disk
    size: int


def state_dir() -> Path:
    """``<base hermes home>/state/md-search`` — never the plugin member dir.

    See the module docstring for why. The ``md-search`` leaf is deliberately
    its own directory: ingest, FTS state and any future locks live together, and
    a profile that wants a private index gets it by pointing ``HERMES_HOME``
    elsewhere.
    """
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
        if home.parent.name == "profiles":
            home = home.parent.parent
    except Exception:
        home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return home / "state" / "md-search"


def db_path() -> Path:
    return state_dir() / "index.sqlite"


def connect() -> sqlite3.Connection:
    """Open (creating if needed) the index and ensure the schema exists."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=15.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS files(
               path TEXT PRIMARY KEY,
               root TEXT NOT NULL,
               sha TEXT NOT NULL,
               chunks INTEGER NOT NULL)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS fts(
               path TEXT NOT NULL, heading TEXT NOT NULL, content TEXT NOT NULL)"""
    )
    con.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS ftsx USING fts5(
               path, heading, content,
               tokenize='unicode61 remove_diacritics 2')"""
    )
    con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    return con


def index_is_present() -> bool:
    """True when an index file exists AND carries at least one indexed file."""
    if not db_path().is_file():
        return False
    try:
        con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return False
    try:
        row = con.execute("SELECT count(*) FROM files").fetchone()
        return bool(row and row[0])
    except sqlite3.Error:
        return False
    finally:
        con.close()


def read_meta(con: sqlite3.Connection, key: str, default: str = "") -> str:
    try:
        row = con.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return str(row[0]) if row and row[0] is not None else default
    except sqlite3.Error:
        return default


def write_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        "INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, value),
    )


# ---------------------------------------------------------------------------
# Corpus walking
# ---------------------------------------------------------------------------


def _qdrant_json() -> dict[str, Any]:
    """The provider's runtime config dict (``<HERMES_HOME>/qdrant.json``).

    Read defensively: the knowledge index must not be taken down by a malformed
    or absent config file, and it must never import the provider module (that
    would drag the whole memory plugin — and its Qdrant client — onto the
    FTS5 fast path, which is exactly what this module must never do).
    """
    import json

    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
        if home.parent.name == "profiles":
            home = home.parent.parent
    except Exception:
        env = os.environ.get("HERMES_HOME", "").strip()
        home = Path(env) if env else Path.home() / ".hermes"
    path = home / "qdrant.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def configured_roots() -> dict[str, tuple[str, set[str]]]:
    """The effective corpus roots: :data:`DEFAULT_ROOTS` + ``md_docs_roots``.

    The ``md_docs_roots`` config key (in ``qdrant.json``) maps a root LABEL to a
    path, and is merged OVER the defaults::

        {"md_docs_roots": {"skills": "/home/me/.hermes/profiles/qd/skills",
                           "sessions": "/home/me/knowledge/qd-sessions"}}

    A label given here replaces that label's default path while KEEPING its
    skip set, so an override can widen a corpus without having to restate which
    directories to avoid. A label absent from :data:`DEFAULT_ROOTS` gets the
    default skip set (dot-directories are skipped regardless — see
    :func:`iter_markdown`). Non-string paths and non-dict values are ignored
    with a warning rather than raising: a typo in a config file must not make
    ``md_search`` raise.
    """
    roots = dict(DEFAULT_ROOTS)
    configured = _qdrant_json().get("md_docs_roots")
    if not isinstance(configured, dict):
        if configured is not None:
            logger.warning(
                "md-search: ignoring md_docs_roots of type %s (want an object "
                "of label -> path)",
                type(configured).__name__,
            )
        return roots
    for label, base in configured.items():
        if not isinstance(label, str) or not isinstance(base, str):
            logger.warning(
                "md-search: ignoring md_docs_roots entry %r=%r (want string "
                "label and string path)",
                label,
                base,
            )
            continue
        path = base.strip()
        if not path:
            logger.warning("md-search: ignoring empty md_docs_roots path for %r", label)
            continue
        roots[label] = (os.path.expanduser(path), roots.get(label, (None, set()))[1])
    return roots


def iter_markdown(roots: dict[str, tuple[str, set[str]]] | None = None
                  ) -> Iterator[CorpusFile]:
    """Yield every ``.md`` file under the configured roots.

    Two skip rules, deliberately both: an explicit ``skip`` set per root, AND
    any directory starting with a dot. The second catches roots nobody enumerated
    exclusions for, which is how a ``.venv`` of vendored readmes silently
    doubles the corpus.
    """
    effective = roots if roots is not None else configured_roots()
    for label, (base, skip) in effective.items():
        base_path = Path(base).expanduser()
        if not base_path.is_dir():
            logger.warning(
                "md-search root %r (%s) is not a directory; skipping", label, base
            )
            continue
        for dirpath, dirnames, filenames in os.walk(base_path):
            dirnames[:] = [
                d for d in dirnames if d not in skip and not d.startswith(".")
            ]
            for filename in sorted(filenames):
                if not filename.endswith(".md"):
                    continue
                full = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                if size > MAX_FILE_BYTES:
                    logger.debug("md-search skipping %s: %d bytes > cap", full, size)
                    continue
                rel = os.path.relpath(full, base_path)
                yield CorpusFile(f"{label}/{rel}", full, size)


def sha_of(path: str) -> str:
    """SHA-256 of a file's bytes — the incremental-ingest change detector."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
    except OSError as exc:
        logger.warning("md-search could not hash %s: %s", path, exc)
        return ""
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, fallback_label: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading_path, content)`` chunks.

    Heading-aware, because a chunk that starts mid-section is much less useful
    than one that carries its own breadcrumb. The heading path is the ``>``
    -joined chain of enclosing headings (``"Guide > Install > Linux"``), so a
    hit reports where it came from without the reader opening the file.

    Frontmatter is stripped: YAML keys are not prose and would otherwise rank
    above real content for keyword queries.

    A section longer than ``MAX_CHUNK_CHARS`` is split on blank lines and
    packed greedily, so a chunk stays a run of whole paragraphs.
    """
    raw = text
    text = _FRONTMATTER.sub("", text, count=1)
    sections: list[tuple[str, str]] = []
    stack: dict[int, str] = {}
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            if "".join(body).strip():
                sections.append((heading, "".join(body).strip()))
            level, title = len(m.group(1)), m.group(2).strip()
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = title
            heading = " > ".join(stack[k] for k in sorted(stack))
            body = []
        else:
            body.append(line + "\n")
    if "".join(body).strip():
        sections.append((heading, "".join(body).strip()))

    out: list[tuple[str, str]] = []
    for heading_path, body_text in sections:
        label = heading_path or fallback_label
        if len(body_text) <= MAX_CHUNK_CHARS:
            out.append((label, body_text))
            continue
        current = ""
        for para in (p for p in re.split(r"\n\s*\n", body_text) if p.strip()):
            if current and len(current) + len(para) + 2 > MAX_CHUNK_CHARS:
                out.append((label, current.strip()))
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current.strip():
            out.append((label, current.strip()))
    # A file of nothing but frontmatter/headings still deserves one entry, or
    # it would be in `files` but unfindable — and "indexed but unreachable"
    # is exactly the kind of quiet hole this gate culture exists to prevent.
    #
    # The fallback content is the RAW text, frontmatter included: a file whose
    # entire substance is its YAML block (skills' `DESCRIPTION.md` — 20 of
    # them) strips to an empty string, which the embedder rejects outright and
    # which FTS5 matches nothing against. The frontmatter description is the
    # only prose such a file has, so keeping it is both non-empty and useful.
    fallback = (raw or "").strip()[:MAX_CHUNK_CHARS]
    return out or [(fallback_label, fallback or fallback_label)]


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


def fts_query(text: str) -> str | None:
    """Turn free user input into a safe FTS5 OR-expression, or None.

    Tokens are extracted with ``\\w+`` and quoted, so FTS5 operators a user
    types (``AND``, ``*``, ``NEAR``, unbalanced quotes) are data, not syntax —
    an injection would otherwise be a syntax error the caller sees as "no
    results". OR rather than AND: a multi-word question should return the chunks
    that match *any* term, ranked by bm25; AND would silently return nothing
    the moment one rare word is missing.
    """
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    if not tokens:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


class LexicalHit(NamedTuple):
    path: str
    heading: str
    snippet: str
    score: float


def search_lexical(query: str, limit: int = 5, *, con: sqlite3.Connection | None = None,
                   root: str = "") -> list[LexicalHit]:
    """bm25-ranked lexical search. Opens the index read-only if not given a connection.

    This function must never import an embedding model — the FTS5-first promise
    is that a lexical answer costs a sqlite query and nothing else.
    """
    match = fts_query(query)
    if not match:
        return []

    own = con is None
    if own:
        if not db_path().is_file():
            return []
        con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True, timeout=10.0)
    try:
        # NO table alias: SQLite resolves both ``MATCH`` and the ``bm25()``
        # argument against the TABLE name, not against a column alias (verified
        # on sqlite 3.53.1 — ``FROM ftsx AS f WHERE f MATCH ?`` fails with
        # "no such column: f" even though the projection can use ``f.path``).
        sql = (
            "SELECT ftsx.path, ftsx.heading, ftsx.content, bm25(ftsx) AS score "
            "FROM ftsx JOIN files m ON m.path = ftsx.path "
            "WHERE ftsx MATCH ? "
        )
        params: list[Any] = [match]
        if root:
            sql += " AND m.root = ? "
            params.append(root)
        # bm25() returns a NEGATIVE score (lower is better); ORDER BY ASC.
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = con.execute(sql, params).fetchall()
        return [
            LexicalHit(
                path=r[0],
                heading=r[1],
                snippet=snippet(r[2], query),
                score=float(r[3]),
            )
            for r in rows
        ]
    except sqlite3.Error as exc:
        logger.warning("md-search lexical query failed: %s", exc)
        return []
    finally:
        if own:
            con.close()


def snippet(content: str, query: str, width: int = 170) -> str:
    """A one-line excerpt centred on the first query term that occurs in it.

    FTS5 can rank a hit but cannot say *why* in a form a reader can use, so the
    caller gets the window around the term. Falls back to the head of the chunk
    when no term matches literally (bm25 matched via the stemmer, or the match
    was in the heading only).
    """
    lowered, text = content.casefold(), content
    pos = -1
    for token in re.findall(r"\w+", query, flags=re.UNICODE):
        pos = lowered.find(token.casefold())
        if pos >= 0:
            break
    if pos < 0:
        return text[:width].replace("\n", " ") + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    return (
        ("…" if start else "")
        + text[start:end].replace("\n", " ")
        + ("…" if end < len(text) else "")
    )


# ---------------------------------------------------------------------------
# Incremental ingest bookkeeping
# ---------------------------------------------------------------------------


def file_row(con: sqlite3.Connection, label: str) -> tuple[str, str, int] | None:
    row = con.execute(
        "SELECT root, sha, chunks FROM files WHERE path=?", (label,)
    ).fetchone()
    return (str(row[0]), str(row[1]), int(row[2])) if row else None


def replace_file_chunks(con: sqlite3.Connection, label: str, root: str, sha: str,
                        chunks: Iterable[tuple[str, str]]) -> int:
    """Drop any previous rows for ``label`` and insert the new chunks.

    Returns the count.

    Replace-by-label rather than diff: a changed file re-indexes wholesale. A
    per-chunk diff would need stable chunk identities across edits, and the win
    does not justify the failure mode where a stale chunk outlives its text.
    """
    con.execute("DELETE FROM ftsx WHERE path = ?", (label,))
    con.execute("DELETE FROM fts WHERE path = ?", (label,))
    count = 0
    for heading, content in chunks:
        cur = con.execute(
            "INSERT INTO fts(path, heading, content) VALUES(?, ?, ?)",
            (label, heading, content),
        )
        # The SAME rowid in both tables is what lets a full rebuild regenerate
        # ftsx from fts alone (see rebuild_fts_from), so it is not optional.
        con.execute(
            "INSERT INTO ftsx(rowid, path, heading, content) VALUES(?, ?, ?, ?)",
            (cur.lastrowid, label, heading, content),
        )
        count += 1
    con.execute(
        "INSERT INTO files(path, root, sha, chunks) VALUES(?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET root=excluded.root, sha=excluded.sha, "
        "chunks=excluded.chunks",
        (label, root, sha, count),
    )
    return count


def rebuild_fts_from(con: sqlite3.Connection) -> None:
    """Regenerate the FTS5 index from the plain ``fts`` table.

    ``fts`` is the copy we can always rebuild from, ``ftsx`` is the one bm25()
    needs. They are written together with a shared rowid, so if the FTS5 side is
    ever suspect (a corrupted segment, a tokenizer upgrade after a Python
    upgrade changed unicode61's behaviour) this restores it without re-reading
    a single markdown file.
    """
    con.execute("DELETE FROM ftsx")
    con.execute(
        "INSERT INTO ftsx(rowid, path, heading, content) "
        "SELECT rowid, path, heading, content FROM fts"
    )
    con.commit()


def drop_file(con: sqlite3.Connection, label: str) -> None:
    con.execute("DELETE FROM ftsx WHERE path = ?", (label,))
    con.execute("DELETE FROM fts WHERE path = ?", (label,))
    con.execute("DELETE FROM files WHERE path = ?", (label,))


def labels_in_index(con: sqlite3.Connection) -> set[str]:
    return {str(r[0]) for r in con.execute("SELECT path FROM files")}
