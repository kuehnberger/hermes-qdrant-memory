#!/usr/bin/env bash
# Mutation check for the WIDENED docs-honesty checks (bare "<n> tools" count
# pattern + §1b peer-table re-measurement).
#
# Why this script exists: the shipped README said "We ship 6 tools" while the
# same file said "7 agent tools" 440 lines earlier, and check_docs_honesty.py
# stayed green because its count pattern required the literal word "agent"
# before "tools". A gate that has never failed is not evidence — this plants
# each defect it was widened for and requires a non-zero exit WITH the expected
# symptom, then restores and re-confirms clean.
#
# Operates on a throwaway COPY of the repo, never the working tree: reverting a
# file in place to run a mutation is how you lose the very test you are
# proving (see the plugin skill, "a mutation check can destroy the test").
#
# Each mutant starts from a FRESHLY EXTRACTED pristine copy rather than an
# in-place undo. The first version of this script restored by sed'ing the
# mutation back, and two of the five restorations silently did not apply — so
# mutant C reported a failure caused by mutant A's leftover, and the final
# "restored" run failed for reasons that had nothing to do with the check.
# `git checkout` is not available (the copy has no .git), and an undo that
# quietly no-ops is worse than no undo: re-extract instead.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
CORE="${HERMES_CORE:-/home/gk/.hermes/hermes-agent}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/docs-gate-mutation-XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

# Wipe and re-extract the working copy from the pristine tree.
reset_tree() {
  rm -rf "$WORK/repo"
  mkdir -p "$WORK/repo"
  tar -C "$REPO" --exclude=.git --exclude=__pycache__ -cf - . \
    | tar -C "$WORK/repo" -xf -
}

run_gate() {
  ( cd "$WORK/repo" && \
    env -u PYTHONPATH HERMES_CORE="$CORE" "$PY" \
    scripts/check_docs_honesty.py 2>&1 )
}

# Run one mutant: $1 label, $2 expected-symptom regex, $3 mutation command.
# The mutation command runs with $WORK/repo as cwd.
run_mutant() {
  local label="$1" want="$2" mutate="$3"
  local out rc

  reset_tree
  echo "=== MUTANT $label ==="
  ( cd "$WORK/repo" && eval "$mutate" ) || {
    echo "!!! mutation $label did not apply"; return 1; }

  out="$(run_gate)"; rc=$?
  echo "$out" | sed -n '/^FAIL/,$p' | head -8

  if [ "$rc" -eq 0 ]; then
    echo "!!! MUTANT $label NOT CAUGHT (rc=0) — the gate does not guard this"
    return 1
  fi
  if ! echo "$out" | grep -qE "$want"; then
    echo "!!! MUTANT $label failed for the WRONG reason (want /$want/)"
    return 1
  fi
  echo "--- caught $label (rc=$rc)"
  return 0
}

status=0

echo "=== BASELINE (must be clean) ==="
reset_tree
base_out="$(run_gate)"; base_rc=$?
echo "$base_out" | tail -4
if [ "$base_rc" -ne 0 ]; then
  echo "!!! baseline is NOT clean — nothing below is meaningful"
  echo "$base_out"
  exit 1
fi

# A: the exact shipped defect — bare "We ship 6 tools", no "agent".
run_mutant A 'claims 6' \
  "sed -i 's/We ship 7 tools and no CLI/We ship 6 tools and no CLI/' README.md" \
  || status=1

# B: the correct-count line degraded (the shape the old pattern DID catch).
run_mutant B 'claims 6' \
  "sed -i 's/- \*\*7 agent tools\*\*/- **6 agent tools**/' README.md" \
  || status=1

# C: peer table Tools row reverted to 5.
run_mutant C "'Tools' row claims" \
  "sed -i 's/^| Tools | 7 | 7 |$/| Tools | 7 | 5 |/' docs/competitor-analysis-entropicmem.md" \
  || status=1

# D: peer table __init__.py line count reverted to the old figure.
run_mutant D "row claims" \
  "sed -i 's/| 1,740 |/| 1,449 |/' docs/competitor-analysis-entropicmem.md" \
  || status=1

# E: reverse rule-6 mismatch — a hook implemented in code, undeclared in the
# manifest, so the loader can never see it.
run_mutant E 'implements 1' \
  "printf '\n\ndef post_setup(**kwargs) -> None:\n    \"\"\"mutant hook\"\"\"\n' >> __init__.py" \
  || status=1

# F: peer table Python-lines cell drifted (independent assertion on the same
# section: the LOC row is measured from the tree, not hardcoded).
run_mutant F "row claims" \
  "sed -i 's/| 3,763 (plugin/| 3,700 (plugin/' docs/competitor-analysis-entropicmem.md" \
  || status=1

# G: the CI shape — `uv sync` leaves a .venv/ inside the checkout AND the
# workflow sparse-checks the hermes core out to `hermes-core/` in the same
# workspace. Counting either as plugin source is what broke §1b twice in CI
# (measured 3,595,681, then 263,421) while every local run passed. Plant BOTH
# and the gate must still be clean: a measurement that moves when an untracked
# artifact appears is not a measurement.
echo "=== MUTANT G: CI workspace shape (.venv + nested core checkout) ==="
reset_tree
mkdir -p "$WORK/repo/.venv/lib/python3.13/site-packages/fake" \
         "$WORK/repo/hermes-core/hermes_cli/subcommands"
# A nested clone is skipped by SHAPE (has its own .git), so this must be a real
# one — and deliberately named something CI would not use, to prove the rule is
# not keyed on the literal "hermes-core".
git init -q "$WORK/repo/third-party-core"
mkdir -p "$WORK/repo/third-party-core/hermes_cli/subcommands"
cp "$CORE/hermes_cli/subcommands/memory.py" \
   "$WORK/repo/third-party-core/hermes_cli/subcommands/memory.py" 2>/dev/null
# actions/checkout gives hermes-core/ its own .git too — so make it a real
# checkout here, or the test is not reproducing CI.
git init -q "$WORK/repo/hermes-core"
for i in $(seq 1 200); do
  printf 'x = 1\n' > "$WORK/repo/.venv/lib/python3.13/site-packages/fake/mod$i.py"
done
for i in $(seq 1 300); do
  printf 'z = 3\n' > "$WORK/repo/hermes-core/hermes_cli/subcommands/m$i.py"
  printf 'z = 3\n' > "$WORK/repo/third-party-core/hermes_cli/subcommands/m$i.py"
done
g_out="$(run_gate)"; g_rc=$?
echo "$g_out" | grep -E 'peer-table|FAIL' | head -3
if [ "$g_rc" -ne 0 ]; then
  echo "!!! MUTANT G: gate failed in the CI workspace shape — the LOC count"
  echo "    still includes build artifacts or a nested checkout. This is"
  echo "    exactly the failure CI reported twice."
  status=1
elif ! echo "$g_out" | grep -q 'python lines=3,763'; then
  echo "!!! MUTANT G: count moved (expected python lines=3,763)"
  status=1
else
  echo "--- G passed: .venv and nested clone both ignored, count unmoved (rc=0)"
fi

echo
echo "=== RESTORED (fresh pristine copy must be clean) ==="
reset_tree
out="$(run_gate)"; rc=$?
echo "$out" | tail -3
if [ "$rc" -ne 0 ]; then
  echo "!!! restored copy is NOT clean — restoration unverified"
  echo "$out"
  status=1
else
  echo "--- restoration verified clean"
fi

echo
if [ "$status" -eq 0 ]; then
  echo "=== ALL MUTATIONS CAUGHT (6 plants + 1 CI-shape) + restoration verified ==="
else
  echo "=== MUTATION CHECK FAILED ==="
fi
exit "$status"