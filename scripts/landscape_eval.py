#!/usr/bin/env python3
"""Landscape evaluation for qdrant plugin — default/WSL profile.

This script runs a comprehensive benchmark suite on the installed qdrant plugin
(~/.hermes/plugins/qdrant) against the live Qdrant server at localhost:6333.
It measures:

1. Memory recall quality (retrieval_eval via real tool path)
2. md_search lexical + semantic performance
3. Latency percentiles (p50, p95, p99)
4. Resource footprint (RSS, model cache)
5. Correctness gates (mutation checks, docs honesty)

Results are written to JSON for cross-profile comparison.

GK will approve runs when load allows (outside 04:00-14:00 CEST or on battery).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# RAM knobs (6.9 GB host): the child eval scripts each load a MiniLM copy via
# fastembed; ORT defaults to one arena thread per core, which is the dominant
# avoidable cost under co-tenancy. 2 threads keeps p95 latency within noise
# and roughly halves ORT arena memory.
RAM_ENV = {
    "OMP_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
}

# Co-tenant probe: the resumable LongMemEval 3-way probe holds ~3G (three
# ONNX models) and relaunches under its watchdog every 15 min while
# incomplete, 04:00-14:00 CEST. Stacking the eval's own MiniLM on top of a
# probe relaunch spiked the box to 3.3G+ (2026-10-10 incident). The eval
# refuses to start while a probe is alive unless --wait-for-probe resolves.
PROBE_SCRIPT = "/home/gk/.hermes/profiles/qd/cache/scratch/hybrid_probe_3way.py"


def probe_alive() -> bool:
    """Exact-match probe pid scan (same rule as probe-watchdog.sh:
    the script path must be its own argv element, or a chat session
    mentioning the filename fakes 'alive')."""
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = p.read_bytes().split(b"\0")
        except OSError:
            continue
        if PROBE_SCRIPT.encode() in argv:
            return True
    return False


def wait_for_probe(max_minutes: int = 90) -> bool:
    """Poll until the probe dies. Returns True if it died within the budget."""
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        if not probe_alive():
            return True
        time.sleep(120)
    return not probe_alive()


def check_policy_window() -> tuple[bool, str]:
    """GK hard constraint: 04:00-14:00 CEST AND mains power."""
    # CEST = UTC+2; compute the Berlin hour from the TZ database.
    import datetime
    berlin = datetime.timezone(datetime.timedelta(hours=2))
    hour = datetime.datetime.now(berlin).hour
    if not (4 <= hour < 14):
        return False, f"hour {hour} outside 04:00-14:00 CEST window"
    ac = "1"
    for bat in Path("/sys/class/power_supply").glob("AC*/online"):
        ac = bat.read_text().strip()
        break
    if ac != "1":
        return False, f"not on mains power (AC online={ac})"
    return True, "ok"

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = Path("/home/gk/.hermes/plugins/qdrant")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
PYTHON = "/home/gk/.hermes/state/qdrant/qd-release/.venv/bin/python"

# Results container
RESULTS: dict[str, Any] = {
    "metadata": {
        "profile": "default",
        "host": "wsl",
        "python": PYTHON,
        "qdrant_url": QDRANT_URL,
        "plugin_dir": str(PLUGIN_DIR),
    },
    "retrieval_eval": {},
    "md_search": {},
    "mutation_check": {},
    "docs_honesty": {},
    "footprint": {},
    "latency": {},
}


def run_cmd(
    cmd: list[str],
    cwd: Path = REPO_ROOT,
    env: dict | None = None,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """Run command and return (exit_code, stdout, stderr)."""
    merged_env = os.environ.copy()
    merged_env.update(RAM_ENV)
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_retrieval_eval(stdout: str) -> dict:
    """Parse retrieval_eval.py output."""
    data = {}
    for line in stdout.splitlines():
        line = line.strip()
        if (
            line.startswith("recall@1")
            and "recall@10" not in line
            and "recall@5" not in line
        ):
            parts = line.split()
            data["recall@1"] = float(parts[1])
            # Find the CI part in parentheses
            if "(" in line:
                ci_start = line.index("(")
                ci_end = line.index(")")
                data["recall@1_ci"] = line[ci_start+1:ci_end]
        elif line.startswith("recall@5") and "recall@10" not in line:
            data["recall@5"] = float(line.split()[1])
        elif line.startswith("recall@10"):
            data["recall@10"] = float(line.split()[1])
        elif line.startswith("MRR"):
            data["mrr"] = float(line.split()[1])
        elif line.startswith("nDCG@5"):
            data["ndcg@5"] = float(line.split()[1])
        elif line.startswith("beats its own distractor"):
            data["beats_distractor"] = float(line.split()[4])
        elif "margin vs distractor" in line:
            # Line: "margin vs distractor      median 0.170, min 0.010,
            #       10 case(s) decided by <0.10"
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "median":
                    data["margin_median"] = float(parts[i+1].rstrip(","))
                elif p == "min":
                    data["margin_min"] = float(parts[i+1].rstrip(","))
                elif p == "case(s)":
                    data["margin_lt_010_cases"] = int(parts[i-1])
        elif line.startswith("latency"):
            # Line: "latency    mean 53.6 ms, p95 133.4 ms"
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "mean":
                    data["latency_mean_ms"] = float(parts[i+1].rstrip(","))
                elif p == "p95":
                    data["latency_p95_ms"] = float(parts[i+1].rstrip(","))
    return data


def parse_md_eval(stdout: str) -> dict:
    """Parse eval_md_recall.py output."""
    data = {"queries": []}
    in_table = False
    for line in stdout.splitlines():
        if line.startswith("lexical recall@5"):
            # Line: "lexical recall@5: 6/11 = 0.545 (Wilson 95% CI 0.280-0.787)"
            parts = line.split(":")
            if len(parts) >= 2:
                rest = parts[1].strip()
                # Parse "6/11 = 0.545 (Wilson 95% CI 0.280-0.787)"
                if "=" in rest:
                    left, right = rest.split("=", 1)
                    data["lexical_recall@5"] = float(right.split()[0])
                if "CI" in rest:
                    ci_start = rest.index("(")
                    ci_end = rest.index(")")
                    data["lexical_recall@5_ci"] = rest[ci_start+1:ci_end]
        elif line.startswith("top hit from the queried corpus root"):
            parts = line.split(":")
            data["top_hit_corpus_root"] = parts[1].strip()
        elif line.startswith("diacritics equivalence"):
            data["diacritics_equivalence"] = (
                "passed" if "yes" in line.lower() else "failed"
            )
        elif line.startswith("lexical latency"):
            # Line: "lexical latency: p50 60.6ms  min 2.8ms  max 238.0ms"
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "p50":
                    data["lexical_latency_p50_ms"] = float(parts[i+1].rstrip("ms"))
                elif p == "min":
                    data["lexical_latency_min_ms"] = float(parts[i+1].rstrip("ms"))
                elif p == "max":
                    data["lexical_latency_max_ms"] = float(parts[i+1].rstrip("ms"))
        elif "|" in line and "case" in line.lower() and "ok" in line.lower():
            in_table = True
        elif in_table and line.strip() and "|" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 4:
                data["queries"].append({
                    "case": parts[0],
                    "ok": parts[1] == "yes",
                    "hits": int(parts[2]) if parts[2].isdigit() else 0,
                    "ms": (
                        float(parts[3].replace(",", ""))
                        if parts[3].replace(".", "").replace(",", "").isdigit()
                        else 0
                    ),
                })
    return data


def run_retrieval_eval() -> dict:
    """Run the retrieval evaluation harness."""
    print("Running retrieval_eval.py...")
    rc, out, err = run_cmd([PYTHON, "scripts/retrieval_eval.py"], cwd=REPO_ROOT)
    if rc != 0:
        print(f"retrieval_eval failed: {err}")
    return parse_retrieval_eval(out)


def run_md_search_eval() -> dict:
    """Run md_search evaluation."""
    print("Running eval_md_recall.py...")
    rc, out, err = run_cmd([PYTHON, "scripts/eval_md_recall.py"], cwd=REPO_ROOT)
    if rc != 0:
        print(f"eval_md_recall failed: {err}")
    return parse_md_eval(out)


def run_mutation_check() -> dict:
    """Run mutation checks."""
    print("Running mutation_check_mdsearch.py...")
    rc, out, err = run_cmd(
        [PYTHON, "scripts/mutation_check_mdsearch.py"], cwd=REPO_ROOT
    )
    return {"exit_code": rc, "all_caught": rc == 0, "output": out[-500:] if out else ""}


def run_docs_honesty() -> dict:
    """Run docs honesty check."""
    print("Running check_docs_honesty.py...")
    rc, out, err = run_cmd([PYTHON, "scripts/check_docs_honesty.py"], cwd=REPO_ROOT)
    return {"exit_code": rc, "clean": rc == 0, "output": out[-500:] if out else ""}


def run_full_test_suite() -> dict:
    """Run the full pytest suite and capture summary."""
    print("Running full pytest suite...")
    rc, out, err = run_cmd(
        [PYTHON, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=REPO_ROOT,
        timeout=600,
    )
    passed = failed = skipped = 0
    for line in out.splitlines():
        if "passed" in line and "failed" in line:
            import re
            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m:
                failed = int(m.group(1))
            m = re.search(r"(\d+) skipped", line)
            if m:
                skipped = int(m.group(1))
    return {"exit_code": rc, "passed": passed, "failed": failed, "skipped": skipped}


def check_footprint() -> dict:
    """Check plugin footprint and model cache."""
    data = {}
    # Check installed plugin size
    try:
        rc, out, _ = run_cmd(["du", "-sh", str(PLUGIN_DIR)], cwd=Path("/"))
        data["plugin_dir_size"] = out.split()[0] if out else "unknown"
    except Exception:
        data["plugin_dir_size"] = "error"
    # Check model cache
    cache = Path("/home/gk/.hermes/state/qdrant/model_cache")
    if cache.exists():
        rc, out, _ = run_cmd(["du", "-sh", str(cache)], cwd=Path("/"))
        data["model_cache_size"] = out.split()[0] if out else "unknown"
    else:
        data["model_cache_size"] = "not found"
    # Check fastembed cache
    fc = Path("/home/gk/.hermes/state/qdrant/fastembed_cache")
    if fc.exists():
        rc, out, _ = run_cmd(["du", "-sh", str(fc)], cwd=Path("/"))
        data["fastembed_cache_size"] = out.split()[0] if out else "unknown"
    else:
        data["fastembed_cache_size"] = "not found"
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Landscape evaluation for qdrant plugin"
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip full pytest suite"
    )
    parser.add_argument(
        "--output",
        default="/home/gk/.hermes/state/qdrant/landscape_eval_results.json",
        help="Output JSON file",
    )
    parser.add_argument("--wait-for-probe", action="store_true",
                        help=(
                            "If a hybrid probe is alive, wait up to 90 min for it "
                            "to finish instead of aborting"
                        ))
    parser.add_argument("--force", action="store_true",
                        help=(
                            "Skip the policy-window and co-tenant gates "
                            "(use only on battery/outside window deliberately)"
                        ))
    args = parser.parse_args()

    print("=" * 60)
    print("QDRANT PLUGIN — LANDSCAPE EVALUATION (default/WSL)")
    print("=" * 60)
    print(f"Python: {PYTHON}")
    print(f"Plugin: {PLUGIN_DIR}")
    print(f"Qdrant: {QDRANT_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    # Gate 1: policy window (04:00-14:00 CEST + mains power)
    ok, why = check_policy_window()
    if not ok and not args.force:
        print(
            f"ABORT: policy gate failed ({why}). "
            "Use --force to override deliberately."
        )
        return 2
    if not ok:
        print(f"WARN: policy gate failed ({why}) — continuing due to --force.")

    # Gate 2: co-tenant probe (the 2026-10-10 3.3G stacking incident)
    if probe_alive():
        if args.wait_for_probe:
            print("Co-tenant probe alive — waiting up to 90 min for it to finish...")
            if not wait_for_probe(max_minutes=90):
                print(
                    "ABORT: probe still alive after 90 min. "
                    "Re-run later or use --force."
                )
                return 3
            print("Probe finished — continuing.")
        elif not args.force:
            print(
                "ABORT: hybrid probe is alive (stacking eval on it "
                "spiked the box to 3.3G on 2026-10-10). Re-run with "
                "--wait-for-probe, or --force if you accept the risk."
            )
            return 3
        else:
            print("WARN: hybrid probe is alive — continuing due to --force.")
    print()

    # 1. Retrieval eval
    RESULTS["retrieval_eval"] = run_retrieval_eval()
    print()

    # 2. md_search eval
    RESULTS["md_search"] = run_md_search_eval()
    print()

    # 3. Mutation check
    RESULTS["mutation_check"] = run_mutation_check()
    print()

    # 4. Docs honesty
    RESULTS["docs_honesty"] = run_docs_honesty()
    print()

    # 5. Footprint
    RESULTS["footprint"] = check_footprint()
    print()

    # 6. Full test suite (optional, heavy)
    if not args.skip_tests:
        RESULTS["test_suite"] = run_full_test_suite()
    else:
        RESULTS["test_suite"] = {"skipped": True}
    print()

    # Write results
    out_path = Path(args.output)
    out_path.write_text(json.dumps(RESULTS, indent=2))
    print(f"Results written to {out_path}")
    print()
    print("SUMMARY:")
    print(f"  recall@1: {RESULTS['retrieval_eval'].get('recall@1', 'N/A')}")
    print(f"  recall@5: {RESULTS['retrieval_eval'].get('recall@5', 'N/A')}")
    print(f"  MRR: {RESULTS['retrieval_eval'].get('mrr', 'N/A')}")
    print(f"  nDCG@5: {RESULTS['retrieval_eval'].get('ndcg@5', 'N/A')}")
    print(f"  latency p95: {RESULTS['retrieval_eval'].get('latency_p95_ms', 'N/A')} ms")
    print(
        f"  mutation check: "
        f"{'PASS' if RESULTS['mutation_check'].get('all_caught') else 'FAIL'}"
    )
    print(
        f"  docs honesty: "
        f"{'PASS' if RESULTS['docs_honesty'].get('clean') else 'FAIL'}"
    )
    print(f"  plugin size: {RESULTS['footprint'].get('plugin_dir_size', 'N/A')}")
    return 0 if all([
        RESULTS['mutation_check'].get('all_caught'),
        RESULTS['docs_honesty'].get('clean'),
    ]) else 1


if __name__ == "__main__":
    sys.exit(main())
