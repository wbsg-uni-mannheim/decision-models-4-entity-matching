"""Bounded confirmation recovery for three diagnosed transient provider failures.

Does not alter frozen inference code, requests, selection, or accounting.
At most three sequential confirmation invocations; only narrowly recognized provider
failure records permit a further invocation after the old process has exited.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
MAX_INVOCATIONS = 3


def retryable(records):
    """Only the observed Jev499, Cloudflare capacity429 and Luna wrapped502."""
    if not records:
        return False
    for record in records:
        attempts = record.get("attempts", [])
        if not attempts:
            return False
        for attempt in attempts:
            response = attempt.get("response") or {}
            error = response.get("error") or {}
            if not isinstance(error, dict):
                return False
            message = error.get("message", "")
            alias, status = record.get("model_alias"), attempt.get("status")
            jev = (alias == "jev" and status == 499 and attempt.get("error") == "HTTP 499"
                   and '\"error_type\":\"internal_error\"' in message)
            cloudflare = (alias == "clef_flash" and status == 429
                          and attempt.get("error") == "HTTP 429"
                          and "Capacity temporarily exceeded" in message)
            luna = (alias == "luna" and status == 200 and error.get("code") == 502
                    and (error.get("metadata") or {}).get("error_type") == "provider_unavailable"
                    and attempt.get("error") == "ValueError: Missing actual usage cost"
                    and attempt.get("cost") is None)
            if not (jev or cloudflare or luna):
                return False
    return True


def failures():
    return {str(p): [json.loads(line) for line in p.read_text().splitlines()]
            for p in (ROOT / "runs/quality").glob("*/*/*/*.failures.jsonl")
            if p.name in {"prompt.failures.jsonl", "test.failures.jsonl"}}


def delta(before, after):
    records = []
    for path, previous in before.items():
        assert after.get(path, [])[:len(previous)] == previous, "Failure journal changed"
    for path, rows in after.items():
        records.extend(rows[len(before.get(path, [])):])
    return records


def verify_ready():
    config = json.loads((ROOT / "configs/expanded_execution.json").read_text())
    assert int(os.environ["SLURM_JOB_ID"]) == config["e2_confirmation_job"], "Not current confirmation job"
    for filename, expected in config["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    budget = json.loads((ROOT / "artifacts/budget.json").read_text())
    assert abs(budget["reserved"]) < 1e-8, "Outstanding writer reservations"
    assert budget["spent"] + .05 <= 100, "Project spending ceiling reached"
    return budget


def main():
    job = os.environ["SLURM_JOB_ID"]
    journal = ROOT / "artifacts/operations" / ("confirmation_recovery_" + job + ".jsonl")
    with (ROOT / "artifacts/operations/confirmation_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not journal.exists(), "Supervisor already invoked; inspect journal before recovery"

        def log(event, **values):
            row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
            with journal.open("a") as out:
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                os.fsync(out.fileno())
            print(json.dumps(row), flush=True)

        for invocation in range(1, MAX_INVOCATIONS + 1):
            budget = verify_ready()
            before = failures()
            log("start", invocation=invocation, budget=budget)
            result = subprocess.run([sys.executable, "-u", "src/expanded_confirmation.py", "run"], cwd=ROOT)
            new = delta(before, failures())
            log("end", invocation=invocation, returncode=result.returncode,
                new_failures=new, budget=json.loads((ROOT / "artifacts/budget.json").read_text()))
            if result.returncode == 0:
                return 0
            if invocation == MAX_INVOCATIONS or not retryable(new):
                log("stopped_for_diagnosis", invocation=invocation)
                return 1
            # subprocess.run returned: the prior confirmation and its thread pools ended.
            verify_ready()
            log("cooldown", seconds=60)
            time.sleep(60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
