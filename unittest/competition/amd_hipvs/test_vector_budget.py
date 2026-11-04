#!/usr/bin/env python3
# Copyright (c) 2026 OceanBase.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Exercise GPU snapshot staging under an enforced vector-memory budget."""
import argparse
import gzip
import json
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import benchmark as bench


MEMORY_SQL = ("SELECT vector_mem_used,vector_mem_hold,vector_mem_limit "
              "FROM oceanbase.__all_virtual_vector_mem_info;")
TABLE = "vector_budget_rows"
MIB = 1024 * 1024


def memory(server, phase, destination):
    text, seconds = server.sql(MEMORY_SQL, database=False, timeout=15, record=False)
    rows = [list(map(int, line.split("\t"))) for line in text.splitlines()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"unexpected vector memory VT result: {text!r}")
    used, hold, limit = rows[0]
    sample = {"at": bench.now(), "monotonic": time.monotonic(), "phase": phase,
              "used": used, "hold": hold, "limit": limit, "client_seconds": seconds}
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample) + "\n")
    return sample


def set_limit(server, value, samples, predicate, timeout=60):
    server.sql(f"ALTER SYSTEM SET vector_memory_limit='{value}';", database=False)
    deadline = time.monotonic() + timeout
    while True:
        sample = memory(server, f"set_limit_{value}", samples)
        if predicate(sample):
            return sample
        if time.monotonic() >= deadline:
            raise RuntimeError(f"vector memory limit did not reach expected value: {sample}")
        time.sleep(0.2)


def create_with_samples(server, sql, samples, phase, timeout, persist):
    result = {"started_at": bench.now(), "sql": sql, "samples": []}
    done = threading.Event()

    def create():
        started = time.perf_counter()
        try:
            result["stdout"], result["client_seconds"] = server.sql(sql, timeout=timeout)
            result["success"] = True
        except BaseException as error:
            result.update({"success": False, "error": repr(error),
                           "exception_type": type(error).__name__})
        finally:
            result["elapsed_seconds"] = time.perf_counter() - started
            result["finished_at"] = bench.now()
            done.set()

    thread = threading.Thread(target=create, name=phase, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout + 20
    while True:
        result["samples"].append(memory(server, phase, samples))
        if len(result["samples"]) % 10 == 0:
            persist()
        if done.wait(0.2):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{phase} did not terminate within bounded CREATE timeout")
    thread.join(timeout=1)
    result["samples"].append(memory(server, phase + "_finished", samples))
    return result


def evidence(server, checkpoint, destination):
    # The memory warning is deliberately collected from the raw log archive:
    # benchmark's hipVS-only excerpt does not include allocator warnings.
    time.sleep(0.5)
    result = server.capture_logs(checkpoint, destination)
    warnings = []
    with gzip.open(result["raw_log_archive"], "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "Memory usage exceeds user limit" in line:
                warnings.append(line.strip())
    result["vector_limit_warning_lines"] = warnings
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=200000)
    parser.add_argument("--mysql", type=Path, default=Path(shutil.which("mysql") or "mysql"))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if sys.platform != "linux" or args.rows < 200000 or args.timeout < 60:
        parser.error("require Linux, rows >= 200000, timeout >= 60")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": bench.now(), "rows": args.rows,
              "dim": 128, "port": 2883, "memory_sql": MEMORY_SQL,
              "memory_samples": str(args.output / "memory-samples.jsonl")}
    samples = args.output / "memory-samples.jsonl"
    server = None
    exit_code = 1

    def persist():
        if server is not None:
            report["server"] = server.describe()
        bench.write_json(args.output / "results.json", report)

    try:
        for name in ("binary", "helper", "mysql"):
            setattr(args, name, getattr(args, name).resolve(strict=True))
        report["artifacts"] = {name: {"path": str(getattr(args, name)),
                                      "sha256": bench.sha256(getattr(args, name))}
                               for name in ("binary", "helper")}
        root = Path(tempfile.mkdtemp(prefix="seekdb-vector-budget-", dir="/root"))
        report["instance_root"] = str(root)
        parameters = {"m": 16, "ef_construction": 200, "ef_search": 1000}
        bench.PARAMETERS = {**bench.PARAMETERS, "memory_budget": "8G"}
        report["server_parameters"] = dict(bench.PARAMETERS)
        server = bench.Server("gpu", 2883, args.binary, args.helper, args.mysql,
                              root / "server", args.output / "server", parameters)
        persist()
        server.start()
        server.sql(f"CREATE DATABASE {bench.DATABASE};", database=False)
        report["effective_configuration"] = {}
        for parameter in ("memory_budget", "memory_limit", "vector_memory_limit"):
            report["effective_configuration"][parameter] = server.sql(
                f"SHOW PARAMETERS LIKE '{parameter}';", database=False)[0]
        initial = memory(server, "initial", samples)
        server.sql("ALTER SYSTEM SET ob_vector_memory_limit_percentage=1;", database=False)
        report["compatibility_percentage_1_memory"] = memory(server, "percentage_1", samples)
        # Current memory config resolves vector_memory_limit directly. Record
        # the compatibility percentage setting, then enforce a byte budget.
        low = set_limit(server, "80M", samples, lambda s: s["limit"] == 80 * MIB)
        report["initial_memory"], report["low_limit_memory"] = initial, low
        persist()
        x, ids, queries = bench.dataset(args.rows, 128, 1, 42)
        import_path = args.output / "import.sql"
        bench.write_import(import_path, TABLE, x, ids)
        report["input"] = {"seed": 42, "sql_sha256": bench.sha256(import_path),
                           "id_min": int(ids.min()), "id_max": int(ids.max())}
        server.sql(f"CREATE TABLE {TABLE}(id BIGINT PRIMARY KEY,emb VECTOR(128));")
        _, report["import_seconds"] = server.sql(path=import_path, timeout=args.timeout)
        report["imported_rows"] = int(server.sql(f"SELECT COUNT(*) FROM {TABLE};")[0].strip())
        if report["imported_rows"] != args.rows:
            raise RuntimeError("imported row count differs")
        baseline = memory(server, "before_low_create", samples)
        report["baseline_memory"] = baseline
        checkpoint = server.checkpoint()
        create_sql = bench.create_index_sql(TABLE, parameters)
        (args.output / "create.sql").write_text(create_sql + "\n", encoding="utf-8")
        report["low_create"] = create_with_samples(server, create_sql, samples, "low_create",
                                                    args.timeout, persist)
        low_create = report["low_create"]
        low_create["evidence"] = evidence(server, checkpoint, args.output / "low-create")
        low_create["peak_tracked_used"] = max(s["used"] for s in low_create["samples"])
        low_create["peak_tracked_growth"] = low_create["peak_tracked_used"] - baseline["used"]
        persist()
        if low_create["success"]:
            raise RuntimeError("low-limit CREATE unexpectedly succeeded")
        memory_error = re.search(r"No memory or server runtime memory limit reached|"
                                 r"Vector index memory usage exceeds user defined limit|"
                                 r"ERROR (?:4013|7603)\b", low_create.get("error", ""), re.I)
        if not memory_error or not low_create["evidence"]["vector_limit_warning_lines"]:
            raise RuntimeError("CREATE failed without recognized SQL memory error and allocator warning")
        if low_create["peak_tracked_growth"] < 32 * MIB:
            raise RuntimeError("sampling did not observe significant staging memory accounting")
        if low_create["evidence"]["helper_records"] or low_create["evidence"]["completion_lines"]:
            raise RuntimeError("low-limit CREATE reached successful hipVS construction")
        report["rows_after_failure"] = int(server.sql(f"SELECT COUNT(*) FROM {TABLE};")[0].strip())
        failed_plan = server.sql("EXPLAIN " + bench.query_sql(TABLE, queries[0]))[0]
        (args.output / "failed-create-explain.txt").write_text(failed_plan, encoding="utf-8")
        report["index_available_after_failure"] = bench.index_plan_ok(failed_plan)
        if report["rows_after_failure"] != args.rows or report["index_available_after_failure"]:
            raise RuntimeError("failed CREATE changed data or exposed its index")
        try:
            server.sql(f"DROP INDEX {bench.INDEX} ON {TABLE};")
            report["drop_failed_index"] = {"success": True}
        except RuntimeError as error:
            if not re.search(r"ERROR 1091\b|Can't DROP.*check that column/key exists", str(error), re.I):
                raise
            report["drop_failed_index"] = {"absent": True, "error": str(error)}
        deadline = time.monotonic() + 180
        while True:
            released = memory(server, "after_failed_index_drop", samples)
            report["released_memory"] = released
            persist()
            if released["used"] <= baseline["used"] + 8 * MIB:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("failed snapshot staging memory was not released")
            time.sleep(1)
        server.sql("ALTER SYSTEM SET ob_vector_memory_limit_percentage=30;", database=False)
        report["restored_limit_memory"] = set_limit(
            server, "3G", samples, lambda s: s["limit"] == 3 * 1024 * MIB)
        checkpoint = server.checkpoint()
        report["recovery_create"] = create_with_samples(
            server, create_sql, samples, "recovery_create", args.timeout, persist)
        recovery = report["recovery_create"]
        recovery["evidence"] = evidence(server, checkpoint, args.output / "recovery-create")
        persist()
        if not recovery["success"]:
            raise RuntimeError("CREATE failed after restoring the vector budget")
        coverage = sum(int(r.get("rows", 0)) for r in recovery["evidence"]["helper_records"]
                       if int(r.get("dim", 0)) == 128)
        recovery["helper_rows_total"] = coverage
        if coverage < args.rows or not recovery["evidence"]["completion_lines"] or recovery["evidence"]["fallback_lines"]:
            raise RuntimeError("recovery CREATE lacks full GPU construction evidence")
        report["recovery_explain"] = bench.plan(server, TABLE, queries[0], args.output / "recovery-explain.txt")
        ann = bench.parse_rows(server.sql(bench.query_sql(TABLE, queries[0]))[0])
        oracle = bench.exact_topk(x, ids, queries)[0]
        report["recovery_ann"] = bench.assess(ann, queries[0], x,
                                            {int(label): row for row, label in enumerate(ids)}, oracle)
        report["rows_after_recovery"] = int(server.sql(f"SELECT COUNT(*) FROM {TABLE};")[0].strip())
        if report["rows_after_recovery"] != args.rows:
            raise RuntimeError("recovery changed data row count")
        report["status"] = "ok"
        exit_code = 0
    except BaseException as error:
        report.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        if server is not None:
            try:
                server.stop()
            except BaseException as error:
                report["cleanup_error"] = repr(error)
                report["status"] = "failed"
                exit_code = 1
            try:
                report["final_logs"] = server.capture_logs({}, args.output / "final-server")
            except BaseException as error:
                report["log_capture_error"] = repr(error)
                report["status"] = "failed"
                exit_code = 1
        report["finished_at"] = bench.now()
        persist()
        print(json.dumps({"status": report["status"], "results": str(args.output / "results.json")}), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
