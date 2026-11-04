#!/usr/bin/env python3
"""Measure candidate seekdb CPU/hipVS CREATE VECTOR INDEX end to end.

Run on the Linux AMD host with Python 3 and numpy. Both instances execute the
same --binary. The output directory stores the inputs, SQL, individual query
results, plans, build logs, machine description, and an incrementally written
results.json. Server data lives in a unique directory under /root; its path is
recorded in results.json. Only processes started by this script are stopped.

--base-f32 and --queries-f32 select packed row-major little-endian float32 files
using --dim columns. Each size uses the first N base rows and --queries query
rows. Files have no header; the supplied dimension determines their row shape.
An omitted file uses its independent seeded random stream instead.
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np


INDEX = "hnsw_idx"
DATABASE = "amd_hipvs_benchmark"
K = 10
# ARRAY/VECTOR float display goes through OB_GCVT_ARG_FLOAT and FLT_DIG=6.
# Storage identity is checked independently with SQL against a .9g literal.
VECTOR_DISPLAY_RTOL = 5.1e-6
STORED_VECTOR_L2_TOLERANCE = 1e-7
PARAMETERS = {
    "memory_budget": "8G",
    "vector_memory_limit": "3G",
    "vector_index_memory_saving_mode": "True",
    "cpu_count": "16",
    "datafile_size": "2G",
    "datafile_maxsize": "8G",
    "log_disk_size": "2G",
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command_info(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"command": command, "exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": command, "error": str(e)}


def machine_info():
    paths = ["/proc/cpuinfo", "/proc/meminfo", "/sys/fs/cgroup/cpu.max",
             "/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/memory.max",
             "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"]
    files = {}
    for name in paths:
        try:
            files[name] = Path(name).read_text()
        except OSError as e:
            files[name] = {"error": str(e)}
    return {
        "uname": list(os.uname()), "python": sys.version, "numpy": np.__version__,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "files": files,
        "rocm_smi": command_info(["rocm-smi", "--showproductname", "--showuse", "--showmeminfo", "vram"]),
        "rocminfo": command_info(["rocminfo"]),
        "thread_environment": {k: os.environ.get(k) for k in
                               ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    }


def dataset(rows, dim, queries, seed):
    # Independent RNG streams keep the query workload independent of the rows.
    x = np.random.default_rng(np.random.SeedSequence([seed, rows, 0])).random(
        (rows, dim), dtype=np.float32)
    q = np.random.default_rng(np.random.SeedSequence([seed, rows, 1])).random(
        (queries, dim), dtype=np.float32)
    offsets = np.arange(rows, dtype=np.int64)
    ids = (np.int64(10**12) + offsets * np.int64(7919)) * np.where(offsets % 2, 1, -1)
    return x, ids.astype(np.int64), q


def load_f32(path, dim, required_rows):
    """Validate and map a raw float32 matrix using the declared dimension."""
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"float32 input is not a regular file: {path}")
    size = path.stat().st_size
    stride = 4 * dim
    if size == 0 or size % stride:
        raise ValueError(f"float32 input bytes {size} must be a positive multiple of 4*dim={stride}: {path}")
    rows = size // stride
    if rows < required_rows:
        raise ValueError(f"float32 input has {rows} rows, requires at least {required_rows}: {path}")
    matrix = np.memmap(path, mode="r", dtype="<f4", shape=(rows, dim))
    for start in range(0, rows, 8192):
        if not np.isfinite(matrix[start:start+8192]).all():
            raise ValueError(f"float32 input contains non-finite values: {path}")
    return matrix, {"path": str(path), "sha256": sha256(path), "bytes": size,
                    "dtype": "little-endian IEEE754 float32", "dim": dim, "rows": rows,
                    "layout": "packed row-major, no header"}


def input_dataset(rows, dim, queries, seed, base_matrix=None, query_matrix=None):
    x, ids, q = dataset(rows, dim, queries, seed)
    if base_matrix is not None:
        if base_matrix.shape[1] != dim or base_matrix.shape[0] < rows:
            raise ValueError("base matrix cannot supply the requested row/dimension shape")
        x = np.array(base_matrix[:rows], dtype=np.float32, copy=True)
    if query_matrix is not None:
        if query_matrix.shape[1] != dim or query_matrix.shape[0] < queries:
            raise ValueError("query matrix cannot supply the requested row/dimension shape")
        q = np.array(query_matrix[:queries], dtype=np.float32, copy=True)
    return x, ids, q


def vector_literal(v):
    # Nine significant digits round-trip each input float32 exactly.
    return "[" + ",".join(format(float(x), ".9g") for x in v) + "]"


def exact_topk(x, ids, queries):
    x64 = x.astype(np.float64)
    results = []
    for query in queries:
        diff = x64 - query.astype(np.float64)
        d2 = np.einsum("ij,ij->i", diff, diff)
        selected = np.argpartition(d2, K - 1)[:K]
        order = np.lexsort((ids[selected], d2[selected]))
        selected = selected[order]
        results.append({"ids": [int(ids[j]) for j in selected],
                        "distances": [float(math.sqrt(d2[j])) for j in selected]})
    return results


def write_import(path, table, x, ids):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for start in range(0, len(ids), 250):
            values = [f"({int(ids[j])},'{vector_literal(x[j])}')"
                      for j in range(start, min(start + 250, len(ids)))]
            f.write(f"INSERT INTO {table}(id,emb) VALUES " + ",".join(values) + ";\n")


def query_sql(table, query, k=K):
    literal = vector_literal(query)
    return (f"SELECT id,l2_distance(emb,'{literal}') AS distance FROM {table} "
            f"ORDER BY distance APPROXIMATE LIMIT {k};")


def create_index_sql(table, parameters):
    return (f"CREATE VECTOR INDEX {INDEX} ON {table}(emb) "
            f"WITH(distance=l2,type=hnsw,m={parameters['m']},"
            f"ef_construction={parameters['ef_construction']},ef_search={parameters['ef_search']});")


def parse_rows(stdout, expected=K):
    rows = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            raise RuntimeError(f"unexpected query row: {line!r}")
        label, distance = int(parts[0]), float(parts[1])
        if not math.isfinite(distance):
            raise RuntimeError("non-finite SQL distance")
        rows.append((label, distance))
    if len(rows) != expected or len({r[0] for r in rows}) != expected:
        raise RuntimeError(f"expected {expected} distinct ANN rows, got {rows!r}")
    if any(rows[i][1] > rows[i+1][1] + 1e-6 for i in range(len(rows)-1)):
        raise RuntimeError("SQL distances are not ordered")
    return rows


def assess(rows, query, x, id_to_row, oracle):
    labels = [r[0] for r in rows]
    if any(label not in id_to_row for label in labels):
        raise RuntimeError("ANN returned an ID absent from the input dataset")
    errors = []
    for label, distance in rows:
        vector = x[id_to_row[label]].astype(np.float64)
        exact = float(np.linalg.norm(vector - query.astype(np.float64)))
        error = abs(exact - distance)
        if error > 1e-4 + 1e-5 * exact:
            raise RuntimeError(f"ID/distance mismatch for {label}: SQL={distance}, exact={exact}")
        errors.append(error)
    return {"ids": labels, "distances": [r[1] for r in rows],
            "recall_at_10": len(set(labels) & set(oracle["ids"])) / K,
            "max_abs_distance_error": max(errors)}


def helper_records(text):
    decoder = json.JSONDecoder()
    records = {}
    for candidate in (text, text.replace('\\"', '"')):
        for match in re.finditer(r'\{\s*"status"\s*:', candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if value.get("status") == "ok" and value.get("backend") == "hipVS":
                records[json.dumps(value, sort_keys=True)] = value
    return list(records.values())


def rebuild_completion_evidence(text):
    ddl = [line for line in text.splitlines() if "succ to wait rebuild vec index" in line]
    manual = []
    for line in text.splitlines():
        if "end do_work" not in line or "ob_vector_index_async_task_util.cpp" not in line:
            continue
        # ObVecIndexTaskStatus fields are integer values. Manual optimization is
        # trigger_type=1, task_type=1; CALL separately waits for task history.
        matches = [re.search(r"\b"+name+r"\s*[:=]\s*"+str(value)+r"\b",line)
                   for name,value in [("trigger_type",1),("task_type",1),("ret_code",0)]]
        if all(matches):
            manual.append(line)
    return {"ddl_wait_lines":ddl,"manual_optimization_success_lines":manual,
            "completed":bool(ddl or manual)}


def index_plan_ok(text):
    return "VECTOR INDEX" in text.upper() and INDEX.lower() in text.lower()


class Server:
    def __init__(self, mode, port, binary, helper, mysql, base, output, index_parameters):
        self.mode, self.port, self.binary = mode, port, binary
        self.helper, self.mysql = helper, mysql
        self.base, self.output = Path(base), Path(output)
        self.base.mkdir()
        self.output.mkdir()
        self.proc = None
        self.generation = 0
        self.commands = self.output / "commands.jsonl"
        self.index_parameters = dict(index_parameters)
        self.env = os.environ.copy()
        self.env.update({"SEEKDB_HIPVS_BUILD": "1" if mode == "gpu" else "0",
                         "SEEKDB_HIPVS_BUILDER": str(helper), "SEEKDB_HIPVS_THREADS": "16",
                         "SEEKDB_HIPVS_TMPDIR": str(self.base), "OMP_NUM_THREADS": "16"})
        self.startup = []

    def describe(self):
        return {"mode": self.mode, "port": self.port, "base_dir": str(self.base),
                "env": {k: v for k, v in self.env.items()
                        if k.startswith("SEEKDB_HIPVS_") or k == "OMP_NUM_THREADS"},
                "session_ob_hnsw_ef_search": self.index_parameters["ef_search"],
                "startup": self.startup}

    def mysql_command(self):
        return [str(self.mysql), "--no-defaults", "--protocol=TCP", "--host=127.0.0.1",
                f"--port={self.port}", "--user=root", "--connect-timeout=2", "--batch", "--raw",
                "--skip-column-names", "--default-character-set=utf8mb4", "--max-allowed-packet=64M"]

    def sql(self, sql=None, path=None, database=True, timeout=7200, record=True):
        cmd = self.mysql_command()
        if database:
            cmd.append(DATABASE)
        started = time.perf_counter()
        # Each new CLI connection needs a DDL timeout longer than the workload.
        prefix = ("SET SESSION ob_query_timeout=7200000000;\n"
                  f"SET SESSION ob_hnsw_ef_search={self.index_parameters['ef_search']};\n")
        if path is None:
            p = subprocess.run(cmd, input=prefix + sql, text=True, capture_output=True,
                               env=self.env, timeout=timeout)
        else:
            # Stream the generated SQL without copying a 100k-row import into RAM.
            with tempfile.TemporaryFile(mode="w+b", dir=self.base) as stream:
                stream.write(prefix.encode())
                with Path(path).open("rb") as source:
                    shutil.copyfileobj(source, stream)
                stream.seek(0)
                p = subprocess.run(cmd, stdin=stream, text=True, capture_output=True,
                                   env=self.env, timeout=timeout)
        duration = time.perf_counter() - started
        if record:
            with self.commands.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"at": now(), "sql": sql, "input_file": str(path) if path else None,
                                    "session_prefix": prefix,
                                    "seconds": duration, "exit": p.returncode,
                                    "stdout": p.stdout, "stderr": p.stderr}) + "\n")
        if p.returncode:
            raise RuntimeError(f"{self.mode} mysql exit {p.returncode}: {p.stderr[-8000:]}")
        return p.stdout, duration

    def start(self):
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"port {self.port} is already in use")
        self.generation += 1
        log = self.output / f"console-{self.generation}.log"
        command = [str(self.binary), "--nodaemon", "--base-dir", str(self.base),
                   "--port", str(self.port), "--log-level", "INFO"]
        for key, value in PARAMETERS.items():
            command += ["--parameter", f"{key}={value}"]
        entry = {"started_at": now(), "command": command, "console": str(log)}
        self.startup.append(entry)
        started = time.perf_counter()
        with log.open("wb") as stream:
            self.proc = subprocess.Popen(command, cwd=self.base, env=self.env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        entry["pid"] = self.proc.pid
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.mode} server exited during startup: {self.proc.returncode}")
            try:
                effective, _ = self.sql("SELECT @@SESSION.ob_hnsw_ef_search;", database=False,
                                        timeout=5, record=False)
            except (RuntimeError, subprocess.TimeoutExpired):
                time.sleep(1)
                continue
            if int(effective.strip()) != self.index_parameters["ef_search"]:
                raise RuntimeError(f"{self.mode} effective session ob_hnsw_ef_search differs from CLI")
            entry["effective_session_ob_hnsw_ef_search"] = int(effective.strip())
            vector_info, _ = self.sql(
                "SELECT vector_mem_used,vector_mem_hold,vector_mem_limit "
                "FROM oceanbase.__all_virtual_vector_mem_info;",
                database=False, timeout=5, record=False)
            used, hold, limit = map(int, vector_info.strip().split("\t"))
            requested = PARAMETERS["vector_memory_limit"]
            match = re.fullmatch(r"([0-9]+)([KMG]?)", requested.upper())
            if match is None:
                raise RuntimeError("unsupported vector_memory_limit unit in benchmark configuration")
            expected_limit = int(match[1]) * {"":1,"K":1024,"M":1024**2,"G":1024**3}[match[2]]
            if limit != expected_limit:
                raise RuntimeError(f"{self.mode} effective vector memory limit {limit} differs from {requested}")
            entry["effective_vector_memory"] = {"used":used,"hold":hold,"limit":limit}
            saving_info, _ = self.sql(
                "SHOW PARAMETERS LIKE 'vector_index_memory_saving_mode';",
                database=False, timeout=5, record=False)
            saving_values = []
            for line in saving_info.splitlines():
                columns = line.split("\t")
                if "vector_index_memory_saving_mode" in columns:
                    position = columns.index("vector_index_memory_saving_mode")
                    if position + 2 < len(columns):
                        saving_values.append(columns[position + 2].lower())
            if not saving_values or any(value not in ("true", "1", "on") for value in saving_values):
                raise RuntimeError(f"{self.mode} memory-saving mode is not enabled: {saving_info}")
            entry["effective_vector_index_memory_saving_mode"] = saving_values
            entry["ready_seconds"] = time.perf_counter() - started
            entry["ready_at"] = now()
            return
        raise RuntimeError(f"{self.mode} server startup timed out")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=15)
            self.startup[-1]["stopped_at"] = now()
            self.startup[-1]["exit"] = self.proc.returncode

    def log_files(self):
        files = [p for p in (self.base / "log").rglob("*") if p.is_file() and not p.is_symlink()]
        return files + list(self.output.glob("console-*.log"))

    def checkpoint(self):
        result = {}
        for path in self.log_files():
            try:
                st = path.stat()
                result[(st.st_dev, st.st_ino)] = st.st_size
            except FileNotFoundError:
                pass
        return result

    def capture_logs(self, checkpoint, destination):
        selected, seen = [], set()
        with gzip.open(str(destination) + ".logs.gz", "wb") as archive:
            for path in self.log_files():
                try:
                    st = path.stat()
                    inode = (st.st_dev, st.st_ino)
                    if inode in seen:
                        continue
                    seen.add(inode)
                    offset = checkpoint.get(inode, 0)
                    if offset > st.st_size:
                        offset = 0
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        archive.write(f"\nFILE {path} OFFSET {offset}\n".encode())
                        for line in stream:
                            archive.write(line)
                            if self.evidence_line(line):
                                selected.append(line.decode("utf-8", errors="replace"))
                except FileNotFoundError:
                    continue
        text = "".join(selected)
        Path(str(destination) + ".hipvs.log").write_text(text, encoding="utf-8")
        return {"raw_log_archive": str(destination) + ".logs.gz",
                "hipvs_excerpt": str(destination) + ".hipvs.log",
                "completion_lines": sum("hipVS HNSW construction complete" in line for line in selected),
                "rebuild_completion_lines": sum("succ to wait rebuild vec index" in line for line in selected),
                "rebuild_completion_evidence": rebuild_completion_evidence(text),
                "fallback_lines": [line.strip() for line in selected if "fallback" in line.lower()],
                "helper_records": helper_records(text)}

    @staticmethod
    def evidence_line(line):
        lower = line.lower()
        return (b"hipvs" in lower or b"rebuild vec index" in lower or b"rebuild vector index" in lower
                or (b"end do_work" in lower and b"ob_vector_index_async_task_util.cpp" in lower))

    def evidence_since(self, checkpoint):
        """Read the evidence subset while polling, without rewriting archives."""
        selected, seen = [], set()
        for path in self.log_files():
            try:
                st = path.stat()
                inode = (st.st_dev, st.st_ino)
                if inode in seen:
                    continue
                seen.add(inode)
                offset = checkpoint.get(inode, 0)
                with path.open("rb") as stream:
                    stream.seek(offset if offset <= st.st_size else 0)
                    for line in stream:
                        if self.evidence_line(line):
                            selected.append(line.decode("utf-8", errors="replace"))
            except FileNotFoundError:
                continue
        return "".join(selected)


def quality(server, table, x, ids, queries, oracle, destination, min_recall=None):
    mapping = {int(label): row for row, label in enumerate(ids)}
    samples = []
    started = time.perf_counter()
    for i, query in enumerate(queries):
        out, duration = server.sql(query_sql(table, query))
        sample = assess(parse_rows(out), query, x, mapping, oracle[i])
        sample.update({"query": i, "client_seconds": duration})
        samples.append(sample)
        # Preserve successful query evidence even if a later query fails.
        write_json(destination, {"status": "running", "samples": samples})
    recalls = [sample["recall_at_10"] for sample in samples]
    result = {"status": "ok", "queries": len(samples), "recall_at_10": statistics.mean(recalls),
              "min_query_recall_at_10": min(recalls),
              "max_abs_distance_error": max(s["max_abs_distance_error"] for s in samples),
              "quality_seconds": time.perf_counter() - started, "samples": samples}
    write_json(destination, result)
    if min_recall is not None and result["recall_at_10"] < min_recall:
        raise RuntimeError(f"{server.mode} recall@10 {result['recall_at_10']} < {min_recall}")
    return result


def plan(server, table, query, path):
    text, _ = server.sql("EXPLAIN " + query_sql(table, query))
    Path(path).write_text(text, encoding="utf-8")
    if not index_plan_ok(text):
        raise RuntimeError(f"{server.mode} EXPLAIN does not use {INDEX}; see {path}")
    return str(path)


def build_round(server, table, rows, round_number, directory, x, ids, queries, oracle,
                record, persist, min_recall, index_parameters):
    directory.mkdir()
    checkpoint = server.checkpoint()
    record.update({"status": "running", "at": now()})
    persist()
    try:
        create = create_index_sql(table, index_parameters)
        (directory / "create.sql").write_text(create + "\n", encoding="utf-8")
        # This interval includes CLI/connection overhead and the first successful
        # approximate query after synchronous CREATE completion.
        start = time.perf_counter()
        _, create_seconds = server.sql(create)
        probe_stdout, probe_seconds = server.sql(query_sql(table, queries[0]))
        probe_rows = parse_rows(probe_stdout)
        record.update({"build_to_queryable_seconds": time.perf_counter() - start,
                       "create_client_seconds": create_seconds, "probe_client_seconds": probe_seconds,
                       "probe_result": probe_rows})
        persist()
        record["explain"] = plan(server, table, queries[0], directory / "explain.txt")
        measured = quality(server, table, x, ids, queries, oracle, directory / "quality.json", min_recall)
        record["quality"] = {key: value for key, value in measured.items() if key != "samples"}
        # Allow the asynchronous logger to drain; this is outside build timing.
        time.sleep(0.5)
        evidence = server.capture_logs(checkpoint, directory / "server")
        record["gpu_evidence"] = evidence
        if server.mode == "gpu":
            records = evidence["helper_records"]
            covered = sum(int(r.get("rows", 0)) for r in records if int(r.get("dim", 0)) == x.shape[1])
            record["gpu_evidence"]["helper_rows_total"] = covered
            if not evidence["completion_lines"] or not records or covered < rows:
                raise RuntimeError(f"GPU round lacks hipVS completion/JSON evidence covering {rows} rows")
            if evidence["fallback_lines"]:
                raise RuntimeError("GPU timed round used a logged CPU fallback")
        elif evidence["helper_records"] or evidence["completion_lines"]:
            raise RuntimeError("CPU round unexpectedly invoked hipVS")
        record["status"] = "ok"
    except BaseException as e:
        record.update({"status": "failed", "error": repr(e)})
        if "gpu_evidence" not in record:
            record["gpu_evidence"] = server.capture_logs(checkpoint, directory / "server")
        raise
    finally:
        record["finished_at"] = now()
        persist()


def dml_check(server, dim, seed, destination, index_parameters):
    """Use an independent small table so mutations cannot affect timed builds."""
    destination.mkdir()
    x, ids, _ = dataset(256, dim, 1, seed)
    table = "incremental_check"
    server.sql(f"CREATE TABLE {table}(id BIGINT PRIMARY KEY,emb VECTOR({dim}));")
    path = destination / "import.sql"
    write_import(path, table, x, ids)
    server.sql(path=path)
    server.sql(create_index_sql(table, index_parameters))
    result = {"status": "running", "steps": [], "excluded_from_build_measurement": True}
    added = np.int64(9000000000000000001)
    zero = np.zeros(dim, dtype=np.float32)
    two = np.full(dim, 2, dtype=np.float32)
    for action, query in [("insert", zero), ("update", two), ("delete", two)]:
        if action == "insert":
            sql = f"INSERT INTO {table} VALUES({int(added)},'{vector_literal(query)}');"
            x = np.vstack([x, query]); ids = np.append(ids, added)
        elif action == "update":
            sql = f"UPDATE {table} SET emb='{vector_literal(query)}' WHERE id={int(added)};"
            x[-1] = query
        else:
            sql = f"DELETE FROM {table} WHERE id={int(added)};"
            x = x[:-1]; ids = ids[:-1]
        server.sql(sql)
        stdout, duration = server.sql(query_sql(table, query))
        rows = parse_rows(stdout)
        expected = exact_topk(x, ids, query[None, :])[0]
        sample = assess(rows, query, x, {int(v): i for i, v in enumerate(ids)}, expected)
        if action != "delete" and (rows[0][0] != int(added) or abs(rows[0][1]) > 1e-6):
            raise RuntimeError(f"{server.mode} {action} is absent from approximate query")
        if action == "delete" and any(label == int(added) for label, _ in rows):
            raise RuntimeError(f"{server.mode} deleted ID remains in approximate query")
        sample.update({"action": action, "sql": sql, "client_seconds": duration})
        if action == "update":
            old_rows = parse_rows(server.sql(query_sql(table, zero))[0])
            if any(label == int(added) for label, _ in old_rows):
                raise RuntimeError("updated vector still occurs at its old location")
            sample["old_location_query"] = old_rows
        result["steps"].append(sample)
        write_json(destination / "results.json", result)
    result["explain"] = plan(server, table, two, destination / "explain.txt")
    result["status"] = "ok"
    write_json(destination / "results.json", result)
    return result


def snapshot_mutation_state(x, ids):
    """Copy the imported snapshot; the benchmark input/oracle stays unchanged."""
    if len(ids) < 10000 or len(ids) != len(x):
        raise ValueError("snapshot DML requires a completed snapshot with at least 10000 rows")
    dim = x.shape[1]
    updated = np.full(dim, -1234, dtype=np.float32)
    inserted = np.full(dim, 1234, dtype=np.float32)
    if np.any(np.all(x == updated, axis=1)) or np.any(np.all(x == inserted, axis=1)):
        raise ValueError("snapshot already contains one of the reserved mutation vectors")
    added_id = 9000000000000000002
    if added_id in set(map(int,ids)):
        raise ValueError("snapshot already contains the reserved mutation primary key")
    changed_x = np.concatenate([updated[None,:], x[2:].copy(), inserted[None,:]])
    changed_ids = np.concatenate([ids[:1].copy(), ids[2:].copy(), np.array([added_id],dtype=np.int64)])
    return {"x":changed_x,"ids":changed_ids,"count":len(ids),
            "updated_id":int(ids[0]),"deleted_id":int(ids[1]),"inserted_id":added_id,
            "updated_vector":updated,"inserted_vector":inserted,
            "old_updated_vector":x[0].copy(),"old_deleted_vector":x[1].copy(),
            "unchanged_id":int(ids[2]),"unchanged_vector":x[2].copy()}


class SqlEvidence:
    """Keep complete SQL and raw results next to a postcheck's assertions."""
    def __init__(self, server, directory, result):
        self.server, self.directory, self.result = server, Path(directory), result
        self.directory.mkdir(parents=True,exist_ok=True)
        self.result.setdefault("sql_steps",[])

    def save(self):
        write_json(self.directory/"results.json",self.result)

    def execute(self, label, sql, timeout=7200):
        entry = {"label":label,"at":now(),"sql":sql,"status":"running"}
        self.result["sql_steps"].append(entry)
        with (self.directory/"statements.sql").open("a",encoding="utf-8") as f:
            f.write(f"-- {label}\n{sql}\n")
        self.save()
        try:
            stdout,duration = self.server.sql(sql,timeout=timeout)
            entry.update({"status":"ok","stdout":stdout,"client_seconds":duration})
            return stdout
        except BaseException as e:
            entry.update({"status":"failed","error":repr(e)})
            raise
        finally:
            self.save()


def validate_stored_vectors(table, expected, evidence, label, absent_ids=()):
    keys = list(expected) + list(absent_ids)
    sql = f"SELECT id,emb FROM {table} WHERE id IN ({','.join(map(str,keys))}) ORDER BY id;"
    raw = evidence.execute(label+"-stored-vectors",sql)
    actual = {}
    for line in raw.splitlines():
        key,text = line.split("\t",1)
        key = int(key)
        if key in actual:
            raise RuntimeError("duplicate primary key in stored vector output")
        actual[key] = np.asarray(json.loads(text),dtype=np.float64)
    if set(actual) != set(expected):
        raise RuntimeError("stored vector IDs differ from expected present/absent primary keys")
    display_errors,sql_distances = {},{}
    for key,vector in expected.items():
        if (actual[key].shape != vector.shape or not np.isfinite(actual[key]).all()
                or not np.allclose(actual[key],vector,rtol=VECTOR_DISPLAY_RTOL,atol=0)):
            raise RuntimeError(f"stored vector display has an incompatible shape/value for primary key {key}")
        display_errors[str(key)] = float(np.max(np.abs(actual[key]-vector.astype(np.float64))))
        # The formatter intentionally rounds float32 to six significant digits.
        # A scalar SQL distance uses the actual stored vector, independent of its
        # human-readable representation and of the approximate vector index.
        exact_sql = (f"SELECT id,L2_DISTANCE(emb,'{vector_literal(vector)}') AS exact_source_l2 "
                     f"FROM {table} WHERE id={key};")
        distance_rows = parse_rows(evidence.execute(label+f"-exact-source-l2-{key}",exact_sql),expected=1)
        returned_id,distance = distance_rows[0]
        sql_distances[str(key)] = distance
        if returned_id != key or distance < 0 or distance > STORED_VECTOR_L2_TOLERANCE:
            raise RuntimeError(f"stored vector differs from exact float32 source for primary key {key}: "
                               f"SQL L2_DISTANCE={distance}, tolerance={STORED_VECTOR_L2_TOLERANCE}")
    return {"stored_ids":sorted(actual),"display_max_abs_value_errors":display_errors,
            "display_relative_tolerance":VECTOR_DISPLAY_RTOL,"sql_exact_source_l2":sql_distances,
            "sql_l2_tolerance":STORED_VECTOR_L2_TOLERANCE,
            "absent_ids":list(absent_ids)}


def validate_snapshot_state(server, table, state, evidence, label):
    count = int(evidence.execute(label+"-count",f"SELECT COUNT(*) FROM {table};").strip())
    if count != state["count"]:
        raise RuntimeError(f"{server.mode} snapshot row count changed: {count} != {state['count']}")
    expected = {state["updated_id"]:state["updated_vector"],
                state["inserted_id"]:state["inserted_vector"],
                state["unchanged_id"]:state["unchanged_vector"]}
    stored = validate_stored_vectors(table,expected,evidence,label,[state["deleted_id"]])
    return {"count":count,**stored,"deleted_id_absent":True}


def snapshot_ann_checks(server, table, state, evidence, label):
    mapping = {int(v):i for i,v in enumerate(state["ids"])}
    cases = [("updated-original-id",state["updated_vector"],state["updated_id"],[]),
             ("inserted-id",state["inserted_vector"],state["inserted_id"],[]),
             ("old-updated-location",state["old_updated_vector"],None,[state["updated_id"]]),
             ("old-deleted-location",state["old_deleted_vector"],None,[state["deleted_id"]])]
    checks = {}
    for name,query,required,forbidden in cases:
        tag = label+"-"+name
        sql = query_sql(table,query)
        text = evidence.execute(tag,sql)
        rows = parse_rows(text)
        oracle = exact_topk(state["x"],state["ids"],query[None,:])[0]
        sample = assess(rows,query,state["x"],mapping,oracle)
        if required is not None and (rows[0][0] != required or abs(rows[0][1]) > 1e-6):
            raise RuntimeError(f"{server.mode} {name} is not returned at its exact new location")
        if any(key in sample["ids"] for key in forbidden):
            raise RuntimeError(f"{server.mode} stale ID returned at {name}")
        # Every result passes through the mutated ID map, so the deleted ID is
        # rejected for all probes rather than only its old vector probe.
        explain = evidence.execute(tag+"-explain","EXPLAIN "+sql)
        explain_path = evidence.directory/(tag+".explain.txt")
        explain_path.write_text(explain,encoding="utf-8")
        if not index_plan_ok(explain):
            raise RuntimeError(f"{server.mode} snapshot mutation probe does not use {INDEX}")
        sample.update({"oracle":oracle,"explain":str(explain_path)})
        checks[name] = sample
    return checks


def snapshot_dml_check(server, table, x, ids, destination):
    input_hashes = {"vectors":hashlib.sha256(x.tobytes()).hexdigest(),
                    "ids":hashlib.sha256(ids.tobytes()).hexdigest()}
    state = snapshot_mutation_state(x,ids)
    result = {"status":"running","excluded_from_build_measurement":True,
              "table":table,"snapshot_rows":len(ids),"original_updated_id":state["updated_id"],
              "original_deleted_id":state["deleted_id"],"inserted_id":state["inserted_id"],
              "benchmark_input_sha256":input_hashes}
    evidence = SqlEvidence(server,destination,result)
    checkpoint = server.checkpoint()
    try:
        original_keys = [state["updated_id"],state["deleted_id"],state["unchanged_id"]]
        result["original_snapshot_storage"] = validate_stored_vectors(
            table,{key:x[i] for i,key in enumerate(original_keys)},evidence,"original-snapshot")
        evidence.execute("update-original-snapshot-id",
            f"UPDATE {table} SET emb='{vector_literal(state['updated_vector'])}' WHERE id={state['updated_id']};")
        evidence.execute("delete-original-snapshot-id",f"DELETE FROM {table} WHERE id={state['deleted_id']};")
        evidence.execute("insert-new-id",
            f"INSERT INTO {table}(id,emb) VALUES({state['inserted_id']},'{vector_literal(state['inserted_vector'])}');")
        result["before_restart_storage"] = validate_snapshot_state(server,table,state,evidence,"before-restart")
        result["before_restart_ann"] = snapshot_ann_checks(server,table,state,evidence,"before-restart")
        evidence.save()
        server.stop()
        server.start()
        result["after_restart_storage"] = validate_snapshot_state(server,table,state,evidence,"after-restart")
        result["after_restart_ann"] = snapshot_ann_checks(server,table,state,evidence,"after-restart")
        result["ann_ids_identical_after_restart"] = all(
            result["before_restart_ann"][case]["ids"] == sample["ids"]
            for case,sample in result["after_restart_ann"].items())
        result["benchmark_inputs_unchanged"] = input_hashes == {
            "vectors":hashlib.sha256(x.tobytes()).hexdigest(),"ids":hashlib.sha256(ids.tobytes()).hexdigest()}
        if not result["benchmark_inputs_unchanged"]:
            raise RuntimeError("snapshot mutation modified the benchmark input arrays")
        result["status"] = "ok"
        return result,state
    except BaseException as e:
        result.update({"status":"failed","error":repr(e)})
        raise
    finally:
        result["log_evidence"] = server.capture_logs(checkpoint,Path(destination)/"server")
        evidence.save()


def rebuild_sql(table):
    # ob_dbms_vector_mysql.cpp defines the fourth argument as delta threshold;
    # ob_vector_index_refresh.cpp:524 bypasses the threshold check when it is 0,
    # submits a rebuild task, and waits for its asynchronous DDL at line 598.
    # Hybrid HNSW dispatches manual optimization at line 610; its task manager
    # also waits for asynchronous task history before CALL returns.
    return f"CALL DBMS_VECTOR.REBUILD_INDEX('{INDEX}','{table}','emb',0);"


def rebuild_check(server, table, state, destination, timeout):
    result = {"status":"running","excluded_from_build_measurement":True,"table":table,
              "operation":"DBMS_VECTOR.REBUILD_INDEX with delta threshold 0",
              "source_references":["src/pl/sys_package/ob_dbms_vector_mysql.cpp:66",
                                   "src/sql/engine/cmd/ob_vector_index_refresh.cpp:524",
                                   "src/sql/engine/cmd/ob_vector_index_refresh.cpp:598",
                                   "src/sql/engine/cmd/ob_vector_index_refresh.cpp:610",
                                   "src/observer/vector_index/ob_vector_index_async_task.cpp:307"],
              "polls":[]}
    evidence = SqlEvidence(server,destination,result)
    checkpoint = server.checkpoint()
    try:
        started = time.perf_counter()
        # CALL waits for the internally asynchronous rebuild DDL. Poll afterward
        # until its new build logs and queryable published index are both seen.
        evidence.execute("force-rebuild",rebuild_sql(table),timeout=timeout)
        result["call_seconds"] = time.perf_counter()-started
        deadline = time.monotonic()+min(timeout,60)
        while True:
            text = server.evidence_since(checkpoint)
            records = helper_records(text)
            full_rows = max([int(r.get("rows",0)) for r in records
                             if int(r.get("dim",0)) == state["x"].shape[1]] or [0])
            completion = rebuild_completion_evidence(text)
            waited = completion["completed"]
            full_snapshot = any(int(r.get("rows",0)) == state["count"]
                                and int(r.get("dim",0)) == state["x"].shape[1] for r in records)
            gpu_ok = ("hipVS HNSW construction complete" in text and full_snapshot
                      ) if server.mode == "gpu" else not records
            poll = {"at":now(),"async_completion_evidence":completion,"helper_records":records,
                    "largest_helper_rows":full_rows,"gpu_evidence_ok":gpu_ok,
                    "fallback_lines":[line for line in text.splitlines() if "fallback" in line.lower()]}
            # A successful old-generation query alone cannot prove rebuild;
            # require it together with the new DDL and full-row GPU evidence.
            probe = query_sql(table,state["inserted_vector"])
            try:
                rows = parse_rows(evidence.execute("rebuild-poll-query",probe,timeout=30))
                explain = evidence.execute("rebuild-poll-explain","EXPLAIN "+probe,timeout=30)
                poll["query_available"] = (rows[0][0] == state["inserted_id"]
                                           and abs(rows[0][1]) <= 1e-6 and index_plan_ok(explain))
                poll["probe_rows"] = rows
            except (RuntimeError,subprocess.TimeoutExpired) as e:
                poll["query_available"] = False
                poll["query_error"] = repr(e)
            result["polls"].append(poll)
            evidence.save()
            if waited and gpu_ok and poll["query_available"]:
                result["after_rebuild_storage"] = validate_snapshot_state(server,table,state,evidence,"after-rebuild")
                result["after_rebuild_ann"] = snapshot_ann_checks(server,table,state,evidence,"after-rebuild")
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("forced rebuild lacks new completion/GPU coverage evidence")
            if server.proc.poll() is not None:
                raise RuntimeError("server exited while waiting for forced rebuild evidence")
            time.sleep(1)
        result["rebuild_to_verified_seconds"] = time.perf_counter()-started
        result["log_evidence"] = server.capture_logs(checkpoint,Path(destination)/"rebuild-server")
        server.stop()
        server.start()
        result["after_restart_storage"] = validate_snapshot_state(server,table,state,evidence,"after-rebuild-restart")
        result["after_restart_ann"] = snapshot_ann_checks(server,table,state,evidence,"after-rebuild-restart")
        result["status"] = "ok"
        return result
    except BaseException as e:
        result.update({"status":"failed","error":repr(e)})
        raise
    finally:
        if "log_evidence" not in result:
            result["log_evidence"] = server.capture_logs(checkpoint,Path(destination)/"rebuild-server")
        evidence.save()


def summarize(records):
    result = {}
    for size in sorted({r["rows"] for r in records}):
        selected = [r for r in records if r["rows"] == size and r.get("status") == "ok"]
        modes = {mode: [r["build_to_queryable_seconds"] for r in selected if r["mode"] == mode]
                 for mode in ("cpu", "gpu")}
        entry = {mode: {"seconds": values, "median_seconds": statistics.median(values)}
                 for mode, values in modes.items() if values}
        if all(modes.values()):
            entry["cpu_median_div_gpu_median"] = statistics.median(modes["cpu"]) / statistics.median(modes["gpu"])
            pairs = {}
            for r in selected:
                pairs.setdefault(r["round"], {})[r["mode"]] = r["build_to_queryable_seconds"]
            entry["paired_cpu_div_gpu_ratios"] = [p["cpu"] / p["gpu"] for p in pairs.values() if len(p) == 2]
            if entry["paired_cpu_div_gpu_ratios"]:
                entry["median_paired_cpu_div_gpu_ratio"] = statistics.median(entry["paired_cpu_div_gpu_ratios"])
        result[str(size)] = entry
    return result


def self_check():
    from unittest.mock import patch

    x, ids, q = dataset(128, 8, 3, 42)
    x2, ids2, q2 = dataset(128, 8, 3, 42)
    assert np.array_equal(x, x2) and np.array_equal(ids, ids2) and np.array_equal(q, q2)
    assert not np.array_equal(x[:3], q)
    assert np.array_equal(np.asarray(json.loads(vector_literal(x[0])), dtype=np.float32), x[0])
    oracle = exact_topk(x, ids, q)
    for query, expected in zip(q, oracle):
        full = sorted((float(np.linalg.norm(row.astype(np.float64)-query)), int(label))
                      for row, label in zip(x, ids))[:K]
        assert expected["ids"] == [r[1] for r in full]
        rows = parse_rows("\n".join(f"{label}\t{distance:.17g}" for distance, label in full))
        assert assess(rows, query, x, {int(v): i for i, v in enumerate(ids)}, expected)["recall_at_10"] == 1
    j = '{"status":"ok","backend":"hipVS","rows":128,"dim":8}'
    assert len(helper_records("log diagnostics="+j+"\n"+j)) == 1
    assert len(helper_records(j.replace('"', '\\"'))) == 1
    async_success = ("ob_vector_index_async_task_util.cpp:1195 end do_work "
                     "task_status={trigger_type:1,task_type:1,ret_code:0}")
    assert rebuild_completion_evidence(async_success)["completed"]
    assert not rebuild_completion_evidence(async_success.replace("ret_code:0","ret_code:-4000"))["completed"]
    assert not rebuild_completion_evidence(async_success.replace("trigger_type:1","trigger_type:0"))["completed"]
    assert index_plan_ok("VECTOR INDEX SCAN t(hnsw_idx)")
    assert not index_plan_ok("TABLE FULL SCAN t")
    records = [{"rows":10000,"round":i,"mode":m,"status":"ok","build_to_queryable_seconds":t}
               for i in range(3) for m,t in [("cpu",2+i),("gpu",1+i)]]
    assert summarize(records)["10000"]["cpu_median_div_gpu_median"] == 1.5
    snapshot_x,snapshot_ids,_ = dataset(10000,8,1,42)
    original_vectors_hash = hashlib.sha256(snapshot_x.tobytes()).hexdigest()
    original_ids_hash = hashlib.sha256(snapshot_ids.tobytes()).hexdigest()
    mutation = snapshot_mutation_state(snapshot_x,snapshot_ids)
    assert len(mutation["x"]) == len(mutation["ids"]) == len(snapshot_ids)
    assert mutation["deleted_id"] not in set(map(int,mutation["ids"]))
    assert mutation["updated_id"] == int(snapshot_ids[0])
    mutation_oracle = exact_topk(mutation["x"],mutation["ids"],
                                np.stack([mutation["updated_vector"],mutation["inserted_vector"]]))
    assert mutation_oracle[0]["ids"][0] == mutation["updated_id"]
    assert mutation_oracle[1]["ids"][0] == mutation["inserted_id"]
    assert mutation_oracle[0]["distances"][0] == mutation_oracle[1]["distances"][0] == 0
    assert hashlib.sha256(snapshot_x.tobytes()).hexdigest() == original_vectors_hash
    assert hashlib.sha256(snapshot_ids.tobytes()).hexdigest() == original_ids_hash
    assert rebuild_sql("vectors_10000") == "CALL DBMS_VECTOR.REBUILD_INDEX('hnsw_idx','vectors_10000','emb',0);"
    rounded_source = np.asarray([.18905849,.000010000049,1234,-1234,0],dtype=np.float32)
    rounded_display = "["+",".join(format(float(v),".6g") for v in rounded_source)+"]"
    assert not np.allclose(np.asarray(json.loads(rounded_display)),rounded_source,rtol=1e-6,atol=1e-7)
    class RoundedStorageFixture:
        distance = 0
        calls = []
        def execute(self,label,sql):
            self.calls.append(sql)
            return "7\t"+rounded_display+"\n" if sql.startswith("SELECT id,emb ") else f"7\t{self.distance}\n"
    fixture = RoundedStorageFixture()
    stored = validate_stored_vectors("fixture",{7:rounded_source},fixture,"rounded")
    assert stored["sql_exact_source_l2"]["7"] == 0
    assert any(vector_literal(rounded_source) in sql for sql in fixture.calls)
    # A real stored change hidden by the same display still fails the SQL check.
    fixture.distance = 2e-7
    try:
        validate_stored_vectors("fixture",{7:rounded_source},fixture,"corrupt")
    except RuntimeError as e:
        assert "SQL L2_DISTANCE" in str(e)
    else:
        raise AssertionError("SQL storage mismatch was masked by rounded display tolerance")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_path, query_path = root/"base.f32", root/"query.f32"
        x.astype("<f4").tofile(base_path)
        q.astype("<f4").tofile(query_path)
        base, base_info = load_f32(base_path, 8, 128)
        query, query_info = load_f32(query_path, 8, 3)
        selected_x, selected_ids, selected_q = input_dataset(64, 8, 2, 42, base, query)
        assert np.array_equal(selected_x, x[:64]) and np.array_equal(selected_q, q[:2])
        random_x, _, random_q = dataset(64, 8, 2, 42)
        assert np.array_equal(input_dataset(64,8,2,42,base,None)[2], random_q)
        assert np.array_equal(input_dataset(64,8,2,42,None,query)[0], random_x)
        assert base_info["rows"] == 128 and query_info["rows"] == 3
        assert base_info["sha256"] == sha256(base_path)
        reference = exact_topk(selected_x, selected_ids, selected_q)
        assert all(set(item["ids"]).issubset(set(map(int, selected_ids))) for item in reference)
        base._mmap.close()
        query._mmap.close()
        bad = root/"bad.f32"
        bad.write_bytes(b"abc")
        for path, dim, needed in [(bad,8,1), (base_path,8,129), (query_path,7,1)]:
            try:
                load_f32(path,dim,needed)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid float32 input was accepted")
        nonfinite = x.copy()
        nonfinite[0,0] = np.nan
        nonfinite.astype("<f4").tofile(bad)
        try:
            load_f32(bad,8,1)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite float32 input was accepted")
        options = {"m":32, "ef_construction":300, "ef_search":1000}
        assert create_index_sql("vectors", options).endswith(
            "WITH(distance=l2,type=hnsw,m=32,ef_construction=300,ef_search=1000);")
        parsed = parse_args(["--binary","/candidate","--helper","/helper","--output","/output",
                             "--sizes","10000","--m","32","--ef-construction","300",
                             "--ef-search","1000","--base-f32",str(base_path),
                             "--queries-f32",str(query_path),"--check-rebuild"])
        assert (parsed.m,parsed.ef_construction,parsed.ef_search) == (32,300,1000)
        assert parsed.base_f32 == base_path and parsed.queries_f32 == query_path
        assert parsed.check_rebuild
        for mode in ("cpu","gpu"):
            fake_server = Server(mode,2881,Path("/candidate"),Path("/helper"),Path("/mysql"),
                                 root/("fake-instance-"+mode),root/("fake-evidence-"+mode),options)
            def fake_mysql(command,**kwargs):
                assert "SET SESSION ob_hnsw_ef_search=1000;\n" in kwargs["input"]
                return subprocess.CompletedProcess(command,0,stdout="1000\n",stderr="")
            with patch.object(subprocess,"run",side_effect=fake_mysql):
                output,_ = fake_server.sql("SELECT @@SESSION.ob_hnsw_ef_search;",database=False)
            assert output.strip() == "1000"
            logged = json.loads(fake_server.commands.read_text())
            assert "SET SESSION ob_hnsw_ef_search=1000;" in logged["session_prefix"]
            assert fake_server.describe()["session_ob_hnsw_ef_search"] == 1000
            assert fake_server.describe()["env"]["OMP_NUM_THREADS"] == "16"
        server = Server.__new__(Server)
        server.base, server.output = root/"instance", root/"evidence"
        (server.base/"log").mkdir(parents=True)
        server.output.mkdir()
        log = server.base/"log"/"seekdb.log"
        log.write_text("previous run " + j + "\n", encoding="utf-8")
        checkpoint = server.checkpoint()
        # Rename a log after the checkpoint to exercise rotated-inode handling.
        log.rename(log.with_name("seekdb.log.1"))
        with log.with_name("seekdb.log.1").open("a", encoding="utf-8") as f:
            f.write("hipVS HNSW construction complete diagnostics=" + j + "\n")
            f.write("succ to wait rebuild vec index task_id=123\n")
        log.write_text("new log file\n", encoding="utf-8")
        captured = server.capture_logs(checkpoint, server.output/"round")
        assert captured["completion_lines"] == 1 and len(captured["helper_records"]) == 1
        assert captured["rebuild_completion_lines"] == 1
        assert "succ to wait rebuild vec index" in server.evidence_since(checkpoint)
        assert "previous run" not in Path(captured["hipvs_excerpt"]).read_text()
        with gzip.open(captured["raw_log_archive"], "rt", encoding="utf-8") as f:
            assert "new log file" in f.read()
    print("self-check passed: deterministic float32 data, independent queries, exact L2 oracle, "
          "reference file validation/SHA256/prefix selection, configurable HNSW SQL, IDs/distances, "
          "per-connection session ef, snapshot mutation input isolation/IDs, forced rebuild SQL, "
          "rounded display vs exact SQL storage checks, JSON evidence parsing, log rotation checkpoints, "
          "plan recognition, paired medians")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path)
    p.add_argument("--helper", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--sizes", default="10000,100000")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-f32", type=Path, help="packed little-endian float32 base matrix, --dim columns")
    p.add_argument("--queries-f32", type=Path, help="packed little-endian float32 query matrix, --dim columns")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=1000)
    p.add_argument("--mysql", type=Path, default=Path(shutil.which("mysql") or "mysql"))
    p.add_argument("--min-recall", type=float, default=None)
    p.add_argument("--skip-postchecks", action="store_true",
                   help="skip restart, original-snapshot mutations and separate DML checks")
    p.add_argument("--check-rebuild", action="store_true",
                   help="after snapshot DML, force DBMS_VECTOR rebuild, verify new build logs, queries and restart")
    p.add_argument("--rebuild-timeout", type=int, default=1800,
                   help="seconds allowed for the optional forced rebuild CALL")
    p.add_argument("--self-check", action="store_true", help="run pure measurement checks without starting a server")
    a = p.parse_args(argv)
    if a.self_check:
        return a
    if any(getattr(a, key) is None for key in ("binary", "helper", "output")):
        p.error("--binary, --helper, and --output are required")
    try:
        a.sizes = [int(s) for s in a.sizes.split(",")]
    except ValueError:
        p.error("--sizes must be comma-separated integers")
    if not a.sizes or len(a.sizes) != len(set(a.sizes)) or any(n not in (10000,100000,1000000) for n in a.sizes):
        p.error("--sizes accepts distinct values from 10000,100000,1000000")
    if a.rounds < 3 or a.queries < 1 or a.dim < 1 or a.seed < 0:
        p.error("require rounds >= 3, queries >= 1, dim >= 1, seed >= 0")
    if not 2 <= a.m <= 256 or not a.m <= a.ef_construction <= 2**31-1 or not K <= a.ef_search <= 160000:
        p.error("require 2 <= m <= 256, m <= ef-construction <= INT32_MAX, 10 <= ef-search <= 160000")
    if a.min_recall is not None and not 0 <= a.min_recall <= 1:
        p.error("--min-recall must lie in [0,1]")
    if a.check_rebuild and a.skip_postchecks:
        p.error("--check-rebuild requires postchecks; remove --skip-postchecks")
    if not 60 <= a.rebuild_timeout <= 7200:
        p.error("--rebuild-timeout must lie in [60,7200]")
    return a


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if sys.platform != "linux":
        raise RuntimeError("run the benchmark on the Linux AMD host")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    index_parameters = {"m":args.m, "ef_construction":args.ef_construction, "ef_search":args.ef_search}
    report = {"status": "running", "started_at": now(), "runs": [], "datasets": {}, "servers": {},
              "timing_definition": "CREATE client start through completion of the first APPROXIMATE query; "
              "excludes data generation/import, startup, EXPLAIN, quality queries, restart, DML and optional rebuild",
              "config": {**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                         "server_parameters": PARAMETERS, "index_parameters":index_parameters}}
    servers = []
    exit_code = 0

    def persist():
        report["servers"] = {s.mode:s.describe() for s in servers}
        report["summary"] = summarize(report["runs"])
        write_json(args.output / "results.json", report)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        persist()
        for name in ("binary", "helper", "mysql"):
            path = getattr(args, name).resolve(strict=True)
            if not path.is_file() or not os.access(path, os.X_OK):
                raise RuntimeError(f"{name} is not an executable regular file: {path}")
            setattr(args, name, path)
        report["artifacts"] = {name:{"path":str(getattr(args,name)),"sha256":sha256(getattr(args,name))}
                               for name in ("binary", "helper", "mysql")}
        report["candidate_version"] = command_info([str(args.binary), "--version"])
        report["machine"] = machine_info()
        base_matrix, query_matrix = None, None
        report["input_sources"] = {}
        for kind, path, required in [("base",args.base_f32,max(args.sizes)),
                                     ("queries",args.queries_f32,args.queries)]:
            if path is not None:
                matrix, metadata = load_f32(path,args.dim,required)
                report["input_sources"][kind] = {"source":"file", **metadata}
                if kind == "base":
                    base_matrix = matrix
                else:
                    query_matrix = matrix
            else:
                report["input_sources"][kind] = {"source":"seeded_random", "seed":args.seed,
                                                 "dtype":"float32", "dim":args.dim}
        persist()
        root = Path(tempfile.mkdtemp(prefix="seekdb-hipvs-benchmark-", dir="/root"))
        report["instance_root"] = str(root)
        for mode, port in [("cpu",2881), ("gpu",2882)]:
            server = Server(mode, port, args.binary, args.helper, args.mysql,
                            root / mode, args.output / mode, index_parameters)
            servers.append(server)
            persist()
            server.start()
            server.sql(f"CREATE DATABASE {DATABASE};", database=False)
            report.setdefault("server_versions", {})[mode] = server.sql("SELECT VERSION();")[0].strip()
            effective = int(server.sql("SELECT @@SESSION.ob_hnsw_ef_search;")[0].strip())
            if effective != args.ef_search:
                raise RuntimeError("effective SQL session ef_search differs between requested and observed values")
            report.setdefault("effective_session_variables", {})[mode] = {"ob_hnsw_ef_search":effective}
            persist()

        for n in args.sizes:
            directory = args.output / f"n{n}"
            directory.mkdir()
            started = time.perf_counter()
            x, ids, queries = input_dataset(n, args.dim, args.queries, args.seed, base_matrix, query_matrix)
            x.astype("<f4", copy=False).tofile(directory / "vectors.f32")
            ids.astype("<i8", copy=False).tofile(directory / "ids.i64")
            queries.astype("<f4", copy=False).tofile(directory / "queries.f32")
            oracle = exact_topk(x, ids, queries)
            write_json(directory / "oracle.json", oracle)
            table = f"vectors_{n}"
            import_path = directory / "import.sql"
            write_import(import_path, table, x, ids)
            info = {"rows":n,"dim":args.dim,"generation_oracle_sql_seconds":time.perf_counter()-started,
                    "base_row_range":[0,n], "query_row_range":[0,args.queries],
                    "input_sha256": {p.name:sha256(p) for p in directory.iterdir() if p.is_file()},
                    "imports":{}}
            report["datasets"][str(n)] = info
            persist()
            for server in servers:
                server.sql(f"CREATE TABLE {table}(id BIGINT PRIMARY KEY,emb VECTOR({args.dim}));")
                _, duration = server.sql(path=import_path)
                count = server.sql(f"SELECT COUNT(*),MIN(id),MAX(id) FROM {table};")[0].strip().split("\t")
                if list(map(int,count)) != [n,int(ids.min()),int(ids.max())]:
                    raise RuntimeError(f"{server.mode} imported row/ID counts differ")
                info["imports"][server.mode] = {"seconds":duration,"count_min_max":count}
                persist()
            for round_number in range(1,args.rounds+1):
                order = servers if round_number%2 else list(reversed(servers))
                for position, server in enumerate(order, 1):
                    if round_number > 1:
                        server.sql(f"DROP INDEX {INDEX} ON {table};")
                    rec = {"rows":n,"round":round_number,"mode":server.mode,"order_position":position,
                           "index_parameters":dict(index_parameters)}
                    report["runs"].append(rec)
                    print(f"{now()} rows={n} round={round_number} mode={server.mode}", flush=True)
                    build_round(server,table,n,round_number,directory/f"round{round_number}-{server.mode}",
                                x,ids,queries,oracle,rec,persist,args.min_recall,index_parameters)
                    print(f"  build_to_queryable_seconds={rec['build_to_queryable_seconds']:.6f} "
                          f"recall@10={rec['quality']['recall_at_10']:.6f}", flush=True)

        if not args.skip_postchecks:
            n = min(args.sizes)
            data_dir = args.output / f"n{n}"
            x = np.fromfile(data_dir/"vectors.f32",dtype="<f4").reshape(n,args.dim)
            ids = np.fromfile(data_dir/"ids.i64",dtype="<i8")
            queries = np.fromfile(data_dir/"queries.f32",dtype="<f4").reshape(-1,args.dim)[:10]
            oracle = json.loads((data_dir/"oracle.json").read_text())[:len(queries)]
            report["postchecks"] = {}
            for server in servers:
                post = args.output / f"postcheck-{server.mode}"
                post.mkdir()
                record = {"status":"running","excluded_from_build_measurement":True}
                report["postchecks"][server.mode] = record
                persist()
                before = quality(server,f"vectors_{n}",x,ids,queries,oracle,post/"before-restart.json",args.min_recall)
                server.stop(); server.start()
                record["restart_explain"] = plan(server,f"vectors_{n}",queries[0],post/"restart-explain.txt")
                after = quality(server,f"vectors_{n}",x,ids,queries,oracle,post/"after-restart.json",args.min_recall)
                record["restart_same_ids"] = all(a["ids"]==b["ids"] for a,b in zip(before["samples"],after["samples"]))
                if not record["restart_same_ids"]:
                    raise RuntimeError(f"{server.mode} query IDs changed after restart")
                record["dml"] = dml_check(server,args.dim,args.seed,post/"dml",index_parameters)
                record["snapshot_dml_results"] = str(post/"snapshot-dml"/"results.json")
                persist()
                record["snapshot_dml"],snapshot_state = snapshot_dml_check(
                    server,f"vectors_{n}",x,ids,post/"snapshot-dml")
                persist()
                if args.check_rebuild:
                    record["rebuild_results"] = str(post/"rebuild"/"results.json")
                    persist()
                    record["rebuild"] = rebuild_check(server,f"vectors_{n}",snapshot_state,
                                                       post/"rebuild",args.rebuild_timeout)
                record["status"] = "ok"
                persist()
        report["status"] = "ok"
    except BaseException as e:
        report.update({"status":"failed","error":repr(e),"traceback":traceback.format_exc()})
        print(report["traceback"], file=sys.stderr, flush=True)
        exit_code = 130 if isinstance(e,KeyboardInterrupt) else 1
    finally:
        for server in reversed(servers):
            try:
                server.stop()
            except BaseException as e:
                report.setdefault("cleanup_errors",[]).append({"mode":server.mode,"error":repr(e)})
                report["status"] = "failed"
                exit_code = 1
            try:
                server.capture_logs({},server.output/"final-server")
            except BaseException as e:
                report.setdefault("log_capture_errors",[]).append({"mode":server.mode,"error":repr(e)})
                report["status"] = "failed"
                exit_code = 1
        report["finished_at"] = now()
        persist()
        print(f"results: {args.output/'results.json'} status={report['status']}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
