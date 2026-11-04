#!/usr/bin/env python3
# Copyright (c) 2025 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile and exercise the production process bridge with a CPU fixture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3],
        help="Repository containing src/oblib and the bridge source",
    )
    parser.add_argument("--cxx", default="g++", help="C++17 compiler executable")
    parser.add_argument(
        "--sanitizers", action="store_true",
        help="Enable AddressSanitizer and UndefinedBehaviorSanitizer",
    )
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Run this process-bridge test on Linux")

    fixtures = Path(__file__).resolve().parent
    oblib = args.repo.resolve() / "src" / "oblib"
    bridge = oblib / "lib" / "vector" / "ob_hipvs_builder.cpp"
    for source in (bridge, fixtures / "bridge_driver.cpp", fixtures / "fake_helper.py"):
        if not source.is_file():
            parser.error(f"Required source is missing: {source}")

    with tempfile.TemporaryDirectory(prefix="seekdb hipvs bridge test ") as directory:
        temporary = Path(directory)
        helper = temporary / "helper with spaces.py"
        helper.write_text(
            (fixtures / "fake_helper.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        helper.chmod(0o700)
        executable = temporary / "bridge driver"
        command = [
            args.cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-I", str(oblib), str(bridge), str(fixtures / "bridge_driver.cpp"),
            "-o", str(executable),
        ]
        if args.sanitizers:
            command += [
                "-g", "-O1", "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
            ]
        else:
            command += ["-O2"]
        print("COMPILE " + shlex.join(command), flush=True)
        subprocess.run(command, check=True, timeout=120)
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("SEEKDB_HIPVS_", "SEEKDB_TEST_"))
        }
        environment.pop("PYTHONOPTIMIZE", None)
        result = subprocess.run(
            [str(executable), str(temporary)], env=environment, timeout=30,
        )
        return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Process-bridge test failed: {error}", file=sys.stderr)
        raise SystemExit(1)
