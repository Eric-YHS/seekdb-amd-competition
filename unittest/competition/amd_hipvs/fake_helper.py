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
import json
import os
import struct
import sys
import time

arguments = sys.argv[1:]
expected = ['--input-f32', '--input-ids', '--output', '--rows', '--dim', '--m', '--ef-construction', '--threads']
assert len(arguments) == 16, arguments
assert arguments[::2] == expected, arguments
options = dict(zip(arguments[::2], arguments[1::2]))
assert options['--rows'] == '2' and options['--dim'] == '2'
assert options['--m'] == '16' and options['--ef-construction'] == '100'
assert options['--threads'] == os.environ.get('SEEKDB_TEST_EXPECT_THREADS', '1')
assert struct.unpack('4f', open(options['--input-f32'], 'rb').read()) == (1, 2, 3, 4)
assert struct.unpack('2q', open(options['--input-ids'], 'rb').read()) == (100, 200)
assert os.stat(os.path.dirname(options['--output'])).st_mode & 0o777 == 0o700
mode = os.environ.get('SEEKDB_TEST_MODE', 'success')
output = options['--output']
if mode in ('nonzero_temp', 'signal_temp', 'timeout_temp'):
    with open(output + '.tmp.test', 'wb') as stream:
        stream.write(b'partial serialized graph\n' * 4096)
    os.symlink(os.environ['SEEKDB_TEST_SENTINEL'], output + '.tmp.symlink')
failure = mode[:-5] if mode.endswith('_temp') else mode
if failure == 'nonzero':
    sys.stderr.write('DISCARD_PREFIX' + 'x' * 6000 + 'KEEP_STDERR_TAIL\n')
    sys.exit(17)
if failure == 'signal':
    import signal
    sys.stderr.write('SIGNAL_STDERR\n')
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGTERM)
if failure == 'timeout':
    open(os.environ['SEEKDB_TEST_PIDFILE'], 'w').write(str(os.getpid()))
    sys.stderr.write('TIMEOUT_STDERR\n')
    sys.stderr.flush()
    time.sleep(30)
if mode == 'symlink':
    os.symlink(options['--input-f32'], output)
elif mode == 'fifo':
    os.mkfifo(output, 0o600)
elif mode == 'truncated':
    open(output, 'wb').write(b'tiny')
elif mode == 'oversized':
    with open(output, 'wb') as stream:
        stream.truncate(2 * 1024 * 1024)
elif mode == 'missing_output':
    pass
else:
    open(output, 'wb').write(b'H' * 128)
sys.stderr.write(json.dumps({'fixture': 'cpu-only', 'mode': mode, 'threads': options['--threads']}) + '\n')
