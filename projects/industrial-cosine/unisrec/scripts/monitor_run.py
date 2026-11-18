#!/usr/bin/env python3
"""Observe a retained Docker training container and atomically export its files.

This script never starts, stops, or removes containers. All Docker access uses
argument lists, and only regular completed checkpoint/output files are published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def load_spec(run_dir):
    spec = read_json(run_dir / 'run_spec.json')
    if not isinstance(spec, dict):
        raise ValueError('Missing or invalid run_spec.json: ' + str(run_dir))
    required = ('container', 'container_run_dir', 'container_checkpoint_dir',
                'container_source_dir', 'host_checkpoint_dir')
    for key in required:
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError('run_spec.json requires a nonempty ' + key)
    if any(not spec[key].startswith('/') for key in required[1:]):
        raise ValueError('Run and checkpoint directories must be absolute paths')
    if not isinstance(spec.get('poll_seconds', 20), (int, float)) or spec.get('poll_seconds', 20) <= 0:
        raise ValueError('poll_seconds must be positive')
    return spec


def docker(arguments, timeout=180):
    result = subprocess.run([os.environ.get('DOCKER', 'docker'), *arguments],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout)
    if result.returncode:
        error = result.stderr.decode(errors='replace').strip()
        raise RuntimeError('docker ' + arguments[0] + ': ' + error[-2000:])
    return result


def inspect_state(spec):
    result = docker(['inspect', '--format', '{{json .State}}', spec['container']], timeout=30)
    state = json.loads(result.stdout)
    if not isinstance(state, dict):
        raise ValueError('Docker inspect did not return a container state')
    return state


def complete_name(name):
    lower = name.lower()
    return not (name.startswith('.') or lower.endswith(('.tmp', '.partial', '.part', '~'))
                or '.partial.' in lower or '.tmp.' in lower)


def publish_directory(spec, source, destination, run_dir, *, checkpoints=False):
    """Stage docker cp on the destination filesystem, then publish each file."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    with tempfile.TemporaryDirectory(prefix='.sync-temp-', dir=run_dir) as temporary:
        staging = Path(temporary) / 'files'
        staging.mkdir()
        docker(['cp', spec['container'] + ':' + source.rstrip('/') + '/.', str(staging)], timeout=300)
        for candidate in sorted(staging.rglob('*')):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(staging)
            if any(not complete_name(part) for part in relative.parts):
                continue
            if checkpoints and not (
                    candidate.name.endswith('.pth')
                    or candidate.name.endswith('.vector_quality.jsonl')
                    or candidate.name.endswith('.result.json')):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate, target)
            copied.append(str(target))
    return copied


def publish_logs(spec, run_dir, *, final=False):
    filename = 'train.log' if final else 'train.tail.log'
    descriptor, temporary = tempfile.mkstemp(prefix='.' + filename + '-', dir=run_dir)
    arguments = [os.environ.get('DOCKER', 'docker'), 'logs']
    if not final:
        arguments += ['--tail', '2000']
    arguments.append(spec['container'])
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            result = subprocess.run(arguments, stdout=handle, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            reason = Path(temporary).read_text(errors='replace')[-2000:]
            raise RuntimeError('docker logs: ' + reason)
        os.replace(temporary, run_dir / filename)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def latest_vector_quality(checkpoint_dir):
    latest = None
    latest_order = None
    for path in Path(checkpoint_dir).rglob('*.vector_quality.jsonl'):
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
            # Live JSONL may have one incomplete final line. Keep the latest
            # valid complete record instead of replacing progress with an error.
            record = None
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                except ValueError:
                    continue
                if isinstance(candidate, dict):
                    record = candidate
                    break
            if record is not None:
                order = (path.stat().st_mtime_ns, str(path))
                if latest_order is None or order > latest_order:
                    latest = {'file': str(path), **record}
                    latest_order = order
        except OSError:
            continue
    return latest


def sync_once(spec, run_dir, *, state=None):
    errors, copied = [], []
    if state is None:
        state = inspect_state(spec)
    for source, destination, checkpoints in (
            (spec['container_run_dir'], run_dir / 'runtime', False),
            (spec['container_checkpoint_dir'], Path(spec['host_checkpoint_dir']), True)):
        try:
            copied.extend(publish_directory(spec, source, destination, run_dir,
                                             checkpoints=checkpoints))
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    final = not state.get('Running') and state.get('Status') not in ('created', 'restarting')
    try:
        publish_logs(spec, run_dir, final=final)
        if final:
            # Retain the convenient bounded log alongside the complete final log.
            publish_logs(spec, run_dir, final=False)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        errors.append(str(error))
    checkpoint_dir = Path(spec['host_checkpoint_dir'])
    weights = sorted(str(path) for path in checkpoint_dir.rglob('*.pth') if path.is_file())
    results = sorted(str(path) for path in checkpoint_dir.rglob('*.result.json') if path.is_file())
    exit_code = int(state.get('ExitCode', 0)) if final else None
    if final:
        completed = exit_code == 0 and bool(weights) and bool(results) and not errors
        status = 'completed' if completed else 'failed'
        if exit_code != 0:
            errors.append('Container training process exited with code ' + str(exit_code))
        if not weights:
            errors.append('No completed .pth checkpoint exported')
        if not results:
            errors.append('No completed .result.json exported')
    else:
        status = 'running'
    previous = read_json(run_dir / 'run_status.json', {})
    record = {
        'status': status, 'container': spec['container'], 'container_state': state.get('Status'),
        'exit_code': exit_code, 'updated_at': now(), 'last_sync_at': now(),
        'monitor_pid': os.getpid(), 'checkpoint_files': weights, 'result_files': results,
        'latest_vector_quality': latest_vector_quality(checkpoint_dir),
        'sync_errors': errors, 'copied_file_count': len(copied),
        'container_retained': True,
    }
    record['started_at'] = previous.get('started_at', state.get('StartedAt') or now())
    if final:
        record['finished_at'] = state.get('FinishedAt') or now()
    atomic_json(run_dir / 'run_status.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('watch', 'status', 'sync'))
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error('run-dir does not exist: ' + str(run_dir))
    if args.command == 'status':
        record = read_json(run_dir / 'run_status.json', {'status': 'not_monitored'})
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0 if record.get('status') != 'failed' else 1
    spec = load_spec(run_dir)
    while True:
        try:
            state = inspect_state(spec)
            record = sync_once(spec, run_dir, state=state)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            record = read_json(run_dir / 'run_status.json', {})
            record.update({'status': 'failed', 'updated_at': now(), 'monitor_pid': os.getpid(),
                           'sync_errors': [str(error)], 'container_retained': True})
            atomic_json(run_dir / 'run_status.json', record)
            print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
            return 1
        if args.command == 'sync' or record['status'] != 'running':
            print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
            return 0 if record['status'] != 'failed' else 1
        time.sleep(float(spec.get('poll_seconds', 20)))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
