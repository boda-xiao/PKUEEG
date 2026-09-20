#!/usr/bin/env python3
"""Run an external BIDS validator and save versioned, strictly parsed evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from bids_report import parse_bids_report


def input_fingerprint(root):
    """Detect input changes; reports are required to remain outside the dataset."""
    rows = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append([rel.as_posix(), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-dir', type=Path, required=True, help='Report directory outside the dataset release')
    parser.add_argument('--validator', default=None, help='Executable name/path; prefers bids-validator-deno, then bids-validator')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing validation evidence')
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == root or root in output_dir.parents:
        raise ValueError('--output-dir must be outside the dataset release directory')
    executable = args.validator or shutil.which('bids-validator-deno') or shutil.which('bids-validator')
    if not executable:
        raise SystemExit('Install a BIDS validator externally or provide --validator')
    paths = [output_dir / name for name in
             ['bids_validator.json', 'bids_validator.stderr', 'bids_validator_run.json']]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Validation evidence exists; pass --overwrite explicitly')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    version = subprocess.run([executable, '--version'], capture_output=True, text=True, check=True, env=env)
    version_text = re.sub(r'\x1b\[[0-9;]*m', '', version.stdout + version.stderr).strip()
    started = datetime.now(timezone.utc).isoformat()
    before = input_fingerprint(root)
    completed = subprocess.run([executable, str(root), '--json'], capture_output=True, text=True, env=env)
    after = input_fingerprint(root)
    if before != after:
        raise RuntimeError('Release inputs changed during BIDS validation; evidence was not installed')
    report = json.loads(completed.stdout)
    parsed = parse_bids_report(report)
    if completed.returncode != 0 and not parsed['error_count']:
        raise RuntimeError(f'Unexpected validator exit {completed.returncode}; refusing a pass')
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()
    evidence = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'validator_version': version_text, 'exit_code': completed.returncode,
                'report_format': parsed['format'], 'error_count': parsed['error_count'],
                'warning_count': parsed['warning_count'], 'inputs_unchanged': True,
                'input_state_sha256': before, 'report_sha256': hashlib.sha256(report_bytes).hexdigest(),
                'scope': 'Raw BIDS scope defined by .bidsignore; not a scientific or legal certification'}
    for path, payload in zip(paths, [report_bytes, completed.stderr.encode(), (json.dumps(evidence, indent=2) + '\n').encode()]):
        temp = path.with_name(path.name + '.tmp')
        temp.write_bytes(payload)
        os.replace(temp, path)
    print(json.dumps(evidence, indent=2))
    if parsed['error_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
