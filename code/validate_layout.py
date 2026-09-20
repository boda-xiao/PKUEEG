#!/usr/bin/env python3
"""Read-only structural/header audit. No fits, full hashes or signal scans."""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import whosmat


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0): shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    else: shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    return {'shape': list(shape), 'dtype': str(dtype), 'object_dtype': bool(dtype.hasobject)}


def audit(root, compare_source=None):
    failures, notes = [], []
    subjects = [f'sub-{s:02d}' for s in range(1, 26)]
    expected_subjects = set(subjects)
    actual_subjects = {p.name for p in root.glob('sub-*') if p.is_dir()}
    if actual_subjects != expected_subjects: failures.append('Raw subject coverage differs from 25 subjects')
    raw_records = []
    for sub in subjects:
        for day, stories in [(1, range(1, 18)), (2, range(18, 34)), (3, range(34, 51))]:
            folder = root/sub/f'ses-day{day}'/'eeg'
            stem = f'{sub}_ses-day{day}_task-audio'
            files = {k: folder/f'{stem}_{v}' for k, v in {
                'header': 'eeg.vhdr', 'marker': 'eeg.vmrk', 'data': 'eeg.eeg',
                'sidecar': 'eeg.json', 'channels': 'channels.tsv', 'events': 'events.tsv'}.items()}
            missing = [str(p.relative_to(root)) for p in files.values() if not p.is_file()]
            if missing: failures.extend('Missing '+x for x in missing); continue
            header, marker = files['header'].read_text(), files['marker'].read_text()
            channels = int(re.search(r'^NumberOfChannels=(\d+)', header, re.M).group(1))
            sfreq = 1e6 / float(re.search(r'^SamplingInterval=([\d.]+)', header, re.M).group(1))
            bytes_per_sample = {'IEEE_FLOAT_32': 4, 'INT_16': 2, 'INT_32': 4}.get(re.search(r'^BinaryFormat=(\S+)', header, re.M).group(1))
            if not bytes_per_sample: failures.append('Unsupported raw binary format: '+stem); continue
            size = files['data'].stat().st_size
            if size % (channels * bytes_per_sample): failures.append('Incomplete raw sample frame: '+stem)
            duration = size / (channels * bytes_per_sample * sfreq)
            for name, text, keys in [('header', header, ('DataFile', 'MarkerFile')), ('marker', marker, ('DataFile',))]:
                for key in keys:
                    match = re.search(r'^'+key+r'=(.+)$', text, re.M)
                    if not match or not (folder/match.group(1).strip()).is_file(): failures.append('Broken BrainVision link: '+stem+'/'+key)
            timestamps = re.findall(r'^Mk\d+=[^\n]*,\d{20}\s*$', marker, re.M)
            if timestamps: failures.append('Acquisition timestamp present in marker: '+stem)
            meta = json.loads(files['sidecar'].read_text())
            if float(meta['SamplingFrequency']) != sfreq: failures.append('Sampling-frequency mismatch: '+stem)
            chans = table(files['channels'])
            if len(chans) != channels: failures.append('Channel count mismatch: '+stem)
            events = table(files['events'])
            story_ids = [int(x['story_id']) for x in events if x.get('story_id') not in ('', 'n/a', None)]
            if sorted(story_ids) != list(stories): failures.append('Story allocation mismatch: '+stem)
            if sum(x['trial_type'] == 'rest' for x in events) != 1: failures.append('Rest interval count mismatch: '+stem)
            for row in events:
                start, length = float(row['onset']), float(row['duration'])
                if start < 0 or length <= 0 or start+length > duration + 1/sfreq:
                    failures.append('Event outside raw data: '+stem)
                stimulus = row.get('stim_file')
                if stimulus not in (None, '', 'n/a') and not (root/'stimuli'/stimulus).is_file(): failures.append('Missing event stimulus: '+stimulus)
            raw_records.append({'participant_id': sub, 'session': f'ses-day{day}', 'channels': channels,
                                'sfreq': sfreq, 'duration_seconds_from_size': duration})

    eeg_counts, eeg_examples = {}, []
    for band in (40, 8):
        base = root/'derivatives'/f'preproc_{band}hz'
        expected = set()
        for sub in subjects:
            for story in range(1, 51):
                day = 1 if story <= 17 else 2 if story <= 33 else 3
                expected.add(f'{sub}/ses-day{day}/eeg/{sub}_ses-day{day}_task-audio_desc-story{story:02d}_eeg.npz')
            for day in range(1, 4): expected.add(f'{sub}/ses-day{day}/eeg/{sub}_ses-day{day}_task-audio_desc-rest_eeg.npz')
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*.npz')}
        missing, extra = sorted(expected-actual), sorted(actual-expected)
        failures.extend(f'{base.name}: missing {x}' for x in missing)
        failures.extend(f'{base.name}: unexpected {x}' for x in extra)
        eeg_counts[str(band)] = {'expected': len(expected), 'found': len(actual), 'missing': missing, 'extra': extra}
        for rel in sorted(actual):
            try:
                with zipfile.ZipFile(base/rel) as z:
                    if set(z.namelist()) != {'eeg_data.npy', 'ch_names.npy'}: failures.append('Unexpected NPZ keys: '+rel)
                    with z.open('eeg_data.npy') as f: data = array_header(f)
                    with z.open('ch_names.npy') as f: names = array_header(f)
                    if data['object_dtype'] or names['object_dtype']: failures.append('Object/pickle NPZ: '+rel)
                    if len(data['shape']) != 2 or data['shape'][0] != names['shape'][0] or data['shape'][1] <= 0: failures.append('Invalid NPZ dimensions: '+rel)
                    if data['dtype'] != 'float32': failures.append('Unexpected EEG dtype: '+rel)
                    if len(eeg_examples) < 4 or (band == 8 and len(eeg_examples) < 8): eeg_examples.append({'file': str((base/rel).relative_to(root)), **data})
            except Exception as e: failures.append('NPZ header error '+rel+': '+str(e))
    indexes = {}
    for relative in ['derivatives/stimulus_features/feature_manifest.tsv', 'derivatives/stimulus_annotations/annotation_index.tsv']:
        rows = table(root/relative)
        ids = [int(x['story_id']) for x in rows]
        missing, references = [], 0
        for row in rows:
            for key, value in row.items():
                if key.endswith('_file') and value not in ('', 'n/a'):
                    references += 1
                    if not (root/value).is_file(): missing.append(value)
        if sorted(ids) != list(range(1, 51)): failures.append('Story coverage mismatch: '+relative)
        failures.extend('Missing indexed file: '+x for x in missing)
        indexes[relative] = {'rows': len(rows), 'file_references': references, 'missing_references': missing}
    feature_base = root/'derivatives/stimulus_features'
    features = Counter()
    for p in feature_base.rglob('*'):
        if p.suffix == '.npy':
            try:
                with p.open('rb') as f: header = array_header(f)
                if header['object_dtype'] or any(x <= 0 for x in header['shape']): failures.append('Invalid NPY header: '+str(p.relative_to(root)))
            except Exception as e: failures.append('NPY header error '+p.name+': '+str(e))
            features[p.parent.relative_to(feature_base).as_posix()] += 1
        elif p.suffix == '.mat':
            try:
                matrices = whosmat(p)
                if not matrices: failures.append('Empty MAT file: '+p.name)
            except Exception as e: failures.append('MAT header error '+p.name+': '+str(e))
            features[p.parent.relative_to(feature_base).as_posix()] += 1
    participants = table(root/'participants.tsv')
    if {x['participant_id'] for x in participants} != expected_subjects or len(participants) != 25: failures.append('Participant-table coverage mismatch')
    missing_demographics = {k: sum(x[k] in ('', 'n/a') for x in participants) for k in ('age', 'sex')}
    scope_path = root/'code/config/release_scope.json'
    release_scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
    behavior_status = release_scope.get('behavioral_accuracy', 'included')
    if behavior_status not in ('included', 'deferred'):
        raise ValueError('Unknown behavioral_accuracy release-scope value')
    behavior, behavior_path, phenotype_rows = [], None, None
    if behavior_status == 'deferred':
        for folder in (root/'phenotype', root/'derivatives/behavior'):
            if folder.exists() and any(p.is_file() for p in folder.rglob('*')):
                failures.append('Behavior files present although release scope says deferred: '+str(folder.relative_to(root)))
        notes.append('Behavioral accuracy is deliberately deferred to a later update; its absence is not a missing-file failure for this release.')
    else:
        behavior_path = root/'derivatives/behavior/behavior_long.tsv'
        if not behavior_path.exists(): behavior_path = root/'phenotype/behavior.tsv'
        behavior = table(behavior_path)
        expected_pairs = {(sub, f'ses-day{day}') for sub in subjects for day in range(1, 4)}
        if len(behavior) != 75 or {(x['participant_id'], x['session_id']) for x in behavior} != expected_pairs: failures.append('Behavior coverage mismatch')
        if any(not 0 <= float(x['comprehension_accuracy']) <= 1 for x in behavior): failures.append('Behavior accuracy out of range')
        phenotype_rows = len(table(root/'phenotype/behavior.tsv'))
        if behavior_path != root/'phenotype/behavior.tsv':
            wide_behavior = table(root/'phenotype/behavior.tsv')
            old_values = {(x['participant_id'],x['session_id']):x['comprehension_accuracy'] for x in behavior}
            wide_values = {(x['participant_id'],f'ses-day{day}'):x[f'comprehension_accuracy_day{day}'] for x in wide_behavior for day in (1,2,3)}
            if len(wide_behavior) != 25 or wide_values != old_values: failures.append('Wide/long behavior values differ')
    if release_scope.get('historical_validation_reports') == 'excluded' and (root/'validation').exists():
        failures.append('Internal validation directory must remain outside the release')
    technical_status = release_scope.get('technical_validation', 'unspecified')
    if technical_status not in ('included', 'deferred', 'unspecified'):
        raise ValueError('Unknown technical_validation release-scope value')
    if technical_status == 'deferred':
        for folder in (root/'code/technical_validation', root/'derivatives/technical_validation'):
            if folder.exists():
                failures.append('Technical-validation directory present although release scope says deferred: '+str(folder.relative_to(root)))
        notes.append('Technical-validation analyses/results are deliberately deferred; structural/BIDS checking scripts remain included.')
    code = list((root/'code').rglob('*.py'))
    for p in code:
        try: ast.parse(p.read_text(), filename=str(p))
        except Exception as e: failures.append('Python syntax: '+p.name+': '+str(e))
    metadata_comparison = {}
    if compare_source:
        for rel in ['phenotype/behavior.tsv', 'phenotype/behavior.json', 'participants.tsv', 'participants.json']:
            if rel.startswith('phenotype/behavior.') and behavior_status == 'deferred': continue
            current = (root/rel)
            if rel.startswith('phenotype/behavior.') and behavior_path != root/'phenotype/behavior.tsv':
                current = behavior_path.with_suffix(Path(rel).suffix)
            if (compare_source/rel).exists(): metadata_comparison[rel] = current.read_bytes() == (compare_source/rel).read_bytes()
    if any(missing_demographics.values()):
        notes.append('Some age/sex metadata remain unavailable in this copy.')
    notes += [
        'word2vec and 100-Hz Mel/wav2vec2/BERT arrays are absent in this legacy release; not BIDS raw-format requirements, but incomplete against the planned expanded package.',
        'Pinned official model downloads and per-session ICA records have not been integrated here.',
        'The old preprocessing entry point has permissive bad-channel/montage matching and omits saved ICA records; the later repaired code was not substituted.',
        'requirements.txt does not declare all dependencies used by the legacy scripts (including scikit-learn, torch and transformers).',
        'CC0 is declared in the legacy root. Redistribution permission and CC0 eligibility of third-party text, synthesized audio and derived components require documented confirmation; no rights were changed by this audit.',
        'No subject-specific electrode coordinates or eye tracking were acquired; their absence is not a missing-data error.',
        'GitHub should contain code/documentation and the eventual OpenNeuro URL/DOI, not the EEG/audio/feature payload.',
    ]
    return {'audit_utc': datetime.now(timezone.utc).isoformat(), 'dataset': root.name,
            'scope': 'structure, metadata, raw byte counts, archive/array headers and code syntax',
            'structure_status': 'FAIL' if failures else 'PASS', 'failures': failures,
            'raw_recordings': len(raw_records), 'raw_header_summary': raw_records,
            'eeg_npz': eeg_counts, 'npz_header_examples': eeg_examples,
            'indexes': indexes, 'feature_array_counts': dict(features),
            'audio_files': len(list((root/'stimuli/audio').glob('*.mp3'))),
            'transcript_files': len(list((root/'stimuli/transcripts').glob('*.txt'))),
            'participants': len(participants), 'missing_demographic_values': missing_demographics,
            'sex_counts': dict(Counter(x['sex'] for x in participants)),
            'behavior_release_status': behavior_status,
            'technical_validation_release_status': technical_status,
            'behavior_rows': len(behavior) if behavior_status == 'included' else None,
            'behavior_long_path': behavior_path.relative_to(root).as_posix() if behavior_path else None,
            'phenotype_rows': phenotype_rows, 'python_files_syntax_checked': len(code),
            'metadata_equal_to_comparison_candidate': metadata_comparison,
            'publication_status': 'NOT_CLEARED', 'publication_notes': notes,
            'not_performed': ['full payload checksums', 'full finite-value scans', 'scientific experiment reruns',
                              'independent verification of behavior against response logs', 'legal certification', 'upload']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--compare-source', type=Path)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    report_path = a.report.resolve()
    root = a.root.resolve()
    if report_path == root or root in report_path.parents:
        raise ValueError('Write audit reports outside the dataset release directory')
    if a.report.exists(): raise FileExistsError(a.report)
    result = audit(a.root.resolve(), a.compare_source)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.report.open('x') as f: json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ['structure_status', 'failures', 'raw_recordings', 'eeg_npz', 'feature_array_counts', 'participants', 'behavior_rows', 'metadata_equal_to_comparison_candidate', 'publication_status']}, indent=2))
    if result['failures']: raise SystemExit(1)


if __name__ == '__main__': main()
