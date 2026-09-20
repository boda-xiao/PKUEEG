"""Shared, fail-closed I/O for the new acoustic alignment and BERT recipe."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def new_output(path, *inputs):
    path = Path(path).resolve()
    for root in (HERE, RELEASE, *inputs):
        root = Path(root).resolve()
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError('Output must be disjoint from the release and all inputs')
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path

def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))

def write_tsv(path, rows, fields=None):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]),
                                delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def save_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def word_sequence_sha256(words):
    """Fingerprint lexical content and order independently of TSV timing columns."""
    return hashlib.sha256(
        json.dumps(words, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
