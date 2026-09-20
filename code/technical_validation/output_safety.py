"""Keep analyses outside the immutable release and reject accidental overwrites."""
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = RELEASE_ROOT.parent / (RELEASE_ROOT.name + '_outputs') / 'technical_validation'

def check_output(path, *inputs, allow_existing=False):
    path = Path(path).resolve()
    for root in (RELEASE_ROOT, *inputs):
        root = Path(root).resolve()
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError(f'Output overlaps a protected input: {path}')
    if path.exists() and not allow_existing:
        raise FileExistsError(f'Use a new output path: {path}')
    return path
