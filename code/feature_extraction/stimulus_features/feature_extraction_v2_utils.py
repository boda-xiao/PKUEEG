"""Release-layout and output-safety helpers for the additive v2 extractors."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parents[2]


def validate_output(release, output):
    """Only a new external output directory is permitted (including symlinks)."""
    release, output = Path(release).resolve(), Path(output).resolve()
    for protected in (release, RELEASE_ROOT.resolve()):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Output must be outside, and must not contain, the release')
    if output.exists():
        raise FileExistsError('Use a new output directory')
    return release, output


def validate_stories(stories):
    values = sorted(set(stories))
    if not values or any(value < 1 or value > 50 for value in values):
        raise ValueError('Story IDs must be in 1..50')
    return values


def released_reference(release, generated_relative_path):
    """Map generated v2 files to current 0909 references, never guess new grids."""
    path = Path(generated_relative_path)
    if len(path.parts) != 3 or path.parts[0] != 'features':
        return None
    family, filename = path.parts[1:]
    directories = {
        'envelope_100hz': 'envelope/envelope_100hz',
        'mel_50hz': 'mel/mel_50hz',
        'wav2vec2_layer9_50hz': 'wav2vec2/wav2vec2_layer9_50hz',
        'word2vec_100hz': 'word2vec_100hz',
    }
    if family not in directories:
        return None
    return Path(release) / 'derivatives/stimulus_features' / directories[family] / filename
