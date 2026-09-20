"""Explicit paths for the public PKUEEG 20260909 BIDS derivative layout."""
from pathlib import Path


def story_day(story: int) -> str:
    if not 1 <= story <= 50:
        raise ValueError(f"Story must be between 1 and 50: {story}")
    return "day1" if story <= 17 else "day2" if story <= 33 else "day3"


def derivative_path(root: Path, subject: str, *, story=None, day=None) -> Path:
    if story is not None:
        inferred = story_day(int(story))
        if day is not None and str(day).removeprefix("ses-") != inferred:
            raise ValueError(f"Story/day mismatch: {story}, {day}")
        day = inferred
        description = f"story{int(story):02d}"
    else:
        day = str(day).removeprefix("ses-")
        if day not in ("day1", "day2", "day3"):
            raise ValueError(f"Invalid rest day: {day}")
        description = "rest"
    session = f"ses-{day}"
    return (Path(root) / subject / session / "eeg" /
            f"{subject}_{session}_task-audio_desc-{description}_eeg.npz")


def normalized_channel(name) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return str(name).strip().upper()
