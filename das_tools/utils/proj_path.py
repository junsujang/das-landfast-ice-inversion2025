"""Where the project's inputs and outputs live.

Every default path in the scripts and notebooks comes from here and is relative
to the project root, so the repository works wherever it is cloned:

    <root>/data                              inputs (small files tracked in git)
    <root>/data/DAS/pisces25_hammer_signals  raw Onyx hammer records
    <root>/data/DAS/pisces_2025_Fs5Hz        decimated hourly files
    <root>/outputs/<pipeline>                everything the code writes (git-ignored)
    <root>/figures                           figures for the paper

To put any of them elsewhere -- e.g. data and results on a cluster's scratch
disk -- set the matching environment variable; it applies to every script and
notebook at once:

    DAS_LFI_PROJECT_ROOT    the root itself
    DAS_LFI_DATA_DIR        <root>/data
    DAS_LFI_HAMMER_DIR      <data>/DAS/pisces25_hammer_signals
    DAS_LFI_HOURLY_DIR      <data>/DAS/pisces_2025_Fs5Hz
    DAS_LFI_OUTPUT_DIR      <root>/outputs
    DAS_LFI_FIGURES_DIR     <root>/figures

e.g.  export DAS_LFI_OUTPUT_DIR=/scratch/$USER/das_lfi_outputs

A script's own command-line flag (--data-root, --results-dir, ...) still wins
over both. Paths are looked up when a function is called, not at import.
"""

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[1]   # .../das_tools


def _is_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "das_tools").is_dir()


def find_project_root(start=None) -> Path:
    """The repository root: the folder holding pyproject.toml and das_tools/.

    Found from `start` if given; otherwise from where das_tools itself is
    installed (an editable install points back into the repo), then from the
    current directory. DAS_LFI_PROJECT_ROOT overrides the search.
    """
    if start is None and os.environ.get("DAS_LFI_PROJECT_ROOT"):
        return Path(os.environ["DAS_LFI_PROJECT_ROOT"]).expanduser().resolve()

    starts = [Path(start)] if start is not None else [_PACKAGE_DIR, Path.cwd()]
    for s in starts:
        s = s.resolve()
        for parent in [s, *s.parents]:
            if _is_root(parent):
                return parent

    raise RuntimeError(
        "Could not find the project root (a folder with pyproject.toml and das_tools/). "
        "Install the package with `pip install -e .` from the repository, run from "
        "inside it, or set DAS_LFI_PROJECT_ROOT."
    )


def _from_env(var: str, default: Path, parts) -> Path:
    base = os.environ.get(var)
    base = Path(base).expanduser() if base else default
    return base.joinpath(*parts)


def data_dir(*parts) -> Path:
    """<root>/data, or DAS_LFI_DATA_DIR."""
    return _from_env("DAS_LFI_DATA_DIR", find_project_root() / "data", parts)


def hammer_das_dir(*parts) -> Path:
    """Raw Onyx hammer-test records, or DAS_LFI_HAMMER_DIR."""
    return _from_env("DAS_LFI_HAMMER_DIR", data_dir("DAS", "pisces25_hammer_signals"), parts)


def hourly_das_dir(*parts) -> Path:
    """Decimated hourly DAS files the beamforming reads, or DAS_LFI_HOURLY_DIR."""
    return _from_env("DAS_LFI_HOURLY_DIR", data_dir("DAS", "pisces_2025_Fs5Hz"), parts)


def output_dir(*parts) -> Path:
    """<root>/outputs, or DAS_LFI_OUTPUT_DIR; e.g. output_dir("bf_inv_disp")."""
    return _from_env("DAS_LFI_OUTPUT_DIR", find_project_root() / "outputs", parts)


def figures_dir(*parts) -> Path:
    """<root>/figures, or DAS_LFI_FIGURES_DIR."""
    return _from_env("DAS_LFI_FIGURES_DIR", find_project_root() / "figures", parts)
