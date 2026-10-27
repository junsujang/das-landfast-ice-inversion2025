"""Small helpers shared by the scripts: argparse booleans and atomic .npz writes."""

import argparse
import os
from pathlib import Path

import numpy as np


def str2bool(v):
    """argparse `type=` for booleans: yes/true/t/1/y and no/false/f/0/n."""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def savez_atomic(out_path: Path, **arrays) -> None:
    """np.savez_compressed via a temp file in the same directory, then replace.

    A job killed by the wall clock or the OOM reaper mid-write otherwise leaves a
    truncated archive that looks like a finished block: `out_path.exists()` skips
    it on the next run, and every reader downstream has to defend against it.
    Writing beside the target and moving it into place makes the output either
    absent or complete -- os.replace is atomic within a filesystem. The temp name
    carries the PID so concurrent array tasks cannot collide on it.
    """
    out_path = Path(out_path)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp{os.getpid()}.npz")

    try:
        # Write through a handle: given a path, np.savez_compressed appends
        # ".npz" unless the name already ends in it, and would rename the temp
        # out from under os.replace.
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
