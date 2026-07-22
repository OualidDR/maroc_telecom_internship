"""
quality/gx_context.py
=======================
Creates (or loads) the Great Expectations Data Context for this project.

GX 1.x has no CLI (`great_expectations init` no longer exists) -- everything
is done through the Python "Fluent" API. This script is the single place
that sets up the context, so every other quality script imports from here
instead of re-initializing GX each time.

Usage:
    python3 quality/gx_context.py     # sanity check: creates context, prints confirmation
"""

from pathlib import Path

import great_expectations as gx

GX_PROJECT_ROOT = Path(__file__).resolve().parent / "gx"


def get_context() -> gx.data_context.AbstractDataContext:
    """Get (or create on first run) a file-backed GX context.

    File-backed means config + validation results persist to disk under
    quality/gx/ -- so results survive between script runs, and are
    inspectable/committable (minus the actual data, which stays out of Git).
    """
    context = gx.get_context(mode="file", project_root_dir=str(GX_PROJECT_ROOT))
    return context


if __name__ == "__main__":
    ctx = get_context()
    print(f"GX Data Context ready at: {GX_PROJECT_ROOT}")
    print(f"Context type: {type(ctx).__name__}")
