"""Configuration-path resolution shared by every HGR command-line entry point."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Union


PathLike = Union[str, os.PathLike[str]]


def resolve_config_path(config_path: PathLike, *, repo_root: PathLike | None = None) -> Path:
    """Resolve a YAML configuration path using one explicit public contract.

    Canonical CLI values are repository-relative paths beginning with ``configs/``
    (for example, ``configs/generation/qm9/gvae.yaml``) or absolute paths.  The
    historical form relative to ``configs/`` remains available with a visible
    compatibility warning so existing internal jobs do not fail silently.
    """

    raw = os.fspath(config_path).strip()
    if not raw:
        raise ValueError("config path must not be empty")

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    supplied = Path(os.path.expandvars(raw)).expanduser()
    if supplied.suffix == "":
        supplied = supplied.with_suffix(".yaml")

    if supplied.is_absolute():
        candidates = [(supplied, False)]
    else:
        candidates = [(root / supplied, False)]
        if not supplied.parts or supplied.parts[0] != "configs":
            candidates.append((root / "configs" / supplied, True))

    attempted = []
    for candidate, legacy in candidates:
        resolved = candidate.resolve()
        attempted.append(str(resolved))
        if resolved.is_file():
            if legacy:
                warnings.warn(
                    "Config paths relative to configs/ are deprecated; pass the explicit "
                    f"repository-relative path 'configs/{supplied.as_posix()}'.",
                    UserWarning,
                    stacklevel=2,
                )
            return resolved

    attempted_text = "\n  - ".join(attempted)
    raise FileNotFoundError(
        f"Config file not found for {raw!r}. Tried:\n  - {attempted_text}"
    )
