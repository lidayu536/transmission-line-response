"""Compatibility shim for local `_ComputingPackages` imports.

The installable package source lives in `src/transmission_line_response`.  This
small shim keeps existing local workflows working when the parent directory of
this repository is placed on `PYTHONPATH`.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PACKAGE = Path(__file__).resolve().parent / "src" / "transmission_line_response"
if not _SOURCE_PACKAGE.exists():
    raise ImportError(f"Cannot find package source directory: {_SOURCE_PACKAGE}")

source_path = str(_SOURCE_PACKAGE)
if source_path not in __path__:
    __path__.insert(0, source_path)

_init_file = _SOURCE_PACKAGE / "__init__.py"
exec(compile(_init_file.read_text(encoding="utf-8"), str(_init_file), "exec"), globals())