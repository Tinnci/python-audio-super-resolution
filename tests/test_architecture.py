from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_weight_store_does_not_depend_on_model_catalog_or_backends() -> None:
    imports = _module_imports(ROOT / "src" / "audio_super_resolution" / "weight_store.py")

    assert ".models" not in imports
    assert ".backends" not in imports
    assert "audio_super_resolution.models" not in imports
    assert "audio_super_resolution.backends" not in imports


def test_lavasr_backend_uses_spec_weight_store_not_model_facade() -> None:
    imports = _module_imports(ROOT / "src" / "audio_super_resolution" / "backends" / "lavasr_compat.py")

    assert "..weight_store" in imports
    assert "..model_weights" not in imports
    assert "..models" not in imports


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add("." * node.level + (node.module or ""))
    return imports
