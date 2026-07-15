from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "neuromorphic"
FORBIDDEN_IMPORT_PREFIXES = {
    "matplotlib",
    "nilearn",
    "plotly",
    "pythreejs",
    "three",
    "trimesh",
    "websocket",
    "websockets",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_network_package_has_no_visualization_or_3d_dependency() -> None:
    violations: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        forbidden = sorted(
            name
            for name in _imports(path)
            if name.split(".", maxsplit=1)[0] in FORBIDDEN_IMPORT_PREFIXES
            or name == "visualization"
            or name.startswith("neuromorphic.visualization")
        )
        if forbidden:
            violations[str(path.relative_to(ROOT))] = forbidden
    assert not violations
    assert not (PACKAGE_ROOT / "visualization").exists()
