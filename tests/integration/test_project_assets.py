from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema.validators import validator_for  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_environment_is_named_brain_and_targets_python_312() -> None:
    environment = _load_yaml(ROOT / "environment.yml")
    assert environment["name"] == "brain"
    dependencies = [str(item) for item in environment["dependencies"]]
    assert any(item.startswith("python=3.12") for item in dependencies)


def test_all_yaml_configuration_files_parse() -> None:
    config_files = sorted((ROOT / "configs").rglob("*.yml")) + sorted(
        (ROOT / "configs").rglob("*.yaml")
    )
    assert config_files, "At least one version-controlled YAML config is required"
    for path in config_files:
        assert _load_yaml(path) is not None, f"Empty configuration: {path}"


def test_all_json_schemas_are_valid() -> None:
    schema_files = sorted((ROOT / "schemas").rglob("*.json"))
    assert schema_files, "At least one JSON Schema draft is required"
    for path in schema_files:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator = validator_for(schema)
        validator.check_schema(schema)


def test_telemetry_schema_remains_an_unfrozen_draft() -> None:
    path = ROOT / "schemas" / "telemetry-v1.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["x-status"] == "DRAFT"
    assert schema["properties"]["schema_version"]["const"] == "telemetry-v1-draft"


def test_p0_templates_remain_explicit_drafts() -> None:
    template_names = (
        "evidence_registry.md",
        "module_hypotheses.md",
        "computational_graph.md",
        "module_lifecycle.md",
        "experiment_protocols.md",
        "baseline_spec.md",
        "statistical_protocol.md",
        "experiment_artifacts.md",
    )
    for name in template_names:
        path = ROOT / "docs" / name
        content = path.read_text(encoding="utf-8")
        headings = [line for line in content.splitlines() if line.startswith("#")]
        assert "DRAFT" in content, f"P0 template must be marked DRAFT: {path}"
        assert len(headings) >= 3, f"P0 template lacks required structure: {path}"
