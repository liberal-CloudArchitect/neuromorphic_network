"""Schema-aligned artificial model telemetry with no visualization dependency."""

from neuromorphic.telemetry.events import (
    SCHEMA_VERSION,
    SCIENTIFIC_DISCLAIMER,
    TelemetryEdge,
    TelemetryEvent,
)
from neuromorphic.telemetry.events_v2 import (
    SCHEMA_VERSION_V2,
    SCIENTIFIC_DISCLAIMER_V2,
    TelemetryV2Event,
    detached_scalar,
)

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V2",
    "SCIENTIFIC_DISCLAIMER",
    "SCIENTIFIC_DISCLAIMER_V2",
    "TelemetryEdge",
    "TelemetryEvent",
    "TelemetryV2Event",
    "detached_scalar",
]
