from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Location:
    file_path: str
    line_start: int | None = None
    line_end: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Location":
        return cls(
            file_path=str(value.get("file_path") or value.get("path") or ""),
            line_start=_integer(value.get("line_start") or value.get("line")),
            line_end=_integer(value.get("line_end")),
        )


@dataclass
class Finding:
    finding_id: str
    subject: str
    scanner: str
    risk_name: str
    description: str
    severity: str | None = None
    score: float | None = None
    tool_name: str | None = None
    function_name: str | None = None
    readme_path: str | None = None
    readme_line: int | None = None
    locations: list[Location] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Finding":
        return cls(
            finding_id=str(value.get("finding_id") or value.get("id") or ""),
            subject=str(value.get("subject") or value.get("server_name") or "unknown"),
            scanner=str(value.get("scanner") or "generic"),
            risk_name=str(value.get("risk_name") or value.get("name") or "UNKNOWN"),
            description=str(value.get("description") or ""),
            severity=_optional_text(value.get("severity")),
            score=_number(value.get("score")),
            tool_name=_optional_text(value.get("tool_name")),
            function_name=_optional_text(value.get("function_name")),
            readme_path=_optional_text(value.get("readme_path")),
            readme_line=_integer(value.get("readme_line")),
            locations=[
                Location.from_dict(item)
                for item in value.get("locations", [])
                if isinstance(item, dict)
            ],
            source=dict(value.get("source") or {}),
        )


@dataclass
class FindingSet:
    scanner: str
    subjects: list[str]
    findings: list[Finding]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "cpi.findings.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FindingSet":
        findings = [
            Finding.from_dict(item)
            for item in value.get("findings", [])
            if isinstance(item, dict)
        ]
        subjects = [str(item) for item in value.get("subjects", []) if str(item)]
        if not subjects:
            subjects = sorted({finding.subject for finding in findings})
        return cls(
            scanner=str(value.get("scanner") or "generic"),
            subjects=subjects,
            findings=findings,
            metadata=dict(value.get("metadata") or {}),
            schema_version=str(value.get("schema_version") or "cpi.findings.v1"),
        )


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
