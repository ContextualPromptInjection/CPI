"""Step 1: use an LLM to normalize complete scanner reports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import Finding, FindingSet, Location
from .payloads import ChatFunction, chat_json


ALLOWED_SUFFIXES = {".txt", ".log", ".json", ".md", ".yaml", ".yml", ".xml", ".html", ".csv"}
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
SOURCE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".go",
    ".rs", ".c", ".h", ".cpp", ".cs", ".swift", ".kt", ".kts", ".php",
    ".rb", ".sh", ".pl", ".r", ".sql", ".lua", ".hs",
}
SOURCE_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}
README_NAMES = {"readme", "readme.md", "readme.mdx", "readme.rst", "readme.txt"}
MAX_CONTEXT_FILES = 12
MAX_CONTEXT_LINES = 240

STEP1_SYSTEM_PROMPT = """
You are the finding-extraction stage of an authorized security-scanner
robustness experiment. Read the complete raw scanner report and normalize every
actual security finding. Do not invent evidence. Return exactly one JSON object
with this schema:
{
  "findings": [
    {
      "risk_name": "string",
      "description": "string",
      "tool_name": "string or null",
      "function_name": "affected function name or null",
      "severity": "string or null",
      "readme_path": "repository-relative README path or null",
      "readme_line": "integer insertion line or null",
      "locations": [
        {
          "file_path": "string",
          "line_start": "integer or null",
          "line_end": "integer or null"
        }
      ]
    }
  ]
}
Preserve multiple locations belonging to one finding. If there is no actual
finding, return {"findings": []}. When repository source evidence is supplied,
use it to resolve a missing MCP tool, affected function, precise source region,
and a suitable README insertion line. Every returned path, symbol, and line
number must refer to the supplied repository evidence. If the report and
repository evidence do not support a placement, return null rather than
inventing one. Output JSON only.
""".strip()


def normalize_path(
    input_path: Path,
    chat: ChatFunction | None = None,
    source_root: Path | None = None,
) -> FindingSet:
    """Normalize each input report with one independent LLM call."""
    files = _input_files(input_path)
    if not files:
        raise FileNotFoundError(f"No supported report files found below {input_path}")
    call = chat or chat_json
    resolved_source_root = source_root.resolve() if source_root is not None else None
    findings: list[Finding] = []
    subjects: list[str] = []
    statuses: list[dict[str, Any]] = []
    llm_calls = 0

    for path in files:
        subject = _subject_name(path)
        subjects.append(subject)
        raw_text = _read_text(path)
        score = _reported_score(raw_text)
        if score == 100:
            statuses.append({"file": str(path), "status": "skipped", "reason": "score == 100"})
            continue

        repository = (
            _repository_for_subject(resolved_source_root, subject, len(files) == 1)
            if resolved_source_root is not None else None
        )
        source_context = _build_source_context(repository, raw_text) if repository is not None else None
        user_prompt = (
            f"File name: {path.name}\n\n"
            "COMPLETE RAW REPORT START\n"
            f"{raw_text}\n"
            "COMPLETE RAW REPORT END"
        )
        if source_context is not None:
            user_prompt += (
                "\n\nREPOSITORY SOURCE EVIDENCE START\n"
                f"{json.dumps(source_context, ensure_ascii=False, indent=2)}\n"
                "REPOSITORY SOURCE EVIDENCE END"
            )
        response = call(STEP1_SYSTEM_PROMPT, user_prompt, 0.0)
        llm_calls += 1
        normalized = _normalize_response(
            response,
            subject,
            path,
            score,
            repository=repository,
            report_text=raw_text,
        )
        findings.extend(normalized)
        statuses.append({"file": str(path), "status": "success", "findings": len(normalized)})

    findings = _deduplicate(findings)
    return FindingSet(
        scanner="llm-normalized",
        subjects=sorted(set(subjects)),
        findings=findings,
        metadata={
            "step": 1,
            "input": str(input_path),
            "source_root": str(resolved_source_root) if resolved_source_root is not None else None,
            "files_read": len(files),
            "llm_calls": llm_calls,
            "files": statuses,
        },
    )


def load_finding_set(path: Path) -> FindingSet:
    value = _read_json_object(path)
    if value.get("schema_version") != "cpi.findings.v1":
        raise ValueError(f"Expected cpi.findings.v1 in {path}")
    return FindingSet.from_dict(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> dict[str, Any]:
    return _read_json_object(path)


def _normalize_response(
    response: dict[str, Any],
    subject: str,
    source_path: Path,
    score: float | None,
    repository: Path | None = None,
    report_text: str = "",
) -> list[Finding]:
    raw_findings = response.get("findings")
    if not isinstance(raw_findings, list):
        raise ValueError("Step 1 model response must contain a findings list")
    result: list[Finding] = []
    for index, item in enumerate(raw_findings, 1):
        if not isinstance(item, dict):
            continue
        risk_name = _text(item.get("risk_name")) or "UNKNOWN"
        description = _text(item.get("description")) or ""
        tool_name = _text(item.get("tool_name"))
        if tool_name is None and repository is not None:
            tool_name = _infer_reported_tool(report_text, repository)
        locations = _resolve_locations(
            item.get("locations"),
            repository,
            report_text,
            tool_name,
            description,
        )
        function_name = _text(item.get("function_name"))
        if function_name is not None and repository is not None:
            function_name = _validated_function_name(repository, locations, function_name)
        if function_name is None and repository is not None:
            function_name = _infer_function_from_locations(repository, locations)
        if tool_name is None and repository is not None:
            tool_name = _infer_tool_from_locations(repository, locations)
        readme_path, readme_line = _resolve_readme_placement(
            repository,
            item.get("readme_path"),
            item.get("readme_line"),
        )
        finding_id = _stable_id(subject, risk_name, description, locations, tool_name)
        result.append(Finding(
            finding_id=finding_id,
            subject=subject,
            scanner="llm-normalized",
            risk_name=risk_name,
            description=description,
            severity=_text(item.get("severity")),
            score=score,
            tool_name=tool_name,
            function_name=function_name,
            readme_path=readme_path,
            readme_line=readme_line,
            locations=locations,
            source={"file": str(source_path), "index": index},
        ))
    return result


def _repository_for_subject(source_root: Path, subject: str, single_report: bool) -> Path:
    candidate = (source_root / subject).resolve()
    if candidate.is_dir() and candidate.is_relative_to(source_root):
        return candidate
    if single_report and source_root.is_dir():
        return source_root
    raise FileNotFoundError(
        f"Repository source not found for report subject {subject!r} below {source_root}"
    )


def _build_source_context(repository: Path, report_text: str) -> dict[str, Any]:
    source_files = _source_files(repository)
    report_normalized = report_text.replace("\\", "/").lower()
    symbols = _reported_symbols(report_text)
    reported_lines = _reported_line_numbers(report_text)
    ranked: list[tuple[int, str, Path, str, list[int]]] = []

    for path in source_files:
        relative = path.relative_to(repository).as_posix()
        path_mentioned = relative.lower() in report_normalized
        name_mentioned = path.name.lower() in report_normalized
        if not path_mentioned and not name_mentioned and not symbols:
            continue
        text = _read_source_text(path)
        symbol_lines = _symbol_lines(text, symbols)
        if path_mentioned:
            rank = 0
        elif name_mentioned:
            rank = 1
        elif symbol_lines:
            rank = 2
        else:
            continue
        hits = list(symbol_lines)
        if path_mentioned or name_mentioned:
            hits.extend(reported_lines)
        ranked.append((rank, relative, path, text, sorted(set(hits))))

    files = []
    for _, relative, _, text, hits in sorted(ranked)[:MAX_CONTEXT_FILES]:
        files.append({
            "path": relative,
            "candidate_lines": hits,
            "source": _render_source_excerpt(text, hits),
        })
    readmes = []
    for path in _readme_files(repository):
        text = _read_source_text(path)
        readmes.append({
            "path": path.relative_to(repository).as_posix(),
            "source": _render_source_excerpt(text, []),
        })
    return {
        "repository": repository.name,
        "source_inventory": [
            path.relative_to(repository).as_posix() for path in source_files[:1000]
        ],
        "referenced_source_files": files,
        "readme_files": readmes,
    }


def _resolve_readme_placement(
    repository: Path | None,
    raw_path: Any,
    raw_line: Any,
) -> tuple[str | None, int | None]:
    if repository is None:
        return _text(raw_path), _integer(raw_line)
    readmes = _readme_files(repository)
    selected: Path | None = None
    requested = _text(raw_path)
    if requested:
        normalized = requested.replace("\\", "/")
        candidate = Path(normalized)
        if not candidate.is_absolute() and ".." not in candidate.parts:
            resolved = (repository / candidate).resolve()
            if resolved.is_relative_to(repository) and resolved.is_file() and not resolved.is_symlink():
                selected = resolved
        if selected is None:
            matches = [path for path in readmes if path.name.lower() == candidate.name.lower()]
            if len(matches) == 1:
                selected = matches[0]
    elif readmes:
        selected = readmes[0]
    if selected is None:
        return None, None
    line = _integer(raw_line)
    line_count = len(_read_source_text(selected).splitlines())
    if line is None or line < 1 or line > line_count + 1:
        line = None
    return selected.relative_to(repository).as_posix(), line


def _resolve_locations(
    raw_locations: Any,
    repository: Path | None,
    report_text: str,
    tool_name: str | None,
    description: str,
) -> list[Location]:
    items = raw_locations if isinstance(raw_locations, list) else []
    if repository is None:
        return [
            Location.from_dict(item)
            for item in items
            if isinstance(item, dict) and _text(item.get("file_path"))
        ]

    locations: list[Location] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = _resolve_source_file(repository, str(item.get("file_path") or item.get("path") or ""))
        if path is None:
            continue
        line_count = len(_read_source_text(path).splitlines())
        line_start = _integer(item.get("line_start") or item.get("line"))
        if line_start is None or line_start < 1 or line_start > max(1, line_count):
            line_start = _best_source_line(path, report_text, tool_name, description)
        line_end = _integer(item.get("line_end"))
        if line_start is not None and (
            line_end is None or line_end < line_start or line_end > max(1, line_count)
        ):
            line_end = line_start
        locations.append(Location(
            file_path=path.relative_to(repository).as_posix(),
            line_start=line_start,
            line_end=line_end,
        ))

    if not locations:
        for path in _reported_source_files(repository, report_text, tool_name, description):
            line = _best_source_line(path, report_text, tool_name, description)
            locations.append(Location(
                file_path=path.relative_to(repository).as_posix(),
                line_start=line,
                line_end=line,
            ))

    deduplicated: dict[tuple[str, int | None, int | None], Location] = {}
    for location in locations:
        deduplicated[(location.file_path, location.line_start, location.line_end)] = location
    return list(deduplicated.values())


def _reported_source_files(
    repository: Path,
    report_text: str,
    tool_name: str | None,
    description: str,
) -> list[Path]:
    normalized = report_text.replace("\\", "/").lower()
    direct: list[Path] = []
    for path in _source_files(repository):
        relative = path.relative_to(repository).as_posix().lower()
        if relative in normalized or path.name.lower() in normalized:
            direct.append(path)
    if direct:
        return direct[:MAX_CONTEXT_FILES]

    symbols = list(dict.fromkeys(filter(None, [
        tool_name,
        *_reported_symbols(report_text),
        *_reported_symbols(description),
    ])))
    matches = [
        path for path in _source_files(repository)
        if _definition_lines(_read_source_text(path), symbols)
    ]
    return matches[:MAX_CONTEXT_FILES]


def _resolve_source_file(repository: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    normalized = raw_path.replace("\\", "/")
    marker = f"/{repository.name}/"
    if marker in normalized:
        normalized = normalized.rsplit(marker, 1)[1]
    candidate = Path(normalized)
    if not candidate.is_absolute() and ".." not in candidate.parts:
        target = (repository / candidate).resolve()
        if target.is_relative_to(repository) and target.is_file():
            return target
    matches = [path for path in _source_files(repository) if path.name == candidate.name]
    return matches[0] if len(matches) == 1 else None


def _best_source_line(
    path: Path,
    report_text: str,
    tool_name: str | None,
    description: str,
) -> int | None:
    text = _read_source_text(path)
    line_count = len(text.splitlines())
    for line in _reported_line_numbers(report_text):
        if 1 <= line <= line_count:
            return line
    symbols = list(dict.fromkeys(filter(None, [
        tool_name,
        *_reported_symbols(report_text),
        *_reported_symbols(description),
    ])))
    definitions = _definition_lines(text, symbols)
    if definitions:
        return definitions[0]
    occurrences = _symbol_lines(text, symbols)
    return occurrences[0] if occurrences else None


def _infer_reported_tool(report_text: str, repository: Path) -> str | None:
    candidates = _reported_symbols(report_text)
    for symbol in candidates:
        if any(_definition_lines(_read_source_text(path), [symbol]) for path in _source_files(repository)):
            return symbol
    return None


def _infer_tool_from_locations(repository: Path, locations: list[Location]) -> str | None:
    for location in locations:
        path = _resolve_source_file(repository, location.file_path)
        if path is None:
            continue
        name = _function_name_near(_read_source_text(path), location.line_start)
        if name:
            return name
    return None


def _validated_function_name(
    repository: Path,
    locations: list[Location],
    function_name: str,
) -> str | None:
    located_files = [
        path
        for location in locations
        for path in [_resolve_source_file(repository, location.file_path)]
        if path is not None
    ]
    candidates = located_files or _source_files(repository)
    return function_name if any(
        _definition_lines(_read_source_text(path), [function_name]) for path in candidates
    ) else None


def _infer_function_from_locations(repository: Path, locations: list[Location]) -> str | None:
    for location in locations:
        path = _resolve_source_file(repository, location.file_path)
        if path is None:
            continue
        name = _function_name_near(_read_source_text(path), location.line_start)
        if name:
            return name
    return None


def _function_name_near(text: str, line_number: int | None) -> str | None:
    lines = text.splitlines()
    if not lines:
        return None
    end = min(len(lines), line_number or len(lines))
    patterns = (
        re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\b"),
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\b"),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="),
        re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*([A-Za-z_$][\w$]*)\s*\("),
    )
    for line in reversed(lines[:end]):
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                return match.group(1)
    return None


def _reported_symbols(text: str) -> list[str]:
    patterns = (
        re.compile(
            r"(?i)\b(?:tool|function|method|handler)\s+(?:named\s+)?[\x60'\"]?([A-Za-z_$][\w$.-]{2,})"
        ),
        re.compile(r"\b([A-Za-z_$][\w$.-]{2,})\s*\("),
        re.compile(r"[\x60'\"]([A-Za-z_$][\w$.-]{2,})[\x60'\"]"),
    )
    ignored = {
        "and", "are", "can", "class", "code", "file", "for", "from", "function",
        "into", "line", "method", "return", "risk", "server", "that", "the", "this",
        "tool", "using", "with",
    }
    result: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            symbol = match.group(1)
            if symbol.lower() not in ignored and symbol not in result:
                result.append(symbol)
            if len(result) == 40:
                return result
    return result


def _reported_line_numbers(text: str) -> list[int]:
    result = []
    for match in re.finditer(r"(?i)\blines?\s*[:#]?\s*(\d+)", text):
        value = int(match.group(1))
        if value not in result:
            result.append(value)
    for match in re.finditer(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+:(\d+)\b", text):
        value = int(match.group(1))
        if value not in result:
            result.append(value)
    return result


def _definition_lines(text: str, symbols: list[str]) -> list[int]:
    lines = text.splitlines()
    result: list[int] = []
    for symbol in symbols:
        escaped = re.escape(symbol)
        patterns = (
            re.compile(rf"^\s*(?:async\s+)?def\s+{escaped}\b"),
            re.compile(rf"^\s*(?:export\s+)?(?:async\s+)?function\s+{escaped}\b"),
            re.compile(rf"^\s*(?:export\s+)?(?:const|let|var)\s+{escaped}\s*="),
            re.compile(rf"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*{escaped}\s*\("),
        )
        for number, line in enumerate(lines, 1):
            if any(pattern.search(line) for pattern in patterns):
                result.append(number)
    return sorted(set(result))


def _symbol_lines(text: str, symbols: list[str]) -> list[int]:
    if not symbols:
        return []
    patterns = [re.compile(rf"(?<![\w$]){re.escape(symbol)}(?![\w$])") for symbol in symbols]
    return [
        number for number, line in enumerate(text.splitlines(), 1)
        if any(pattern.search(line) for pattern in patterns)
    ]


def _render_source_excerpt(text: str, hits: list[int]) -> str:
    lines = text.splitlines()
    if len(lines) <= MAX_CONTEXT_LINES:
        selected = range(1, len(lines) + 1)
    else:
        line_numbers: set[int] = set()
        for hit in hits:
            line_numbers.update(range(max(1, hit - 30), min(len(lines), hit + 30) + 1))
        if not line_numbers:
            line_numbers.update(range(1, min(len(lines), MAX_CONTEXT_LINES) + 1))
        selected = sorted(line_numbers)[:MAX_CONTEXT_LINES]
    return "\n".join(f"{number}: {lines[number - 1]}" for number in selected)


def _source_files(repository: Path) -> list[Path]:
    return [
        path for path in sorted(repository.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and not any(part in SOURCE_SKIP_DIRS for part in path.relative_to(repository).parts)
        and path.stat().st_size <= 1_048_576
    ]


def _readme_files(repository: Path) -> list[Path]:
    return sorted(
        (
            path for path in repository.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.name.lower() in README_NAMES
            and not any(part in SOURCE_SKIP_DIRS for part in path.relative_to(repository).parts)
            and path.stat().st_size <= 1_048_576
        ),
        key=lambda path: (len(path.relative_to(repository).parts), path.as_posix().lower()),
    )[:4]


def _read_source_text(path: Path) -> str:
    try:
        return _read_text(path)
    except UnicodeError:
        return ""


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        item for item in path.rglob("*")
        if item.is_file() and (item.suffix.lower() in ALLOWED_SUFFIXES or not item.suffix)
    )


def _read_text(path: Path) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    raise UnicodeError(f"Cannot decode {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _reported_score(text: str) -> float | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None

    def find(item: Any) -> float | None:
        if isinstance(item, dict):
            if "score" in item:
                try:
                    return float(item["score"])
                except (TypeError, ValueError):
                    pass
            for child in item.values():
                score = find(child)
                if score is not None:
                    return score
        elif isinstance(item, list):
            for child in item:
                score = find(child)
                if score is not None:
                    return score
        return None

    return find(value)


def _stable_id(
    subject: str,
    risk_name: str,
    description: str,
    locations: list[Location],
    tool_name: str | None,
) -> str:
    material = json.dumps(
        [subject, risk_name, description, [asdict(item) for item in locations], tool_name],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _deduplicate(findings: list[Finding]) -> list[Finding]:
    return sorted(
        {finding.finding_id: finding for finding in findings}.values(),
        key=lambda finding: (finding.subject, finding.finding_id),
    )


def _subject_name(path: Path) -> str:
    return path.name[: -len(path.suffix)] if path.suffix else path.name


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
