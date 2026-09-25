from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


README_NAMES = ("README.md", "README.MD", "readme.md", "README.rst", "README.txt", "README")
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}
SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json"}
JAVASCRIPT_SUFFIXES = {".js", ".mjs", ".cjs"}
TYPESCRIPT_SUFFIXES = {".jsx", ".ts", ".tsx"}
COMMENT_PREFIX = {
    ".py": "#", ".sh": "#", ".rb": "#", ".pl": "#", ".r": "#",
    ".js": "//", ".jsx": "//", ".ts": "//", ".tsx": "//", ".mjs": "//", ".cjs": "//",
    ".java": "//", ".go": "//", ".rs": "//", ".c": "//", ".h": "//", ".cpp": "//",
    ".cs": "//", ".swift": "//", ".kt": "//", ".kts": "//", ".php": "//",
    ".sql": "--", ".lua": "--", ".hs": "--", ".html": "<!--", ".xml": "<!--",
}


class InjectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCandidate:
    path: Path
    start: int
    end: int
    line: int
    tool_name: str
    kind: str
    literal: str


def inject_repositories(
    plan: dict[str, Any],
    source_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    findings = plan.get("findings")
    if plan.get("schema_version") != "cpi.plan.v1" or not isinstance(findings, list):
        raise ValueError("Expected a cpi.plan.v1 object with a findings list")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in findings:
        if isinstance(item, dict):
            grouped[str(item.get("subject") or "unknown")].append(item)
    if not grouped:
        raise ValueError("The plan contains no findings")

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise InjectionError("Output root must not be inside the source repository root")
    output_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "cpi.injection-report.v1",
        "metadata": {"step": 3, "llm_calls": 0},
        "surfaces": ["tool_description", "readme", "code_comment"],
        "repositories": [],
        "totals": {"repositories": 0, "readme": 0, "tool_description": 0, "code_comment": 0, "unresolved": 0},
    }
    single_subject = len(grouped) == 1
    for subject, subject_findings in sorted(grouped.items()):
        source_repo = source_root / subject
        if not source_repo.is_dir() and single_subject and source_root.is_dir():
            source_repo = source_root
        if not source_repo.is_dir():
            raise FileNotFoundError(f"Source repository not found for {subject}: {source_repo}")
        target_repo = output_root / subject
        _prepare_target(source_repo, target_repo, output_root, overwrite)
        repository_report = _inject_one(subject, target_repo, subject_findings)
        report["repositories"].append(repository_report)
        report["totals"]["repositories"] += 1
        for key in ("readme", "tool_description", "code_comment", "unresolved"):
            report["totals"][key] += repository_report["counts"][key]
    return report


def _prepare_target(source: Path, target: Path, output_root: Path, overwrite: bool) -> None:
    resolved_target = target.resolve()
    if resolved_target.parent != output_root or not target.name:
        raise InjectionError(f"Unsafe output target: {target}")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {target}; pass overwrite=True to replace it")
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns(*SKIP_DIRS))


def _inject_one(subject: str, repo: Path, findings: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"readme": 0, "tool_description": 0, "code_comment": 0, "unresolved": 0}
    unresolved: list[dict[str, str]] = []
    changed_files: set[str] = set()

    readme_plans: dict[Path, list[tuple[int, str, str]]] = defaultdict(list)
    for finding in findings:
        payload = _payload(finding, "readme")
        if not payload:
            continue
        readme = _resolve_readme(repo, subject, finding)
        line = _integer(finding.get("readme_line")) or 1
        readme_plans[readme].append((line, payload, str(finding.get("finding_id") or "")))
    for path, annotations in readme_plans.items():
        applied = _insert_readme(path, annotations)
        counts["readme"] += applied
        if applied:
            changed_files.add(path.relative_to(repo).as_posix())

    comment_plans: dict[Path, list[tuple[int, str, str]]] = defaultdict(list)
    for finding in findings:
        payload = _payload(finding, "code_comment")
        if not payload:
            continue
        locations = finding.get("locations") if isinstance(finding.get("locations"), list) else []
        site: tuple[Path | None, int | None] = (None, None)
        for location in locations:
            if isinstance(location, dict):
                site = _resolve_comment_site(repo, subject, finding, location)
                if site[0] is not None and site[1] is not None:
                    break
        if site[0] is None:
            target = _find_unique_function_source(repo, finding)
            site = (target, _fallback_comment_line(target, finding)) if target is not None else (None, None)
        target, line = site
        if target is None or line is None:
            counts["unresolved"] += 1
            unresolved.append({"finding_id": str(finding.get("finding_id")), "surface": "code_comment"})
            continue
        comment_plans[target].append((line, payload, str(finding.get("finding_id") or "")))
    for path, annotations in comment_plans.items():
        applied, failed = _insert_code_comments(path, annotations)
        counts["code_comment"] += applied
        counts["unresolved"] += len(failed)
        unresolved.extend({"finding_id": item, "surface": "code_comment"} for item in failed)
        if applied:
            changed_files.add(path.relative_to(repo).as_posix())

    candidates = _discover_tool_candidates(repo)
    replacements: dict[ToolCandidate, list[str]] = defaultdict(list)
    for finding in findings:
        payload = _payload(finding, "tool_description")
        if not payload:
            continue
        selected = _select_tool_candidates(repo, candidates, finding)
        if not selected:
            counts["unresolved"] += 1
            unresolved.append({"finding_id": str(finding.get("finding_id")), "surface": "tool_description"})
            continue
        for candidate in selected:
            replacements[candidate].append(payload)
    applied_paths, applied_count = _apply_tool_replacements(replacements)
    counts["tool_description"] += applied_count
    changed_files.update(path.relative_to(repo).as_posix() for path in applied_paths)

    return {
        "subject": subject,
        "output": str(repo),
        "counts": counts,
        "changed_files": sorted(changed_files),
        "unresolved": unresolved,
    }


def _payload(finding: dict[str, Any], surface: str) -> str:
    payloads = finding.get("payloads")
    return str(payloads.get(surface) or "").strip() if isinstance(payloads, dict) else ""


def _find_readme(repo: Path) -> Path:
    for name in README_NAMES:
        candidate = repo / name
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    fallback = repo / "README.md"
    if fallback.is_symlink():
        raise InjectionError(f"Refusing to edit symlinked README: {fallback}")
    return fallback


def _resolve_readme(repo: Path, subject: str, finding: dict[str, Any]) -> Path:
    requested = _resolve_location(repo, subject, str(finding.get("readme_path") or ""))
    if requested is not None and requested.name.lower() in {name.lower() for name in README_NAMES}:
        return requested
    return _find_readme(repo)


def _insert_readme(path: Path, annotations: list[tuple[int, str, str]]) -> int:
    text, encoding = _read_text(path) if path.exists() else ("", "utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    applied = 0
    seen = set()
    pending: list[tuple[int, str]] = []
    for requested_line, payload, _ in annotations:
        if not payload or payload in text or payload in seen:
            continue
        seen.add(payload)
        effective_line = requested_line if 1 <= requested_line <= len(lines) + 1 else 1
        pending.append((effective_line, payload))
    for line_number, payload in sorted(pending, key=lambda item: item[0], reverse=True):
        rendered = payload.splitlines() + [""]
        lines[line_number - 1:line_number - 1] = rendered
        applied += 1
    if applied:
        updated = newline.join(lines)
        if text.endswith(("\n", "\r")) or not updated.endswith(newline):
            updated += newline
        path.write_text(updated, encoding=encoding)
    return applied


def _resolve_location(repo: Path, subject: str, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    normalized = raw_path.replace("\\", "/")
    marker = f"/{subject}/"
    if marker in normalized:
        normalized = normalized.rsplit(marker, 1)[1]
    candidate = Path(normalized)
    if candidate.is_absolute():
        return None
    target = (repo / candidate).resolve()
    if not target.is_relative_to(repo.resolve()) or not target.is_file():
        return None
    return target


def _resolve_comment_site(
    repo: Path,
    subject: str,
    finding: dict[str, Any],
    location: dict[str, Any],
) -> tuple[Path | None, int | None]:
    target = _resolve_location(repo, subject, str(location.get("file_path") or ""))
    line = _integer(location.get("line_start"))
    if target is not None:
        declaration = _comment_declaration_line(_read_text(target)[0], _finding_symbols(finding))
        if declaration is not None:
            return target, declaration
        line_count = len(_read_text(target)[0].splitlines())
        if line is not None and 1 <= line <= max(1, line_count):
            return target, line
        return target, _fallback_comment_line(target, finding)

    target = _find_unique_function_source(repo, finding)
    return (target, _fallback_comment_line(target, finding)) if target is not None else (None, None)


def _find_unique_function_source(repo: Path, finding: dict[str, Any]) -> Path | None:
    symbols = _finding_symbols(finding)
    if not symbols:
        return None
    matches = []
    for path in sorted(repo.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix.lower() not in COMMENT_PREFIX
            or any(part in SKIP_DIRS for part in path.relative_to(repo).parts)
            or path.stat().st_size > 2 * 1024 * 1024
        ):
            continue
        text, _ = _read_text(path)
        if _declaration_line(text, symbols) is not None:
            matches.append(path)
    return matches[0] if len(matches) == 1 else None


def _fallback_comment_line(path: Path, finding: dict[str, Any]) -> int:
    text, _ = _read_text(path)
    symbols = _finding_symbols(finding)
    declaration = _comment_declaration_line(text, symbols)
    if declaration is not None:
        return declaration
    for number, line in enumerate(text.splitlines(), 1):
        if any(re.search(rf"(?<![\w$]){re.escape(symbol)}(?![\w$])", line) for symbol in symbols):
            return number
    return _safe_file_level_line(path, text)


def _finding_symbols(finding: dict[str, Any]) -> list[str]:
    symbol = str(finding.get("function_name") or "").strip()
    return [symbol] if re.fullmatch(r"[A-Za-z_$][\w$.-]*", symbol) else []


def _declaration_line(text: str, symbols: list[str]) -> int | None:
    for symbol in symbols:
        escaped = re.escape(symbol)
        patterns = (
            re.compile(rf"^\s*(?:async\s+)?def\s+{escaped}\b"),
            re.compile(rf"^\s*(?:export\s+)?(?:async\s+)?function\s+{escaped}\b"),
            re.compile(rf"^\s*(?:export\s+)?(?:const|let|var)\s+{escaped}\s*="),
            re.compile(rf"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*{escaped}\s*\("),
        )
        for number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns):
                return number
    return None


def _comment_declaration_line(text: str, symbols: list[str]) -> int | None:
    declaration = _declaration_line(text, symbols)
    if declaration is None:
        return None
    lines = text.splitlines()
    while declaration > 1 and lines[declaration - 2].lstrip().startswith("@"):
        declaration -= 1
    return declaration


def _safe_file_level_line(path: Path, text: str) -> int:
    lines = text.splitlines()
    offset = 0
    if lines and lines[0].startswith("#!"):
        offset = 1
    if path.suffix.lower() == ".py":
        while offset < min(2, len(lines)) and re.search(r"coding[:=]\s*[-\w.]+", lines[offset]):
            offset += 1
    if path.suffix.lower() == ".xml" and lines and lines[0].lstrip().startswith("<?xml"):
        offset = max(offset, 1)
    return offset + 1


def _insert_code_comments(path: Path, annotations: list[tuple[int, str, str]]) -> tuple[int, list[str]]:
    prefix = COMMENT_PREFIX.get(path.suffix.lower())
    if prefix is None:
        return 0, [finding_id for _, _, finding_id in annotations]
    text, encoding = _read_text(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    applied = 0
    failed: list[str] = []
    for line_number, payload, finding_id in sorted(annotations, reverse=True):
        if line_number < 1 or line_number > len(lines) + 1:
            failed.append(finding_id)
            continue
        target_line = lines[line_number - 1] if line_number <= len(lines) else ""
        indent = re.match(r"\s*", target_line).group(0)
        comment = _format_comment(prefix, payload)
        rendered = indent + comment
        if rendered in lines[max(0, line_number - 2) : line_number + 1]:
            continue
        lines.insert(line_number - 1, rendered)
        applied += 1
    if applied:
        trailing = newline if text.endswith(("\n", "\r")) else ""
        updated = newline.join(lines) + trailing
        _validate_text(path, updated)
        path.write_text(updated, encoding=encoding)
    return applied, failed


def _format_comment(prefix: str, payload: str) -> str:
    if prefix == "<!--":
        return f"<!-- {payload} -->"
    return f"{prefix} {payload}"


def _discover_tool_candidates(repo: Path) -> list[ToolCandidate]:
    result: list[ToolCandidate] = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(repo).parts):
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        text, _ = _read_text(path)
        if path.suffix.lower() == ".py":
            result.extend(_python_candidates(path, text))
        result.extend(_text_candidates(path, text))
    deduped: dict[tuple[Path, int, int], ToolCandidate] = {}
    for candidate in result:
        deduped.setdefault((candidate.path, candidate.start, candidate.end), candidate)
    return list(deduped.values())


def _python_candidates(path: Path, text: str) -> list[ToolCandidate]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    result: list[ToolCandidate] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(_is_tool_decorator(item) for item in node.decorator_list):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                start, end = _ast_span(text, node.body[0].value)
                result.append(ToolCandidate(path, start, end, node.body[0].lineno, node.name, "python_docstring", text[start:end]))
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "description" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                            start, end = _ast_span(text, keyword.value)
                            result.append(ToolCandidate(path, start, end, keyword.value.lineno, node.name, "decorator_description", text[start:end]))
        if isinstance(node, ast.Call) and _call_name(node.func).split(".")[-1] == "tool":
            name = None
            description = None
            for keyword in node.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    name = keyword.value.value
                if keyword.arg == "description" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    description = keyword.value
            if name and description:
                start, end = _ast_span(text, description)
                result.append(ToolCandidate(path, start, end, description.lineno, name, "python_tool_call", text[start:end]))
    return result


STRING_LITERAL = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`'


def _text_candidates(path: Path, text: str) -> list[ToolCandidate]:
    pattern = re.compile(
        rf"(?:[\"']?(?:name|toolName)[\"']?)\s*:\s*(?P<name>{STRING_LITERAL})"
        rf"(?P<middle>[^{{}}]{{0,1200}}?)(?:[\"']?description[\"']?)\s*:\s*(?P<description>{STRING_LITERAL})",
        flags=re.DOTALL,
    )
    result: list[ToolCandidate] = []
    for match in pattern.finditer(text):
        literal = match.group("description")
        start, end = match.span("description")
        result.append(ToolCandidate(
            path, start, end, text.count("\n", 0, start) + 1,
            _unquote(match.group("name")), "object_description", literal,
        ))
    return result


def _select_tool_candidates(repo: Path, candidates: list[ToolCandidate], finding: dict[str, Any]) -> list[ToolCandidate]:
    tool_name = str(finding.get("tool_name") or "").strip().lower()
    if tool_name:
        exact = [item for item in candidates if item.tool_name.lower() == tool_name]
        if exact:
            return _nearest(exact, repo, finding)
    location_paths = {
        path for location in finding.get("locations", []) if isinstance(location, dict)
        for path in [_relative_location(repo, str(location.get("file_path") or ""))] if path
    }
    same_file = [item for item in candidates if item.path.relative_to(repo).as_posix() in location_paths]
    if same_file:
        return _nearest(same_file, repo, finding)
    return candidates if len(candidates) == 1 else []


def _nearest(candidates: list[ToolCandidate], repo: Path, finding: dict[str, Any]) -> list[ToolCandidate]:
    locations = finding.get("locations") if isinstance(finding.get("locations"), list) else []
    scored: list[tuple[int, ToolCandidate]] = []
    for candidate in candidates:
        score = 10_000
        rel = candidate.path.relative_to(repo).as_posix()
        for location in locations:
            if not isinstance(location, dict) or _relative_location(repo, str(location.get("file_path") or "")) != rel:
                continue
            line = _integer(location.get("line_start"))
            score = min(score, abs(candidate.line - line) if line is not None else 0)
        scored.append((score, candidate))
    best = min(score for score, _ in scored)
    return [candidate for score, candidate in scored if score == best]


def _relative_location(repo: Path, raw: str) -> str | None:
    normalized = raw.replace("\\", "/")
    marker = f"/{repo.name}/"
    if marker in normalized:
        return normalized.rsplit(marker, 1)[1]
    path = Path(normalized)
    return path.as_posix() if normalized and not path.is_absolute() and ".." not in path.parts else None


def _apply_tool_replacements(replacements: dict[ToolCandidate, list[str]]) -> tuple[set[Path], int]:
    by_path: dict[Path, list[tuple[ToolCandidate, list[str]]]] = defaultdict(list)
    for candidate, payloads in replacements.items():
        by_path[candidate.path].append((candidate, payloads))
    changed: set[Path] = set()
    applied = 0
    for path, items in by_path.items():
        text, encoding = _read_text(path)
        updated = text
        for candidate, payloads in sorted(items, key=lambda item: item[0].start, reverse=True):
            replacement = _append_literal(candidate.literal, payloads)
            if replacement == candidate.literal:
                continue
            updated = updated[: candidate.start] + replacement + updated[candidate.end :]
            applied += 1
        if updated != text:
            _validate_text(path, updated)
            path.write_text(updated, encoding=encoding)
            changed.add(path)
    return changed, applied


def _append_literal(literal: str, payloads: list[str]) -> str:
    match = re.match(r"(?s)^([rRuU]*)(\"\"\"|'''|\"|'|`)(.*)\2$", literal)
    if not match:
        raise InjectionError(f"Unsupported description literal: {literal[:80]!r}")
    prefix, quote, content = match.groups()
    additions = [item for item in dict.fromkeys(payloads) if item and item not in content]
    if not additions:
        return literal
    addition = "\n\n" + "\n\n".join(additions)
    if quote in {"\"\"\"", "'''", "`"}:
        if quote == "`":
            addition = addition.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
        else:
            addition = addition.replace(quote, "\\" + quote)
        return f"{prefix}{quote}{content.rstrip()}{addition}{quote}"
    escaped = addition.replace("\\", "\\\\").replace("\r", "").replace("\n", "\\n").replace(quote, "\\" + quote)
    return f"{prefix}{quote}{content}{escaped}{quote}"


def _is_tool_decorator(node: ast.AST) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return _call_name(target).split(".")[-1] == "tool"


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _ast_span(text: str, node: ast.AST) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(line_number: int, byte_column: int) -> int:
        line = lines[line_number - 1]
        prefix = line.encode("utf-8")[:byte_column].decode("utf-8")
        return starts[line_number - 1] + len(prefix)

    return offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)


def _unquote(literal: str) -> str:
    if literal.startswith("`"):
        return literal[1:-1]
    try:
        value = ast.literal_eval(literal)
        return value if isinstance(value, str) else ""
    except Exception:
        return ""


def _read_text(path: Path) -> tuple[str, str]:
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeError:
            continue
    raise UnicodeError(f"Cannot decode {path}")


def _validate_text(path: Path, text: str) -> None:
    if path.suffix.lower() == ".py":
        ast.parse(text, filename=str(path))
    elif path.suffix.lower() == ".json":
        json.loads(text)
    elif path.suffix.lower() in JAVASCRIPT_SUFFIXES:
        _validate_javascript(path, text)
    elif path.suffix.lower() in TYPESCRIPT_SUFFIXES:
        _validate_typescript(path, text)


def _validate_javascript(path: Path, text: str) -> None:
    node = shutil.which("node")
    if node is None:
        raise InjectionError(f"Node.js is required to syntax-check {path}")
    suffix = path.suffix.lower()
    modes = ("module",) if suffix == ".mjs" else ("commonjs",) if suffix == ".cjs" else ("commonjs", "module")
    errors = []
    for mode in modes:
        process = subprocess.run(
            [node, "--check", f"--input-type={mode}", "-"],
            input=text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode == 0:
            return
        errors.append(process.stderr.strip())
    raise InjectionError(f"JavaScript syntax check failed for {path}: {' | '.join(errors)}")


def _validate_typescript(path: Path, text: str) -> None:
    node = shutil.which("node")
    if node is None:
        raise InjectionError(f"Node.js is required to syntax-check {path}")
    node_modules = [
        str(parent / "node_modules")
        for parent in (path.parent, *path.parents)
        if (parent / "node_modules").is_dir()
    ]
    environment = os.environ.copy()
    if node_modules:
        existing = environment.get("NODE_PATH")
        environment["NODE_PATH"] = os.pathsep.join(node_modules + ([existing] if existing else []))
    parser = r"""
const fs = require("fs");
const ts = require("typescript");
const suffix = process.argv[1];
const kinds = {
  ".js": ts.ScriptKind.JS,
  ".jsx": ts.ScriptKind.JSX,
  ".ts": ts.ScriptKind.TS,
  ".tsx": ts.ScriptKind.TSX,
};
const source = fs.readFileSync(0, "utf8");
const file = ts.createSourceFile("cpi-check" + suffix, source, ts.ScriptTarget.Latest, true, kinds[suffix]);
if (file.parseDiagnostics.length) {
  for (const item of file.parseDiagnostics) {
    const point = file.getLineAndCharacterOfPosition(item.start || 0);
    console.error((point.line + 1) + ":" + (point.character + 1) + " " + ts.flattenDiagnosticMessageText(item.messageText, "\n"));
  }
  process.exit(1);
}
"""
    process = subprocess.run(
        [node, "-e", parser, path.suffix.lower()],
        input=text,
        text=True,
        cwd=path.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip()
        if "Cannot find module 'typescript'" in detail:
            raise InjectionError(
                f"TypeScript is required to syntax-check {path}; install it in the repository or make it available through NODE_PATH"
            )
        raise InjectionError(f"TypeScript syntax check failed for {path}: {detail}")


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
