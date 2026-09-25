"""Command-line interface for the three-stage CPI pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .adapters import load_finding_set, normalize_path, read_json_object, write_json
from .injector import InjectionError, inject_repositories
from .payloads import generate_payloads


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpi",
        description="Normalize findings, construct three payloads, and deploy them.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    step1 = commands.add_parser("step1", help="LLM extraction and localization of raw findings.")
    step1.add_argument("--input", type=_path, required=True)
    step1.add_argument("--source-root", type=_path, required=True)
    step1.add_argument("--output", type=_path, required=True)

    step2 = commands.add_parser(
        "step2",
        help="Three LLM calls per finding: tool description, README, and code comment.",
    )
    step2.add_argument("--input", type=_path, required=True)
    step2.add_argument("--output", type=_path, required=True)

    step3 = commands.add_parser("step3", help="Deterministically deploy all three payloads.")
    step3.add_argument("--plan", type=_path, required=True)
    step3.add_argument("--source-root", type=_path, required=True)
    step3.add_argument("--output-root", type=_path, required=True)
    step3.add_argument("--overwrite", action="store_true")
    step3.add_argument("--report", type=_path)

    return parser


def _run(args: argparse.Namespace) -> dict:
    if args.command == "step1":
        findings = normalize_path(args.input, source_root=args.source_root)
        write_json(args.output, findings.to_dict())
        return {
            "step": 1,
            "output": str(args.output),
            "llm_calls": findings.metadata["llm_calls"],
            "findings": len(findings.findings),
        }

    if args.command == "step2":
        findings = load_finding_set(args.input)
        plan = generate_payloads(findings)
        write_json(args.output, plan)
        return {
            "step": 2,
            "output": str(args.output),
            "llm_calls": plan["metadata"]["llm_calls"],
            "payload_sets": len(plan["findings"]),
        }

    if args.command == "step3":
        plan = read_json_object(args.plan)
        report = inject_repositories(
            plan,
            source_root=args.source_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        if args.report:
            write_json(args.report, report)
        return report

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except (InjectionError, OSError, RuntimeError, ValueError) as exc:
        print(f"cpi: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
