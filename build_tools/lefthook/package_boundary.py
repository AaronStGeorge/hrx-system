# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks that Bazel packages do not own sources in descendant directories."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as element_tree
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

BUILD_FILENAMES = ("BUILD.bazel", "BUILD")
POLICY_TARGET_NAME = "package_boundary_policy"
VIOLATION_DISPLAY_LIMIT = 20


class PackageBoundaryQueryError(RuntimeError):
    """Raised when Bazel cannot describe the touched packages."""


@dataclass(frozen=True)
class PackageBoundaryPolicy:
    allowed_descendant_paths: tuple[str, ...]
    reason: str
    location: str


@dataclass(frozen=True)
class PackageBoundaryResult:
    package: str
    violations: tuple[str, ...]
    policy_errors: tuple[str, ...]


@dataclass(frozen=True)
class PackageBoundaryInspection:
    results: tuple[PackageBoundaryResult, ...]

    @property
    def ok(self) -> bool:
        return not any(
            result.violations or result.policy_errors for result in self.results
        )


def _parse_label(label: str) -> tuple[str, str] | None:
    if not label.startswith("//") or ":" not in label:
        return None
    package, target = label[2:].split(":", 1)
    return package, target


def _package_label(package: str) -> str:
    return f"//{package}" if package else "//"


def _query_pattern(package: str) -> str:
    return f"//{package}:*" if package else "//:*"


def query_expression(packages: set[str]) -> str:
    """Returns a query expression selecting rules and source files in packages."""
    return "set(%s)" % " ".join(_query_pattern(package) for package in sorted(packages))


def touched_packages(repository_root: Path, paths: list[str]) -> set[str]:
    """Finds the nearest current Bazel package owning each changed path."""
    packages = set()
    for path in paths:
        relative_path = PurePosixPath(path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"path is not repository-relative: {path}")
        absolute_path = repository_root.joinpath(*relative_path.parts)
        directory = absolute_path if absolute_path.is_dir() else absolute_path.parent
        while directory == repository_root or repository_root in directory.parents:
            if any((directory / name).is_file() for name in BUILD_FILENAMES):
                relative_package = directory.relative_to(repository_root)
                packages.add(
                    "" if relative_package == Path(".") else relative_package.as_posix()
                )
                break
            if directory == repository_root:
                break
            directory = directory.parent
    return packages


def _policy_from_rule(rule: element_tree.Element) -> PackageBoundaryPolicy:
    reason_element = rule.find("./string[@name='reason']")
    reason = reason_element.get("value", "") if reason_element is not None else ""
    paths_element = rule.find("./list[@name='allowed_descendant_paths']")
    allowed_descendant_paths = ()
    if paths_element is not None:
        allowed_descendant_paths = tuple(
            element.get("value", "") for element in paths_element.findall("./string")
        )
    return PackageBoundaryPolicy(
        allowed_descendant_paths=allowed_descendant_paths,
        reason=reason,
        location=rule.get("location", "<unknown location>"),
    )


def _valid_pattern(pattern: str) -> str | None:
    if not pattern:
        return "path pattern is empty"
    if pattern != pattern.strip():
        return f"path pattern has surrounding whitespace: {pattern!r}"
    if pattern == "**":
        return None
    if "\\" in pattern or pattern.startswith("/") or ":" in pattern:
        return f"path pattern is not package-relative: {pattern!r}"
    path = pattern[:-3] if pattern.endswith("/**") else pattern
    if "*" in path:
        return f"path pattern only supports a terminal '/**': {pattern!r}"
    parts = path.split("/")
    if any(not part or part in (".", "..") for part in parts):
        return f"path pattern is not canonical: {pattern!r}"
    if not pattern.endswith("/**") and "/" not in pattern:
        return f"exact path does not name a descendant file: {pattern!r}"
    return None


def _pattern_matches(pattern: str, descendant_path: str) -> bool:
    if pattern == "**":
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return descendant_path.startswith(prefix + "/")
    return descendant_path == pattern


def _pattern_covers_pattern(covering: str, covered: str) -> bool:
    if covering == "**":
        return True
    if not covering.endswith("/**"):
        return covering == covered
    prefix = covering[:-3]
    if covered.endswith("/**"):
        covered = covered[:-3]
    return covered == prefix or covered.startswith(prefix + "/")


def _policy_errors(
    policy: PackageBoundaryPolicy,
    descendant_paths: set[str],
) -> tuple[list[str], tuple[str, ...]]:
    errors = []
    location = policy.location
    if not policy.reason.strip():
        errors.append(f"{location}: package boundary reason is empty")
    if not policy.allowed_descendant_paths:
        errors.append(f"{location}: allowed_descendant_paths is empty")

    valid_patterns = []
    for pattern in policy.allowed_descendant_paths:
        error = _valid_pattern(pattern)
        if error:
            errors.append(f"{location}: {error}")
        else:
            valid_patterns.append(pattern)

    seen_patterns = set()
    unique_patterns = []
    for pattern in valid_patterns:
        if pattern in seen_patterns:
            errors.append(f"{location}: duplicate path pattern: {pattern!r}")
        else:
            seen_patterns.add(pattern)
            unique_patterns.append(pattern)

    for pattern in unique_patterns:
        covering_patterns = [
            candidate
            for candidate in unique_patterns
            if candidate != pattern and _pattern_covers_pattern(candidate, pattern)
        ]
        if covering_patterns:
            errors.append(
                f"{location}: path pattern {pattern!r} is already covered by "
                f"{min(covering_patterns, key=len)!r}"
            )

    for pattern in unique_patterns:
        if not any(_pattern_matches(pattern, path) for path in descendant_paths):
            errors.append(
                f"{location}: stale path pattern matches no sources: {pattern!r}"
            )

    return errors, tuple(unique_patterns)


def inspect_query_xml(xml_stream: BinaryIO) -> PackageBoundaryInspection:
    """Inspects streamed Bazel query XML for package boundary violations."""
    descendant_paths_by_package: dict[str, set[str]] = defaultdict(set)
    policies: dict[str, PackageBoundaryPolicy] = {}
    parser_stack: list[element_tree.Element] = []

    for event, element in element_tree.iterparse(xml_stream, events=("start", "end")):
        if event == "start":
            parser_stack.append(element)
            continue

        if len(parser_stack) == 2:
            parsed_label = _parse_label(element.get("name", ""))
            if element.tag == "source-file" and parsed_label is not None:
                package, target = parsed_label
                if "/" in target:
                    descendant_paths_by_package[package].add(target)
            elif element.tag == "rule" and parsed_label is not None:
                package, target = parsed_label
                if target == POLICY_TARGET_NAME:
                    policies[package] = _policy_from_rule(element)
            element.clear()
            parser_stack[0].clear()
        parser_stack.pop()

    results = []
    for package in sorted(descendant_paths_by_package.keys() | policies.keys()):
        descendant_paths = descendant_paths_by_package.get(package, set())
        policy = policies.get(package)
        errors: list[str] = []
        allowed_patterns: tuple[str, ...] = ()
        if policy is not None:
            errors, allowed_patterns = _policy_errors(policy, descendant_paths)
        violations = tuple(
            sorted(
                path
                for path in descendant_paths
                if not any(
                    _pattern_matches(pattern, path) for pattern in allowed_patterns
                )
            )
        )
        results.append(
            PackageBoundaryResult(
                package=package,
                violations=violations,
                policy_errors=tuple(errors),
            )
        )
    return PackageBoundaryInspection(results=tuple(results))


def _query_packages(
    repository_root: Path,
    packages: set[str],
    *,
    verbose: bool,
) -> PackageBoundaryInspection:
    command = [
        "bazel",
        "query",
        query_expression(packages),
        "--output=xml",
        "--noshow_progress",
    ]
    if verbose:
        print("  " + shlex.join(command))
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise PackageBoundaryQueryError("Bazel query did not provide stdout")
        try:
            inspection = inspect_query_xml(process.stdout)
        except element_tree.ParseError as error:
            process.kill()
            process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace").strip()
            detail = f"\n{stderr}" if stderr else ""
            raise PackageBoundaryQueryError(
                f"unable to parse Bazel query XML: {error}{detail}"
            ) from error
        finally:
            process.stdout.close()
        return_code = process.wait()
        if return_code:
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace").strip()
            detail = f"\n{stderr}" if stderr else ""
            raise PackageBoundaryQueryError(
                f"Bazel query exited with status {return_code}{detail}"
            )
    return inspection


def _print_inspection(inspection: PackageBoundaryInspection) -> None:
    for result in inspection.results:
        package_label = _package_label(result.package)
        for error in result.policy_errors:
            print(error)
        if not result.violations:
            continue
        print(
            f"{package_label} owns {len(result.violations)} source file(s) below "
            "a descendant directory:"
        )
        for path in result.violations[:VIOLATION_DISPLAY_LIMIT]:
            repository_path = f"{result.package}/{path}" if result.package else path
            print(f"  {repository_path}")
        hidden_count = len(result.violations) - VIOLATION_DISPLAY_LIMIT
        if hidden_count > 0:
            print(f"  ... and {hidden_count} more")
    if not inspection.ok:
        print(
            "Add a BUILD.bazel to each independently owned directory. For an "
            "indivisible source bundle, declare narrow allowed_descendant_paths "
            "and a reason with iree_package_boundary()."
        )


def check_touched_packages(
    repository_root: Path,
    paths: list[str],
    *,
    verbose: bool,
) -> bool:
    """Queries and checks the Bazel packages owning the selected paths."""
    try:
        packages = touched_packages(repository_root, paths)
    except ValueError as error:
        print(error)
        return False
    if not packages:
        return True
    try:
        inspection = _query_packages(
            repository_root,
            packages,
            verbose=verbose,
        )
    except PackageBoundaryQueryError as error:
        print(error)
        return False
    _print_inspection(inspection)
    return inspection.ok
