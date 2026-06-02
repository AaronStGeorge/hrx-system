# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loads project requirement policy from declarative `.bzl` files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Requirement:
    id: str
    label: str
    phase: str
    enabled_by: str | None = None
    cmake_condition: str | None = None
    cmake_label: str | None = None
    skip_contract: str | None = None


@dataclass(frozen=True)
class PackagePolicy:
    packages: list[str]
    build_requirements: list[Requirement]
    run_requirements: list[Requirement]
    resource_group: str | None = None


@dataclass(frozen=True)
class CMakeRequirementCondition:
    cmake_condition: str


@dataclass(frozen=True)
class CollectedPackagePolicy:
    build_requirements: list[Requirement]
    run_requirements: list[Requirement]
    resource_group: str | None

    def cmake_conditions(self) -> list[CMakeRequirementCondition]:
        return [
            CMakeRequirementCondition(requirement.cmake_condition)
            for requirement in self.build_requirements
            if requirement.cmake_condition
        ]

    def tags(self, include_run_requirements: bool) -> list[str]:
        tags = [
            f"iree-build-requirement={requirement.id}"
            for requirement in self.build_requirements
        ]
        if include_run_requirements:
            for requirement in self.run_requirements:
                tags.append(f"iree-run-requirement={requirement.id}")
                if requirement.cmake_label:
                    tags.append(requirement.cmake_label)
        return tags


@dataclass(frozen=True)
class ProjectRequirementPolicy:
    package_policies: list[PackagePolicy]

    def collect(self, package_name: str) -> CollectedPackagePolicy:
        build_requirements: list[Requirement] = []
        run_requirements: list[Requirement] = []
        resource_group = None
        resource_group_specificity = -1
        for policy in self.package_policies:
            matching_patterns = [
                pattern
                for pattern in policy.packages
                if _matches(pattern, package_name)
            ]
            if not matching_patterns:
                continue
            _append_unique_requirements(build_requirements, policy.build_requirements)
            _append_unique_requirements(run_requirements, policy.run_requirements)
            if policy.resource_group:
                specificity = max(_specificity(pattern) for pattern in matching_patterns)
                if specificity >= resource_group_specificity:
                    resource_group = policy.resource_group
                    resource_group_specificity = specificity
        return CollectedPackagePolicy(
            build_requirements=build_requirements,
            run_requirements=run_requirements,
            resource_group=resource_group,
        )


def build_requirement(id: str, label: str, enabled_by: str, cmake_condition: str):
    return Requirement(
        id=id,
        label=label,
        phase="build",
        enabled_by=enabled_by,
        cmake_condition=cmake_condition,
    )


def run_requirement(id: str, label: str, cmake_label: str, skip_contract: str):
    return Requirement(
        id=id,
        label=label,
        phase="run",
        cmake_label=cmake_label,
        skip_contract=skip_contract,
    )


def package_policy(
    packages,
    build_requirements=None,
    run_requirements=None,
    resource_group=None,
):
    return PackagePolicy(
        packages=list(packages),
        build_requirements=list(build_requirements or []),
        run_requirements=list(run_requirements or []),
        resource_group=resource_group,
    )


def append_cmake_conditions(target_compatible_with, conditions):
    if not conditions:
        return target_compatible_with
    if target_compatible_with is None:
        target_compatible_with = []
    return target_compatible_with + conditions


def strip_load_statements(source: str) -> str:
    lines = []
    skipping_load = False
    paren_depth = 0
    for line in source.splitlines():
        stripped_line = line.lstrip()
        if not skipping_load and stripped_line.startswith("load("):
            paren_depth = line.count("(") - line.count(")")
            skipping_load = paren_depth > 0
            continue
        if skipping_load:
            paren_depth += line.count("(") - line.count(")")
            skipping_load = paren_depth > 0
            continue
        lines.append(line)
    return "\n".join(lines)


@lru_cache(maxsize=None)
def load_project_policy(repo_root: str, project: str) -> ProjectRequirementPolicy:
    requirements_dir = Path(repo_root) / project / "requirements"
    env = {
        "build_requirement": build_requirement,
        "run_requirement": run_requirement,
    }
    _exec_bzl(requirements_dir / "defs.bzl", env)
    env["package_policy"] = package_policy
    _exec_bzl(requirements_dir / "package_policy.bzl", env)
    return ProjectRequirementPolicy(
        package_policies=list(env.get("PACKAGE_POLICIES", [])),
    )


def _exec_bzl(path: Path, env: dict):
    source = strip_load_statements(path.read_text(encoding="utf-8"))
    exec(compile(source, str(path), "exec"), env)


def _matches(pattern: str, package_name: str) -> bool:
    if pattern.endswith("/..."):
        prefix = pattern[:-4]
        return package_name == prefix or package_name.startswith(prefix + "/")
    return package_name == pattern


def _specificity(pattern: str) -> int:
    if pattern.endswith("/..."):
        return len(pattern) - 4
    return len(pattern)


def _append_unique_requirements(target, requirements):
    ids = {requirement.id for requirement in target}
    for requirement in requirements:
        if requirement.id not in ids:
            target.append(requirement)
            ids.add(requirement.id)

