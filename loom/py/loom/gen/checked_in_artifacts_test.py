# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path
from unittest import mock

from loom.gen import checked_in_artifacts
from loom.gen.support.generated_file import (
    GeneratedFileFamily,
    GeneratedFileMaintenanceResult,
    GeneratedFileSet,
)


def test_registered_artifact_ownership_is_disjoint() -> None:
    owners: dict[str, str] = {}

    for family in checked_in_artifacts.checked_in_artifact_families():
        for path in (*family.file_set.output_paths, *family.file_set.obsolete_paths):
            assert path not in owners, f"{path} is owned by both {owners[path]} and {family.description}"
            owners[path] = family.description

    assert owners


def test_update_selects_family_owning_selected_output() -> None:
    first_family = GeneratedFileFamily(
        description="first",
        regenerate_command="generate first",
        file_set=GeneratedFileSet.from_mapping({"generated/first.txt": "first\n"}),
    )
    second_family = GeneratedFileFamily(
        description="second",
        regenerate_command="generate second",
        file_set=GeneratedFileSet.from_mapping({"generated/second.txt": "second\n"}),
    )
    with (
        mock.patch.object(
            checked_in_artifacts,
            "checked_in_artifact_families",
            return_value=(first_family, second_family),
        ) as checked_in_artifact_families,
        mock.patch.object(
            checked_in_artifacts._bootstrap,
            "find_repo_root",
            return_value=mock.sentinel.repository_root,
        ),
        mock.patch.object(
            checked_in_artifacts,
            "maintain_generated_file_families",
            return_value=GeneratedFileMaintenanceResult(True),
        ) as maintain_generated_file_families,
    ):
        result = checked_in_artifacts.maintain_checked_in_artifacts("update", selected_paths=["generated/second.txt"])

    assert result.ok
    checked_in_artifact_families.assert_called_once_with(repository_root=mock.sentinel.repository_root)
    maintain_generated_file_families.assert_called_once_with(
        mock.sentinel.repository_root,
        (second_family,),
        mode="update",
    )


def test_update_regenerates_family_affected_by_deleted_input(
    tmp_path: Path,
) -> None:
    affected_family = GeneratedFileFamily(
        description="affected",
        regenerate_command="generate affected",
        file_set=GeneratedFileSet.from_mapping({"generated/affected.txt": "new affected\n"}),
        input_roots=("schema/affected",),
    )
    unrelated_family = GeneratedFileFamily(
        description="unrelated",
        regenerate_command="generate unrelated",
        file_set=GeneratedFileSet.from_mapping({"generated/unrelated.txt": "new unrelated\n"}),
        input_roots=("schema/unrelated",),
    )
    affected_output = tmp_path / "generated/affected.txt"
    unrelated_output = tmp_path / "generated/unrelated.txt"
    affected_output.parent.mkdir(parents=True)
    affected_output.write_text("old affected\n", encoding="utf-8")
    unrelated_output.write_text("old unrelated\n", encoding="utf-8")
    with (
        mock.patch.object(
            checked_in_artifacts,
            "checked_in_artifact_families",
            return_value=(affected_family, unrelated_family),
        ),
        mock.patch.object(
            checked_in_artifacts._bootstrap,
            "find_repo_root",
            return_value=tmp_path,
        ),
    ):
        result = checked_in_artifacts.maintain_checked_in_artifacts("update", selected_paths=["schema/affected/deleted.py"])

    assert result.ok
    assert result.changed_paths == ("generated/affected.txt",)
    assert affected_output.read_text(encoding="utf-8") == "new affected\n"
    assert unrelated_output.read_text(encoding="utf-8") == "old unrelated\n"


def test_main_selects_check_and_update_modes() -> None:
    with mock.patch.object(
        checked_in_artifacts,
        "maintain_checked_in_artifacts",
        return_value=GeneratedFileMaintenanceResult(True),
    ) as maintain_checked_in_artifacts:
        assert checked_in_artifacts.main(["--check"]) == 0
        assert checked_in_artifacts.main(["--in-place"]) == 0

    assert maintain_checked_in_artifacts.call_args_list == [
        mock.call("check"),
        mock.call("update"),
    ]
