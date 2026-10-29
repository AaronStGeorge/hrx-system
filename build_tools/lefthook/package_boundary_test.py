# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from build_tools.lefthook import package_boundary


def inspect(xml: str) -> package_boundary.PackageBoundaryInspection:
    return package_boundary.inspect_query_xml(io.BytesIO(xml.encode("utf-8")))


class PackageBoundaryTest(unittest.TestCase):
    def test_touched_packages_use_nearest_current_build_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "BUILD.bazel").touch()
            (root / "project/nested").mkdir(parents=True)
            (root / "project/BUILD.bazel").touch()
            (root / "project/nested/BUILD.bazel").touch()

            self.assertEqual(
                package_boundary.touched_packages(
                    root,
                    [
                        "README.md",
                        "project/source.cc",
                        "project/orphan/source.cc",
                        "project/nested/source.cc",
                    ],
                ),
                {"", "project", "project/nested"},
            )

    def test_deleted_build_file_falls_back_to_parent_package(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "BUILD.bazel").touch()
            (root / "project/deleted").mkdir(parents=True)
            (root / "project/BUILD.bazel").touch()

            self.assertEqual(
                package_boundary.touched_packages(
                    root,
                    ["project/deleted/BUILD.bazel"],
                ),
                {"project"},
            )

    def test_query_expression_includes_rules_and_source_files(self):
        self.assertEqual(
            package_boundary.query_expression({"project/nested", ""}),
            "set(//:* //project/nested:*)",
        )

    def test_descendant_source_without_policy_is_rejected(self):
        inspection = inspect(
            """<?xml version="1.0"?>
            <query>
              <source-file name="//project:direct.cc"/>
              <source-file name="//project:test/source.cc"/>
            </query>
            """
        )

        self.assertFalse(inspection.ok)
        self.assertEqual(len(inspection.results), 1)
        self.assertEqual(inspection.results[0].package, "project")
        self.assertEqual(inspection.results[0].violations, ("test/source.cc",))
        self.assertEqual(inspection.results[0].policy_errors, ())

    def test_narrow_policy_does_not_hide_other_descendant_trees(self):
        inspection = inspect(
            """<?xml version="1.0"?>
            <query>
              <source-file name="//project:include/api.h"/>
              <source-file name="//project:src/implementation.cc"/>
              <rule class="_package_boundary_policy"
                    location="project/BUILD.bazel:10:1"
                    name="//project:package_boundary_policy">
                <list name="allowed_descendant_paths">
                  <string value="include/**"/>
                </list>
                <string name="reason" value="Installed public header layout."/>
              </rule>
            </query>
            """
        )

        self.assertFalse(inspection.ok)
        self.assertEqual(
            inspection.results[0].violations,
            ("src/implementation.cc",),
        )
        self.assertEqual(inspection.results[0].policy_errors, ())

    def test_exact_and_whole_package_policies_are_supported(self):
        exact_inspection = inspect(
            """<?xml version="1.0"?>
            <query>
              <source-file name="//project:bundle/data.bin"/>
              <rule class="_package_boundary_policy"
                    location="project/BUILD.bazel:10:1"
                    name="//project:package_boundary_policy">
                <list name="allowed_descendant_paths">
                  <string value="bundle/data.bin"/>
                </list>
                <string name="reason" value="Embedded fixture payload."/>
              </rule>
            </query>
            """
        )
        whole_package_inspection = inspect(
            """<?xml version="1.0"?>
            <query>
              <source-file name="//project:first/data.bin"/>
              <source-file name="//project:second/data.bin"/>
              <rule class="_package_boundary_policy"
                    location="project/BUILD.bazel:10:1"
                    name="//project:package_boundary_policy">
                <list name="allowed_descendant_paths">
                  <string value="**"/>
                </list>
                <string name="reason" value="Indivisible generated SDK layout."/>
              </rule>
            </query>
            """
        )

        self.assertTrue(exact_inspection.ok)
        self.assertTrue(whole_package_inspection.ok)

    def test_policy_requires_canonical_live_nonredundant_paths(self):
        inspection = inspect(
            """<?xml version="1.0"?>
            <query>
              <source-file name="//project:include/api.h"/>
              <rule class="_package_boundary_policy"
                    location="project/BUILD.bazel:10:1"
                    name="//project:package_boundary_policy">
                <list name="allowed_descendant_paths">
                  <string value="include/**"/>
                  <string value="include/api.h"/>
                  <string value="missing/**"/>
                  <string value="bad/*/path"/>
                </list>
                <string name="reason" value=""/>
              </rule>
            </query>
            """
        )

        self.assertFalse(inspection.ok)
        errors = "\n".join(inspection.results[0].policy_errors)
        self.assertIn("package boundary reason is empty", errors)
        self.assertIn("only supports a terminal '/**'", errors)
        self.assertIn("'include/api.h' is already covered by 'include/**'", errors)
        self.assertIn("stale path pattern matches no sources: 'missing/**'", errors)

    def test_policy_without_descendant_sources_is_stale(self):
        inspection = inspect(
            """<?xml version="1.0"?>
            <query>
              <rule class="_package_boundary_policy"
                    location="project/BUILD.bazel:10:1"
                    name="//project:package_boundary_policy">
                <list name="allowed_descendant_paths">
                  <string value="bundle/**"/>
                </list>
                <string name="reason" value="Embedded fixture payload."/>
              </rule>
            </query>
            """
        )

        self.assertFalse(inspection.ok)
        self.assertIn(
            "stale path pattern matches no sources: 'bundle/**'",
            inspection.results[0].policy_errors[0],
        )


if __name__ == "__main__":
    unittest.main()
