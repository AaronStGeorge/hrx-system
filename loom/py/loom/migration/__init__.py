# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Migration metadata and validation for Loom source and bytecode formats."""

from loom.migration.manifest import (
    CURRENT_BYTECODE_VERSION,
    CURRENT_MANIFEST,
    CURRENT_TEXT_BASELINE,
    DIAGNOSTIC_ERROR,
    DIAGNOSTIC_WARNING,
    REQUIRED_ACTIVE_RULE_TEST_KINDS,
    RULE_TEST_CURRENT_NOOP,
    RULE_TEST_CURRENT_PARSE,
    RULE_TEST_IDEMPOTENCE,
    RULE_TEST_LEGACY_REWRITE,
    LoomRelease,
    MigrationDiagnostic,
    MigrationLintReport,
    MigrationManifest,
    MigrationRuleMetadata,
    TextBaseline,
    lint_current_manifest,
    lint_legacy_formats,
    migration_rule_metadata_from_ops,
)
from loom.migration.source import (
    MigrationSourceDiagnostic,
    SourceDocument,
    SourceEdit,
    apply_source_edits,
)

__all__ = [
    "CURRENT_BYTECODE_VERSION",
    "CURRENT_MANIFEST",
    "CURRENT_TEXT_BASELINE",
    "DIAGNOSTIC_ERROR",
    "DIAGNOSTIC_WARNING",
    "REQUIRED_ACTIVE_RULE_TEST_KINDS",
    "RULE_TEST_CURRENT_NOOP",
    "RULE_TEST_CURRENT_PARSE",
    "RULE_TEST_IDEMPOTENCE",
    "RULE_TEST_LEGACY_REWRITE",
    "LoomRelease",
    "MigrationDiagnostic",
    "MigrationLintReport",
    "MigrationManifest",
    "MigrationRuleMetadata",
    "MigrationSourceDiagnostic",
    "SourceDocument",
    "SourceEdit",
    "TextBaseline",
    "apply_source_edits",
    "lint_legacy_formats",
    "lint_current_manifest",
    "migration_rule_metadata_from_ops",
]
