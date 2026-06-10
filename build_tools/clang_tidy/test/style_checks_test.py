#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import unittest

import clang_tidy_test

_ARGS = clang_tidy_test.parse_arguments()


class StyleChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_direct_goto_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-direct-goto",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "direct goto is not allowed",
                "goto cleanup",
                "IREE_CLANG_TIDY_STYLE_LOCAL_GOTO_MACRO",
                "[iree-direct-goto]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_style_computed_goto",
            ],
        )

    def test_refcount_lifecycle_requires_void_and_null_safe_release(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-refcount-lifecycle",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "refcounted retain/release function "
                "iree_clang_tidy_style_refcount_status_retain must return void",
                "refcounted retain/release function "
                "iree_clang_tidy_style_refcount_status_release must return void",
                "refcounted release function "
                "iree_clang_tidy_style_refcount_unguarded_release "
                "must be null-safe",
                "iree_atomic_ref_count_t field pending_submissions must model "
                "object lifetime",
                "[iree-refcount-lifecycle]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_style_refcount_void_retain",
                "iree_clang_tidy_style_refcount_void_release",
                "iree_clang_tidy_style_refcount_early_null_release",
                "iree_clang_tidy_style_refcount_inline_null_release",
                "iree_clang_tidy_style_refcount_likely_null_release",
                "iree_clang_tidy_style_refcount_local_counter_release",
                "iree_clang_tidy_style_refcount_lookup_retain",
                "iree_clang_tidy_style_virtual_memory_release",
            ],
        )

    def test_guarded_release_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-guarded-release",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "release functions are null-safe",
                "iree_clang_tidy_style_resource_release",
                "if (resource) iree_clang_tidy_style_resource_release(resource);",
                "if (resource != NULL) {",
                "[iree-guarded-release]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_style_null_guard_ignored",
                "iree_clang_tidy_style_different_predicate_ignored",
                "iree_clang_tidy_style_extra_behavior_ignored",
                "iree_clang_tidy_style_deinitialize_ignored",
                "iree_clang_tidy_style_destroy_ignored",
                "iree_clang_tidy_style_multi_argument_release_ignored",
                "iree_clang_tidy_style_non_pointer_release_ignored",
                "iree_clang_tidy_style_indirect_release_ignored",
            ],
        )

    def test_guarded_release_is_fixed(self):
        output, fixed_source = clang_tidy_test.run_clang_tidy_fix(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-guarded-release",
            source=clang_tidy_test.source_path(__file__, "style_checks.c"),
            compiler_args=["-std=gnu11"],
        )
        self.assertContainsAll(
            output,
            [
                "release functions are null-safe",
                "[iree-guarded-release]",
            ],
        )
        self.assertIn(
            """void iree_clang_tidy_style_guarded_release(
    iree_clang_tidy_style_resource_t* resource) {
  iree_clang_tidy_style_resource_release(resource);
}""",
            fixed_source,
        )
        self.assertIn(
            """void iree_clang_tidy_style_guarded_release_with_clear(
    iree_clang_tidy_style_resource_t* resource) {
  iree_clang_tidy_style_resource_release(resource);
  resource = NULL;
}""",
            fixed_source,
        )
        self.assertNotIn(
            "if (resource) iree_clang_tidy_style_resource_release(resource);",
            fixed_source,
        )
        self.assertNotIn("if (resource != NULL) {", fixed_source)


if __name__ == "__main__":
    unittest.main()
