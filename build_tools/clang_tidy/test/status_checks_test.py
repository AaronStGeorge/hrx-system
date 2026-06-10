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


class StatusChecksTest(clang_tidy_test.ClangTidyAssertions):
    def test_discarded_status_result_is_diagnosed(self):
        output = clang_tidy_test.run_clang_tidy(
            clang_tidy=_ARGS.clang_tidy,
            plugin=_ARGS.plugin,
            checks="-*,iree-status-discarded",
            source=clang_tidy_test.source_path(__file__, "status_checks.c"),
        )
        self.assertContainsAll(
            output,
            [
                "iree_clang_tidy_status_dropped_source",
                "iree_clang_tidy_status_void_cast_source",
                "[iree-status-discarded]",
            ],
        )
        self.assertContainsNone(
            output,
            [
                "iree_clang_tidy_status_assigned_source",
                "iree_clang_tidy_status_ignored_source",
                "iree_clang_tidy_status_returned_source",
            ],
        )


if __name__ == "__main__":
    unittest.main()
