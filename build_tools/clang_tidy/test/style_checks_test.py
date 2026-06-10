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


if __name__ == "__main__":
    unittest.main()
