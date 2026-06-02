# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import execution


class ExecutionUnitTest(unittest.TestCase):
    def test_substitution_preserves_unknown_braces(self):
        with tempfile.TemporaryDirectory() as directory:
            tool_path = Path(directory) / "tool"
            substituter = execution._Substituter(
                {"tmp": directory},
                {"fixture": tool_path},
            )
            self.assertEqual(
                substituter.substitute("{tmp}/file {json: true} {tool:fixture}"),
                f"{directory}/file {{json: true}} {tool_path}",
            )

    def test_contains_ordering(self):
        runner = execution.ExecutionRunner(tools={})
        runner._check_contains(
            "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["alpha", "gamma"]
        )
        with self.assertRaises(execution.CaseFailure):
            runner._check_contains(
                "case", "step", "stdout", "alpha\nbeta\ngamma\n", ["gamma", "alpha"]
            )


if __name__ == "__main__":
    unittest.main()
