# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

COMMIT_MSG_PATH = Path(__file__).with_name("commit_msg.py")
COMMIT_MSG_SPEC = importlib.util.spec_from_file_location("commit_msg", COMMIT_MSG_PATH)
if COMMIT_MSG_SPEC is None or COMMIT_MSG_SPEC.loader is None:
    raise RuntimeError(f"unable to load {COMMIT_MSG_PATH}")
commit_msg = importlib.util.module_from_spec(COMMIT_MSG_SPEC)
sys.modules[COMMIT_MSG_SPEC.name] = commit_msg
COMMIT_MSG_SPEC.loader.exec_module(commit_msg)


class CommitMessageTest(unittest.TestCase):
    def diagnostic_texts(self, message: str) -> list[str]:
        return [
            diagnostic.text
            for diagnostic in commit_msg.validate_commit_message_text(message)
        ]

    def test_allows_normal_paragraphs_and_loom_tools(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Update loom-check help\n"
                "\n"
                "Document loom-compile and loom-opt workflows.\n"
            ),
            [],
        )

    def test_rejects_literal_newline_escape(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Verify sanitizer race sync ordering\n"
                "\n"
                "First paragraph.\\n\\nSecond paragraph.\n"
            ),
            [r"\n"],
        )

    def test_rejects_literal_carriage_return_escape(self):
        self.assertEqual(
            self.diagnostic_texts("Subject\n\nBody with \\r escape.\n"),
            [r"\r"],
        )

    def test_rejects_bead_ids(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Wire race checks\n"
                "\n"
                "Covers loom-20r1y, loom-qof73, loom-20r1y.1, and bd-123.\n"
            ),
            ["loom-20r1y", "loom-qof73", "loom-20r1y.1", "bd-123"],
        )

    def test_ignores_git_comment_lines(self):
        self.assertEqual(
            self.diagnostic_texts(
                "[Loom] Update checks\n"
                "\n"
                "# literal \\n and loom-20r1y in comments are ignored\n"
            ),
            [],
        )

    def test_validates_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            message_path = Path(temporary_directory) / "COMMIT_EDITMSG"
            message_path.write_text(
                "Subject\n\nReferences loom-20r1y.\n",
                encoding="utf-8",
            )

            diagnostics = commit_msg.validate_commit_message_file(message_path)

        self.assertEqual(
            [diagnostic.text for diagnostic in diagnostics], ["loom-20r1y"]
        )


if __name__ == "__main__":
    unittest.main()
