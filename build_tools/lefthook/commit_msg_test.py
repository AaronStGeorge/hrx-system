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
    def diagnostic_texts(
        self,
        message: str,
        *,
        changed_paths: list[str] | None = None,
    ) -> list[str]:
        return [
            diagnostic.text
            for diagnostic in commit_msg.validate_commit_message_text(
                message,
                changed_paths=changed_paths or [],
            )
        ]

    def diagnostic_messages(
        self,
        message: str,
        *,
        changed_paths: list[str] | None = None,
    ) -> list[str]:
        return [
            diagnostic.message
            for diagnostic in commit_msg.validate_commit_message_text(
                message,
                changed_paths=changed_paths or [],
            )
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
            self.diagnostic_texts("[Infra] Subject\n\nBody with \\r escape.\n"),
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
                "[Infra] Subject\n\nReferences loom-20r1y.\n",
                encoding="utf-8",
            )

            diagnostics = commit_msg.validate_commit_message_file(message_path)

        self.assertEqual(
            [diagnostic.text for diagnostic in diagnostics], ["loom-20r1y"]
        )

    def test_rejects_missing_subject_tag_with_path_suggestions(self):
        diagnostics = commit_msg.validate_commit_message_text(
            "Update race lowering\n\nBody.\n",
            changed_paths=[
                "loom/src/loom/target/arch/amdgpu/lower/sanitizer.c",
                "runtime/src/iree/hal/drivers/amdgpu/device.c",
            ],
        )

        self.assertEqual(
            [diagnostic.message for diagnostic in diagnostics],
            ["subject line must start with a bracketed subsystem tag"],
        )
        self.assertIn("[Loom/AMDGPU]", diagnostics[0].hint)
        self.assertIn("[HAL/AMDGPU]", diagnostics[0].hint)

    def test_rejects_missing_subject_description(self):
        self.assertEqual(
            self.diagnostic_messages("[Infra]\n\nBody.\n"),
            ["subject line must include a description after the tag"],
        )

    def test_rejects_long_subject(self):
        subject = "[Infra] " + "x" * 80
        self.assertEqual(
            self.diagnostic_messages(subject + "\n\nBody.\n"),
            ["subject line is 88 characters; keep it at or below 72"],
        )

    def test_rejects_long_body_lines_outside_code_blocks(self):
        long_line = "This body line is intentionally too long for commit prose " + (
            "because it should be wrapped."
        )
        self.assertEqual(
            self.diagnostic_messages(f"[Infra] Subject\n\n{long_line}\n"),
            [
                f"line is {len(long_line)} characters; keep commit message "
                "body lines at or below 72"
            ],
        )

    def test_allows_long_lines_inside_code_blocks(self):
        long_line = "x" * 120
        self.assertEqual(
            self.diagnostic_messages(
                f"[Infra] Subject\n\n```text\n{long_line}\n```\nWrapped prose.\n"
            ),
            [],
        )

    def test_ranks_tag_suggestions_from_paths(self):
        self.assertEqual(
            commit_msg.tag_suggestions_for_paths(
                [
                    "loom/src/loom/target/arch/amdgpu/lower/sanitizer.c",
                    "loom/py/loom/dialect/sanitizer/defs.py",
                    "libhrx/src/libhrx/runtime.c",
                    "build_tools/lefthook/commit_msg.py",
                ]
            )[:4],
            ["[Loom]", "[Loom/AMDGPU]", "[HRX]", "[Infra]"],
        )


if __name__ == "__main__":
    unittest.main()
