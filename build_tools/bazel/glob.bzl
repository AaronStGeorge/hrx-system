# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checked source-list helpers."""

def _quoted_file_list(files):
    return "\n".join([
        '        "%s",' % file
        for file in files
    ])

def iree_checked_glob(files, include, exclude = [], allow_empty = True, exclude_directories = 1):
    """Returns an explicit file list after checking it matches a glob.

    This keeps BUILD files as the source of truth for ordered file lists while
    still catching added or removed files during package loading. It is useful
    for test suites and generated metadata where a raw glob would hide review
    signal, but a stale hand-maintained list would silently drop coverage.

    Args:
      files: Explicit file list returned unchanged when it matches the glob.
      include: Glob include patterns.
      exclude: Glob exclude patterns.
      allow_empty: Whether an empty glob match is valid.
      exclude_directories: Native glob directory exclusion mode.

    Returns:
      The `files` list unchanged.
    """
    glob_files = native.glob(
        include = include,
        exclude = exclude,
        allow_empty = allow_empty,
        exclude_directories = exclude_directories,
    )

    sorted_files = sorted(files)
    if sorted_files == glob_files:
        return files

    missing = [
        file
        for file in glob_files
        if file not in files
    ]
    extra = [
        file
        for file in files
        if file not in glob_files
    ]
    fail(
        "\n".join([
            "iree_checked_glob explicit file list does not match native.glob.",
            "Expected: %s" % glob_files,
            "Got: %s" % files,
            "Missing: %s" % missing,
            "Extra: %s" % extra,
            "Paste this into the explicit file list:",
            _quoted_file_list(glob_files),
        ]),
    )
