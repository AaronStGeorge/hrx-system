# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel markers consumed by IREE's Bazel-to-CMake tooling."""

def iree_cmake_extra_content(content = "", inline = False):
    """Declares extra CMake text associated with the containing Bazel package.

    Bazel ignores these declarations. The Bazel-to-CMake generator treats them
    as structured markers when producing CMake files from BUILD files.

    Args:
      content: CMake text to preserve in generated files.
      inline: Whether `content` should be emitted near the generated target
        rather than near the top of the generated CMake file.
    """
    return struct(
        content = content,
        inline = inline,
    )
