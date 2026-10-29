# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Package-local exceptions for intentionally nested Bazel source trees."""

_POLICY_TARGET_NAME = "package_boundary_policy"

def _package_boundary_policy_impl(_ctx):
    return []

_package_boundary_policy = rule(
    implementation = _package_boundary_policy_impl,
    attrs = {
        "allowed_descendant_paths": attr.string_list(
            mandatory = True,
            doc = (
                "Exact package-relative files, directory prefixes ending in " +
                "'/**', or '**' for an indivisible package-wide bundle."
            ),
        ),
        "reason": attr.string(
            mandatory = True,
            doc = "Why the nested files form one package ownership unit.",
        ),
    },
)

def iree_package_boundary(
        allowed_descendant_paths,
        reason,
        name = _POLICY_TARGET_NAME):
    """Declares reviewed nested source ownership for the current package.

    Args:
      allowed_descendant_paths: Exact package-relative files, directory prefixes
        ending in "/**", or "**" when the entire package is an indivisible
        source bundle.
      reason: Why splitting the allowed paths into Bazel subpackages would
        misrepresent their ownership.
      name: Fixed metadata target name used by the package-boundary checker.
    """
    if name != _POLICY_TARGET_NAME:
        fail("iree_package_boundary name must be %r" % _POLICY_TARGET_NAME)
    _package_boundary_policy(
        name = name,
        allowed_descendant_paths = allowed_descendant_paths,
        reason = reason,
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )
