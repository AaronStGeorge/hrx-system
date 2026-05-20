# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared C/C++ Bazel macros for IREE repositories.

This package owns repo-wide C/C++ mechanics that are deliberately independent of
any top-level project directory. Project-specific wrappers should load these
macros and add only the dependencies, flags, or validation that belong to that
project.
"""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

def _merge_string_lists(first, second):
    if first == None:
        first = []
    if second == None:
        second = []
    return first + second

_CC_COMPILATION_ATTRS = {
    "copts": attr.string_list(
        doc = "Additional compiler options for this target.",
    ),
    "defines": attr.string_list(
        doc = "Preprocessor definitions propagated to dependents.",
    ),
    "includes": attr.string_list(
        doc = "Include directories propagated to dependents.",
    ),
    "local_defines": attr.string_list(
        doc = "Preprocessor definitions used only while compiling this target.",
    ),
    "system_includes": attr.string_list(
        doc = """Include directories that should be system includes outside Bazel.

        Bazel's C/C++ rules do not expose a separate public system include
        attribute, so these paths are forwarded through `includes`. Other
        generators can preserve the caller's distinction.
        """,
    ),
}

_CC_LINK_ATTRS = {
    "linkopts": attr.string_list(
        doc = "Additional linker options for this target.",
    ),
    "linkstatic": attr.bool(
        doc = "Whether to prefer static linking for this target.",
    ),
}

_CC_DEP_ATTRS = {
    "data": attr.label_list(
        allow_files = True,
        doc = "Runtime data dependencies for this target.",
    ),
    "deps": attr.label_list(
        doc = "C/C++ library dependencies for this target.",
    ),
}

_CC_SOURCE_ATTRS = {
    "hdrs": attr.label_list(
        allow_files = True,
        doc = "Public headers provided by this target.",
    ),
    "srcs": attr.label_list(
        allow_files = True,
        doc = "Source files compiled by this target.",
    ),
    "textual_hdrs": attr.label_list(
        allow_files = True,
        doc = "Headers that are textually included and not compiled independently.",
    ),
}

_CC_BINARY_SOURCE_ATTRS = {
    "srcs": attr.label_list(
        allow_files = True,
        doc = "Source files compiled by this target.",
    ),
}

def _add_if_not_none(target, name, value):
    if value != None:
        target[name] = value

def _merge_dicts(*dicts):
    result = {}
    for source in dicts:
        result.update(source)
    return result

def _cc_attrs(
        srcs,
        hdrs,
        textual_hdrs,
        deps,
        data,
        copts,
        defines,
        local_defines,
        includes,
        system_includes,
        linkopts,
        linkstatic):
    result = {
        "includes": _merge_string_lists(includes, system_includes),
    }
    _add_if_not_none(result, "srcs", srcs)
    _add_if_not_none(result, "hdrs", hdrs)
    _add_if_not_none(result, "textual_hdrs", textual_hdrs)
    _add_if_not_none(result, "deps", deps)
    _add_if_not_none(result, "data", data)
    _add_if_not_none(result, "copts", copts)
    _add_if_not_none(result, "defines", defines)
    _add_if_not_none(result, "local_defines", local_defines)
    _add_if_not_none(result, "linkopts", linkopts)
    _add_if_not_none(result, "linkstatic", linkstatic)
    return result

def _iree_cc_library_impl(
        name,
        visibility,
        srcs,
        hdrs,
        textual_hdrs,
        deps,
        data,
        copts,
        defines,
        local_defines,
        includes,
        system_includes,
        linkopts,
        linkstatic,
        alwayslink,
        **kwargs):
    cc_attrs = _cc_attrs(
        srcs = srcs,
        hdrs = hdrs,
        textual_hdrs = textual_hdrs,
        deps = deps,
        data = data,
        copts = copts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        system_includes = system_includes,
        linkopts = linkopts,
        linkstatic = linkstatic,
    )
    cc_attrs = _merge_dicts(kwargs, cc_attrs)
    _add_if_not_none(cc_attrs, "alwayslink", alwayslink)
    cc_library(
        name = name,
        visibility = visibility,
        **cc_attrs
    )

iree_cc_library = macro(
    doc = """Defines a shared IREE C/C++ library target.

    The shared wrapper intentionally adds no project policy. It exists so that
    project wrappers can share Bazel-specific mechanics, and so that attributes
    needed by non-Bazel generators can remain explicit at call sites.
    """,
    implementation = _iree_cc_library_impl,
    inherit_attrs = "common",
    attrs = _merge_dicts(
        _CC_COMPILATION_ATTRS,
        _CC_DEP_ATTRS,
        _CC_SOURCE_ATTRS,
        _CC_LINK_ATTRS,
        {
            "alwayslink": attr.bool(
                doc = "Whether to force this library into binaries that depend on it.",
            ),
        },
    ),
)

def _iree_cc_binary_impl(
        name,
        visibility,
        srcs,
        deps,
        data,
        copts,
        defines,
        local_defines,
        includes,
        system_includes,
        linkopts,
        linkstatic,
        args,
        **kwargs):
    cc_attrs = _cc_attrs(
        srcs = srcs,
        hdrs = None,
        textual_hdrs = None,
        deps = deps,
        data = data,
        copts = copts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        system_includes = system_includes,
        linkopts = linkopts,
        linkstatic = linkstatic,
    )
    cc_attrs = _merge_dicts(kwargs, cc_attrs)
    _add_if_not_none(cc_attrs, "args", args)
    cc_binary(
        name = name,
        visibility = visibility,
        **cc_attrs
    )

iree_cc_binary = macro(
    doc = """Defines a shared IREE C/C++ binary target.""",
    implementation = _iree_cc_binary_impl,
    inherit_attrs = "common",
    attrs = _merge_dicts(
        _CC_COMPILATION_ATTRS,
        _CC_DEP_ATTRS,
        _CC_BINARY_SOURCE_ATTRS,
        _CC_LINK_ATTRS,
        {
            "args": attr.string_list(
                doc = "Command-line arguments used when this binary is run by Bazel.",
            ),
        },
    ),
)

def _iree_cc_test_impl(
        name,
        visibility,
        srcs,
        deps,
        data,
        copts,
        defines,
        local_defines,
        includes,
        system_includes,
        linkopts,
        linkstatic,
        args,
        env,
        **kwargs):
    cc_attrs = _cc_attrs(
        srcs = srcs,
        hdrs = None,
        textual_hdrs = None,
        deps = deps,
        data = data,
        copts = copts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        system_includes = system_includes,
        linkopts = linkopts,
        linkstatic = linkstatic,
    )
    cc_attrs = _merge_dicts(kwargs, cc_attrs)
    _add_if_not_none(cc_attrs, "args", args)
    _add_if_not_none(cc_attrs, "env", env)
    cc_test(
        name = name,
        visibility = visibility,
        **cc_attrs
    )

iree_cc_test = macro(
    doc = """Defines a shared IREE C/C++ test target.""",
    implementation = _iree_cc_test_impl,
    inherit_attrs = "common",
    attrs = _merge_dicts(
        _CC_COMPILATION_ATTRS,
        _CC_DEP_ATTRS,
        _CC_BINARY_SOURCE_ATTRS,
        _CC_LINK_ATTRS,
        {
            "args": attr.string_list(
                doc = "Command-line arguments passed to the test binary.",
            ),
            "env": attr.string_dict(
                doc = "Environment variables passed to the test binary.",
            ),
        },
    ),
)
