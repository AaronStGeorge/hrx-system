# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for exposing existing executables as binaries or tests."""

IreeExecutableInfo = provider(
    doc = "Metadata for an executable alias or test wrapper.",
    fields = {
        "data": "depset of additional runtime data files.",
        "env": "Environment variables expanded against the wrapper runfiles.",
        "output": "Executable symlink produced by the wrapper.",
        "src": "Wrapped executable label.",
    },
)

def _merge_runfiles(ctx):
    runfiles = ctx.runfiles(files = ctx.files.data)
    return runfiles.merge_all(
        [ctx.attr.src[DefaultInfo].default_runfiles] +
        [target[DefaultInfo].default_runfiles for target in ctx.attr.data],
    )

def _expand_env(ctx):
    return {
        key: ctx.expand_location(value, ctx.attr.data + [ctx.attr.src])
        for key, value in ctx.attr.env.items()
    }

def _executable_output(ctx):
    output_name = ctx.attr.out
    if not output_name:
        output_name = ctx.label.name
    output = ctx.actions.declare_file(output_name)
    ctx.actions.symlink(
        is_executable = True,
        output = output,
        target_file = ctx.executable.src,
    )
    return output

def _iree_executable_alias_impl(ctx):
    output = _executable_output(ctx)
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = _merge_runfiles(ctx),
        ),
        IreeExecutableInfo(
            data = depset(ctx.files.data),
            env = {},
            output = output,
            src = ctx.attr.src.label,
        ),
    ]

def _iree_executable_test_impl(ctx):
    output = _executable_output(ctx)
    expanded_env = _expand_env(ctx)
    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = _merge_runfiles(ctx),
        ),
        IreeExecutableInfo(
            data = depset(ctx.files.data),
            env = expanded_env,
            output = output,
            src = ctx.attr.src.label,
        ),
        testing.TestEnvironment(expanded_env),
    ]

_SHARED_ATTRS = {
    "data": attr.label_list(
        allow_files = True,
        doc = "Runtime data dependencies available to the wrapped executable.",
    ),
    "out": attr.string(
        doc = "Output executable filename. Defaults to the target name.",
    ),
    "src": attr.label(
        allow_files = True,
        cfg = "target",
        doc = "Executable target or file to expose.",
        executable = True,
        mandatory = True,
    ),
}

_TEST_ATTRS = dict(_SHARED_ATTRS)
_TEST_ATTRS["env"] = attr.string_dict(
    doc = "Environment variables passed to the wrapped executable. Values may use $(location) for src or data labels.",
)

iree_executable_alias = rule(
    implementation = _iree_executable_alias_impl,
    attrs = _SHARED_ATTRS,
    doc = "Exposes an executable target or file as another executable target.",
    executable = True,
)

iree_executable_test = rule(
    implementation = _iree_executable_test_impl,
    attrs = _TEST_ATTRS,
    doc = "Runs an executable target or file directly as a Bazel test.",
    test = True,
)
