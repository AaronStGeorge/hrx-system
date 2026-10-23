# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The native command and shared-library closure installed by the Loom wheel."""

NATIVE_COMMANDS = [
    "loom-benchmark",
    "loom-check",
    "loom-compile",
    "loom-format",
    "loom-import-cxx",
    "loom-link",
    "loom-opt",
    "loom-run",
    "loom-serve",
    "loom-test",
]

_CONFIGURATION = {
    str(Label("//loom/config:shared_tools")): True,
    str(Label("//loom/config/import:enable")): ["cxx"],
    str(Label("//loom/config/import/cxx:embed_includes")): True,
}

def _configuration_impl(_settings, _attr):
    return _CONFIGURATION

_configuration = transition(
    implementation = _configuration_impl,
    inputs = [],
    outputs = _CONFIGURATION.keys(),
)

def _native_files_impl(ctx):
    files = {}
    for library in ctx.attr.libraries:
        for file in library[DefaultInfo].files.to_list():
            if file.basename in files:
                fail("Duplicate native library destination: " + file.basename)
            files[file.basename] = file
    for tool in ctx.attr.tools:
        executable = tool[DefaultInfo].files_to_run.executable
        if executable == None:
            fail("Native command has no executable: " + str(tool.label))
        if executable.basename in files:
            fail("Duplicate native command destination: " + executable.basename)
        files[executable.basename] = executable
    for tool in ctx.attr.tools:
        # C++ runfiles include libraries and their loader aliases: solib links
        # on Unix and per-executable DLL copies on Windows. Install each name
        # once, retaining toolchain runtimes such as Windows AddressSanitizer.
        for file in tool[DefaultInfo].default_runfiles.files.to_list():
            files.setdefault(file.basename, file)

    outputs = files.values()
    if ctx.executable.elf_patcher:
        # Bazel's solib paths serve build-tree execution. The wheel owns a flat
        # installation with matching companions beside every native command.
        outputs = []
        for file in files.values():
            output = ctx.actions.declare_file(ctx.label.name + "/" + file.basename)
            ctx.actions.run(
                executable = ctx.executable.elf_patcher,
                arguments = ["--set-rpath", "$ORIGIN", "--output", output.path, file.path],
                inputs = [file],
                outputs = [output],
                mnemonic = "LoomWheelRelocate",
                progress_message = "Relocating Loom wheel native file %s" % file.basename,
            )
            outputs.append(output)
    return [DefaultInfo(files = depset(outputs))]

loom_native_files = rule(
    implementation = _native_files_impl,
    cfg = _configuration,
    attrs = {
        "elf_patcher": attr.label(
            executable = True,
            cfg = "exec",
        ),
        "libraries": attr.label_list(
            default = [
                "//loom/binding/c:iree_shared",
                "//loom/binding/c:loomc_shared",
            ],
        ),
        "tools": attr.label_list(
            default = ["//loom/src/loom/tools/" + name for name in NATIVE_COMMANDS],
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
