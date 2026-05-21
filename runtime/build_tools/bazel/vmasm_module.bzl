# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime VM assembly module generation."""

load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

IreeRuntimeVmasmModuleInfo = provider(
    doc = "Metadata for a VM assembly module generated as VM bytecode.",
    fields = {
        "args": "Command-line arguments passed to the VM assembly tool.",
        "module": "Generated VM bytecode module file.",
        "src": "VM assembly source file.",
        "tool": "VM assembly executable label.",
    },
)

_VMASM_TOOL = Label("//runtime/src/iree/tools:iree-as-module")

def _iree_runtime_vmasm_module_impl(ctx):
    module = ctx.outputs.module
    src = ctx.file.src
    command_args = [
        "--output=%s" % module.path,
        src.path,
    ]

    args = ctx.actions.args()
    args.add_all(command_args)

    ctx.actions.run(
        arguments = [args],
        executable = ctx.executable.assemble_tool,
        inputs = depset([src]),
        mnemonic = "IreeVmasmModule",
        outputs = [module],
        progress_message = "Assembling VM bytecode module for %{label}",
    )

    return [
        DefaultInfo(files = depset([module])),
        IreeRuntimeVmasmModuleInfo(
            args = command_args,
            module = module,
            src = src,
            tool = ctx.attr.assemble_tool.label,
        ),
    ]

_iree_runtime_vmasm_module = rule(
    implementation = _iree_runtime_vmasm_module_impl,
    attrs = {
        "assemble_tool": attr.label(
            cfg = "exec",
            default = _VMASM_TOOL,
            executable = True,
            doc = "Executable tool used to assemble textual VM assembly.",
        ),
        "module": attr.output(
            mandatory = True,
            doc = "Generated VM bytecode module file.",
        ),
        "src": attr.label(
            allow_single_file = [".vmasm"],
            mandatory = True,
            doc = "Textual VM assembly source file.",
        ),
    },
    doc = "Assembles textual VM assembly into an IREE VM bytecode module.",
)

def iree_runtime_vmasm_module(
        name,
        src,
        module_name = None,
        assemble_tool = _VMASM_TOOL,
        c_identifier = None,
        deps = None,
        testonly = False,
        visibility = None,
        **kwargs):
    """Builds an IREE VM bytecode module from textual VM assembly.

    The primary target produces a `.vmfb` file. Supplying `c_identifier` also
    creates a `<name>_c` C library embedding the generated module with
    `iree_c_embed_data`.

    Args:
      name: Generated VM bytecode module target name.
      src: Textual VM assembly source file.
      module_name: Generated VM bytecode module filename. Defaults to
        `<name>.vmfb`.
      assemble_tool: Executable VM assembly tool.
      c_identifier: Optional C identifier prefix for an embedded module library.
      deps: Additional dependencies for the optional embedded module library.
      testonly: Whether generated targets are test-only.
      visibility: Visibility for generated targets.
      **kwargs: Additional common attributes forwarded to generated targets.
    """
    if module_name == None:
        module_name = "%s.vmfb" % name
    if deps == None:
        deps = []
    if not module_name.endswith(".vmfb"):
        fail("module_name must end with .vmfb, got %r" % module_name)

    _iree_runtime_vmasm_module(
        name = name,
        assemble_tool = assemble_tool,
        module = module_name,
        src = src,
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

    if c_identifier:
        iree_c_embed_data(
            name = "%s_c" % name,
            c_file_output = "%s_c.c" % name,
            deps = deps,
            flatten = True,
            h_file_output = "%s_c.h" % name,
            identifier = c_identifier,
            srcs = [":" + name],
            testonly = testonly,
            visibility = visibility,
            **kwargs
        )
