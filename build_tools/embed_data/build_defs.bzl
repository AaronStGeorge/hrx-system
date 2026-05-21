# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for embedding data files into C modules."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

IreeCEmbedDataInfo = provider(
    doc = "Metadata for C embed-data generation.",
    fields = {
        "args": "Command-line arguments passed to the embed-data generator.",
        "outputs": "depset of generated C and header outputs.",
        "srcs": "depset of input data files.",
        "tool": "Embed-data generator executable label.",
    },
)

def _iree_c_embed_data_sources_impl(ctx):
    outputs = [
        ctx.outputs.c_file,
        ctx.outputs.h_file,
    ]

    command_args = [
        "--identifier=%s" % ctx.attr.identifier,
        "--output_impl=%s" % ctx.outputs.c_file.path,
        "--output_header=%s" % ctx.outputs.h_file.path,
    ]
    if ctx.attr.strip_prefix:
        command_args.append("--strip_prefix=%s" % ctx.attr.strip_prefix)
    if ctx.attr.flatten:
        command_args.append("--flatten")

    args = ctx.actions.args()
    args.add_all(command_args)
    args.add_all(ctx.files.srcs)

    ctx.actions.run(
        arguments = [args],
        executable = ctx.executable.generator,
        inputs = depset(ctx.files.srcs),
        mnemonic = "IreeCEmbedData",
        outputs = outputs,
        progress_message = "Embedding C data for %{label}",
    )

    return [
        DefaultInfo(files = depset(outputs)),
        IreeCEmbedDataInfo(
            args = command_args + [src.path for src in ctx.files.srcs],
            outputs = depset(outputs),
            srcs = depset(ctx.files.srcs),
            tool = ctx.attr.generator.label,
        ),
    ]

_iree_c_embed_data_sources = rule(
    implementation = _iree_c_embed_data_sources_impl,
    attrs = {
        "c_file": attr.output(
            mandatory = True,
            doc = "Generated C implementation file.",
        ),
        "flatten": attr.bool(
            doc = "Whether embedded table-of-contents names should drop directory components.",
        ),
        "generator": attr.label(
            cfg = "exec",
            default = Label("//build_tools/embed_data:embed_data"),
            executable = True,
            doc = "Executable tool used to generate the embedded C module.",
        ),
        "h_file": attr.output(
            mandatory = True,
            doc = "Generated C header file.",
        ),
        "identifier": attr.string(
            mandatory = True,
            doc = "C identifier prefix used for generated functions.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Data files to embed, in table-of-contents order.",
        ),
        "strip_prefix": attr.string(
            doc = "Verbatim path prefix stripped from table-of-contents names before flattening.",
        ),
    },
    doc = "Generates C source and header files containing embedded data.",
)

def iree_c_embed_data(
        name,
        srcs,
        c_file_output,
        h_file_output,
        testonly = False,
        strip_prefix = None,
        flatten = False,
        identifier = None,
        generator = Label("//build_tools/embed_data:embed_data"),
        deps = None,
        **kwargs):
    """Embeds data files into a C library.

    The generated header declares:

    ```
    typedef struct iree_file_toc_t {
      const char* name;
      const char* data;
      size_t size;
    } iree_file_toc_t;

    const iree_file_toc_t* <identifier>_create(void);
    static inline size_t <identifier>_size(void);
    ```

    The returned table has one entry per input file followed by a `{NULL, NULL,
    0}` sentinel. Each data buffer has an additional NUL byte after the reported
    `size` so text consumers can safely form C strings while binary consumers
    still use the explicit size.

    Args:
      name: Generated `cc_library` target name.
      srcs: Data files to embed, in table-of-contents order.
      c_file_output: Generated C implementation filename.
      h_file_output: Generated C header filename.
      testonly: Whether generated targets are test-only.
      strip_prefix: Verbatim path prefix stripped from table-of-contents names
        before optional flattening.
      flatten: Whether table-of-contents names should drop directory components.
      identifier: C identifier prefix. Defaults to `name`.
      generator: Executable embed-data generator.
      deps: Additional `cc_library` dependencies.
      **kwargs: Additional attributes forwarded to the generated `cc_library`.
    """
    if deps == None:
        deps = []
    if identifier == None:
        identifier = name

    _iree_c_embed_data_sources(
        name = name + "_generate",
        c_file = c_file_output,
        flatten = flatten,
        generator = generator,
        h_file = h_file_output,
        identifier = identifier,
        srcs = srcs,
        strip_prefix = strip_prefix or "",
        testonly = testonly,
        visibility = ["//visibility:private"],
    )

    cc_library(
        name = name,
        hdrs = [h_file_output],
        srcs = [c_file_output],
        deps = deps,
        testonly = testonly,
        **kwargs
    )
