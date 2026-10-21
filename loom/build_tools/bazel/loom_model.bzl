# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Model-card selected program linking and native deployment artifacts."""

_LINK_TOOLCHAIN = Label("//loom/build_tools/bazel:link_toolchain_type")

def _loom_model_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name)
    files = {}
    inputs = [ctx.file.card]
    assets = []
    for target, name in (ctx.attr.files | ctx.attr.assets).items():
        artifacts = target[DefaultInfo].files.to_list()
        if len(artifacts) != 1:
            fail("%s must supply exactly one file for %s" % (target.label, name))
        if name in files:
            fail("duplicate logical file %s" % name)
        artifact = artifacts[0]
        files[name] = {"path": artifact.path, "runtime_path": artifact.short_path}
        inputs.append(artifact)
        if target in ctx.attr.assets:
            assets.append(artifact)
    if len(ctx.attr.files) + len(ctx.attr.assets) != len(files):
        fail("files and assets must have disjoint bindings")
    plan = ctx.actions.declare_file(ctx.label.name + ".plan.json")
    ctx.actions.write(plan, json.encode({
        "assets": ctx.attr.assets.values(),
        "card": ctx.file.card.path,
        "configuration": ctx.attr.configuration,
        "encoding": ctx.attr.encoding,
        "files": files,
        "name": ctx.attr.model_name,
        "output": output.path,
        "programs": ctx.attr.programs if ctx.attr.programs else None,
        "runtime_output": output.short_path,
    }))
    linker = ctx.toolchains[_LINK_TOOLCHAIN].tool
    ctx.actions.run(
        executable = ctx.executable._builder,
        arguments = ["--plan", plan.path, "--linker", linker.files_to_run.executable.path],
        inputs = depset(inputs + [plan]),
        tools = [linker.files_to_run],
        outputs = [output],
        mnemonic = "LoomModel",
        progress_message = "Linking model configuration %s" % ctx.label,
    )
    return [DefaultInfo(
        files = depset([output]),
        runfiles = ctx.runfiles(files = assets + [output]),
    )]

loom_model = rule(
    implementation = _loom_model_impl,
    attrs = {
        "assets": attr.label_keyed_string_dict(allow_files = True, doc = "Runtime asset files mapped to logical model-card paths."),
        "card": attr.label(mandatory = True, allow_single_file = [".json"]),
        "configuration": attr.string(mandatory = True),
        "encoding": attr.string(default = "bc", values = ["text", "bc"]),
        "files": attr.label_keyed_string_dict(allow_files = True, doc = "Declared files mapped to logical model-card paths."),
        "model_name": attr.string(mandatory = True),
        "programs": attr.string(),
        "_builder": attr.label(default = Label("//loom/py/loom/model:build"), executable = True, cfg = "exec"),
    },
    toolchains = [_LINK_TOOLCHAIN],
)
