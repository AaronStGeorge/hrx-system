# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python SDK packaging with an explicit compatible model-catalog association."""

load("@rules_python//python:packaging.bzl", "py_wheel")

def _metadata_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(output, json.encode({"catalog": ctx.attr.catalog or None}))
    return [DefaultInfo(files = depset([output]))]

_metadata = rule(
    implementation = _metadata_impl,
    attrs = {"catalog": attr.string()},
)

def loom_python_wheel(name, version, catalog = "", testonly = False):
    """Builds an SDK wheel without publishing it.

    Args:
      name: Target name.
      version: Explicit Python distribution version.
      catalog: Compatible github:OWNER/REPOSITORY@TAG association; empty for a
        development SDK requiring an explicit discovery source.
      testonly: Whether this wheel is a test fixture.
    """
    _metadata(name = name + "_metadata", catalog = catalog, testonly = testonly)
    py_wheel(
        name = name,
        testonly = testonly,
        description_file = Label("//loom/packaging/python:README.md"),
        distribution = "hrx-loom",
        entry_points = {"console_scripts": ["loom-fetch = loom.model.fetch_main:main"]},
        extra_distinfo_files = {
            Label("//:LICENSE"): "licenses/LICENSE",
            ":" + name + "_metadata": "loom.json",
        },
        extra_requires = {"hf": ["huggingface_hub>=2.2,<3"]},
        homepage = "https://github.com/ROCm/hrx-system",
        license = "Apache-2.0 WITH LLVM-exception",
        python_requires = ">=3.12",
        python_tag = "py3",
        stamp = 0,
        strip_path_prefixes = ["loom/py"],
        summary = "Loom Python authoring tools and portable model acquisition",
        twine_binary = None,
        version = version,
        deps = [
            Label("//loom/packaging/python:package"),
            Label("//loom/packaging/python:schemas"),
        ],
    )
