# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Model composition retains source contracts through dependency analysis."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load("//loom/build_tools/bazel:defs.bzl", "LoomModelInfo", "loom_library", "loom_model")

_PACKAGE = "loom/build_tools/bazel/test/"
_CARD = "//loom/src/loom/serve/fixture/text:native.card.json"

def _declare_composition(name, form):
    loom_library(
        name = name + "_leaf",
        srcs = ["link_kernels.loom", "dependency_checks.loom"],
        tags = ["manual"],
    )
    loom_library(
        name = name + "_primary",
        srcs = ["model_provider.cxx"],
        data = ["model_provider.h"],
        input_format = "cxx",
        inputopts = ["cxx:fp-model=fast"],
        deps = [":" + name + "_leaf"],
        tags = ["manual"],
    )
    loom_library(
        name = name + "_branch",
        srcs = ["profile_benchmarks.loom"],
        deps = [":" + name + "_primary", ":" + name + "_leaf"],
        tags = ["manual"],
    )
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_primary"],
        libraries = [":" + name + "_branch"],
        form = form,
        programs = "selected",
        preset = "check",
        model_name = "fixture",
        encoding = "preserve",
        tags = ["manual"],
    )

def _expect_composition(env, target, form):
    info = target[LoomModelInfo]
    stem = target.label.name.removesuffix("_subject")
    primary = {
        "data": [_PACKAGE + "model_provider.h"],
        "format": "loombc" if form == "module" else "cxx",
        "path": _PACKAGE + (stem + "_primary.loombc" if form == "module" else "model_provider.cxx"),
    }
    if form == "sources":
        primary["options"] = ["cxx:fp-model=fast"]
    libraries = [
        {"format": "loombc", "path": _PACKAGE + stem + "_branch.loombc"},
        {"format": "loombc", "path": _PACKAGE + stem + "_leaf.loombc"},
    ] if form == "module" else [
        _PACKAGE + "profile_benchmarks.loom",
        _PACKAGE + "link_kernels.loom",
        _PACKAGE + "dependency_checks.loom",
    ]
    expected = {"selected": {"inputs": [primary], "libraries": libraries}}
    if info.programs != expected:
        env.fail("expected %r, got %r" % (expected, info.programs))
    env.expect.that_str(info.card).equals("loom/src/loom/serve/fixture/text/native.card.json")
    paths = [file.short_path for file in target[OutputGroupInfo].model_files.to_list()]

    # The auxiliary file remains runtime data even when the owning source has
    # become bytecode. A primary reached through a library is emitted once.
    expected_files = [primary["path"], _PACKAGE + "model_provider.h"] + [
        item["path"] if type(item) == "dict" else item
        for item in libraries
    ]
    for path in expected_files:
        env.expect.that_collection(paths).contains(_PACKAGE + stem + "_subject/" + path)
    env.expect.that_int(len(paths)).equals(len(expected_files) + 2)

def _test_sources(name, **kwargs):
    _declare_composition(name, "sources")
    analysis_test(name = name, target = ":" + name + "_subject", impl = _sources_impl, **kwargs)

def _sources_impl(env, target):
    _expect_composition(env, target, "sources")

def _test_modules(name, **kwargs):
    _declare_composition(name, "module")
    analysis_test(name = name, target = ":" + name + "_subject", impl = _modules_impl, **kwargs)

def _modules_impl(env, target):
    _expect_composition(env, target, "module")

def _test_bytecode_is_an_original_source(name, **kwargs):
    loom_library(name = name + "_module", srcs = ["link_kernels.loom"], tags = ["manual"])
    loom_library(name = name + "_provider", srcs = [":" + name + "_module"], tags = ["manual"])
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_provider"],
        programs = "source",
        preset = "check",
        model_name = "fixture",
        tags = ["manual"],
    )
    analysis_test(name = name, target = ":" + name + "_subject", impl = _bytecode_source_impl, **kwargs)

def _bytecode_source_impl(env, target):
    stem = target.label.name.removesuffix("_subject")
    env.expect.that_collection(target[LoomModelInfo].programs["source"]["inputs"]).contains_exactly([
        _PACKAGE + stem + "_module.loombc",
    ])

def _test_external_files_fail(name, **kwargs):
    loom_library(
        name = name + "_provider",
        srcs = ["@bazel_tools//tools/build_defs/repo:http.bzl"],
        tags = ["manual"],
    )
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_provider"],
        programs = "source",
        preset = "check",
        model_name = "fixture",
        tags = ["manual"],
    )
    analysis_test(name = name, target = ":" + name + "_subject", expect_failure = True, impl = _external_files_impl, **kwargs)

def _external_files_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("requires a local repository file"))

def _test_conflicting_importers_fail(name, **kwargs):
    for suffix, format_name in [("a", "cxx"), ("b", "loom")]:
        loom_library(
            name = name + "_" + suffix,
            srcs = ["model_provider.cxx"],
            input_format = format_name,
            tags = ["manual"],
        )
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_a"],
        libraries = [":" + name + "_b"],
        programs = "source",
        preset = "check",
        model_name = "fixture",
        tags = ["manual"],
    )
    analysis_test(name = name, target = ":" + name + "_subject", expect_failure = True, impl = _conflicting_importers_impl, **kwargs)

def _conflicting_importers_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("conflicting model provider"))

def loom_model_rules_test_suite(name):
    test_suite(name = name, tests = [
        _test_sources,
        _test_modules,
        _test_bytecode_is_an_original_source,
        _test_external_files_fail,
        _test_conflicting_importers_fail,
    ])
