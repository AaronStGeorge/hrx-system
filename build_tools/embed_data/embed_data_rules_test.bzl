# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for C embed-data generation."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load(":build_defs.bzl", "IreeCEmbedDataInfo")

def _expect_basename(env, files, expected_basename):
    for file in files:
        if file.basename == expected_basename:
            return
    env.fail("expected basename %r in %r" % (expected_basename, files))

def _expect_arg_with_suffix(env, args, suffix):
    for arg in args:
        if arg.endswith(suffix):
            return
    env.fail("expected argument ending with %r in %r" % (suffix, args))

def _test_embed_data_action_contract(name, **kwargs):
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_embed_data_action_contract_impl,
        target = ":flat_embed_generate",
        **kwargs
    )

def _test_embed_data_action_contract_impl(env, target):
    info = target[IreeCEmbedDataInfo]
    _expect_basename(env, info.outputs.to_list(), "flat_embed.c")
    _expect_basename(env, info.outputs.to_list(), "flat_embed.h")
    _expect_basename(env, info.srcs.to_list(), "crlf.bin")
    _expect_basename(env, info.srcs.to_list(), "first.txt")
    _expect_basename(env, info.srcs.to_list(), "second.txt")
    if not str(info.tool).endswith("//build_tools/embed_data:embed_data"):
        env.fail("unexpected embed-data tool %s" % info.tool)

    actions = target[TestingAspectInfo].actions
    env.expect.that_int(len(actions)).equals(1)
    action = actions[0]
    env.expect.that_str(action.mnemonic).equals("IreeCEmbedData")
    _expect_arg_with_suffix(env, action.argv, "--identifier=flat_embed")
    _expect_arg_with_suffix(env, action.argv, "--flatten")
    _expect_arg_with_suffix(env, action.argv, "flat_embed.c")
    _expect_arg_with_suffix(env, action.argv, "flat_embed.h")
    _expect_basename(env, action.inputs.to_list(), "first.txt")
    _expect_basename(env, action.inputs.to_list(), "second.txt")

def embed_data_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_embed_data_action_contract,
        ],
    )
