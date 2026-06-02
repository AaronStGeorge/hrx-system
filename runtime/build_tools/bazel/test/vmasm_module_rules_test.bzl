# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for runtime VMASM Bazel rules."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo", "util")
load(
    "//runtime/build_tools/bazel:vmasm_module.bzl",
    "IreeRuntimeVmasmModuleInfo",
    "iree_runtime_vmasm_module",
)

def _expect_arg_with_suffix(env, args, suffix):
    for arg in args:
        if arg.endswith(suffix):
            return
    env.fail("expected argument ending with %r in %r" % (suffix, args))

def _test_vmasm_module_action_contract(name, **kwargs):
    util.helper_target(
        iree_runtime_vmasm_module,
        name = name + "_subject",
        c_identifier = "iree_runtime_vmasm_module_test",
        src = "vmasm_module_test.vmasm",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        impl = _test_vmasm_module_action_contract_impl,
        target = name + "_subject",
        **kwargs
    )

def _test_vmasm_module_action_contract_impl(env, target):
    info = target[IreeRuntimeVmasmModuleInfo]
    env.expect.that_str(info.module.basename).equals(
        "test_vmasm_module_action_contract_subject.vmfb",
    )
    env.expect.that_str(info.src.basename).equals("vmasm_module_test.vmasm")
    if not str(info.tool).endswith(
        "//runtime/src/iree/tools/iree-as-module:iree-as-module",
    ):
        env.fail("unexpected VMASM tool %s" % info.tool)

    actions = target[TestingAspectInfo].actions
    env.expect.that_int(len(actions)).equals(1)
    action = actions[0]
    env.expect.that_str(action.mnemonic).equals("IreeVmasmModule")
    _expect_arg_with_suffix(
        env,
        action.argv,
        "test_vmasm_module_action_contract_subject.vmfb",
    )
    _expect_arg_with_suffix(env, action.argv, "vmasm_module_test.vmasm")

def vmasm_module_rules_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_vmasm_module_action_contract,
        ],
    )
