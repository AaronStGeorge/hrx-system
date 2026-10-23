# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Actual command-line expansion for the exported Windows cross toolchain."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")

_CommandInfo = provider("Expanded commands and enabled cross-toolchain capabilities.", fields = {
    # Fully expanded C++ compile arguments from the production configuration.
    "arguments": "Compile arguments.",
    # Whether the unsupported native-Windows export helper is disabled.
    "explicit_exports": "Automatic exports disabled.",
    # Whether the ELF-style ThinLTO indexing/backend graph is enabled.
    "split_lto": "ELF ThinLTO enabled.",
})

def _command_impl(ctx):
    compiler = ctx.attr.compiler[cc_common.CcToolchainInfo]
    configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = compiler,
        requested_features = ctx.attr.profile,
    )
    variables = cc_common.create_compile_variables(
        cc_toolchain = compiler,
        feature_configuration = configuration,
        source_file = "probe.cc",
        output_file = "probe.obj",
        user_compile_flags = ctx.attr.copts,
    )
    return [_CommandInfo(
        arguments = cc_common.get_memory_inefficient_command_line(
            feature_configuration = configuration,
            action_name = ACTION_NAMES.cpp_compile,
            variables = variables,
        ),
        explicit_exports = cc_common.is_enabled(feature_configuration = configuration, feature_name = "no_windows_export_all_symbols"),
        split_lto = cc_common.is_enabled(feature_configuration = configuration, feature_name = "thin_lto"),
    )]

_command = rule(
    implementation = _command_impl,
    attrs = {
        "compiler": attr.label(default = ":compiler", providers = [cc_common.CcToolchainInfo]),
        "copts": attr.string_list(),
        "profile": attr.string_list(),
    },
    fragments = ["cpp"],
)

def _test_optimized(name, **kwargs):
    _command(
        name = name + "_subject",
        copts = ["/O1"],
        profile = ["opt", "optimize_for_speed", "thin_lto_link"],
    )
    analysis_test(name = name, target = name + "_subject", impl = _test_optimized_impl, **kwargs)

def _test_optimized_impl(env, target):
    command = target[_CommandInfo]
    for flag in ["/clang:-O3", "/clang:-flto=thin", "/O1"]:
        env.expect.that_collection(command.arguments).contains(flag)
    if command.arguments.index("/clang:-O3") > command.arguments.index("/O1"):
        env.fail("toolchain profile overrides explicit caller optimization")
    env.expect.that_bool(command.explicit_exports).equals(True)
    env.expect.that_bool(command.split_lto).equals(False)

def _test_sanitizer(name, **kwargs):
    _command(name = name + "_subject", profile = ["fastbuild", "sanitizer_debug_info"])
    analysis_test(name = name, target = name + "_subject", impl = _test_sanitizer_impl, **kwargs)

def _test_sanitizer_impl(env, target):
    command = target[_CommandInfo]
    env.expect.that_collection(command.arguments).contains("/Z7")
    env.expect.that_collection(command.arguments).not_contains("/clang:-flto=thin")
    env.expect.that_collection(command.arguments).not_contains("/clang:-O3")

def _test_static_release_crt(name, **kwargs):
    _command(name = name + "_subject", profile = ["dbg", "static_link_msvcrt_release"])
    analysis_test(name = name, target = name + "_subject", impl = _test_static_release_crt_impl, **kwargs)

def _test_static_release_crt_impl(env, target):
    arguments = target[_CommandInfo].arguments
    env.expect.that_collection(arguments).contains("/MT")
    env.expect.that_collection(arguments).not_contains("/MTd")

def _test_static_debug_crt(name, **kwargs):
    _command(name = name + "_subject", profile = ["dbg", "static_link_msvcrt"])
    analysis_test(name = name, target = name + "_subject", impl = _test_static_debug_crt_impl, **kwargs)

def _test_static_debug_crt_impl(env, target):
    arguments = target[_CommandInfo].arguments
    env.expect.that_collection(arguments).contains("/MTd")
    env.expect.that_collection(arguments).not_contains("/MT")

def toolchain_config_test_suite(name):
    test_suite(
        name = name,
        tests = [_test_optimized, _test_sanitizer, _test_static_release_crt, _test_static_debug_crt],
    )
