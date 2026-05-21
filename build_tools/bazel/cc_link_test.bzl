# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis tests for shared C/C++ platform link options."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load(":cc_link.bzl", "cc_link")

LinkOptsInfo = provider(
    doc = "Selected link options for a configured test subject.",
    fields = {
        "flags": "Configured link option strings.",
    },
)

def _linkopts_subject_impl(ctx):
    return [LinkOptsInfo(flags = ctx.attr.linkopts)]

_linkopts_subject = rule(
    implementation = _linkopts_subject_impl,
    attrs = {
        "linkopts": attr.string_list(
            doc = "Configurable link options to expose to analysis tests.",
        ),
    },
)

def _platform_label(name, os_name):
    return Label("//build_tools/bazel:%s_%s_platform" % (name, os_name))

def _define_platform(name, os_name):
    native.platform(
        name = "%s_%s_platform" % (name, os_name),
        constraint_values = ["@platforms//os:" + os_name],
        tags = ["manual"],
    )

def _expect_link_flags(env, target, expected):
    actual = target[LinkOptsInfo].flags
    if actual != expected:
        env.fail("expected link flags %r, got %r" % (expected, actual))

def _link_test(name, linkopts, os_name, impl):
    _define_platform(name, os_name)
    _linkopts_subject(
        name = name + "_subject",
        linkopts = linkopts,
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        attr_values = {
            "timeout": "short",
        },
        config_settings = {
            "//command_line_option:platforms": [_platform_label(name, os_name)],
        },
        impl = impl,
        target = name + "_subject",
    )

def _test_pthreads_linux_links(name):
    _link_test(
        name = name,
        impl = _test_pthreads_linux_links_impl,
        linkopts = cc_link.pthreads_linkopts(),
        os_name = "linux",
    )

def _test_pthreads_linux_links_impl(env, target):
    _expect_link_flags(env, target, ["-pthread"])

def _test_pthreads_wasi_does_not_link(name):
    _link_test(
        name = name,
        impl = _test_pthreads_wasi_does_not_link_impl,
        linkopts = cc_link.pthreads_linkopts(),
        os_name = "wasi",
    )

def _test_pthreads_wasi_does_not_link_impl(env, target):
    _expect_link_flags(env, target, [])

def _test_pthreads_windows_does_not_link(name):
    _link_test(
        name = name,
        impl = _test_pthreads_windows_does_not_link_impl,
        linkopts = cc_link.pthreads_linkopts(),
        os_name = "windows",
    )

def _test_pthreads_windows_does_not_link_impl(env, target):
    _expect_link_flags(env, target, [])

def _test_dl_linux_links(name):
    _link_test(
        name = name,
        impl = _test_dl_linux_links_impl,
        linkopts = cc_link.dl_linkopts(),
        os_name = "linux",
    )

def _test_dl_linux_links_impl(env, target):
    _expect_link_flags(env, target, ["-ldl"])

def _test_dl_macos_does_not_link(name):
    _link_test(
        name = name,
        impl = _test_dl_macos_does_not_link_impl,
        linkopts = cc_link.dl_linkopts(),
        os_name = "macos",
    )

def _test_dl_macos_does_not_link_impl(env, target):
    _expect_link_flags(env, target, [])

def _test_rt_linux_links(name):
    _link_test(
        name = name,
        impl = _test_rt_linux_links_impl,
        linkopts = cc_link.rt_linkopts(),
        os_name = "linux",
    )

def _test_rt_linux_links_impl(env, target):
    _expect_link_flags(env, target, ["-lrt"])

def _test_rt_windows_does_not_link(name):
    _link_test(
        name = name,
        impl = _test_rt_windows_does_not_link_impl,
        linkopts = cc_link.rt_linkopts(),
        os_name = "windows",
    )

def _test_rt_windows_does_not_link_impl(env, target):
    _expect_link_flags(env, target, [])

def cc_link_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_pthreads_linux_links,
            _test_pthreads_wasi_does_not_link,
            _test_pthreads_windows_does_not_link,
            _test_dl_linux_links,
            _test_dl_macos_does_not_link,
            _test_rt_linux_links,
            _test_rt_windows_does_not_link,
        ],
    )
