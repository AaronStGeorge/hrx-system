# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Windows cross-toolchain features owned by the exporting HRX module."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "with_feature_set")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/toolchain:windows_cc_toolchain_config.bzl", "all_compile_actions", "all_link_actions", _cc_toolchain_config = "cc_toolchain_config")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")

def _static_crt_feature():
    flag_sets = []
    for actions, release, debug in [
        ([ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile], "/MT", "/MTd"),
        (all_link_actions, "/DEFAULTLIB:libcmt.lib", "/DEFAULTLIB:libcmtd.lib"),
    ]:
        flag_sets.extend([
            flag_set(
                actions = actions,
                flag_groups = [flag_group(flags = [release])],
                with_features = [
                    with_feature_set(not_features = ["dbg"]),
                    with_feature_set(features = ["static_link_msvcrt_release"]),
                ],
            ),
            flag_set(
                actions = actions,
                flag_groups = [flag_group(flags = [debug])],
                with_features = [with_feature_set(
                    features = ["dbg"],
                    not_features = ["static_link_msvcrt_release"],
                )],
            ),
        ])
    return feature(name = "static_link_msvcrt", flag_sets = flag_sets)

def _cross_configuration_impl(ctx):
    base = ctx.attr.base[CcToolchainConfigInfo]
    names = {item.name: True for item in base._features_DO_NOT_USE}
    additions = [
        feature(
            name = "optimize_for_speed",
            flag_sets = [flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/clang:-O3"])],
            )],
        ),
        feature(
            name = "thin_lto_link",
            # COFF ThinLTO is performed by lld-link, not ELF index/backend actions.
            flag_sets = [flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["/clang:-flto=thin"])],
            )],
        ),
        feature(
            name = "sanitizer_debug_info",
            flag_sets = [flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/Z7"])],
            )],
        ),
        feature(name = "static_link_msvcrt_release", implies = ["static_link_msvcrt"]),
        # The base configuration already expanded legacy features and actions.
        feature(name = "no_legacy_features"),
    ]
    replacements = {
        # Bazel's automatic DLL export helper is a Windows executable. A Linux
        # cross toolchain uses explicit exports and cannot execute that helper.
        "no_windows_export_all_symbols": feature(name = "no_windows_export_all_symbols", enabled = True),
    }
    if "static_link_msvcrt_release" not in names:
        replacements["static_link_msvcrt"] = _static_crt_feature()
    features = [replacements.get(item.name, item) for item in base._features_DO_NOT_USE]
    position = [item.name for item in features].index("user_compile_flags")
    features = features[:position] + [item for item in additions if item.name not in names] + features[position:]

    # Keep the provider representation boundary here. Consumer rules and their
    # CcInfo/toolchain identities still belong to the selected rules_cc module.
    optional = {}
    if hasattr(base, "disallowed_copts_infos"):
        optional["disallowed_copts_infos"] = base.disallowed_copts_infos
    return [cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = base._action_configs_DO_NOT_USE,
        artifact_name_patterns = base._artifact_name_patterns_DO_NOT_USE,
        toolchain_identifier = base.toolchain_id,
        compiler = base.compiler,
        target_cpu = base.target_cpu,
        target_libc = base.target_libc,
        target_system_name = base.target_system_name,
        abi_version = base.abi_version,
        abi_libc_version = base.abi_libc_version,
        cxx_builtin_include_directories = base.cxx_builtin_include_directories,
        tool_paths = base.tool_paths,
        make_variables = base.make_variables,
        builtin_sysroot = base.builtin_sysroot,
        **optional
    )]

_cross_configuration = rule(
    implementation = _cross_configuration_impl,
    attrs = {
        "base": attr.label(mandatory = True, providers = [CcToolchainConfigInfo]),
    },
    provides = [CcToolchainConfigInfo],
)

def windows_cross_toolchain_config(name, **kwargs):
    """Defines a clang-cl cross configuration without root-only rules_cc patches.

    Args:
      name: Name of the composed toolchain configuration target.
      **kwargs: Attributes of rules_cc's Windows cc_toolchain_config rule.
    """
    _cc_toolchain_config(name = name + "_base", **kwargs)
    _cross_configuration(
        name = name,
        base = ":" + name + "_base",
        **{key: kwargs[key] for key in ["visibility", "tags", "testonly", "target_compatible_with"] if key in kwargs}
    )
