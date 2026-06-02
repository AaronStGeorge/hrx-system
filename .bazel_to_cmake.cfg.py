# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import bazel_to_cmake_converter
import bazel_to_cmake_requirements
import bazel_to_cmake_targets


DEFAULT_ROOT_DIRS = ["runtime/src/iree", "libhrx"]

REPO_MAP = {
    "@iree_core": "",
}


class CustomBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _custom_initialize(self):
        self._runtime_requirement_policy = (
            bazel_to_cmake_requirements.load_project_policy(
                self._repo_root,
                "runtime",
            )
        )

    def _drop_selects(self, values):
        if isinstance(values, bazel_to_cmake_converter.MixedDeps):
            return values.unconditional
        return values

    def _package_name(self):
        return os.path.relpath(self._build_dir, self._repo_root).replace("\\", "/")

    def _runtime_package_policy(self):
        return self._runtime_requirement_policy.collect(self._package_name())

    def _apply_runtime_cmake_policy(self, kwargs, include_run_requirements=False):
        policy = self._runtime_package_policy()
        kwargs = dict(kwargs)
        kwargs["target_compatible_with"] = (
            bazel_to_cmake_requirements.append_cmake_conditions(
                kwargs.get("target_compatible_with"),
                policy.cmake_conditions(),
            )
        )
        policy_tags = policy.tags(include_run_requirements=include_run_requirements)
        if policy_tags or kwargs.get("tags"):
            tags = list(kwargs.get("tags") or [])
            tags.extend(policy_tags)
            kwargs["tags"] = tags
        if include_run_requirements and policy.resource_group and not kwargs.get(
            "resource_group"
        ):
            kwargs["resource_group"] = policy.resource_group
        return kwargs

    def iree_select(self, selector):
        return self.select(selector)

    def hrx_cc_library(self, deps=[], **kwargs):
        self.cc_library(deps=deps + ["//runtime/src:defines", "//libhrx:defines"], **kwargs)

    def hrx_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(deps=deps + ["//runtime/src:defines", "//libhrx:defines"], **kwargs)

    def hrx_cc_test(self, deps=[], resource_group=None, **kwargs):
        self.cc_test(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            resource_group=resource_group,
            **kwargs,
        )

    def hrx_cc_benchmark(self, deps=[], **kwargs):
        self.cc_binary_benchmark(
            deps=deps
            + [
                "//runtime/src:defines",
                "//libhrx:defines",
                "//third_party:google_benchmark",
            ],
            **kwargs,
        )

    def hrx_cc_shared_library(self, deps=[], **kwargs):
        kwargs["copts"] = self._drop_selects(kwargs.get("copts"))
        kwargs["hdrs"] = [
            hdr
            for hdr in (kwargs.get("hdrs") or [])
            if hdr != "//libhrx/src:iree_hal_compat.h"
        ]
        self.cc_library(
            deps=deps + ["//runtime/src:defines", "//libhrx:defines"],
            shared=True,
            **kwargs,
        )

    def hrx_cts_test(self, name, deps=[], **kwargs):
        srcs = kwargs.get("srcs")
        copts = kwargs.get("copts")
        defines = kwargs.get("defines")
        includes = kwargs.get("includes")
        tags = kwargs.get("tags")
        full_deps = [
            ":core",
            "//third_party:catch2",
        ] + deps + ["//runtime/src:defines", "//libhrx:defines"]
        name_block = self._convert_string_arg_block("NAME", "hrx_cts_" + name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        deps_block, platform_deps_block = self._convert_platform_select_deps("hrx_cts_" + name, full_deps)
        includes_block = self._convert_includes_block(includes)
        args_block = self._convert_string_list_block(
            "ARGS",
            [
                "--hrx-library",
                "$<TARGET_FILE:libhrx::src::libhrx::hrx>",
            ],
        )
        if platform_deps_block:
            self._converter.body += platform_deps_block
        labels_block = self._convert_string_list_block("LABELS", tags)
        self._converter.body += (
            f"hrx_cc_test(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{deps_block}"
            f"{args_block}"
            f"{includes_block}"
            f"{labels_block}"
            f")\n\n"
        )

    def iree_runtime_cc_library(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.cc_library(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_binary(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.cc_binary(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_test(self, deps=[], resource_group=None, **kwargs):
        if resource_group:
            kwargs["resource_group"] = resource_group
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        resource_group = kwargs.pop("resource_group", None)
        tags = kwargs.get("tags", [])
        if not resource_group and tags:
            for tag in tags:
                if tag.startswith("resource_group:"):
                    resource_group = tag[len("resource_group:") :]
                    break
        self.cc_test(
            deps=deps + ["//runtime/src:defines"],
            resource_group=resource_group,
            **kwargs,
        )

    def iree_runtime_cc_benchmark(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self.cc_binary_benchmark(
            deps=deps
            + [
                "//runtime/src:defines",
                "//third_party:google_benchmark",
            ],
            **kwargs,
        )

    def iree_runtime_cc_fuzz(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.iree_cc_fuzz(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_executable_test(self, src, **kwargs):
        self.native_test(src=src, **kwargs)

    def iree_runtime_flatbuffer_c_library(
        self, flatcc_includes=None, deps=None, **kwargs
    ):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        if deps:
            kwargs["deps"] = deps
        self.iree_flatbuffer_c_library(includes=flatcc_includes, **kwargs)

    def iree_checked_glob(self, files, **kwargs):
        return files

    def iree_generated_files(self, name, srcs, outs, args, output_args, tool, **kwargs):
        if tool != ":generate_vm_isa":
            raise NotImplementedError(f"iree_generated_files tool: {tool}")
        cmd_parts = ["python3 $(rootpath generate_vm_isa.py)"]
        cmd_parts.extend(arg.replace("$(location ", "$(rootpath ") for arg in args)
        for out in outs:
            cmd_parts.extend([output_args[out], f"$(execpath {out})"])
        self.iree_genrule(
            name=name,
            srcs=["generate_vm_isa.py"] + srcs,
            outs=outs,
            cmd=" ".join(cmd_parts),
        )

    def iree_runtime_vmasm_module(self, deps=[], **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        self.iree_vmasm_module(deps=deps + ["//runtime/src:defines"], **kwargs)

    def _emit_iree_hal_cts_test_suite(self, kwargs):
        if "backends" in kwargs and "backends_lib" not in kwargs:
            kwargs["backends_lib"] = kwargs.pop("backends")
        super().iree_hal_cts_test_suite(**kwargs)

    def iree_runtime_hal_cts_test_suite(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self._emit_iree_hal_cts_test_suite(kwargs)

    def iree_hal_cts_test_suite(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self._emit_iree_hal_cts_test_suite(kwargs)

    def iree_hal_cts_testdata(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        super().iree_hal_cts_testdata(**kwargs)

    def iree_amdgpu_binary(self, **kwargs):
        kwargs = self._apply_runtime_cmake_policy(kwargs)
        super().iree_amdgpu_binary(**kwargs)


class CustomTargetConverter(bazel_to_cmake_targets.TargetConverter):
    def _initialize(self):
        self._update_target_mappings(
            {
                "//runtime/src:defines": [],
                "//libhrx:defines": ["libhrx_defs"],
                "//libhrx/src/libhrx:hrx_static": [
                    "libhrx::src::libhrx::hrx"
                ],
                # Temporary AQL profile SDK header shape. See the comments in
                # build_tools/third_party/rocm_headers.cmake and
                # third_party/BUILD.bazel: this should become a real
                # ROCm-provided package/target once upstream publishes package
                # metadata for include/aqlprofile-sdk.
                "//third_party:aqlprofile_sdk_headers": [
                    "iree::third_party::aqlprofile_sdk_headers"
                ],
                "//third_party:hsa_runtime_headers": [
                    "iree::third_party::hsa_runtime_headers"
                ],
                "//third_party:libbacktrace": [
                    "${IREE_LIBBACKTRACE_TARGET}"
                ],
                "//third_party:flatcc": ["iree-flatcc-cli"],
                "//third_party:flatcc_compiler": ["iree-flatcc-cli"],
                "//third_party:flatcc_parsing": [
                    "iree::third_party::flatcc_parsing"
                ],
                "//third_party:flatcc_runtime": [
                    "iree::third_party::flatcc_runtime"
                ],
                "//third_party:catch2": ["iree::third_party::catch2"],
            }
        )

    def _convert_unmatched_target(self, target: str) -> str:
        if target.startswith("//libhrx"):
            cmake_path = self._convert_to_cmake_path(target)
            if cmake_path == "libhrx":
                return ["libhrx"]
            if cmake_path.startswith("libhrx::"):
                cmake_path = cmake_path[len("libhrx::") :]
            return ["libhrx::" + cmake_path]
        return ["iree::" + self._convert_to_cmake_path(target)]
