# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_converter
import bazel_to_cmake_targets


DEFAULT_ROOT_DIRS = ["runtime/src/iree"]

REPO_MAP = {
    "@iree_core": "",
}


class CustomBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def iree_runtime_cc_library(self, deps=[], **kwargs):
        self.cc_library(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_binary(self, deps=[], **kwargs):
        self.cc_binary(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_test(self, deps=[], resource_group=None, **kwargs):
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
        self.cc_binary_benchmark(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_cc_fuzz(self, deps=[], **kwargs):
        self.iree_cc_fuzz(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_flatbuffer_c_library(
        self, flatcc_includes=None, deps=None, **kwargs
    ):
        if deps:
            kwargs["deps"] = deps
        self.iree_flatbuffer_c_library(includes=flatcc_includes, **kwargs)

    def iree_runtime_vmasm_module(self, deps=[], **kwargs):
        self.iree_vmasm_module(deps=deps + ["//runtime/src:defines"], **kwargs)

    def iree_runtime_hal_cts_test_suite(self, **kwargs):
        self.iree_hal_cts_test_suite(**kwargs)

    def iree_execution_test_suite(self, **kwargs):
        # The reduced HRX runtime CMake bring-up builds the tools needed by
        # runtime targets, but does not port the YAML execution-test harness yet.
        pass


class CustomTargetConverter(bazel_to_cmake_targets.TargetConverter):
    def _initialize(self):
        self._update_target_mappings(
            {
                "//runtime/src:defines": [],
                # Temporary AQL profile SDK header shape. See the comments in
                # build_tools/cmake/hrx_dependencies.cmake and
                # build_tools/third_party/BUILD.bazel: this should become a
                # real ROCm-provided package/target once upstream publishes
                # package metadata for include/aqlprofile-sdk.
                "//build_tools/third_party:aqlprofile_sdk_headers": [
                    "aqlprofile-sdk::headers"
                ],
                "@flatcc//:compiler": ["iree-flatcc-cli"],
                "@flatcc//:flatcc": ["flatcc"],
                "@flatcc//:parsing": ["flatcc::parsing"],
                "@flatcc//:runtime": ["flatcc::runtime"],
            }
        )

    def _convert_unmatched_target(self, target: str) -> str:
        return ["iree::" + self._convert_to_cmake_path(target)]
