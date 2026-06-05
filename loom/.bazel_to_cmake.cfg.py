# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

import bazel_to_cmake_config
import bazel_to_cmake_converter
import bazel_to_cmake_requirements

_LOOM_CONFIG_CMAKE_OPTIONS = {
    "//loom/config/emit:amdgpu": "LOOM_EMIT_AMDGPU",
    "//loom/config/emit:ireevm": "LOOM_EMIT_IREEVM",
    "//loom/config/emit:llvmir": "LOOM_EMIT_LLVMIR",
    "//loom/config/emit:spirv": "LOOM_EMIT_SPIRV",
    "//loom/config/emit:wasm": "LOOM_EMIT_WASM",
    "//loom/config/execute:amdgpu": "LOOM_EXECUTE_AMDGPU",
    "//loom/config/execute:ireevm": "LOOM_EXECUTE_IREEVM",
    "//loom/config/execute:spirv_vulkan": "LOOM_EXECUTE_SPIRV_VULKAN",
    "//loom/config/import:mlir": "LOOM_IMPORT_MLIR",
    "//loom/config/import:tilelang": "LOOM_IMPORT_TILELANG",
    "//loom/config/target:amdgpu": "LOOM_TARGET_AMDGPU",
    "//loom/config/target:ireevm": "LOOM_TARGET_IREEVM",
    "//loom/config/target:spirv": "LOOM_TARGET_SPIRV",
    "//loom/config/target:wasm": "LOOM_TARGET_WASM",
    "//loom/config/target:x86": "LOOM_TARGET_X86",
    "//loom/config/target/arch:amdgpu": "LOOM_TARGET_ARCH_AMDGPU",
    "//loom/config/target/arch:ireevm": "LOOM_TARGET_ARCH_IREEVM",
    "//loom/config/target/arch:spirv": "LOOM_TARGET_ARCH_SPIRV",
    "//loom/config/target/arch:wasm": "LOOM_TARGET_ARCH_WASM",
    "//loom/config/target/arch:x86": "LOOM_TARGET_ARCH_X86",
}

_LOW_DESCRIPTOR_ROOTPATH_PATTERN = re.compile(r"\$\(rootpath ([^)]+)\)")


class LoomBuildFileFunctions(bazel_to_cmake_converter.BuildFileFunctions):
    def _custom_initialize(self):
        self._loom_low_descriptor_archive_source_vars = {}
        self._loom_low_descriptor_archive_targets = {}
        self._loom_requirement_policy = bazel_to_cmake_requirements.load_project_policy(
            self._repo_root,
            "loom",
        )

    def _package_name(self):
        return os.path.relpath(self._build_dir, self._repo_root).replace("\\", "/")

    def _loom_package_policy(self):
        return self._loom_requirement_policy.collect(self._package_name())

    def _apply_loom_cmake_policy(self, kwargs, include_run_requirements=False):
        policy = self._loom_package_policy()
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
        if (
            include_run_requirements
            and policy.resource_group
            and not kwargs.get("resource_group")
        ):
            kwargs["resource_group"] = policy.resource_group
        return kwargs

    def _apply_loom_target_compatible_with(self, target_compatible_with):
        policy = self._loom_package_policy()
        return bazel_to_cmake_requirements.append_cmake_conditions(
            target_compatible_with,
            policy.cmake_conditions(),
        )

    def _cmake_guard_condition(self, target_compatible_with):
        if not target_compatible_with:
            return None
        if isinstance(target_compatible_with, bazel_to_cmake_converter.ConditionSelect):
            compatible_conditions = []
            for label, value in target_compatible_with.conditions.items():
                if label == "//conditions:default":
                    continue
                if value == []:
                    condition = self._convert_select_condition(label)
                    if condition:
                        compatible_conditions.append(condition)
            if compatible_conditions:
                return " OR ".join(compatible_conditions)
            return None
        conditions = [
            self._convert_select_condition(label) for label in target_compatible_with
        ]
        if not all(conditions):
            raise NotImplementedError(
                f"target_compatible_with: {target_compatible_with}"
            )
        return " AND ".join(conditions)

    def _guard_cmake_text(self, text, target_compatible_with):
        condition = self._cmake_guard_condition(target_compatible_with)
        if not condition:
            return text
        return f"\nif({condition})\n{text}endif()\n\n"

    def _convert_select_condition(self, label):
        if label in _LOOM_CONFIG_CMAKE_OPTIONS:
            return _LOOM_CONFIG_CMAKE_OPTIONS[label]
        return super()._convert_select_condition(label)

    def _should_emit_python_target(self):
        return self._current_package().startswith("loom/py/loom")

    def _python_package_dirs(self):
        return [
            "${PROJECT_SOURCE_DIR}/loom/py",
            "${PROJECT_BINARY_DIR}/loom/py",
        ]

    def _emit_rewritten_cmake_rule(self, old_rule_name, new_rule_name, emit):
        body_start = len(self._converter.body)
        emit()
        emitted_body = self._converter.body[body_start:]
        self._converter.body = self._converter.body[:body_start] + emitted_body.replace(
            f"{old_rule_name}(",
            f"{new_rule_name}(",
            1,
        )

    def loom_cc_library(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self._emit_rewritten_cmake_rule(
            "iree_cc_library",
            "loom_cc_library",
            lambda: self.cc_library(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                **kwargs,
            ),
        )

    def loom_cc_binary(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self._emit_rewritten_cmake_rule(
            "iree_cc_binary",
            "loom_cc_binary",
            lambda: self.cc_binary(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                **kwargs,
            ),
        )

    def loom_cc_test(self, deps=[], resource_group=None, **kwargs):
        if resource_group:
            kwargs["resource_group"] = resource_group
        kwargs = self._apply_loom_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        resource_group = kwargs.pop("resource_group", None)
        self._emit_rewritten_cmake_rule(
            "iree_cc_test",
            "loom_cc_test",
            lambda: self.cc_test(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                resource_group=resource_group,
                **kwargs,
            ),
        )

    def loom_cc_benchmark(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(
            kwargs,
            include_run_requirements=True,
        )
        self._emit_rewritten_cmake_rule(
            "iree_cc_binary_benchmark",
            "loom_cc_benchmark",
            lambda: self.cc_binary_benchmark(
                deps=deps
                + [
                    "//runtime/src:defines",
                    "//loom/src:defines",
                ],
                **kwargs,
            ),
        )

    def loom_cc_fuzz(self, deps=[], **kwargs):
        kwargs = self._apply_loom_cmake_policy(kwargs)
        self._emit_rewritten_cmake_rule(
            "iree_cc_fuzz",
            "loom_cc_fuzz",
            lambda: self.iree_cc_fuzz(
                deps=deps + ["//runtime/src:defines", "//loom/src:defines"],
                **kwargs,
            ),
        )

    def loom_assert_no_dependencies(self, **kwargs):
        pass

    def _cmake_identifier(self, name):
        return re.sub(r"[^A-Za-z0-9_]", "_", name)

    def _parse_external_file_label(self, label):
        match = re.fullmatch(r"@([^/]+)//(?:(.*):)?([^:]+)", label)
        if not match:
            raise ValueError(f"Unsupported external file label: {label}")
        repo_name = match.group(1)
        package = match.group(2) or ""
        filename = match.group(3)
        if package:
            return repo_name, f"{package}/{filename}"
        return repo_name, filename

    def _convert_low_descriptor_file_label(self, label):
        if label.startswith("@"):
            repo_name, filename = self._parse_external_file_label(label)
            try:
                source_var = self._loom_low_descriptor_archive_source_vars[repo_name]
            except KeyError as exc:
                raise ValueError(
                    f"Low descriptor input {label} requires a prior "
                    f"loom_low_descriptor_data_archive for repo '{repo_name}'"
                ) from exc
            return f"${{{source_var}}}/{filename}"
        return self._normalize_label(label)

    def _convert_low_descriptor_input_label(self, label):
        inputs = [self._convert_low_descriptor_file_label(label)]
        if label.startswith("@"):
            repo_name, _ = self._parse_external_file_label(label)
            inputs.append(self._loom_low_descriptor_archive_targets[repo_name])
        return inputs

    def _convert_low_descriptor_arg(self, arg):
        def replace_rootpath(match):
            return self._convert_low_descriptor_file_label(match.group(1))

        return _LOW_DESCRIPTOR_ROOTPATH_PATTERN.sub(replace_rootpath, arg)

    def _convert_generated_args(self, args):
        converted_args = []
        for arg in args or []:
            arg = self._convert_low_descriptor_arg(arg)
            converted_args.extend(self._convert_location_arg(arg))
        return converted_args

    def _convert_generated_inputs(self, inputs):
        converted_inputs = []
        for label in inputs or []:
            for converted_input in self._convert_low_descriptor_input_label(label):
                if converted_input not in converted_inputs:
                    converted_inputs.append(converted_input)
        return converted_inputs

    def loom_generated_textual_header(
        self,
        name,
        generator,
        output,
        output_flag,
        args=None,
        inputs=None,
        comment=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = self._convert_single_target_block("GENERATOR", generator)
        output_block = self._convert_string_arg_block("OUTPUT", output)
        output_flag_block = self._convert_string_arg_block("OUTPUT_FLAG", output_flag)
        args_block = self._convert_string_list_block(
            "ARGS", self._convert_generated_args(args) or None, sort=False
        )
        inputs_block = self._convert_string_list_block(
            "INPUTS", self._convert_generated_inputs(inputs) or None, sort=False
        )
        comment_block = self._convert_string_arg_block("COMMENT", comment)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"loom_generated_textual_header(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{output_block}"
            f"{output_flag_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{comment_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_generated_cc_library(
        self,
        name,
        generator,
        source=None,
        srcs=None,
        generated_src_flags=None,
        generated_srcs=None,
        hdrs=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        args=None,
        inputs=None,
        extra_output_flags=None,
        extra_outputs=None,
        deps=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = self._convert_single_target_block("GENERATOR", generator)
        source_block = self._convert_string_arg_block("SOURCE", source)
        srcs_block = self._convert_srcs_block(srcs, block_name="SRCS")
        generated_src_flags_block = self._convert_string_list_block(
            "GENERATED_SRC_FLAGS", generated_src_flags or None, sort=False
        )
        generated_srcs_block = self._convert_string_list_block(
            "GENERATED_SRCS", generated_srcs or None, sort=False
        )
        hdrs_block = self._convert_srcs_block(hdrs, block_name="HDRS")
        generated_hdr_flags_block = self._convert_string_list_block(
            "GENERATED_HDR_FLAGS", generated_hdr_flags or None, sort=False
        )
        generated_hdrs_block = self._convert_string_list_block(
            "GENERATED_HDRS", generated_hdrs or None, sort=False
        )
        args_block = self._convert_string_list_block(
            "ARGS", self._convert_generated_args(args) or None, sort=False
        )
        inputs_block = self._convert_string_list_block(
            "INPUTS", self._convert_generated_inputs(inputs) or None, sort=False
        )
        extra_output_flags_block = self._convert_string_list_block(
            "EXTRA_OUTPUT_FLAGS", extra_output_flags or None, sort=False
        )
        extra_outputs_block = self._convert_string_list_block(
            "EXTRA_OUTPUTS", extra_outputs or None, sort=False
        )
        deps_block = self._convert_target_list_block("DEPS", deps)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"loom_generated_cc_library(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{source_block}"
            f"{srcs_block}"
            f"{generated_src_flags_block}"
            f"{generated_srcs_block}"
            f"{hdrs_block}"
            f"{generated_hdr_flags_block}"
            f"{generated_hdrs_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{extra_output_flags_block}"
            f"{extra_outputs_block}"
            f"{deps_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_low_descriptor_data_archive(
        self,
        name,
        repo_name,
        urls,
        sha256,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        source_var = "_LOOM_%s_SOURCE_DIR" % self._cmake_identifier(name).upper()
        archive_target = "loom_%s" % self._cmake_identifier(name)
        self._loom_low_descriptor_archive_source_vars[repo_name] = source_var
        self._loom_low_descriptor_archive_targets[repo_name] = archive_target

        source_dir = "${PROJECT_BINARY_DIR}/_deps/%s-src" % repo_name
        source_dir_block = self._convert_string_arg_block(
            "SOURCE_DIR", "${%s}" % source_var
        )
        name_block = self._convert_string_arg_block("NAME", archive_target, quote=False)
        urls_block = self._convert_string_list_block("URLS", urls, sort=False)
        sha256_block = self._convert_string_arg_block("SHA256", sha256)
        self._converter.header += self._guard_cmake_text(
            "\n"
            f"set({source_var}\n"
            f'  "{source_dir}"\n'
            f")\n"
            f"loom_low_descriptor_data_archive(\n"
            f"{name_block}"
            f"{source_dir_block}"
            f"{urls_block}"
            f"{sha256_block}"
            f")\n\n",
            target_compatible_with,
        )

    def _emit_loom_target_table_cc_library(
        self,
        name,
        generator=None,
        source=None,
        header=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        header_only=False,
        cmake_rule_name="loom_target_table_cc_library",
        args=None,
        inputs=None,
        deps=None,
        ids_deps=None,
        exclude_from_cmake_all=False,
        tags=None,
        testonly=None,
        visibility=None,
        **kwargs,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        generator_block = (
            self._convert_single_target_block("GENERATOR", generator)
            if generator is not None
            else ""
        )
        source_block = self._convert_string_arg_block("SOURCE", source)
        header_block = self._convert_string_arg_block("HEADER", header)
        generated_hdr_flags_block = self._convert_string_list_block(
            "GENERATED_HDR_FLAGS", generated_hdr_flags, sort=False
        )
        generated_hdrs_block = self._convert_string_list_block(
            "GENERATED_HDRS", generated_hdrs, sort=False
        )
        args_block = self._convert_string_list_block(
            "ARGS", self._convert_generated_args(args) or None, sort=False
        )
        inputs_block = self._convert_string_list_block(
            "INPUTS", self._convert_generated_inputs(inputs) or None, sort=False
        )
        deps_block = self._convert_target_list_block("DEPS", deps)
        ids_deps_block = self._convert_target_list_block("IDS_DEPS", ids_deps)
        exclude_from_all_block = self._convert_option_block(
            "EXCLUDE_FROM_ALL", exclude_from_cmake_all
        )
        header_only_block = self._convert_option_block("HEADER_ONLY", header_only)

        self._converter.body += (
            f"{cmake_rule_name}(\n"
            f"{name_block}"
            f"{generator_block}"
            f"{source_block}"
            f"{header_block}"
            f"{generated_hdr_flags_block}"
            f"{generated_hdrs_block}"
            f"{args_block}"
            f"{inputs_block}"
            f"{deps_block}"
            f"{ids_deps_block}"
            f"{exclude_from_all_block}"
            f"{header_only_block}"
            f")\n\n"
        )

    def loom_target_table_cc_library(
        self,
        name,
        generator=None,
        source=None,
        header=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        header_only=False,
        args=None,
        inputs=None,
        deps=None,
        ids_deps=None,
        exclude_from_cmake_all=False,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._emit_platform_guard_begin(target_compatible_with)
        self._emit_loom_target_table_cc_library(
            name=name,
            generator=generator,
            source=source,
            header=header,
            generated_hdr_flags=generated_hdr_flags,
            generated_hdrs=generated_hdrs,
            header_only=header_only,
            args=args,
            inputs=inputs,
            deps=deps,
            ids_deps=ids_deps,
            exclude_from_cmake_all=exclude_from_cmake_all,
            cmake_rule_name="loom_target_table_cc_library",
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_low_descriptor_cc_library(
        self,
        name,
        generator=None,
        source=None,
        header=None,
        generated_hdr_flags=None,
        generated_hdrs=None,
        header_only=False,
        args=None,
        inputs=None,
        deps=None,
        ids_deps=None,
        exclude_from_cmake_all=False,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._emit_platform_guard_begin(target_compatible_with)
        self._emit_loom_target_table_cc_library(
            name=name,
            generator=generator,
            source=source,
            header=header,
            generated_hdr_flags=generated_hdr_flags,
            generated_hdrs=generated_hdrs,
            header_only=header_only,
            args=args,
            inputs=inputs,
            deps=deps,
            ids_deps=ids_deps,
            exclude_from_cmake_all=exclude_from_cmake_all,
            cmake_rule_name="loom_low_descriptor_cc_library",
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_low_descriptor_exclude_from_cmake_all(
        self,
        cc_libraries=None,
        targets=None,
        target_compatible_with=None,
    ):
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        cc_libraries_block = self._convert_string_list_block(
            "CC_LIBRARIES", cc_libraries, sort=True
        )
        targets_block = self._convert_string_list_block("TARGETS", targets, sort=True)
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"loom_low_descriptor_exclude_from_all(\n"
            f"{cc_libraries_block}"
            f"{targets_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _loom_check_test_base_name(self, src):
        extension = ".loom-test"
        if not src.endswith(extension):
            raise ValueError(
                f"loom_check_test source must use the {extension} extension: {src}"
            )
        return src[: -len(extension)]

    def _convert_loom_check_data(self, data):
        if not data:
            return None
        converted_data = []
        for label in data:
            converted_data.append(self._normalize_label(label))
        return converted_data

    def loom_check_runner_binary(
        self,
        name,
        src,
        deps=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        runner_deps = (deps or []) + ["//loom/src/loom/tools/loom-check:main"]
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block([src])
        deps_block, platform_deps_block = self._convert_platform_select_deps(
            name, runner_deps
        )
        if platform_deps_block:
            self._converter.body += platform_deps_block
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_cc_binary(\n{name_block}{srcs_block}{deps_block})\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_check_test(
        self,
        name,
        src,
        data=None,
        tags=None,
        runner="//loom/src/loom/tools/loom-check",
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        self._loom_check_test_base_name(src)
        combined_tags = (tags or []) + ["loom-check"]
        name_block = self._convert_string_arg_block("NAME", name)
        test_binary_block = self._convert_single_target_block("SRC", runner)
        args_block = self._convert_string_list_block(
            "ARGS", [f"${{CMAKE_CURRENT_SOURCE_DIR}}/{src}"]
        )
        data_block = self._convert_string_list_block(
            "DATA", self._convert_loom_check_data(data)
        )
        labels_block = self._convert_string_list_block("LABELS", combined_tags)
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_native_test(\n"
            f"{name_block}"
            f"{args_block}"
            f"{test_binary_block}"
            f"{data_block}"
            f"{labels_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def loom_check_test_suite(
        self,
        name,
        srcs,
        data=None,
        tags=None,
        runner="//loom/src/loom/tools/loom-check",
        test_name_prefix_to_strip="",
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_compatible_with = self._apply_loom_target_compatible_with(
            target_compatible_with
        )
        test_binary_block = self._convert_single_target_block("SRC", runner)
        data_block = self._convert_string_list_block(
            "DATA", self._convert_loom_check_data(data)
        )
        combined_tags = (tags or []) + ["loom-check"]
        labels_block = self._convert_string_list_block("LABELS", combined_tags)
        for src in srcs:
            test_name_src = self._loom_check_test_base_name(src)
            if test_name_prefix_to_strip and test_name_src.startswith(
                test_name_prefix_to_strip
            ):
                test_name_src = test_name_src[len(test_name_prefix_to_strip) :]
            test_name = test_name_src.replace("/", "_")
            name_block = self._convert_string_arg_block("NAME", test_name)
            args_block = self._convert_string_list_block(
                "ARGS", [f"${{CMAKE_CURRENT_SOURCE_DIR}}/{src}"]
            )
            self._emit_platform_guard_begin(target_compatible_with)
            self._converter.body += (
                f"iree_native_test(\n"
                f"{name_block}"
                f"{args_block}"
                f"{test_binary_block}"
                f"{data_block}"
                f"{labels_block}"
                f")\n\n"
            )
            self._emit_platform_guard_end(target_compatible_with)


def convert_unmatched_target(converter, target):
    cmake_path = converter._convert_to_cmake_path(target)
    if cmake_path == "loom":
        return ["loom"]
    if cmake_path == "loom::src::loom":
        return ["loom"]
    if cmake_path.startswith("loom::src::loom::"):
        cmake_path = cmake_path[len("loom::src::loom::") :]
        return ["loom::" + cmake_path]
    if cmake_path == "loom::src":
        return ["loom"]
    if cmake_path.startswith("loom::"):
        cmake_path = cmake_path[len("loom::") :]
    return ["loom::" + cmake_path]


PROJECT_CONFIG = bazel_to_cmake_config.ProjectConfig(
    name="loom",
    package_prefixes=["loom"],
    build_file_functions=LoomBuildFileFunctions,
    target_mappings={
        "//loom/src:defines": [],
    },
    convert_unmatched_target=convert_unmatched_target,
)
