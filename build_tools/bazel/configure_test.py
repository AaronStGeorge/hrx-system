# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

CONFIGURE_BAZEL_PATH = Path(__file__).with_name("configure.py")


def load_configure_bazel_module():
    spec = importlib.util.spec_from_file_location(
        "configure_bazel", CONFIGURE_BAZEL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {CONFIGURE_BAZEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConfigureBazelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configure_bazel = load_configure_bazel_module()

    def make_rocm_root(self, temporary_directory: str) -> Path:
        rocm_root = Path(temporary_directory) / "rocm"
        (rocm_root / "include").mkdir(parents=True)
        return rocm_root

    def test_direct_script_entrypoint_loads_project_contributors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "configured.bazelrc"
            subprocess.run(
                [
                    sys.executable,
                    str(CONFIGURE_BAZEL_PATH),
                    "--output",
                    str(output),
                    "-DLOOM_TARGET_ALL_CPU=ON",
                ],
                check=True,
            )

            config = output.read_text(encoding="utf-8")

        self.assertIn("# Loom target selection: cpu", config)
        self.assertIn("build --//loom/config/arch:enable=vm,x86", config)

    def test_loom_image_configuration_is_portable_and_independent(self):
        for arguments in (
            ["-DLOOM_SERVE_IMAGE_INPUT=OFF", "-DLOOM_SERVE_IMAGE_CODECS=png;jpeg"],
            [
                "--//loom/config/serve:image_input=false",
                "--//loom/config/serve:image_codecs=png,jpeg",
            ],
        ):
            with self.subTest(arguments=arguments):
                request = self.configure_bazel.parse_arguments(arguments)
                config = self.configure_bazel.generate_config(request)
                self.assertIn("--//loom/config/serve:image_input=false", config)
                self.assertIn("--//loom/config/serve:image_codecs=png,jpeg", config)
                self.assertIn("--//loom/config/execute:enable=iree_hal", config)

    def test_loom_image_configuration_rejects_unknown_and_mixed_options(self):
        for arguments in (
            ["-DLOOM_SERVE_IMAGE_CODECS=png;typo"],
            ["--//loom/config/serve:image_codecs=typo"],
            ["-DLOOM_SERVE_IMAGE_INPUT=ON", "--//loom/config/serve:image_input=false"],
            [
                "-DLOOM_SERVE_IMAGE_CODECS=png",
                "--//loom/config/serve:image_codecs=jpeg",
            ],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.configure_bazel.generate_config(
                    self.configure_bazel.parse_arguments(arguments)
                )

    def test_loom_audio_codec_selection_is_portable(self):
        for codecs in ((), ("wav",), ("mp3", "flac"), ("wav", "mp3", "flac")):
            for argument in (
                "-DLOOM_SERVE_AUDIO_CODECS=" + ";".join(codecs),
                "--//loom/config/serve:audio_codecs=" + ",".join(codecs),
            ):
                with self.subTest(argument=argument):
                    config = self.configure_bazel.generate_config(
                        self.configure_bazel.parse_arguments([argument])
                    )
                    self.assertIn(
                        "build --//loom/config/serve:audio_codecs="
                        + ",".join(codecs)
                        + "\n",
                        config,
                    )

    def test_loom_audio_configuration_rejects_unknown_and_mixed_options(self):
        for arguments in (
            ["-DLOOM_SERVE_AUDIO_CODECS=wav;typo"],
            ["--//loom/config/serve:audio_codecs=typo"],
            ["-DLOOM_SERVE_AUDIO_CODECS=wav", "--//loom/config/serve:audio_codecs=mp3"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.configure_bazel.generate_config(
                    self.configure_bazel.parse_arguments(arguments)
                )

    def assert_rocm_path(self, config: str, rocm_root: Path) -> None:
        self.assertIn(
            self.configure_bazel.bazelrc_line(
                "common",
                "--repo_env=IREE_ROCM_PATH="
                + self.configure_bazel.as_bazel_path(rocm_root),
            ),
            config,
        )

    def test_bazel_paths_use_forward_slashes_on_windows(self):
        path = PureWindowsPath(r"C:\home\runner\rocm-root")

        self.assertEqual(
            self.configure_bazel.as_bazel_path(path),
            "C:/home/runner/rocm-root",
        )

    def test_non_windows_host_does_not_probe_windows_capabilities(self):
        symbolic_link_probe = mock.Mock(return_value=False)

        self.configure_bazel.require_windows_bazel_host(
            platform_name="linux",
            symbolic_link_probe=symbolic_link_probe,
        )

        symbolic_link_probe.assert_not_called()

    def test_windows_host_accepts_required_capabilities(self):
        self.configure_bazel.require_windows_bazel_host(
            platform_name="win32",
            symbolic_link_probe=lambda: True,
        )

    def test_windows_host_reports_missing_symbolic_link_capability(self):
        with self.assertRaisesRegex(SystemExit, "Symbolic-link creation") as context:
            self.configure_bazel.require_windows_bazel_host(
                platform_name="win32",
                symbolic_link_probe=lambda: False,
            )

        message = str(context.exception)
        self.assertIn("Developer Mode", message)
        self.assertIn("Create symbolic links", message)
        self.assertIn("base image", message)
        self.assertNotIn("LongPathsEnabled", message)

    def test_portable_project_options_configure_amdgpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "-DIREE_HAL_DRIVER_AMDGPU=ON",
                    f"-DIREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=task,amdgpu,amd-gpu,amd-xdna\n",
            config,
        )
        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assert_rocm_path(config, rocm_root)
        self.assertNotIn("--deleted_packages", config)

    def test_libamdf_can_be_enabled_without_amdgpu(self):
        args = self.configure_bazel.parse_arguments(
            ["-DIREE_HAL_DRIVER_AMD=OFF", "-DAMDF_BUILD=ON"]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//libamdf/config:enabled=true", config)
        self.assertIn("build --//runtime/config/hal:drivers=task\n", config)

    def test_native_amd_defaults_and_explicit_opt_outs(self):
        for options, drivers, provider in (
            ([], "task,amd-gpu,amd-xdna", True),
            (["-DIREE_HAL_DRIVER_AMD_XDNA=OFF"], "task,amd-gpu", True),
            (["-DIREE_HAL_DRIVER_AMD_GPU=OFF"], "task,amd-xdna", True),
            (["-DIREE_HAL_DRIVER_AMD=OFF"], "task", False),
            (["--//runtime/config/hal:drivers=task"], "task", False),
        ):
            with self.subTest(options=options):
                config = self.configure_bazel.generate_config(
                    self.configure_bazel.parse_arguments(options)
                )
                self.assertIn(
                    f"build --//runtime/config/hal:drivers={drivers}\n", config
                )
                self.assertIn(
                    f"build --//libamdf/config:enabled={str(provider).lower()}\n",
                    config,
                )

    def test_amd_driver_rejects_explicitly_disabled_native_provider(self):
        args = self.configure_bazel.parse_arguments(
            ["-DIREE_HAL_DRIVER_AMD=ON", "-DAMDF_BUILD=OFF"]
        )
        with self.assertRaisesRegex(SystemExit, "conflicts with"):
            self.configure_bazel.generate_config(args)

    def test_amd_driver_resolves_native_provider(self):
        args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_AMD=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=task,amd-gpu,amd-xdna\n", config
        )
        self.assertIn("build --//libamdf/config:enabled=true", config)

    def test_amd_driver_native_families_are_independent(self):
        families = self.configure_bazel.AMDF_FAMILIES
        for mask in range(1 << len(families)):
            selected = [family for i, family in enumerate(families) if mask & (1 << i)]
            with self.subTest(families=selected):
                args = self.configure_bazel.parse_arguments(
                    ["-DIREE_HAL_DRIVER_AMD=ON"]
                    + [
                        f"-DAMDF_FAMILY_{family.upper()}="
                        + ("ON" if family in selected else "OFF")
                        for family in families
                    ]
                )
                config = self.configure_bazel.generate_config(args)
                self.assertIn(
                    "build --//runtime/config/hal:drivers=task,amd-gpu,amd-xdna\n",
                    config,
                )
                self.assertIn(
                    "build --//libamdf/config:families=" + ",".join(selected),
                    config,
                )

    def test_amd_engines_override_umbrella_defaults(self):
        for umbrella in (False, True):
            for gpu in (None, False, True):
                for xdna in (None, False, True):
                    for reverse in (False, True):
                        with self.subTest(
                            umbrella=umbrella, gpu=gpu, xdna=xdna, reverse=reverse
                        ):
                            defines = [
                                f"-DIREE_HAL_DRIVER_AMD={'ON' if umbrella else 'OFF'}"
                            ]
                            selected = []
                            for engine, enabled in (("GPU", gpu), ("XDNA", xdna)):
                                if enabled is not None:
                                    defines.append(
                                        f"-DIREE_HAL_DRIVER_AMD_{engine}={'ON' if enabled else 'OFF'}"
                                    )
                                effective = umbrella if enabled is None else enabled
                                if effective:
                                    selected.append("amd-" + engine.lower())
                            if reverse:
                                defines.reverse()
                            args = self.configure_bazel.parse_arguments(defines)
                            config = self.configure_bazel.generate_config(args)
                            self.assertIn(
                                "build --//runtime/config/hal:drivers="
                                + ",".join(["task", *selected])
                                + "\n",
                                config,
                            )
                            self.assertIn(
                                "build --//libamdf/config:enabled="
                                + str(bool(selected)).lower()
                                + "\n",
                                config,
                            )
                            self.assertIn(
                                "build --//libamdf/config:families=rdna,cdna,xdna\n",
                                config,
                            )

    def test_native_amd_engine_selections(self):
        for selection, expected in (
            ("amd", "amd-gpu,amd-xdna"),
            ("amd-gpu", "amd-gpu"),
            ("amd-xdna", "amd-xdna"),
            ("amd-gpu,amd-xdna", "amd-gpu,amd-xdna"),
        ):
            with self.subTest(selection=selection):
                args = self.configure_bazel.parse_arguments(
                    [
                        "--//runtime/config/hal:drivers=" + selection,
                    ]
                )
                config = self.configure_bazel.generate_config(args)
                self.assertIn(
                    "build --//runtime/config/hal:drivers=" + expected + "\n", config
                )
                self.assertIn("build --//libamdf/config:enabled=true\n", config)

    def test_disabled_amd_engines_do_not_require_libamdf(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DIREE_HAL_DRIVER_AMD=ON",
                "-DIREE_HAL_DRIVER_AMD_GPU=OFF",
                "-DIREE_HAL_DRIVER_AMD_XDNA=OFF",
                "-DAMDF_BUILD=OFF",
            ]
        )
        config = self.configure_bazel.generate_config(args)
        self.assertIn("build --//runtime/config/hal:drivers=task\n", config)
        self.assertIn("build --//libamdf/config:enabled=false\n", config)

    def test_libamdf_explicit_disable_is_independent_of_hal_driver(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DIREE_HAL_DRIVER_AMDGPU=ON",
                "-DIREE_HAL_DRIVER_AMD=OFF",
                "-DAMDF_BUILD=OFF",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//libamdf/config:enabled=false", config)

    def test_libamdf_portable_family_options_select_packages(self):
        args = self.configure_bazel.parse_arguments(
            ["-DAMDF_FAMILY_CDNA=OFF", "-DAMDF_FAMILY_XDNA=OFF"]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//libamdf/config:families=rdna", config)

    def test_libamdf_native_options_select_library_and_families(self):
        args = self.configure_bazel.parse_arguments(
            [
                "--//libamdf/config:enabled=true",
                "--//libamdf/config:families=cdna,xdna",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//libamdf/config:enabled=true", config)
        self.assertIn("build --//libamdf/config:families=cdna,xdna", config)

    def test_libamdf_portable_and_native_options_conflict(self):
        args = self.configure_bazel.parse_arguments(
            ["-DAMDF_BUILD=ON", "--//libamdf/config:enabled=true"]
        )

        with self.assertRaisesRegex(SystemExit, "Do not mix portable"):
            self.configure_bazel.generate_config(args)

    def test_libamdf_rejects_unknown_native_family(self):
        args = self.configure_bazel.parse_arguments(
            ["--//libamdf/config:families=banana"]
        )

        with self.assertRaisesRegex(SystemExit, "Unknown libamdf family"):
            self.configure_bazel.generate_config(args)

    def test_native_bazel_options_configure_amdgpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "--//runtime/config/hal:drivers=amdgpu,task",
                    f"--repo_env=IREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=task,amdgpu",
            config,
        )
        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_ROCM_DEPENDENCY_MODE=package", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assert_rocm_path(config, rocm_root)
        self.assertNotIn("--deleted_packages", config)

    def test_rocm_dependency_mode_overrides_rocm_path_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "--//runtime/config/hal:drivers=amdgpu,task",
                    "--repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned",
                    f"--repo_env=IREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_ROCM_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assert_rocm_path(config, rocm_root)

    def test_rdma_defaults_off_and_accepts_portable_or_native_selection(self):
        for arguments, expected in [
            ([], "false"),
            (["-DIREE_NET_RDMA=ON"], "true"),
            (["-DIREE_NET_RDMA=OFF"], "false"),
            (["--//runtime/config/net:rdma=true"], "true"),
            (["--//runtime/config/net:rdma=false"], "false"),
        ]:
            with self.subTest(arguments=arguments):
                args = self.configure_bazel.parse_arguments(arguments)
                config = self.configure_bazel.generate_config(args)
                self.assertIn(f"build --//runtime/config/net:rdma={expected}", config)
                self.assertIn(
                    "build --//runtime/config/hal:drivers=task,amd-gpu,amd-xdna\n",
                    config,
                )

    def test_rdma_rejects_mixed_selection_spellings(self):
        with self.assertRaisesRegex(SystemExit, "Do not mix portable -DIREE_NET_RDMA"):
            args = self.configure_bazel.parse_arguments(
                ["-DIREE_NET_RDMA=ON", "--//runtime/config/net:rdma=false"]
            )
            self.configure_bazel.generate_config(args)

    def test_portable_project_options_configure_webgpu(self):
        args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_WEBGPU=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=task,webgpu,amd-gpu,amd-xdna\n",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertNotIn("runtime/src/iree/hal/drivers/webgpu,", config)
        self.assertNotIn("runtime/src/iree/hal/drivers/webgpu/registration", config)

    def test_portable_project_options_configure_vulkan(self):
        args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_VULKAN=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=task,vulkan,amd-gpu,amd-xdna\n",
            config,
        )
        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertNotIn("IREE_ROCM_PATH", config)

    def test_api_options_have_equivalent_native_and_portable_spellings(self):
        portable = self.configure_bazel.parse_arguments(
            ["-DIREE_ENABLE_VULKAN=ON", "-DIREE_ENABLE_D3D12=OFF"]
        )
        native = self.configure_bazel.parse_arguments(
            [
                "--//build_tools/vulkan/config:enabled=true",
                "--//build_tools/d3d12/config:enabled=false",
            ]
        )
        config = self.configure_bazel.generate_config(portable)
        self.assertEqual(config, self.configure_bazel.generate_config(native))
        self.assertIn("build --//build_tools/vulkan/config:enabled=true", config)
        self.assertIn("build --//build_tools/d3d12/config:enabled=false", config)
        self.assertIn(
            "build --//runtime/config/hal:drivers=task,amd-gpu,amd-xdna\n", config
        )

    def test_d3d12_can_be_selected_without_vulkan_or_a_hal_driver(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DIREE_HAL_DRIVER_AMD=OFF",
                "-DIREE_ENABLE_D3D12=ON",
                "-DIREE_ENABLE_VULKAN=OFF",
            ]
        )
        config = self.configure_bazel.generate_config(args)
        self.assertIn("build --//build_tools/vulkan/config:enabled=false", config)
        self.assertIn("build --//build_tools/d3d12/config:enabled=true", config)
        self.assertIn("build --//runtime/config/hal:drivers=task\n", config)

    def test_hal_selection_does_not_mutate_explicit_api_request(self):
        args = self.configure_bazel.parse_arguments(
            ["-DIREE_HAL_DRIVER_VULKAN=ON", "-DIREE_ENABLE_VULKAN=OFF"]
        )
        config = self.configure_bazel.generate_config(args)
        self.assertIn(
            "build --//runtime/config/hal:drivers=task,vulkan,amd-gpu,amd-xdna\n",
            config,
        )
        self.assertIn("build --//build_tools/vulkan/config:enabled=false", config)

    def test_api_options_reject_ambiguous_or_invalid_inputs(self):
        for api in ("vulkan", "d3d12"):
            portable = f"-DIREE_ENABLE_{api.upper()}"
            native = f"--//build_tools/{api}/config:enabled"
            for options in (
                [portable + "=ON", native + "=false"],
                [portable + "=unknown"],
                [native + "=unknown"],
                [native],
            ):
                with self.subTest(options=options):
                    args = self.configure_bazel.parse_arguments(options)
                    with self.assertRaises(SystemExit):
                        self.configure_bazel.generate_config(args)

    def test_environment_rocm_path_configures_amdgpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)
            args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_AMDGPU=ON"])

            with mock.patch.dict(
                self.configure_bazel.os.environ,
                {"IREE_ROCM_PATH": str(rocm_root)},
                clear=True,
            ):
                config = self.configure_bazel.generate_config(args)

        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assert_rocm_path(config, rocm_root)

    def test_pinned_amdgpu_without_rocm_path_uses_no_device_toolchain(self):
        args = self.configure_bazel.parse_arguments(["-DIREE_HAL_DRIVER_AMDGPU=ON"])

        with mock.patch.dict(self.configure_bazel.os.environ, {}, clear=True):
            config = self.configure_bazel.generate_config(args)

        self.assertIn("common --repo_env=IREE_DEPENDENCY_MODE=pinned", config)
        self.assertIn(
            "common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none",
            config,
        )
        self.assertNotIn("IREE_ROCM_PATH", config)

    def test_package_rocm_driver_without_rocm_path_fails(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DIREE_HAL_DRIVER_AMDGPU=ON",
                "-DIREE_ROCM_DEPENDENCY_MODE=package",
            ]
        )

        with mock.patch.dict(self.configure_bazel.os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "package dependency mode"):
                self.configure_bazel.generate_config(args)

    def test_unknown_dependency_mode_fails(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DIREE_DEPENDENCY_MODE=banana",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "IREE_DEPENDENCY_MODE"):
            self.configure_bazel.generate_config(args)

    def test_default_loom_scope_is_dependency_satisfied(self):
        args = self.configure_bazel.parse_arguments([])
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/arch:enable=aie2p,amdgpu,spirv,vm,x86",
            config,
        )
        self.assertIn("build --//loom/config/emit:enable=elf,spirv,vm", config)
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("build --//loom/config/import:enable=", config)
        self.assertIn("# Loom target selection: default", config)

    def test_family_preset_replaces_the_default_fallback(self):
        args = self.configure_bazel.parse_arguments(["-DLOOM_TARGET_ALL_CPU=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//loom/config/arch:enable=vm,x86", config)
        self.assertIn("build --//loom/config/emit:enable=elf,vm", config)
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("# Loom target selection: cpu", config)

    def test_family_presets_compose_by_union(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_TARGET_ALL_CPU=ON",
                "-DLOOM_TARGET_ALL_AMD_NPU=ON",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//loom/config/arch:enable=aie2p,vm,x86", config)
        self.assertIn("build --//loom/config/emit:enable=elf,vm", config)
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)
        self.assertIn("# Loom target selection: cpu,amd_npu", config)

    def test_default_can_be_replaced_by_explicit_leaves(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_TARGET_DEFAULT=OFF",
                "-DLOOM_ARCH_X86=ON",
                "-DLOOM_EMIT_ELF=ON",
                "-DLOOM_EXECUTE_IREE_HAL=ON",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//loom/config/arch:enable=x86", config)
        self.assertIn("build --//loom/config/emit:enable=elf", config)
        self.assertIn("build --//loom/config/execute:enable=iree_hal", config)
        self.assertIn("# Loom target selection: explicit leaves", config)

    def test_portable_wasm_options_configure_complete_scope(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_ARCH_WASM=ON",
                "-DLOOM_EMIT_WASM=ON",
                "-DLOOM_EXECUTE_WASM_HOST=ON",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/arch:enable=aie2p,amdgpu,spirv,vm,wasm,x86",
            config,
        )
        self.assertIn("build --//loom/config/emit:enable=elf,spirv,vm,wasm", config)
        self.assertIn(
            "build --//loom/config/execute:enable=iree_hal,iree_vm,wasm_host",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none", config)
        self.assertNotIn("IREE_ROCM_PATH", config)

    def test_native_exact_options_configure_all_capability_axes(self):
        args = self.configure_bazel.parse_arguments(
            [
                "--//loom/config/arch:enable=vm,x86",
                "--//loom/config/emit:enable=elf,vm",
                "--//loom/config/execute:enable=iree_hal,iree_vm",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//loom/config/arch:enable=vm,x86", config)
        self.assertIn("build --//loom/config/emit:enable=elf,vm", config)
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)

    def test_disabled_leaf_does_not_subtract_from_a_preset(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_ARCH_AMDGPU=OFF",
                "-DLOOM_EMIT_SPIRV=OFF",
                "-DLOOM_EXECUTE_IREE_HAL=OFF",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//loom/config/arch:enable=aie2p,amdgpu,spirv,vm,x86",
            config,
        )
        self.assertIn("build --//loom/config/emit:enable=elf,spirv,vm", config)
        self.assertIn("build --//loom/config/execute:enable=iree_hal,iree_vm", config)

    def test_native_capability_lists_must_be_complete(self):
        args = self.configure_bazel.parse_arguments(["--//loom/config/arch:enable=x86"])

        with self.assertRaisesRegex(SystemExit, "must name complete"):
            self.configure_bazel.generate_config(args)

    def test_target_preset_conflicts_with_native_capability_lists(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_TARGET_ALL_CPU=ON",
                "--//loom/config/arch:enable=x86",
                "--//loom/config/emit:enable=elf",
                "--//loom/config/execute:enable=iree_hal",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "Do not mix Loom target presets"):
            self.configure_bazel.generate_config(args)

    def test_portable_loom_import_option_configures_importer_scope(self):
        args = self.configure_bazel.parse_arguments(["-DLOOM_IMPORT_TILELANG=ON"])
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//loom/config/import:enable=tilelang", config)

    def test_native_loom_import_option_configures_importer_scope(self):
        args = self.configure_bazel.parse_arguments(
            ["--//loom/config/import:enable=mlir,tilelang"]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn("build --//loom/config/import:enable=mlir,tilelang", config)

    def test_native_amdgpu_targets_configure_processor_scope(self):
        args = self.configure_bazel.parse_arguments(
            [
                "--//runtime/src/iree/hal/drivers/amdgpu:targets=gfx942,gfx125X-all",
                "--//loom/config/arch/amdgpu:targets=gfx942,gfx125X-all",
            ]
        )
        config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/src/iree/hal/drivers/amdgpu:targets=gfx125X-all,gfx942",
            config,
        )
        self.assertIn(
            "build --//loom/config/arch/amdgpu:targets=gfx125X-all,gfx942",
            config,
        )

    def test_portable_and_native_loom_arch_options_conflict(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_ARCH_AMDGPU=ON",
                "--//loom/config/arch:enable=spirv",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "Do not mix portable"):
            self.configure_bazel.generate_config(args)

    def test_portable_and_native_loom_import_options_conflict(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_IMPORT_TILELANG=ON",
                "--//loom/config/import:enable=mlir",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "Do not mix portable"):
            self.configure_bazel.generate_config(args)

    def test_portable_and_native_loom_execute_options_conflict(self):
        args = self.configure_bazel.parse_arguments(
            [
                "-DLOOM_EXECUTE_IREE_HAL=ON",
                "--//loom/config/execute:enable=iree_hal",
            ]
        )

        with self.assertRaisesRegex(SystemExit, "Do not mix portable"):
            self.configure_bazel.generate_config(args)

    def test_removed_driver_dialect_fails(self):
        args = self.configure_bazel.parse_arguments(["--enable-driver=amdgpu"])

        with self.assertRaisesRegex(SystemExit, "was removed"):
            self.configure_bazel.generate_config(args)

    def test_unknown_portable_define_fails(self):
        args = self.configure_bazel.parse_arguments(["-DCMAKE_BUILD_TYPE=Debug"])

        with self.assertRaisesRegex(SystemExit, "Unsupported Bazel configure option"):
            self.configure_bazel.generate_config(args)


if __name__ == "__main__":
    unittest.main()
