# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,amdgpu",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)
        self.assertNotIn("runtime/src/iree/hal/drivers/amdgpu,", config)

    def test_native_bazel_options_configure_amdgpu(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = self.make_rocm_root(temporary_directory)

            args = self.configure_bazel.parse_arguments(
                [
                    "--//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null",
                    f"--repo_env=IREE_ROCM_PATH={rocm_root}",
                ]
            )
            config = self.configure_bazel.generate_config(args)

        self.assertIn(
            "build --//runtime/config/hal:drivers=local-sync,local-task,null,amdgpu",
            config,
        )
        self.assertIn("common --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm", config)
        self.assertIn(f"common --repo_env=IREE_ROCM_PATH={rocm_root}", config)

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
