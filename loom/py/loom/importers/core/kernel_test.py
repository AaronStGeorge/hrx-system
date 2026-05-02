# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.importers.core import (
    KernelModuleSpec,
    create_kernel_module,
    kernel_module_ops,
    print_loom_module,
)
from loom.verify import verify_module


def test_create_kernel_module_uses_supported_amdgpu_target_record() -> None:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset="hip -mcpu=gfx1100",
            export_symbol="kernel",
            callee="kernel",
            arguments=[],
        )
    )
    with shell.builder.insertion_block(shell.body_block):
        shell.builder.kernel.return_()

    diagnostics = verify_module(
        shell.module,
        ops=kernel_module_ops("hip -mcpu=gfx1100"),
    )
    diagnostics.raise_if_errors()
    assert (
        print_loom_module(
            shell.module,
            ops=kernel_module_ops("hip -mcpu=gfx1100"),
        )
        == """amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export(\"kernel\") workgroup_size(1, 1, 1) @kernel() {
  kernel.return
}
"""
    )


def test_create_kernel_module_uses_generic_target_when_cpu_has_no_row() -> None:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset="hip -mcpu=gfx942",
            export_symbol="kernel",
            callee="kernel",
            arguments=[],
        )
    )
    with shell.builder.insertion_block(shell.body_block):
        shell.builder.kernel.return_()

    diagnostics = verify_module(
        shell.module,
        ops=kernel_module_ops("hip -mcpu=gfx942"),
    )
    diagnostics.raise_if_errors()
    assert (
        print_loom_module(
            shell.module,
            ops=kernel_module_ops("hip -mcpu=gfx942"),
        )
        == """target.generic<reference> @hip_mcpu_gfx942

kernel.def target(@hip_mcpu_gfx942) export(\"kernel\") workgroup_size(1, 1, 1) @kernel() {
  kernel.return
}
"""
    )
