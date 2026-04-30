# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from loom.importers.core import LoomImportError
from loom.importers.mlir.api import import_iree_ir
from loom.importers.mlir.importer import (
    MlirImportOptions,
    choose_chunk,
    import_mlir_module,
    print_loom_module,
    split_input_file,
)


def test_split_input_chunk_selection_by_export() -> None:
    chunks = split_input_file(
        """
module {
  func.func @a()
}
// -----
module {
  hal.executable.export public @b ordinal(0)
  func.func @impl()
}
"""
    )

    assert choose_chunk(chunks, "b") == chunks[1]


def test_optional_iree_smoke_imports_scf_kernel() -> None:
    try:
        import_iree_ir(prefer_abi3_extensions=True)
    except LoomImportError as exc:
        pytest.skip(str(exc))
    source = """
#map = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [1, 1, 1] subgroup_size = 1>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">

module {
  hal.executable private @smoke {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @for_smoke ordinal(0) layout(#pipeline_layout)
      builtin.module {
        func.func @for_smoke() attributes {translation_info = #translation} {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %acc = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<4xf32>
          %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg0 = %acc) -> (f32) {
            %idx = affine.apply #map(%i)
            %next = arith.addf %arg0, %arg0 : f32
            scf.yield %next : f32
          }
          return
        }
      }
    }
  }
}
"""

    result = import_mlir_module(
        source,
        options=MlirImportOptions(
            kernel="for_smoke",
            include_report=True,
            prefer_abi3_extensions=True,
        ),
    )

    text = print_loom_module(result.module)
    assert 'kernel.def target(@rocm_hsaco_fb) export("for_smoke")' in text
    assert "scf.for" in text
    assert result.report is not None


def test_optional_iree_unsupported_dialect_result_reports_error() -> None:
    try:
        import_iree_ir(prefer_abi3_extensions=True)
    except LoomImportError as exc:
        pytest.skip(str(exc))
    source = """
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [1, 1, 1] subgroup_size = 1>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">

module {
  hal.executable private @smoke {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @mma_smoke ordinal(0) layout(#pipeline_layout)
      builtin.module {
        func.func @mma_smoke() attributes {translation_info = #translation} {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f16
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<4xf16>
          %mma = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf16, "COp">
          return
        }
      }
    }
  }
}
"""

    with pytest.raises(LoomImportError, match="gpu.subgroup_mma_constant_matrix"):
        import_mlir_module(
            source,
            options=MlirImportOptions(
                kernel="mma_smoke",
                prefer_abi3_extensions=True,
            ),
        )
