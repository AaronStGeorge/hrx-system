# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-of-truth rows for ordinary SPIR-V scalar storage-buffer memory."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.spirv.features import feature_bits_value


@dataclass(frozen=True, slots=True)
class StorageBufferScalar:
    source_type: str
    suffix: str
    scalar_enum: str
    byte_width: int
    feature_atoms: tuple[str, ...] = ()

    @property
    def feature_bits(self) -> int:
        return feature_bits_value(self.feature_atoms)


STORAGE_BUFFER_SCALARS = (
    StorageBufferScalar(
        source_type="i8",
        suffix="i8",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S8",
        byte_width=1,
        feature_atoms=("int8", "storage_buffer_8bit_access"),
    ),
    StorageBufferScalar(
        source_type="i16",
        suffix="i16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S16",
        byte_width=2,
        feature_atoms=("int16", "storage_buffer_16bit_access"),
    ),
    StorageBufferScalar(
        source_type="i32",
        suffix="i32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S32",
        byte_width=4,
    ),
    StorageBufferScalar(
        source_type="i64",
        suffix="i64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_S64",
        byte_width=8,
    ),
    StorageBufferScalar(
        source_type="f16",
        suffix="f16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F16",
        byte_width=2,
        feature_atoms=("float16", "storage_buffer_16bit_access"),
    ),
    StorageBufferScalar(
        source_type="bf16",
        suffix="bf16",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_BF16",
        byte_width=2,
        feature_atoms=("bfloat16_type_khr", "storage_buffer_16bit_access"),
    ),
    StorageBufferScalar(
        source_type="f32",
        suffix="f32",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F32",
        byte_width=4,
    ),
    StorageBufferScalar(
        source_type="f64",
        suffix="f64",
        scalar_enum="LOOM_SPIRV_SCALAR_TYPE_F64",
        byte_width=8,
        feature_atoms=("float64",),
    ),
)


def storage_buffer_scalar_by_source_type(
    source_type: str,
) -> StorageBufferScalar | None:
    for row in STORAGE_BUFFER_SCALARS:
        if row.source_type == source_type:
            return row
    return None
