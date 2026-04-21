# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernel dialect: kernel execution and synchronization operations."""

from loom.dialect.kernel.defs import (
    ALL_KERNEL_OPS,
    ALL_KERNEL_TYPES,
    KernelAsyncDirection,
    KernelDimension,
    KernelMemorySpace,
    KernelOrdering,
    KernelScope,
    kernel_async_cluster_gather,
    kernel_async_cluster_gather_mask,
    kernel_async_copy,
    kernel_async_copy_mask,
    kernel_async_gather,
    kernel_async_gather_mask,
    kernel_async_group,
    kernel_async_group_type,
    kernel_async_tensor_load_to_lds,
    kernel_async_tensor_store_from_lds,
    kernel_async_token_type,
    kernel_async_wait,
    kernel_barrier,
    kernel_ops,
    kernel_tensor_lds_descriptor,
    kernel_tensor_lds_descriptor_type,
    kernel_workgroup_id,
    kernel_workitem_id,
)

__all__ = [
    "kernel_ops",
    "ALL_KERNEL_OPS",
    "ALL_KERNEL_TYPES",
    "KernelAsyncDirection",
    "KernelDimension",
    "KernelMemorySpace",
    "KernelOrdering",
    "KernelScope",
    "kernel_async_cluster_gather",
    "kernel_async_cluster_gather_mask",
    "kernel_async_copy",
    "kernel_async_copy_mask",
    "kernel_async_gather",
    "kernel_async_gather_mask",
    "kernel_async_group",
    "kernel_async_tensor_load_to_lds",
    "kernel_async_tensor_store_from_lds",
    "kernel_async_group_type",
    "kernel_async_token_type",
    "kernel_async_wait",
    "kernel_barrier",
    "kernel_tensor_lds_descriptor",
    "kernel_tensor_lds_descriptor_type",
    "kernel_workgroup_id",
    "kernel_workitem_id",
]
