# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration for source-tree IREE CI command groups."""

IREE_TARGET_DIRECTORIES = ("runtime", "loom")

SANITIZER_TEST_CONFIGS = ("asan", "ubsan", "tsan")
SANITIZER_BUILD_CONFIGS = ("msan",)
CPU_XFAIL_TARGETS = ()
CPU_SANITIZERS_XFAIL_TARGETS = (
    "-//runtime/src/iree/async/platform/io_uring/cts/...",
    "-//runtime/src/iree/builtins/device/tools:libdevice_test",
    "-//runtime/src/iree/hal:string_util_test",
    "-//runtime/src/iree/hal/local/elf/...",
    "-//runtime/src/iree/hal/local:profile_test",
    "-//runtime/src/iree/hal/replay:execute_test",
    "-//runtime/src/iree/io/formats/gguf:gguf_parser_test",
    "-//runtime/src/iree/tokenizer/...",
    "-//runtime/src/iree/tooling/profile:cli_test",
    "-//runtime/src/iree/vm:list_test",
)

AMDGPU_DRIVER_TARGETS = ("//runtime/src/iree/hal/drivers/amdgpu/...",)
AMDGPU_RESOURCE_TAG = "iree-run-requirement=runtime.resource.amd_gpu"
AMDGPU_XFAIL_TARGETS = (
    "-//runtime/src/iree/hal/drivers/amdgpu/cts/...",
    "-//runtime/src/iree/hal/drivers/amdgpu:allocator_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:host_queue_command_buffer_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:pm4_command_buffer_benchmark_test",
)
AMDGPU_SANITIZERS_XFAIL_TARGETS = (
    "-//runtime/src/iree/hal/drivers/amdgpu/cts/...",
    "-//runtime/src/iree/hal/drivers/amdgpu:allocator_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:host_queue_command_buffer_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:host_queue_pending_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:host_queue_staging_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:host_queue_submission_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:pm4_command_buffer_benchmark_test",
    "-//runtime/src/iree/hal/drivers/amdgpu:slab_provider_test",
    "-//runtime/src/iree/hal/drivers/amdgpu/util:blit_benchmark_test",
    "-//runtime/src/iree/hal/drivers/amdgpu/util:block_pool_test",
    "-//runtime/src/iree/hal/drivers/amdgpu/util:pm4_emitter_test",
    "-//runtime/src/iree/hal/drivers/amdgpu/util:queue_benchmark_test",
)
