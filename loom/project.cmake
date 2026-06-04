# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake"
)

if(NOT DEFINED LOOM_BUILD)
  option(LOOM_BUILD
    "Build Loom compiler libraries, tools, bindings, and tests." ON)
endif()

option(LOOM_EMIT_AMDGPU
  "Enables the AMDGPU Loom artifact emitter." OFF)
option(LOOM_EMIT_IREEVM
  "Enables the IREE VM Loom artifact emitter." ON)
option(LOOM_EMIT_LLVMIR
  "Enables the LLVM IR Loom artifact emitter." OFF)
option(LOOM_EMIT_SPIRV
  "Enables the SPIR-V Loom artifact emitter." OFF)
option(LOOM_EMIT_WASM
  "Enables the WebAssembly Loom artifact emitter." OFF)

option(LOOM_EXECUTE_AMDGPU
  "Enables Loom execution tests that require AMDGPU runtime support." OFF)
option(LOOM_EXECUTE_IREEVM
  "Enables Loom execution tests that use IREE VM runtime support." ON)
option(LOOM_EXECUTE_SPIRV_VULKAN
  "Enables Loom execution tests that require SPIR-V/Vulkan runtime support." OFF)

option(LOOM_IMPORT_MLIR
  "Enables the Loom MLIR importer package and importer tests." OFF)
option(LOOM_IMPORT_TILELANG
  "Enables the Loom TileLang importer package and importer tests." OFF)

option(LOOM_TARGET_AMDGPU
  "Enables the AMDGPU Loom target architecture." OFF)
option(LOOM_TARGET_IREEVM
  "Enables the IREE VM Loom target architecture." ON)
option(LOOM_TARGET_SPIRV
  "Enables the SPIR-V Loom target architecture." OFF)
option(LOOM_TARGET_WASM
  "Enables the WebAssembly Loom target architecture." OFF)
option(LOOM_TARGET_X86
  "Enables the x86 Loom target architecture." OFF)

function(loom_configure_project)
endfunction()
