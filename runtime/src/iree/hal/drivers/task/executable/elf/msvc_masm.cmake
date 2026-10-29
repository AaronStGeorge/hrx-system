# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(WIN32 AND IREE_ARCH STREQUAL "x86_64")
  # CMake + MASM does not work well and CMake ends up passing all our C/C++
  # flags confusing MASM. We invoke MASM directly (ml.exe/ml64.exe) to keep it
  # quiet.
  iree_cc_library_add_external_objects(iree_hal_drivers_task_executable_elf_arch
    "${CMAKE_CURRENT_BINARY_DIR}/arch/x86_64_msvc.obj")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/arch/x86_64_msvc.obj
    DEPENDS ${_IREE_PACKAGE_SOURCE_DIR}/arch/x86_64_msvc.asm
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/arch
    COMMAND ml64 /nologo /Zi /c /Fo ${CMAKE_CURRENT_BINARY_DIR}/arch/x86_64_msvc.obj ${_IREE_PACKAGE_SOURCE_DIR}/arch/x86_64_msvc.asm
    VERBATIM
  )
endif()
