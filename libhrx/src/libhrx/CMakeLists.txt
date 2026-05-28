# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
#
# libhrx.so — LTO'd shared library, only hrx_* symbols exported.

set(HRX_SOURCES
    status.c
    runtime.c
    device.c
    allocator.c
    semaphore.c
    stream.c
    buffer.c
    pool.c
    buffer_view.c
    executable.c
    fence.c
    module.c
    transfer.c
    host_allocator.c
    queue_ops.c
    value_list.c
)

set(LIBHRX_INCLUDE_DIR "${LIBHRX_SOURCE_DIR}/include")

add_library(hrx SHARED ${HRX_SOURCES})
target_include_directories(hrx
    PUBLIC
        $<BUILD_INTERFACE:${LIBHRX_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include/hrx>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBHRX_SOURCE_DIR}/src
)
target_link_libraries(hrx PRIVATE
    ${CMAKE_DL_LIBS}
    flatcc::parsing
    iree_async_util_proactor_pool
    iree_hal_drivers_local_task_task_driver
    iree_hal_local_loaders_registration_registration
    iree_hal_utils_profile_file
    iree_modules_hal_hal
    iree_vm_bytecode_module
    iree_vm_vm
)
if(IREE_HAL_DRIVER_AMDGPU)
  target_link_libraries(hrx PRIVATE
      iree_hal_drivers_amdgpu_registration_registration
  )
endif()
target_compile_definitions(hrx PRIVATE HRX_BUILDING_SHARED)
if(IREE_HAL_DRIVER_AMDGPU)
  target_compile_definitions(hrx PRIVATE HRX_HAS_IREE_AMDGPU_DRIVER)
endif()
set_target_properties(hrx PROPERTIES
    OUTPUT_NAME hrx
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# Version script to hide all non-hrx_* symbols.
set_target_properties(hrx PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${LIBHRX_SOURCE_DIR}/cmake/hrx_exports.lds")

# LTO (interprocedural optimization).
include(CheckIPOSupported)
check_ipo_supported(RESULT _hrx_ipo_supported OUTPUT _hrx_ipo_error)
if(_hrx_ipo_supported)
  set_target_properties(hrx PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  message(STATUS "hrx-runtime: LTO enabled for libhrx.so")
else()
  message(WARNING "hrx-runtime: LTO not supported: ${_hrx_ipo_error}")
endif()
