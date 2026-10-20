// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/cts/cpu_executable.h"

#include "iree/hal/drivers/task/executable/executable.h"
#include "iree/hal/drivers/task/executable/library/util.h"

namespace iree::hal::amd::cts {
namespace {

struct CpuAddExecutable : iree_hal_task_executable_t {
  static int Add(const iree_hal_executable_environment_v0_t*,
                 const iree_hal_executable_dispatch_state_v0_t* dispatch,
                 const iree_hal_executable_workgroup_state_v0_t* workgroup) {
    uint32_t addend;
    memcpy(&addend, dispatch->constants.data, sizeof(addend));
    auto* values = static_cast<uint32_t*>(dispatch->binding_ptrs[0]);
    values[workgroup->workgroup_id_x] += addend;
    return 0;
  }

  static const iree_hal_executable_library_v0_t& Library() {
    static const iree_hal_executable_dispatch_v0_t functions[] = {Add};
    static const char* const names[] = {"add_u32"};
    static const auto attributes = [] {
      iree_hal_executable_dispatch_attrs_v0_t value = {
          .binding_count = 1,
          .workgroup_size_x = 1,
          .workgroup_size_y = 1,
          .workgroup_size_z = 1,
          .constant_byte_length = sizeof(uint32_t),
      };
      return value;
    }();
    static const auto library = [] {
      iree_hal_executable_library_v0_t value = {};
      value.exports.count = 1;
      value.exports.ptrs = functions;
      value.exports.attrs = &attributes;
      value.exports.names = names;
      return value;
    }();
    return library;
  }

  explicit CpuAddExecutable(const iree_hal_queue_family_t* family) {
    static const auto vtable = [] {
      iree_hal_task_executable_vtable_t value = {};
      value.base.destroy = +[](iree_hal_executable_t* base) {
        auto* executable = reinterpret_cast<CpuAddExecutable*>(base);
        iree_hal_task_executable_deinitialize(executable);
        delete executable;
      };
      value.base.function_count =
          +[](iree_hal_executable_t*) -> iree_host_size_t {
        return Library().exports.count;
      };
      value.base.function_info =
          +[](iree_hal_executable_t*, iree_hal_executable_function_t function,
              iree_hal_executable_function_info_t* out_info) {
            return iree_hal_executable_library_export_info(&Library(), function,
                                                           out_info);
          };
      value.base.function_parameters =
          +[](iree_hal_executable_t*, iree_hal_executable_function_t function,
              iree_host_size_t capacity,
              iree_hal_executable_function_parameter_t* out_parameters) {
            return iree_hal_executable_library_export_parameters(
                &Library(), function, capacity, out_parameters);
          };
      value.issue_call =
          +[](iree_hal_task_executable_t* executable, iree_host_size_t,
              const iree_hal_executable_dispatch_state_v0_t* dispatch,
              const iree_hal_executable_workgroup_state_v0_t* workgroup,
              uint32_t) {
            Add(&executable->environment, dispatch, workgroup);
            return iree_ok_status();
          };
      return value;
    }();
    iree_hal_task_executable_initialize(family, &vtable,
                                        iree_allocator_system(), this);
    dispatch_attrs = Library().exports.attrs;
    dispatch_ptrs = Library().exports.ptrs;
    export_count = Library().exports.count;
    export_names = Library().exports.names;
  }
};

}  // namespace

iree_hal_executable_t* CreateCpuAddExecutable(
    const iree_hal_queue_family_t* family) {
  return &(new CpuAddExecutable(family))->base;
}

}  // namespace iree::hal::amd::cts
