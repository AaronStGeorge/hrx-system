// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/vmvx/module.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"

#define IREE_VMVX_MODULE_VERSION_0_0 0x00000000u
#define IREE_VMVX_MODULE_VERSION_LATEST IREE_VMVX_MODULE_VERSION_0_0

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_vmvx_module_state_t {
  // Host allocator used for this context-local module state.
  iree_allocator_t host_allocator;
  // Logical processor identifier for the active VMVX workgroup.
  uint32_t processor_id;
} iree_vmvx_module_state_t;

static void IREE_API_PTR iree_vmvx_module_destroy(void* base_module) {
  (void)base_module;
}

static iree_status_t IREE_API_PTR
iree_vmvx_module_alloc_state(void* self, iree_allocator_t host_allocator,
                             iree_vm_module_state_t** out_module_state) {
  (void)self;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vmvx_module_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;

  *out_module_state = (iree_vm_module_state_t*)state;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void IREE_API_PTR
iree_vmvx_module_free_state(void* self, iree_vm_module_state_t* module_state) {
  (void)self;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vmvx_module_state_t* state = (iree_vmvx_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t IREE_API_PTR iree_vmvx_module_fork_state(
    void* self, iree_vm_module_state_t* parent_state,
    iree_allocator_t allocator, iree_vm_module_state_t** out_child_state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_vmvx_module_alloc_state(self, allocator, out_child_state);
  if (iree_status_is_ok(status)) {
    iree_vmvx_module_state_t* parent = (iree_vmvx_module_state_t*)parent_state;
    iree_vmvx_module_state_t* child =
        (iree_vmvx_module_state_t*)*out_child_state;
    child->processor_id = parent->processor_id;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Exports
//===----------------------------------------------------------------------===//

#define IREE_VMVX_ABI_EXPORT(function_name, arg_types, ret_types)        \
  IREE_VM_ABI_EXPORT(function_name, iree_vmvx_module_state_t, arg_types, \
                     ret_types)

IREE_VMVX_ABI_EXPORT(iree_vmvx_module_reserved, v, v) {
  (void)stack;
  (void)module;
  (void)state;
  (void)args;
  (void)rets;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of iree_vmvx_module_exports_.
static const iree_vm_native_function_ptr_t iree_vmvx_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_struct, arg_types, ret_types) \
  {                                                                  \
      .shim = (iree_vm_native_function_shim_t)                       \
          iree_vm_shim_##arg_struct##_##ret_types,                   \
      .target = (iree_vm_native_function_target_t)(target_fn),       \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t iree_vmvx_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_vmvx_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_struct, arg_types, ret_types) \
  {                                                                  \
      .local_name = iree_string_view_literal(name),                  \
      .calling_convention =                                          \
          iree_string_view_literal("0" #arg_types "_" #ret_types),   \
      .attr_count = 0,                                               \
      .attrs = NULL,                                                 \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_vmvx_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_vmvx_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t iree_vmvx_module_descriptor_ = {
    .name = iree_string_view_literal("vmvx"),
    .version = IREE_VMVX_MODULE_VERSION_LATEST,
    .attr_count = 0,
    .attrs = NULL,
    .dependency_count = 0,
    .dependencies = NULL,
    .import_count = 0,  // workaround for 0-length C struct
    .imports = iree_vmvx_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_vmvx_module_exports_),
    .exports = iree_vmvx_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_vmvx_module_funcs_),
    .functions = iree_vmvx_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_vmvx_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_vmvx_module_destroy,
      .alloc_state = iree_vmvx_module_alloc_state,
      .free_state = iree_vmvx_module_free_state,
      .fork_state = iree_vmvx_module_fork_state,
  };

  iree_host_size_t total_size = iree_vm_native_module_size();
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_vmvx_module_descriptor_, instance, host_allocator,
      base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  *out_module = base_module;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vmvx_module_state_update_workgroup_state(
    iree_vm_module_state_t* module_state, uint32_t processor_id) {
  iree_vmvx_module_state_t* state = (iree_vmvx_module_state_t*)module_state;
  state->processor_id = processor_id;
}
