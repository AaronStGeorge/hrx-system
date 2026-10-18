// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/bank.h"

struct loom_serve_parameter_bank_t {
  // Allocator owning this bank and its trailing buffer/plan arrays.
  iree_allocator_t host_allocator;
  // Number of domain slots, including NULL slots during partial construction.
  iree_host_size_t count;
  // Owned physical group; every invocation residency borrows this registration.
  loom_serve_residency_group_t* group;
  // Owned fixed-root views followed by count owned weight-plan pointers.
  iree_hal_buffer_t* buffers[];
};

static loom_serve_weights_t** loom_serve_parameter_bank_plans(
    const loom_serve_parameter_bank_t* bank) {
  return (loom_serve_weights_t**)(bank->buffers + bank->count);
}

iree_status_t loom_serve_parameter_bank_allocate(
    loom_serve_device_t* device, const loom_cmd_program_t* program,
    iree_allocator_t host_allocator, loom_serve_parameter_bank_t** out_bank) {
  *out_bank = NULL;
  const uint32_t count = program ? program->requirements.fixed_buffer_count : 0;
  if (!count) {
    return iree_ok_status();
  }
  loom_serve_parameter_bank_t* bank = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*bank), count,
      sizeof(iree_hal_buffer_t*) + sizeof(loom_serve_weights_t*),
      (void**)&bank));
  bank->host_allocator = host_allocator;
  bank->count = count;
  *out_bank = bank;
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_parameter_root_t reflected =
        loom_cmd_program_parameter_root_at(program, i);
    const loom_serve_weight_root_t root = {
        program, reflected, &bank->buffers[reflected.fixed_buffer_index]};
    status =
        loom_serve_weights_allocate(loom_serve_device_handle(device),
                                    loom_serve_device_transfer_queue(device),
                                    loom_serve_device_dispatch_queue(device),
                                    loom_serve_device_storage_pool(device),
                                    loom_serve_device_memory_pool(device),
                                    loom_serve_device_workspace(device), 0, 1,
                                    &root, host_allocator, &plans[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_residency_group_allocate(
        loom_serve_device_residency_cache(device), count, plans, host_allocator,
        &bank->group);
  }
  return status;
}

iree_host_size_t loom_serve_parameter_bank_count(
    const loom_serve_parameter_bank_t* bank) {
  return bank ? bank->count : 0;
}

loom_serve_weights_t* loom_serve_parameter_bank_weights(
    const loom_serve_parameter_bank_t* bank, iree_host_size_t index) {
  return loom_serve_parameter_bank_plans(bank)[index];
}

iree_hal_buffer_t* const* loom_serve_parameter_bank_buffers(
    const loom_serve_parameter_bank_t* bank) {
  return bank ? bank->buffers : NULL;
}

loom_serve_residency_group_t* loom_serve_parameter_bank_group(
    const loom_serve_parameter_bank_t* bank) {
  return bank ? bank->group : NULL;
}

void loom_serve_parameter_bank_retire(loom_serve_parameter_bank_t* bank,
                                      void (*wake)(void* user_data),
                                      void* user_data) {
  if (!bank) {
    return;
  }
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  for (iree_host_size_t i = 0; i < bank->count; ++i) {
    loom_serve_weights_set_retirement_observer(plans[i], wake, user_data);
    iree_hal_buffer_release(bank->buffers[i]);
    bank->buffers[i] = NULL;
  }
}

bool loom_serve_parameter_bank_is_retired(loom_serve_parameter_bank_t* bank) {
  if (!bank) {
    return true;
  }
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  bool retired = true;
  for (iree_host_size_t i = 0; i < bank->count; ++i) {
    retired &= loom_serve_weights_is_retired(plans[i]);
  }
  return retired;
}

void loom_serve_parameter_bank_await_retirement(
    loom_serve_parameter_bank_t* bank) {
  if (!bank) {
    return;
  }
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  for (iree_host_size_t i = 0; i < bank->count; ++i) {
    loom_serve_weights_await_retirement(plans[i]);
  }
}

iree_status_t loom_serve_parameter_bank_free(
    loom_serve_parameter_bank_t* bank) {
  if (!bank) {
    return iree_ok_status();
  }
  loom_serve_residency_group_free(bank->group);
  bank->group = NULL;
  loom_serve_weights_t** plans = loom_serve_parameter_bank_plans(bank);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < bank->count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_weights_free(plans[i]);
    if (iree_status_is_ok(status)) {
      plans[i] = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_allocator_free(bank->host_allocator, bank);
  }
  return status;
}
