// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/base.h"

#include <string.h>

#include "iree/base/api.h"
#include "loomc/status.h"

static loomc_status_t loomc_system_allocator_ctl(
    void* self, loomc_allocator_command_t command, const void* params,
    void** inout_ptr) {
  (void)self;
  iree_allocator_t allocator = iree_allocator_system();
  return (loomc_status_t)allocator.ctl(
      allocator.self, (iree_allocator_command_t)command, params, inout_ptr);
}

loomc_allocator_t loomc_allocator_system(void) {
  return (loomc_allocator_t){
      .self = NULL,
      .ctl = loomc_system_allocator_ctl,
  };
}

loomc_allocator_t loomc_allocator_or_system(loomc_allocator_t allocator) {
  if (allocator.ctl) {
    return allocator;
  }
  return loomc_allocator_system();
}

loomc_status_t loomc_allocator_malloc(loomc_allocator_t allocator,
                                      loomc_host_size_t byte_length,
                                      void** out_ptr) {
  if (out_ptr == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_ptr must not be NULL");
  }
  *out_ptr = NULL;
  allocator = loomc_allocator_or_system(allocator);
  loomc_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, LOOMC_ALLOCATOR_COMMAND_CALLOC, &params,
                       out_ptr);
}

loomc_status_t loomc_allocator_malloc_uninitialized(
    loomc_allocator_t allocator, loomc_host_size_t byte_length,
    void** out_ptr) {
  if (out_ptr == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_ptr must not be NULL");
  }
  *out_ptr = NULL;
  allocator = loomc_allocator_or_system(allocator);
  loomc_allocator_alloc_params_t params = {
      .byte_length = byte_length,
  };
  return allocator.ctl(allocator.self, LOOMC_ALLOCATOR_COMMAND_MALLOC, &params,
                       out_ptr);
}

void loomc_allocator_free(loomc_allocator_t allocator, void* ptr) {
  if (ptr == NULL) {
    return;
  }
  allocator = loomc_allocator_or_system(allocator);
  void* inout_ptr = ptr;
  allocator.ctl(allocator.self, LOOMC_ALLOCATOR_COMMAND_FREE, NULL, &inout_ptr);
}
