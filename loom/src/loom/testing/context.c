// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/context.h"

#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/pool/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

typedef const loom_op_vtable_t* const* (*loom_testing_dialect_vtables_fn_t)(
    iree_host_size_t* out_count);

typedef struct loom_testing_dialect_registration_t {
  // Numeric dialect ID from the generated op registry.
  loom_dialect_id_t dialect_id;
  // Static generated vtable accessor for this dialect.
  loom_testing_dialect_vtables_fn_t vtables_fn;
} loom_testing_dialect_registration_t;

static const loom_testing_dialect_registration_t loom_testing_all_dialects[] = {
    {LOOM_DIALECT_TEST, loom_test_dialect_vtables},
    {LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables},
    {LOOM_DIALECT_FUNC, loom_func_dialect_vtables},
    {LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables},
    {LOOM_DIALECT_POOL, loom_pool_dialect_vtables},
    {LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables},
    {LOOM_DIALECT_SCF, loom_scf_dialect_vtables},
    {LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables},
    {LOOM_DIALECT_VIEW, loom_view_dialect_vtables},
    {LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables},
    {LOOM_DIALECT_INDEX, loom_index_dialect_vtables},
    {LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables},
    {LOOM_DIALECT_LLVMIR, loom_llvmir_dialect_vtables},
};

static iree_status_t loom_testing_context_register_dialect(
    loom_context_t* context,
    const loom_testing_dialect_registration_t* registration) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = registration->vtables_fn(&count);
  if (count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dialect %u has %" PRIhsz
                            " ops, exceeding the uint16_t registry cap",
                            (unsigned)registration->dialect_id, count);
  }
  return loom_context_register_dialect(context, registration->dialect_id,
                                       vtables, (uint16_t)count);
}

iree_status_t loom_testing_context_register_all_dialects(
    loom_context_t* context) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(loom_testing_all_dialects);
       ++i) {
    IREE_RETURN_IF_ERROR(loom_testing_context_register_dialect(
        context, &loom_testing_all_dialects[i]));
  }
  return loom_context_register_builtin_encoding_vtables(context);
}

iree_status_t loom_testing_context_initialize_all(iree_allocator_t allocator,
                                                  loom_context_t* out_context) {
  loom_context_initialize(allocator, out_context);
  iree_status_t status =
      loom_testing_context_register_all_dialects(out_context);
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(out_context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(out_context);
  }
  return status;
}
