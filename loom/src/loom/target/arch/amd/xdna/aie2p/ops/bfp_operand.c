// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/ops/bfp_operand.h"

#include "iree/base/string_view.h"

bool loom_aie2p_bfp16ebs8_operand_type_isa(const loom_module_t* module,
                                           loom_type_t type) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  const loom_string_id_t name_id = loom_type_dialect_name_id(type);
  return name_id != LOOM_STRING_ID_INVALID && name_id < module->strings.count &&
         iree_string_view_equal(
             loom_string_table_get(&module->strings, name_id),
             IREE_SV("aie2p.bfp16ebs8.operand"));
}
