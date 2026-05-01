// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 AVX512 source-to-low memory lowering.

#include <stdint.h>

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/lower/internal.h"

static bool loom_x86_memory_space_is_object_memory(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return true;
    default:
      return false;
  }
}

static bool loom_x86_i64_fits_disp32(int64_t value) {
  return value >= INT32_MIN && value <= INT32_MAX;
}

typedef enum loom_x86_memory_value_kind_e {
  LOOM_X86_MEMORY_VALUE_NONE = 0,
  LOOM_X86_MEMORY_VALUE_XMM32 = 1,
  LOOM_X86_MEMORY_VALUE_ZMM32 = 2,
} loom_x86_memory_value_kind_t;

static loom_x86_memory_value_kind_t loom_x86_select_memory_value_kind(
    const loom_low_source_memory_access_plan_t* plan) {
  if (plan->element_byte_count != 4 || plan->vector_lane_byte_stride != 4) {
    return LOOM_X86_MEMORY_VALUE_NONE;
  }
  switch (plan->vector_lane_count) {
    case 4:
      return LOOM_X86_MEMORY_VALUE_XMM32;
    case 16:
      return LOOM_X86_MEMORY_VALUE_ZMM32;
    default:
      return LOOM_X86_MEMORY_VALUE_NONE;
  }
}

typedef enum loom_x86_memory_address_kind_e {
  LOOM_X86_MEMORY_ADDRESS_STATIC = 0,
  LOOM_X86_MEMORY_ADDRESS_INDEXED = 1,
} loom_x86_memory_address_kind_t;

typedef struct loom_x86_memory_access_plan_t {
  // Target-independent memory decomposition shared across targets.
  loom_low_source_memory_access_plan_t source;
  // Descriptor row selected for the x86 memory packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Low register type produced by load packets, or none for store packets.
  loom_type_t load_result_type;
  // Module string ID for the disp32 memory operand attribute.
  loom_string_id_t disp32_attr_name_id;
  // Module string ID for the indexed scale memory operand attribute.
  loom_string_id_t scale_attr_name_id;
  // Selected x86 register-width value form.
  loom_x86_memory_value_kind_t value_kind;
  // Selected x86 addressing form.
  loom_x86_memory_address_kind_t address_kind;
  // x86 index scale for LOOM_X86_MEMORY_ADDRESS_INDEXED.
  uint8_t index_scale;
} loom_x86_memory_access_plan_t;

static bool loom_x86_dynamic_stride_as_address_scale(int64_t byte_stride,
                                                     uint8_t* out_scale) {
  IREE_ASSERT_ARGUMENT(out_scale);
  switch (byte_stride) {
    case 1:
    case 2:
    case 4:
    case 8:
      *out_scale = (uint8_t)byte_stride;
      return true;
    default:
      *out_scale = 0;
      return false;
  }
}

static uint64_t loom_x86_memory_access_descriptor_id(
    const loom_op_t* source_op, const loom_x86_memory_access_plan_t* plan) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_STATIC) {
        return plan->value_kind == LOOM_X86_MEMORY_VALUE_XMM32
                   ? X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_LOAD_XMM
                   : X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_LOAD_ZMM;
      }
      return plan->value_kind == LOOM_X86_MEMORY_VALUE_XMM32
                 ? X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_LOAD_INDEXED_XMM
                 : X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_LOAD_INDEXED_ZMM;
    case LOOM_OP_VECTOR_STORE:
      if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_STATIC) {
        return plan->value_kind == LOOM_X86_MEMORY_VALUE_XMM32
                   ? X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_STORE_XMM
                   : X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_STORE_ZMM;
      }
      return plan->value_kind == LOOM_X86_MEMORY_VALUE_XMM32
                 ? X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_STORE_INDEXED_XMM
                 : X86_AVX512_CORE_DESCRIPTOR_ID_AVX512_VMOVDQU32_STORE_INDEXED_ZMM;
    default:
      IREE_ASSERT_UNREACHABLE("unexpected x86 memory source op kind");
      return LOOM_LOW_DESCRIPTOR_ID_NONE;
  }
}

static iree_status_t loom_x86_select_memory_load_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_memory_access_plan_t* plan, loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  *out_type = loom_type_none();
  if (source_op->kind != LOOM_OP_VECTOR_LOAD) {
    return iree_ok_status();
  }
  if (plan->value_kind == LOOM_X86_MEMORY_VALUE_XMM32) {
    return loom_x86_make_xmm_register_type(context, out_type);
  }
  return loom_x86_make_zmm_register_type(context, out_type);
}

static iree_status_t loom_x86_select_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_x86_memory_access_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_plan = (loom_x86_memory_access_plan_t){
      .load_result_type = loom_type_none(),
      .disp32_attr_name_id = LOOM_STRING_ID_INVALID,
      .scale_attr_name_id = LOOM_STRING_ID_INVALID,
  };
  *out_selected = false;

  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_low_source_memory_access_plan_build(
          module, loom_low_lower_context_fact_table(context), source_op,
          &out_plan->source, &diagnostic)) {
    return iree_ok_status();
  }
  out_plan->value_kind = loom_x86_select_memory_value_kind(&out_plan->source);
  if (out_plan->value_kind == LOOM_X86_MEMORY_VALUE_NONE ||
      !loom_x86_memory_space_is_object_memory(out_plan->source.memory_space) ||
      !loom_x86_i64_fits_disp32(out_plan->source.static_byte_offset) ||
      !loom_low_source_memory_value_is_block_argument(
          module, out_plan->source.root_value_id)) {
    return iree_ok_status();
  }
  if (!loom_low_source_memory_access_is_dynamic(&out_plan->source)) {
    out_plan->address_kind = LOOM_X86_MEMORY_ADDRESS_STATIC;
    out_plan->index_scale = 0;
  } else {
    const loom_low_source_memory_dynamic_term_t* term =
        loom_low_source_memory_access_single_dynamic_term(&out_plan->source);
    if (!term ||
        term->source != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE) {
      return iree_ok_status();
    }
    out_plan->address_kind = LOOM_X86_MEMORY_ADDRESS_INDEXED;
    if (!loom_x86_dynamic_stride_as_address_scale(term->byte_stride,
                                                  &out_plan->index_scale)) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, loom_x86_memory_access_descriptor_id(source_op, out_plan),
      &out_plan->descriptor));
  IREE_RETURN_IF_ERROR(loom_x86_select_memory_load_result_type(
      context, source_op, out_plan, &out_plan->load_result_type));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("disp32"), &out_plan->disp32_attr_name_id));
  if (out_plan->address_kind == LOOM_X86_MEMORY_ADDRESS_INDEXED) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module, IREE_SV("scale"), &out_plan->scale_attr_name_id));
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_x86_select_avx512_op(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE: {
      loom_x86_memory_access_plan_t selected_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_x86_select_memory_access(
          context, source_op, &selected_plan, &selected));
      if (selected) {
        loom_x86_memory_access_plan_t* plan = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan), (void**)&plan));
        *plan = selected_plan;
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static loom_named_attr_t loom_x86_make_disp32_attr(loom_string_id_t name_id,
                                                   int64_t value) {
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  IREE_ASSERT(loom_x86_i64_fits_disp32(value));
  return (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
}

static loom_named_attr_t loom_x86_make_scale_attr(loom_string_id_t name_id,
                                                  uint8_t value) {
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  IREE_ASSERT(value == 1 || value == 2 || value == 4 || value == 8);
  return (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
}

static iree_status_t loom_x86_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_memory_access_plan_t* plan) {
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source.root_value_id, &base));

  loom_named_attr_t attrs[2] = {
      loom_x86_make_disp32_attr(plan->disp32_attr_name_id,
                                plan->source.static_byte_offset),
  };
  loom_value_id_t operands[2] = {base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 1;
  iree_host_size_t attr_count = 1;
  if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_INDEXED) {
    const loom_low_source_memory_dynamic_term_t* term =
        loom_low_source_memory_access_single_dynamic_term(&plan->source);
    IREE_ASSERT(term);
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, term->index, &operands[1]));
    attrs[1] =
        loom_x86_make_scale_attr(plan->scale_attr_name_id, plan->index_scale);
    operand_count = 2;
    attr_count = 2;
  }
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &plan->load_result_type, 1,
      NULL, 0, source_op->location, &load_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_record_source_memory_access(
      context, load_op, &plan->source));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(load_op), 0));
}

static iree_status_t loom_x86_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_memory_access_plan_t* plan) {
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source.root_value_id, &base));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &value));

  loom_named_attr_t attrs[2] = {
      loom_x86_make_disp32_attr(plan->disp32_attr_name_id,
                                plan->source.static_byte_offset),
  };
  loom_value_id_t operands[3] = {value, base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  iree_host_size_t attr_count = 1;
  if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_INDEXED) {
    const loom_low_source_memory_dynamic_term_t* term =
        loom_low_source_memory_access_single_dynamic_term(&plan->source);
    IREE_ASSERT(term);
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, term->index, &operands[2]));
    attrs[1] =
        loom_x86_make_scale_attr(plan->scale_attr_name_id, plan->index_scale);
    operand_count = 3;
    attr_count = 2;
  }
  loom_op_t* store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), NULL, 0, NULL, 0,
      source_op->location, &store_op));
  return loom_low_lower_record_source_memory_access(context, store_op,
                                                    &plan->source);
}

iree_status_t loom_x86_emit_avx512_op(void* user_data,
                                      loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_low_lower_bind_value_alias(
          context, loom_buffer_view_buffer(source_op),
          loom_buffer_view_result(source_op));
    case LOOM_OP_VECTOR_LOAD:
      return loom_x86_lower_vector_load(
          context, source_op,
          (const loom_x86_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_x86_lower_vector_store(
          context, source_op,
          (const loom_x86_memory_access_plan_t*)plan.target_data);
    default:
      IREE_CHECK_UNREACHABLE();
  }
}
