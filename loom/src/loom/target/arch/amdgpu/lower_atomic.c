// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/context.h"
#include "loom/ops/atomic.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/target/arch/amdgpu/lower_memory_internal.h"

typedef uint32_t loom_amdgpu_atomic_rejection_flags_t;

#define LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP ((uint32_t)1u << 0)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND ((uint32_t)1u << 1)
#define LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE ((uint32_t)1u << 2)
#define LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT ((uint32_t)1u << 3)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND ((uint32_t)1u << 5)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE ((uint32_t)1u << 6)
#define LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT ((uint32_t)1u << 7)
#define LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING ((uint32_t)1u << 8)
#define LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE ((uint32_t)1u << 9)
#define LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY ((uint32_t)1u << 10)
#define LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING ((uint32_t)1u << 11)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE ((uint32_t)1u << 12)
#define LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE ((uint32_t)1u << 13)

typedef struct loom_amdgpu_atomic_diagnostic_t {
  // Rejection bits explaining why a source atomic is not legal.
  loom_amdgpu_atomic_rejection_flags_t rejection_bits;
} loom_amdgpu_atomic_diagnostic_t;

typedef struct loom_amdgpu_atomic_rejection_detail_t {
  // Rejection bit matched by this diagnostic row.
  loom_amdgpu_atomic_rejection_flags_t rejection_bit;
  // User-facing diagnostic detail for the matched rejection bit.
  iree_string_view_t detail;
} loom_amdgpu_atomic_rejection_detail_t;

typedef enum loom_amdgpu_atomic_value_kind_e {
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32 = 0,
  LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32 = 1,
} loom_amdgpu_atomic_value_kind_t;

typedef struct loom_amdgpu_atomic_descriptor_candidate_t {
  // Source atomic operation form matched by this row.
  loom_amdgpu_atomic_operation_kind_t operation_kind;
  // Source atomic arithmetic kind matched by this row.
  uint8_t atomic_kind;
  // Source scalar value type required by this row.
  loom_amdgpu_atomic_value_kind_t value_kind;
  // Stable descriptor ID selected when present in the descriptor set.
  uint64_t descriptor_id;
} loom_amdgpu_atomic_descriptor_candidate_t;

static const loom_amdgpu_atomic_descriptor_candidate_t
    kAmdgpuAtomicDescriptorCandidates[] = {
        {
            .operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE,
            .atomic_kind = LOOM_ATOMIC_KIND_ADDI,
            .value_kind = LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_U32,
        },
        {
            .operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE,
            .atomic_kind = LOOM_ATOMIC_KIND_ADDF,
            .value_kind = LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_F32,
        },
        {
            .operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_RMW,
            .atomic_kind = LOOM_ATOMIC_KIND_ADDI,
            .value_kind = LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_RTN_U32,
        },
        {
            .operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_RMW,
            .atomic_kind = LOOM_ATOMIC_KIND_ADDF,
            .value_kind = LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_ADD_RTN_F32,
        },
};

static const loom_amdgpu_atomic_rejection_detail_t
    kAmdgpuAtomicRejectionDetails[] = {
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP,
            .detail = IREE_SVL("source op is not a supported atomic access"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND,
            .detail =
                IREE_SVL("source atomic operation kind is not representable"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic lowering requires workgroup memory"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic lowering requires an LDS buffer root"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic lowering requires one 32-bit element"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic lowering currently supports addi/addf"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic kind does not match the source value type"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT,
            .detail =
                IREE_SVL("AMDGPU LDS atomic value must be available as a VGPR"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING,
            .detail = IREE_SVL("AMDGPU LDS atomic lowering currently supports "
                               "relaxed ordering"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE,
            .detail = IREE_SVL("AMDGPU LDS atomic lowering currently supports "
                               "workgroup scope"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic lowering does not support cache policies"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING,
            .detail = IREE_SVL("selected descriptor set does not provide an "
                               "LDS atomic packet"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE,
            .detail = IREE_SVL(
                "AMDGPU LDS atomic descriptor has no usable offset immediate"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE,
            .detail =
                IREE_SVL("AMDGPU LDS atomic static offset cannot be encoded"),
        },
};

static bool loom_amdgpu_view_atomic_isa(const loom_op_t* op) {
  return loom_view_atomic_reduce_isa(op) || loom_view_atomic_rmw_isa(op);
}

static loom_value_id_t loom_amdgpu_atomic_value(const loom_op_t* op) {
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_value(op);
  }
  return loom_view_atomic_rmw_value(op);
}

static uint8_t loom_amdgpu_atomic_kind(const loom_op_t* op) {
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_kind(op);
  }
  return loom_view_atomic_rmw_kind(op);
}

static uint8_t loom_amdgpu_atomic_ordering(const loom_op_t* op) {
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_ordering(op);
  }
  return loom_view_atomic_rmw_ordering(op);
}

static uint8_t loom_amdgpu_atomic_scope(const loom_op_t* op) {
  if (loom_view_atomic_reduce_isa(op)) {
    return loom_view_atomic_reduce_scope(op);
  }
  return loom_view_atomic_rmw_scope(op);
}

static bool loom_amdgpu_atomic_value_kind_matches(
    loom_type_t value_type, loom_amdgpu_atomic_value_kind_t value_kind) {
  switch (value_kind) {
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32:
      return loom_amdgpu_type_is_i32(value_type);
    case LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32:
      return loom_amdgpu_type_is_f32(value_type);
  }
  return false;
}

static bool loom_amdgpu_atomic_value_can_feed_vgpr_operand(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_amdgpu_atomic_descriptor_candidate_t* candidate) {
  if (candidate->value_kind == LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32) {
    return true;
  }
  const loom_value_id_t value_id = loom_amdgpu_atomic_value(source_op);
  int64_t unused_value = 0;
  return loom_amdgpu_module_value_prefers_vgpr(module, value_id) ||
         loom_amdgpu_module_value_as_i32_constant(module, value_id,
                                                  &unused_value);
}

static bool loom_amdgpu_atomic_source_plan_proves_workgroup_root(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source) {
  if (source->root_value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* root_value =
      loom_module_value(module, source->root_value_id);
  if (loom_value_is_block_arg(root_value)) {
    return false;
  }
  const loom_op_t* root_op = loom_value_def_op(root_value);
  return root_op != NULL && loom_buffer_alloca_isa(root_op);
}

static bool loom_amdgpu_atomic_select_descriptor(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_atomic_operation_kind_t operation_kind, loom_type_t value_type,
    uint64_t* out_descriptor_id, loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  const uint8_t atomic_kind = loom_amdgpu_atomic_kind(source_op);
  bool found_kind = false;
  bool found_type = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuAtomicDescriptorCandidates); ++i) {
    const loom_amdgpu_atomic_descriptor_candidate_t* candidate =
        &kAmdgpuAtomicDescriptorCandidates[i];
    if (candidate->operation_kind != operation_kind ||
        candidate->atomic_kind != atomic_kind) {
      continue;
    }
    found_kind = true;
    if (!loom_amdgpu_atomic_value_kind_matches(value_type,
                                               candidate->value_kind)) {
      continue;
    }
    found_type = true;
    if (!loom_amdgpu_atomic_value_can_feed_vgpr_operand(module, source_op,
                                                        candidate)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_PLACEMENT;
      return false;
    }
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_lookup_descriptor_by_id(
            descriptor_set, candidate->descriptor_id);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
    *out_descriptor_id = candidate->descriptor_id;
    return true;
  }
  diagnostic->rejection_bits |= found_kind && !found_type
                                    ? LOOM_AMDGPU_ATOMIC_REJECTION_VALUE_TYPE
                                    : LOOM_AMDGPU_ATOMIC_REJECTION_ATOMIC_KIND;
  return false;
}

static bool loom_amdgpu_atomic_select_offset(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t descriptor_id,
    loom_amdgpu_atomic_plan_t* plan,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      descriptor_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_ATOMIC_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info) ||
      offset_info.unit_byte_count == 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_IMMEDIATE;
    return false;
  }
  if (plan->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t static_byte_offset = (uint64_t)plan->source.static_byte_offset;
  if ((static_byte_offset % offset_info.unit_byte_count) != 0) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  const uint64_t encoded_offset =
      static_byte_offset / offset_info.unit_byte_count;
  if (encoded_offset > offset_info.unsigned_max) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OFFSET_RANGE;
    return false;
  }
  plan->immediate_offset = (int64_t)encoded_offset;
  return true;
}

static bool loom_amdgpu_atomic_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_atomic_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* memory_diagnostic,
    loom_amdgpu_atomic_diagnostic_t* diagnostic) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(source_diagnostic);
  IREE_ASSERT_ARGUMENT(memory_diagnostic);
  IREE_ASSERT_ARGUMENT(diagnostic);
  *out_plan = (loom_amdgpu_atomic_plan_t){
      .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
  };
  *source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *memory_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};
  *diagnostic = (loom_amdgpu_atomic_diagnostic_t){0};
  if (!loom_amdgpu_view_atomic_isa(source_op)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SOURCE_OP;
    return false;
  }

  if (!loom_low_source_memory_access_plan_build(module, fact_table, source_op,
                                                &out_plan->source,
                                                source_diagnostic)) {
    return false;
  }
  switch (out_plan->source.operation_kind) {
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE:
      out_plan->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE;
      break;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW:
      out_plan->operation_kind = LOOM_AMDGPU_ATOMIC_OPERATION_RMW;
      break;
    default:
      diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_OPERATION_KIND;
      return false;
  }
  if (out_plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_MEMORY_SPACE;
    return false;
  }
  if (!loom_amdgpu_atomic_source_plan_proves_workgroup_root(
          module, &out_plan->source)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_WORKGROUP_ROOT;
    return false;
  }
  if (out_plan->source.element_byte_count != 4 ||
      out_plan->source.vector_lane_count != 1 ||
      out_plan->source.vector_lane_byte_stride != 4) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SHAPE;
    return false;
  }
  if (loom_amdgpu_memory_cache_policy_is_present(
          &out_plan->source.cache_policy)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_CACHE_POLICY;
    return false;
  }
  if (loom_amdgpu_atomic_ordering(source_op) != LOOM_VIEW_ORDERING_RELAXED) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_ORDERING;
    return false;
  }
  if (loom_amdgpu_atomic_scope(source_op) != LOOM_VIEW_SCOPE_WORKGROUP) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_ATOMIC_REJECTION_SCOPE;
    return false;
  }

  loom_amdgpu_memory_access_plan_t memory_access = {
      .source = out_plan->source,
  };
  if (!loom_amdgpu_memory_access_select_dynamic_index_kind(
          module, &memory_access, memory_diagnostic)) {
    return false;
  }
  out_plan->dynamic_index_kind = memory_access.dynamic_index_kind;

  const loom_type_t value_type =
      loom_module_value_type(module, loom_amdgpu_atomic_value(source_op));
  if (!loom_amdgpu_atomic_select_descriptor(
          module, descriptor_set, source_op, out_plan->operation_kind,
          value_type, &out_plan->descriptor_id, diagnostic)) {
    return false;
  }
  return loom_amdgpu_atomic_select_offset(
      descriptor_set, out_plan->descriptor_id, out_plan, diagnostic);
}

static iree_string_view_t loom_amdgpu_atomic_rejection_detail(
    loom_amdgpu_atomic_rejection_flags_t rejection_bits) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuAtomicRejectionDetails); ++i) {
    const loom_amdgpu_atomic_rejection_detail_t* row =
        &kAmdgpuAtomicRejectionDetails[i];
    if (iree_any_bit_set(rejection_bits, row->rejection_bit)) {
      return row->detail;
    }
  }
  return IREE_SV("source atomic is not representable");
}

bool loom_amdgpu_select_view_atomic_plan(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_amdgpu_atomic_plan_t* out_plan) {
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  return loom_amdgpu_atomic_select(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan,
      &source_diagnostic, &memory_diagnostic, &diagnostic);
}

static iree_status_t loom_amdgpu_lookup_atomic_value_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t source_value = loom_amdgpu_atomic_value(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source_value);
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, source_value, out_low_value);
  }

  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));
  loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU LDS atomic selected a non-VGPR dynamic update value");
}

iree_status_t loom_amdgpu_lower_view_atomic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_lookup_atomic_value_as_vgpr(context, source_op, &low_value));

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->source.view_value_id, &low_resource));

  loom_amdgpu_memory_access_plan_t access = {
      .source = plan->source,
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
      .dynamic_index_kind = plan->dynamic_index_kind,
      .immediate_offset = plan->immediate_offset,
      .vgpr_count = 1,
      .packet_byte_count = 4,
      .descriptor_id = plan->descriptor_id,
  };
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
      context, source_op, &access, low_resource, &low_vaddr));

  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, &access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  if (plan->operation_kind == LOOM_AMDGPU_ATOMIC_OPERATION_RMW) {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, loom_view_atomic_rmw_result(source_op),
        &result_type));
    loom_value_id_t operands[] = {low_vaddr, low_value};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, plan->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_view_atomic_rmw_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  loom_value_id_t operands[] = {low_vaddr, low_value};
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}

iree_status_t loom_amdgpu_low_legality_verify_view_atomic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_view_atomic_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_atomic_plan_t plan = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_amdgpu_atomic_diagnostic_t diagnostic = {0};
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (loom_amdgpu_atomic_select(
          module, loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op, &plan,
          &source_diagnostic, &memory_diagnostic, &diagnostic)) {
    return iree_ok_status();
  }

  iree_string_view_t detail = IREE_SV("source atomic is not representable");
  if (source_diagnostic.rejection_bits != 0) {
    detail = loom_low_source_memory_access_rejection_detail(
        source_diagnostic.rejection_bits);
  } else if (memory_diagnostic.rejection_bits != 0) {
    detail = loom_amdgpu_memory_access_rejection_detail(
        memory_diagnostic.rejection_bits);
  } else {
    detail = loom_amdgpu_atomic_rejection_detail(diagnostic.rejection_bits);
  }
  return loom_target_low_legality_reject(context, provider, op,
                                         IREE_SV("atomic"),
                                         loom_op_name(module, op), detail);
}
