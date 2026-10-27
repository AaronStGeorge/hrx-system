// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/vector_slice.h"

#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/descriptors/vector_slice.h"
#include "loom/target/arch/x86/register_classes.h"

enum {
  LOOM_X86_VECTOR_SLICE_PLAN = 0x7838736Cu,
  LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT = 16,
};

typedef enum loom_x86_vector_slice_mechanism_e {
  // The result aliases the source or projects its low register class.
  LOOM_X86_VECTOR_SLICE_PROJECT = 0,
  // One XMM quarter is shifted to expose a window in its low bytes.
  LOOM_X86_VECTOR_SLICE_SHIFT_XMM = 1,
  // Adjacent XMM quarters are aligned to expose a crossing window.
  LOOM_X86_VECTOR_SLICE_ALIGN_XMM = 2,
  // One aligned XMM quarter is extracted from a wide source.
  LOOM_X86_VECTOR_SLICE_EXTRACT_XMM = 3,
  // A ZMM source is aligned at a 32-bit lane and projected to YMM.
  LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM = 4,
  // A ZMM source is aligned at a 64-bit lane and projected to YMM.
  LOOM_X86_VECTOR_SLICE_ALIGN_QWORD_ZMM = 5,
  // The high YMM half is extracted from a ZMM source.
  LOOM_X86_VECTOR_SLICE_EXTRACT_YMM = 6,
  // Two byte-aligned XMM windows are assembled into one YMM result.
  LOOM_X86_VECTOR_SLICE_ASSEMBLE_YMM = 7,
} loom_x86_vector_slice_mechanism_t;

typedef struct loom_x86_vector_slice_shape_t {
  // Declared bit width of one logical vector element.
  uint8_t element_bit_width;
  // Physical register class carrying the source value.
  loom_x86_register_class_t source_register_class;
  // Physical register class carrying the result value.
  loom_x86_register_class_t result_register_class;
  // Logical byte offset of the result window in the source.
  uint8_t source_byte_offset;
  // Logical result payload length in bytes.
  uint8_t result_byte_count;
} loom_x86_vector_slice_shape_t;

typedef struct loom_x86_vector_slice_plan_t {
  // Selected physical register-window mechanism.
  loom_x86_vector_slice_mechanism_t mechanism;
  // Physical register class carrying the source value.
  loom_x86_register_class_t source_register_class;
  // Physical register class carrying the result value.
  loom_x86_register_class_t result_register_class;
  // First 128-bit source quarter consumed by an XMM window.
  uint8_t source_quarter;
  // Byte shift within the first source quarter.
  uint8_t quarter_byte_offset;
  // Immediate consumed by a full-width ZMM alignment.
  uint8_t alignment_lane_offset;
} loom_x86_vector_slice_plan_t;

typedef struct loom_x86_vector_slice_emitter_t {
  // Active lowering context.
  loom_low_lower_context_t* context;
  // Source slice supplying the result binding and source location.
  const loom_op_t* source_op;
  // Descriptor vocabulary for the selected x86 view.
  const loom_x86_vector_slice_descriptors_t* descriptors;
  // XMM intermediate type.
  loom_type_t xmm_type;
  // YMM intermediate type.
  loom_type_t ymm_type;
  // ZMM intermediate type when available.
  loom_type_t zmm_type;
  // Interned immediate field name "bytes".
  loom_string_id_t bytes_name;
  // Interned immediate field name "lane".
  loom_string_id_t lane_name;
  // Interned immediate field name "lanes".
  loom_string_id_t lanes_name;
} loom_x86_vector_slice_emitter_t;

static bool loom_x86_vector_slice_match_shape(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_x86_vector_slice_descriptors_t* descriptors,
    loom_x86_vector_slice_shape_t* out_shape) {
  *out_shape = (loom_x86_vector_slice_shape_t){0};
  if (descriptors == NULL || !loom_vector_slice_isa(source_op) ||
      loom_vector_slice_offsets(source_op).count != 0) {
    return false;
  }
  const loom_attribute_t offsets = loom_vector_slice_static_offsets(source_op);
  if (offsets.kind != LOOM_ATTR_I64_ARRAY || offsets.count != 1 ||
      offsets.i64_array[0] < 0) {
    return false;
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_slice_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_slice_result(source_op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      loom_type_rank(source_type) != 1 || loom_type_rank(result_type) != 1 ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type) ||
      !loom_type_element_type_equals(source_type, result_type)) {
    return false;
  }

  uint32_t element_bit_width = 0;
  if (!loom_x86_register_vector_element_bit_width(
          loom_type_element_type(source_type), &element_bit_width)) {
    return false;
  }
  const int64_t source_lane_count =
      loom_type_dim_static_size_at(source_type, 0);
  const int64_t result_lane_count =
      loom_type_dim_static_size_at(result_type, 0);
  if (source_lane_count <= 0 || result_lane_count <= 0) {
    return false;
  }
  const uint64_t source_bit_width =
      (uint64_t)source_lane_count * element_bit_width;
  const uint64_t result_bit_width =
      (uint64_t)result_lane_count * element_bit_width;
  if (source_bit_width > descriptors->maximum_vector_bit_width ||
      result_bit_width > descriptors->maximum_vector_bit_width) {
    return false;
  }

  loom_x86_register_class_t source_register_class;
  loom_x86_register_class_t result_register_class;
  if (!loom_x86_register_class_for_vector_bit_width((uint32_t)source_bit_width,
                                                    &source_register_class) ||
      !loom_x86_register_class_for_vector_bit_width((uint32_t)result_bit_width,
                                                    &result_register_class)) {
    return false;
  }

  const uint64_t lane_offset = (uint64_t)offsets.i64_array[0];
  const uint64_t source_byte_offset = lane_offset * element_bit_width / 8u;
  const uint64_t source_byte_count = source_bit_width / 8u;
  const uint64_t result_byte_count = result_bit_width / 8u;
  if (source_byte_offset + result_byte_count > source_byte_count ||
      source_byte_offset > UINT8_MAX || result_byte_count > UINT8_MAX) {
    return false;
  }

  *out_shape = (loom_x86_vector_slice_shape_t){
      .element_bit_width = (uint8_t)element_bit_width,
      .source_register_class = source_register_class,
      .result_register_class = result_register_class,
      .source_byte_offset = (uint8_t)source_byte_offset,
      .result_byte_count = (uint8_t)result_byte_count,
  };
  return true;
}

static loom_x86_vector_slice_plan_t loom_x86_vector_slice_select_mechanism(
    const loom_x86_vector_slice_shape_t* shape) {
  loom_x86_vector_slice_plan_t plan = {
      .mechanism = LOOM_X86_VECTOR_SLICE_PROJECT,
      .source_register_class = shape->source_register_class,
      .result_register_class = shape->result_register_class,
      .source_quarter =
          shape->source_byte_offset / LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT,
      .quarter_byte_offset =
          shape->source_byte_offset % LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT,
  };
  if (shape->source_byte_offset == 0) {
    return plan;
  }
  if (shape->source_register_class == LOOM_X86_REGISTER_CLASS_XMM) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_SHIFT_XMM;
    return plan;
  }
  if (shape->result_register_class == LOOM_X86_REGISTER_CLASS_XMM) {
    if (plan.quarter_byte_offset == 0) {
      plan.mechanism = LOOM_X86_VECTOR_SLICE_EXTRACT_XMM;
    } else if (plan.quarter_byte_offset + shape->result_byte_count <=
               LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT) {
      plan.mechanism = LOOM_X86_VECTOR_SLICE_SHIFT_XMM;
    } else {
      plan.mechanism = LOOM_X86_VECTOR_SLICE_ALIGN_XMM;
    }
    return plan;
  }

  IREE_ASSERT_EQ(shape->source_register_class, LOOM_X86_REGISTER_CLASS_ZMM);
  IREE_ASSERT_EQ(shape->result_register_class, LOOM_X86_REGISTER_CLASS_YMM);
  if (shape->source_byte_offset == 32) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_EXTRACT_YMM;
  } else if (shape->element_bit_width == 64) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_ALIGN_QWORD_ZMM;
    plan.alignment_lane_offset = shape->source_byte_offset / 8u;
  } else if (shape->element_bit_width >= 32 ||
             shape->source_byte_offset % 4u == 0) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM;
    plan.alignment_lane_offset = shape->source_byte_offset / 4u;
  } else {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_ASSEMBLE_YMM;
  }
  return plan;
}

iree_status_t loom_x86_query_vector_slice_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  if (environment->vector_lane_projection.source_lane_count != 0) {
    return iree_ok_status();
  }
  loom_x86_vector_slice_shape_t shape;
  if (loom_x86_vector_slice_match_shape(
          environment->module, source_op,
          loom_x86_vector_slice_descriptors(environment->descriptor_set),
          &shape)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

bool loom_x86_vector_slice_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_X86_VECTOR_SLICE_PLAN;
}

iree_status_t loom_x86_select_vector_slice_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  loom_x86_vector_slice_shape_t shape;
  if (!loom_x86_vector_slice_match_shape(
          loom_low_lower_context_module(context), source_op,
          loom_x86_vector_slice_descriptors(
              loom_low_lower_context_descriptor_set(context)),
          &shape)) {
    return iree_ok_status();
  }

  loom_x86_vector_slice_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = loom_x86_vector_slice_select_mechanism(&shape);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, shape.result_register_class, 1, &result_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_vector_slice_result(source_op), result_type));
  *out_plan =
      loom_low_lower_plan_make(LOOM_X86_VECTOR_SLICE_PLAN, retained_plan);
  return iree_ok_status();
}

void loom_x86_mark_vector_slice_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  (void)plan;
  loom_low_lower_require_source_value_storage(
      context, loom_vector_slice_source(source_op));
}

void loom_x86_describe_vector_slice_plan(
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  const loom_x86_vector_slice_plan_t* slice = plan.target_data;
  const bool zero_instruction =
      slice->mechanism == LOOM_X86_VECTOR_SLICE_PROJECT;
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = zero_instruction ? IREE_SV("vector-slice.register-alias")
                                   : IREE_SV("vector-slice.register-window"),
  };
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emitter_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_x86_vector_slice_emitter_t* out_emitter) {
  *out_emitter = (loom_x86_vector_slice_emitter_t){
      .context = context,
      .source_op = source_op,
      .descriptors = loom_x86_vector_slice_descriptors(
          loom_low_lower_context_descriptor_set(context)),
      .zmm_type = loom_type_none(),
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_XMM, 1, &out_emitter->xmm_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_YMM, 1, &out_emitter->ymm_type));
  if (out_emitter->descriptors->maximum_vector_bit_width == 512) {
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_ZMM, 1, &out_emitter->zmm_type));
  }
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("bytes"),
                                                  &out_emitter->bytes_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("lane"),
                                                  &out_emitter->lane_name));
  return loom_builder_intern_string(builder, IREE_SV("lanes"),
                                    &out_emitter->lanes_name);
}

static const loom_low_descriptor_t* loom_x86_vector_slice_descriptor(
    const loom_x86_vector_slice_emitter_t* emitter, uint32_t ordinal) {
  IREE_ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  return &loom_low_lower_context_descriptor_set(emitter->context)
              ->descriptors[ordinal];
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emit_descriptor(
    loom_x86_vector_slice_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_string_id_t immediate_name, uint8_t immediate, loom_type_t result_type,
    loom_value_id_t* out_result) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          loom_x86_vector_slice_descriptor(emitter, descriptor_ordinal),
  };
  const loom_named_attr_t attr = {
      .name_id = immediate_name,
      .value = loom_attr_i64(immediate),
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      emitter->context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(&attr, 1), &result_type, 1, NULL, 0,
      emitter->source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_x86_vector_slice_project(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_x86_register_class_t source_register_class,
    loom_x86_register_class_t result_register_class,
    loom_value_id_t* out_result) {
  if (source_register_class == result_register_class) {
    *out_result = source;
    return iree_ok_status();
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, result_register_class, 1, &result_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_slice_build(loom_low_lower_context_builder(context), source, 0,
                           result_type, source_op->location, &low_op));
  *out_result = loom_low_slice_result(low_op);
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_x86_vector_slice_extract_xmm(
    loom_x86_vector_slice_emitter_t* emitter, loom_value_id_t source,
    loom_x86_register_class_t source_register_class, uint8_t source_quarter,
    loom_value_id_t* out_result) {
  if (source_quarter == 0) {
    return loom_x86_vector_slice_project(
        emitter->context, emitter->source_op, source, source_register_class,
        LOOM_X86_REGISTER_CLASS_XMM, out_result);
  }
  const uint32_t descriptor_ordinal =
      source_register_class == LOOM_X86_REGISTER_CLASS_YMM
          ? emitter->descriptors->extract_xmm_from_ymm
          : emitter->descriptors->extract_xmm_from_zmm;
  return loom_x86_vector_slice_emit_descriptor(
      emitter, descriptor_ordinal, &source, 1, emitter->lane_name,
      source_quarter, emitter->xmm_type, out_result);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emit_xmm_window(loom_x86_vector_slice_emitter_t* emitter,
                                      loom_value_id_t source,
                                      const loom_x86_vector_slice_plan_t* slice,
                                      bool crosses_quarter,
                                      loom_value_id_t* out_result) {
  loom_value_id_t lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
      emitter, source, slice->source_register_class, slice->source_quarter,
      &lower));
  if (!crosses_quarter) {
    return loom_x86_vector_slice_emit_descriptor(
        emitter, emitter->descriptors->shift_xmm, &lower, 1,
        emitter->bytes_name, slice->quarter_byte_offset, emitter->xmm_type,
        out_result);
  }

  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
      emitter, source, slice->source_register_class, slice->source_quarter + 1,
      &upper));
  const loom_value_id_t operands[] = {upper, lower};
  return loom_x86_vector_slice_emit_descriptor(
      emitter, emitter->descriptors->align_xmm, operands,
      IREE_ARRAYSIZE(operands), emitter->bytes_name, slice->quarter_byte_offset,
      emitter->xmm_type, out_result);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emit_assembled_ymm(
    loom_x86_vector_slice_emitter_t* emitter, loom_value_id_t source,
    const loom_x86_vector_slice_plan_t* slice, loom_value_id_t* out_result) {
  loom_value_id_t quarters[3] = {LOOM_VALUE_ID_INVALID};
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(quarters); ++i) {
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
        emitter, source, LOOM_X86_REGISTER_CLASS_ZMM, slice->source_quarter + i,
        &quarters[i]));
  }

  loom_value_id_t windows[2] = {LOOM_VALUE_ID_INVALID};
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(windows); ++i) {
    const loom_value_id_t operands[] = {quarters[i + 1], quarters[i]};
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
        emitter, emitter->descriptors->align_xmm, operands,
        IREE_ARRAYSIZE(operands), emitter->bytes_name,
        slice->quarter_byte_offset, emitter->xmm_type, &windows[i]));
  }

  loom_value_id_t assembled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
      emitter->context, emitter->source_op, source, LOOM_X86_REGISTER_CLASS_ZMM,
      LOOM_X86_REGISTER_CLASS_YMM, &assembled));
  for (uint8_t lane = 0; lane < IREE_ARRAYSIZE(windows); ++lane) {
    const loom_value_id_t operands[] = {assembled, windows[lane]};
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
        emitter, emitter->descriptors->insert_xmm_into_ymm, operands,
        IREE_ARRAYSIZE(operands), emitter->lane_name, lane, emitter->ymm_type,
        &assembled));
  }
  *out_result = assembled;
  return iree_ok_status();
}

iree_status_t loom_x86_emit_vector_slice_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  const loom_x86_vector_slice_plan_t* slice = plan.target_data;
  const loom_value_id_t source =
      loom_low_lower_lookup_value(context, loom_vector_slice_source(source_op));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (slice->mechanism == LOOM_X86_VECTOR_SLICE_PROJECT) {
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
        context, source_op, source, slice->source_register_class,
        slice->result_register_class, &result));
    return loom_low_lower_bind_value(
        context, loom_vector_slice_result(source_op), result);
  }

  loom_x86_vector_slice_emitter_t emitter;
  IREE_RETURN_IF_ERROR(
      loom_x86_vector_slice_emitter_initialize(context, source_op, &emitter));
  switch (slice->mechanism) {
    case LOOM_X86_VECTOR_SLICE_PROJECT: {
      IREE_ASSERT_UNREACHABLE("project slice handled before emitter setup");
      return iree_ok_status();
    }
    case LOOM_X86_VECTOR_SLICE_SHIFT_XMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_xmm_window(
          &emitter, source, slice, /*crosses_quarter=*/false, &result));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_ALIGN_XMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_xmm_window(
          &emitter, source, slice, /*crosses_quarter=*/true, &result));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_EXTRACT_XMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
          &emitter, source, slice->source_register_class, slice->source_quarter,
          &result));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM:
    case LOOM_X86_VECTOR_SLICE_ALIGN_QWORD_ZMM: {
      const uint32_t descriptor_ordinal =
          slice->mechanism == LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM
              ? emitter.descriptors->align_dword_zmm
              : emitter.descriptors->align_qword_zmm;
      const loom_value_id_t operands[] = {source, source};
      loom_value_id_t aligned = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
          &emitter, descriptor_ordinal, operands, IREE_ARRAYSIZE(operands),
          emitter.lanes_name, slice->alignment_lane_offset, emitter.zmm_type,
          &aligned));
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
          context, source_op, aligned, LOOM_X86_REGISTER_CLASS_ZMM,
          LOOM_X86_REGISTER_CLASS_YMM, &result));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_EXTRACT_YMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
          &emitter, emitter.descriptors->extract_ymm_from_zmm, &source, 1,
          emitter.lane_name, 1, emitter.ymm_type, &result));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_ASSEMBLE_YMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_assembled_ymm(
          &emitter, source, slice, &result));
      break;
    }
  }
  return loom_low_lower_bind_value(context, loom_vector_slice_result(source_op),
                                   result);
}
