// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/type_propagation.h"

#include <string.h>

#include "loom/analysis/type_refinement.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/type_refinement.h"

#define LOOM_TYPE_PROPAGATION_INITIAL_VALUE_CAPACITY 64
#define LOOM_TYPE_PROPAGATION_INITIAL_LIST_CAPACITY 32

typedef struct loom_type_value_span_t {
  // Value ids in the span. May point at op trailing storage, block args, or
  // this span's inline single_value field.
  const loom_value_id_t* values;

  // Number of value ids in values.
  uint16_t count;

  // Inline storage for fixed single-value field references.
  loom_value_id_t single_value;
} loom_type_value_span_t;

struct loom_type_propagator_t {
  // Module whose value table is indexed by the dense scratch arrays.
  loom_module_t* module;

  // Scratch arena owning transaction arrays and temporary overflow dimensions.
  iree_arena_allocator_t* arena;

  // Number of value ids addressable by the dense arrays below.
  iree_host_size_t value_capacity;

  // Current transaction generation for candidate and worklist marks.
  uint32_t transaction_generation;

  // Current function-owner generation for block-argument owner marks.
  uint32_t owner_generation;

  // Candidate type for each value touched in the active transaction.
  loom_type_t* candidate_types;

  // Generation mark indicating that candidate_types[value_id] is live.
  uint32_t* candidate_generations;

  // Generation mark indicating that a value is already queued.
  uint32_t* queued_generations;

  // Parent op owning the region that defines each block argument, when known.
  loom_op_t** owner_ops;

  // Generation mark indicating that owner_ops[value_id] is live.
  uint32_t* owner_generations;

  // Values whose candidate type differs from the module type.
  loom_value_id_t* touched_values;

  // Number of live entries in touched_values.
  iree_host_size_t touched_count;

  // Allocated entry count for touched_values.
  iree_host_size_t touched_capacity;

  // Value worklist used to expand the candidate closure.
  loom_value_id_t* value_worklist;

  // Number of live entries in value_worklist.
  iree_host_size_t value_worklist_count;

  // Allocated entry count for value_worklist.
  iree_host_size_t value_worklist_capacity;

  // True when a candidate contradicts another candidate or existing type.
  bool conflict;
};

struct loom_type_transfer_context_t {
  // Propagator owning the active transaction.
  loom_type_propagator_t* propagator;
};

static bool loom_type_propagator_valid_value_id(
    const loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  return value_id != LOOM_VALUE_ID_INVALID &&
         (iree_host_size_t)value_id < propagator->module->values.count;
}

static iree_status_t loom_type_propagator_ensure_value_capacity(
    loom_type_propagator_t* propagator) {
  iree_host_size_t value_count = propagator->module->values.count;
  if (value_count <= propagator->value_capacity) return iree_ok_status();

  iree_host_size_t old_capacity = propagator->value_capacity;
  iree_host_size_t new_capacity = value_count + value_count / 2;
  if (new_capacity < LOOM_TYPE_PROPAGATION_INITIAL_VALUE_CAPACITY) {
    new_capacity = LOOM_TYPE_PROPAGATION_INITIAL_VALUE_CAPACITY;
  }

  loom_type_t* candidate_types = NULL;
  uint32_t* candidate_generations = NULL;
  uint32_t* queued_generations = NULL;
  loom_op_t** owner_ops = NULL;
  uint32_t* owner_generations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*candidate_types),
      (void**)&candidate_types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*candidate_generations),
      (void**)&candidate_generations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*queued_generations),
      (void**)&queued_generations));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*owner_ops), (void**)&owner_ops));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      propagator->arena, new_capacity, sizeof(*owner_generations),
      (void**)&owner_generations));
  memset(candidate_types, 0, new_capacity * sizeof(*candidate_types));
  memset(candidate_generations, 0,
         new_capacity * sizeof(*candidate_generations));
  memset(queued_generations, 0, new_capacity * sizeof(*queued_generations));
  memset(owner_ops, 0, new_capacity * sizeof(*owner_ops));
  memset(owner_generations, 0, new_capacity * sizeof(*owner_generations));

  if (old_capacity > 0) {
    memcpy(candidate_types, propagator->candidate_types,
           old_capacity * sizeof(*candidate_types));
    memcpy(candidate_generations, propagator->candidate_generations,
           old_capacity * sizeof(*candidate_generations));
    memcpy(queued_generations, propagator->queued_generations,
           old_capacity * sizeof(*queued_generations));
    memcpy(owner_ops, propagator->owner_ops, old_capacity * sizeof(*owner_ops));
    memcpy(owner_generations, propagator->owner_generations,
           old_capacity * sizeof(*owner_generations));
  }

  propagator->candidate_types = candidate_types;
  propagator->candidate_generations = candidate_generations;
  propagator->queued_generations = queued_generations;
  propagator->owner_ops = owner_ops;
  propagator->owner_generations = owner_generations;
  propagator->value_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_grow_value_list(
    loom_type_propagator_t* propagator, loom_value_id_t** list,
    iree_host_size_t count, iree_host_size_t* capacity) {
  iree_host_size_t minimum_capacity = count + 1;
  if (minimum_capacity <= *capacity) return iree_ok_status();
  iree_host_size_t new_capacity =
      *capacity ? *capacity * 2 : LOOM_TYPE_PROPAGATION_INITIAL_LIST_CAPACITY;
  if (new_capacity < minimum_capacity) new_capacity = minimum_capacity;
  return iree_arena_grow_array(propagator->arena, count, new_capacity,
                               sizeof(**list), capacity, (void**)list);
}

iree_status_t loom_type_propagator_allocate(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_type_propagator_t** out_propagator) {
  if (!module || !arena || !out_propagator) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "type propagator allocation requires module, arena, and output");
  }
  loom_type_propagator_t* propagator = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*propagator), (void**)&propagator));
  memset(propagator, 0, sizeof(*propagator));
  propagator->module = module;
  propagator->arena = arena;
  propagator->transaction_generation = 1;
  propagator->owner_generation = 1;
  IREE_RETURN_IF_ERROR(loom_type_propagator_ensure_value_capacity(propagator));
  *out_propagator = propagator;
  return iree_ok_status();
}

static void loom_type_propagator_next_transaction(
    loom_type_propagator_t* propagator) {
  ++propagator->transaction_generation;
  if (propagator->transaction_generation == 0) {
    memset(propagator->candidate_generations, 0,
           propagator->value_capacity *
               sizeof(*propagator->candidate_generations));
    memset(
        propagator->queued_generations, 0,
        propagator->value_capacity * sizeof(*propagator->queued_generations));
    propagator->transaction_generation = 1;
  }
  propagator->touched_count = 0;
  propagator->value_worklist_count = 0;
  propagator->conflict = false;
}

static void loom_type_propagator_next_owner_generation(
    loom_type_propagator_t* propagator) {
  ++propagator->owner_generation;
  if (propagator->owner_generation == 0) {
    memset(propagator->owner_generations, 0,
           propagator->value_capacity * sizeof(*propagator->owner_generations));
    propagator->owner_generation = 1;
  }
}

static iree_status_t loom_type_propagator_note_owner(
    loom_type_propagator_t* propagator, loom_value_id_t value_id,
    loom_op_t* owner_op) {
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_ensure_value_capacity(propagator));
  propagator->owner_ops[value_id] = owner_op;
  propagator->owner_generations[value_id] = propagator->owner_generation;
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_record_op_region_owners(
    loom_type_propagator_t* propagator, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region || region->block_count == 0) continue;
    loom_block_t* entry = loom_region_entry_block(region);
    for (uint16_t arg_index = 0; arg_index < entry->arg_count; ++arg_index) {
      IREE_RETURN_IF_ERROR(loom_type_propagator_note_owner(
          propagator, loom_block_arg_id(entry, arg_index), op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_record_region_tree_owners(
    loom_type_propagator_t* propagator, loom_region_t* region) {
  if (!region) return iree_ok_status();
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(
          loom_type_propagator_record_op_region_owners(propagator, op));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t region_index = 0; region_index < op->region_count;
           ++region_index) {
        IREE_RETURN_IF_ERROR(loom_type_propagator_record_region_tree_owners(
            propagator, regions[region_index]));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_type_propagator_prepare_function(
    loom_type_propagator_t* propagator, loom_func_like_t function) {
  if (!propagator || !propagator->module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "initialized type propagator required");
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_ensure_value_capacity(propagator));
  loom_type_propagator_next_owner_generation(propagator);
  return loom_type_propagator_record_region_tree_owners(
      propagator, loom_func_like_body(function));
}

static bool loom_type_propagator_has_candidate(
    const loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  return (iree_host_size_t)value_id < propagator->value_capacity &&
         propagator->candidate_generations[value_id] ==
             propagator->transaction_generation;
}

static loom_type_t loom_type_propagator_value_type(
    const loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    return loom_type_none();
  }
  if (loom_type_propagator_has_candidate(propagator, value_id)) {
    return propagator->candidate_types[value_id];
  }
  return loom_module_value_type(propagator->module, value_id);
}

static iree_status_t loom_type_propagator_enqueue_value(
    loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  if (propagator->queued_generations[value_id] ==
      propagator->transaction_generation) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_grow_value_list(
      propagator, &propagator->value_worklist, propagator->value_worklist_count,
      &propagator->value_worklist_capacity));
  propagator->queued_generations[value_id] = propagator->transaction_generation;
  propagator->value_worklist[propagator->value_worklist_count++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_mark_touched(
    loom_type_propagator_t* propagator, loom_value_id_t value_id) {
  if (!loom_type_propagator_has_candidate(propagator, value_id)) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_grow_value_list(
        propagator, &propagator->touched_values, propagator->touched_count,
        &propagator->touched_capacity));
    propagator->touched_values[propagator->touched_count++] = value_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_refine_shape_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  uint8_t rank = loom_type_rank(candidate_type);
  uint64_t candidate_dimensions[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t i = 0; i < rank; ++i) {
    candidate_dimensions[i] = loom_type_dim(candidate_type, i);
  }
  return loom_type_refine_shape_with_dims(current_type, candidate_dimensions,
                                          rank, arena, out_type, out_result);
}

static iree_status_t loom_type_propagator_refine_encoding_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  return loom_type_refine_encoding_with_attachment(
      current_type, candidate_type.encoding_id, candidate_type.encoding_flags,
      arena, out_type, out_result);
}

static iree_status_t loom_type_propagator_refine_element_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result) {
  *out_type = current_type;
  *out_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  if (loom_type_kind(current_type) == loom_type_kind(candidate_type)) {
    return loom_type_refine_element_with_candidate(current_type, candidate_type,
                                                   arena, out_type, out_result);
  }
  if (loom_type_element_type(current_type) !=
      loom_type_element_type(candidate_type)) {
    *out_result = LOOM_TYPE_REFINEMENT_CONFLICT;
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_refine_property_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    loom_constraint_property_t property, iree_arena_allocator_t* arena,
    loom_type_t* out_type, loom_type_refinement_result_t* out_result) {
  switch ((enum loom_constraint_property_e)property) {
    case LOOM_PROPERTY_TYPE:
      return loom_type_refine_with_candidate(current_type, candidate_type,
                                             arena, out_type, out_result);
    case LOOM_PROPERTY_ELEMENT_TYPE:
      return loom_type_propagator_refine_element_with_candidate(
          current_type, candidate_type, arena, out_type, out_result);
    case LOOM_PROPERTY_ENCODING:
      return loom_type_propagator_refine_encoding_with_candidate(
          current_type, candidate_type, arena, out_type, out_result);
    case LOOM_PROPERTY_SHAPE:
      return loom_type_propagator_refine_shape_with_candidate(
          current_type, candidate_type, arena, out_type, out_result);
    case LOOM_PROPERTY_KIND:
      *out_type = current_type;
      *out_result =
          loom_type_kind(current_type) == loom_type_kind(candidate_type)
              ? LOOM_TYPE_REFINEMENT_UNCHANGED
              : LOOM_TYPE_REFINEMENT_CONFLICT;
      return iree_ok_status();
    case LOOM_PROPERTY_RANK:
      *out_type = current_type;
      *out_result =
          loom_type_rank(current_type) == loom_type_rank(candidate_type)
              ? LOOM_TYPE_REFINEMENT_UNCHANGED
              : LOOM_TYPE_REFINEMENT_CONFLICT;
      return iree_ok_status();
    default:
      *out_type = current_type;
      *out_result = LOOM_TYPE_REFINEMENT_UNCHANGED;
      return iree_ok_status();
  }
}

static iree_status_t loom_type_propagator_seed_candidate(
    loom_type_propagator_t* propagator, loom_value_id_t value_id,
    loom_type_t candidate_type, loom_constraint_property_t property) {
  if (propagator->conflict) return iree_ok_status();
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_ensure_value_capacity(propagator));

  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_propagator_refine_property_with_candidate(
      current_type, candidate_type, property, propagator->arena, &refined_type,
      &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
      loom_type_equal(refined_type, current_type)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_type_propagator_mark_touched(propagator, value_id));
  propagator->candidate_types[value_id] = refined_type;
  propagator->candidate_generations[value_id] =
      propagator->transaction_generation;
  return loom_type_propagator_enqueue_value(propagator, value_id);
}

loom_type_t loom_type_transfer_value_type(
    const loom_type_transfer_context_t* context, loom_value_id_t value_id) {
  if (!context || !context->propagator) return loom_type_none();
  return loom_type_propagator_value_type(context->propagator, value_id);
}

iree_status_t loom_type_transfer_seed_candidate(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    loom_type_t candidate_type, loom_constraint_property_t property) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  return loom_type_propagator_seed_candidate(context->propagator, value_id,
                                             candidate_type, property);
}

iree_status_t loom_type_transfer_seed_shape_dims(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    const uint64_t* candidate_dimensions, uint8_t candidate_rank) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  loom_type_propagator_t* propagator = context->propagator;
  if (propagator->conflict ||
      !loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_shape_with_dims(
      current_type, candidate_dimensions, candidate_rank, propagator->arena,
      &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED) return iree_ok_status();
  return loom_type_propagator_seed_candidate(propagator, value_id, refined_type,
                                             LOOM_PROPERTY_SHAPE);
}

iree_status_t loom_type_transfer_seed_encoding_attachment(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    uint16_t candidate_encoding_id,
    loom_encoding_flags_t candidate_encoding_flags) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  loom_type_propagator_t* propagator = context->propagator;
  if (propagator->conflict ||
      !loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  loom_type_t refined_type = current_type;
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_encoding_with_attachment(
      current_type, candidate_encoding_id, candidate_encoding_flags,
      propagator->arena, &refined_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    propagator->conflict = true;
    return iree_ok_status();
  }
  if (result == LOOM_TYPE_REFINEMENT_UNCHANGED) return iree_ok_status();
  return loom_type_propagator_seed_candidate(propagator, value_id, refined_type,
                                             LOOM_PROPERTY_ENCODING);
}

iree_status_t loom_type_transfer_seed_static_structure_from_type(
    loom_type_transfer_context_t* context, loom_value_id_t value_id,
    loom_type_t candidate_type) {
  if (!context || !context->propagator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "type transfer requires an active context");
  }
  loom_type_propagator_t* propagator = context->propagator;
  if (propagator->conflict ||
      !loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  if (loom_type_kind(candidate_type) == LOOM_TYPE_NONE) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
      propagator, value_id, candidate_type, LOOM_PROPERTY_KIND));
  if (propagator->conflict) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
      propagator, value_id, candidate_type, LOOM_PROPERTY_ELEMENT_TYPE));
  if (propagator->conflict) return iree_ok_status();

  loom_type_t current_type =
      loom_type_propagator_value_type(propagator, value_id);
  if (loom_type_is_shaped(current_type) &&
      loom_type_is_shaped(candidate_type) &&
      loom_type_rank(current_type) == loom_type_rank(candidate_type)) {
    uint8_t rank = loom_type_rank(candidate_type);
    uint64_t candidate_dimensions[LOOM_TYPE_MAX_RANK] = {0};
    bool has_static_candidate_dimension = false;
    for (uint8_t i = 0; i < rank; ++i) {
      uint64_t candidate_dimension = loom_type_dim(candidate_type, i);
      if (loom_dim_is_dynamic(candidate_dimension)) {
        candidate_dimensions[i] = loom_type_dim(current_type, i);
      } else {
        candidate_dimensions[i] = candidate_dimension;
        has_static_candidate_dimension = true;
      }
    }
    if (has_static_candidate_dimension) {
      IREE_RETURN_IF_ERROR(loom_type_transfer_seed_shape_dims(
          context, value_id, candidate_dimensions, rank));
      if (propagator->conflict) return iree_ok_status();
    }
  }

  if (loom_type_has_static_encoding(candidate_type)) {
    IREE_RETURN_IF_ERROR(loom_type_transfer_seed_encoding_attachment(
        context, value_id, candidate_type.encoding_id,
        candidate_type.encoding_flags));
  }
  return iree_ok_status();
}

static bool loom_type_propagator_field_is_variadic(
    const loom_op_vtable_t* vtable, loom_field_ref_t field_ref) {
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  switch (category) {
    case LOOM_FIELD_OPERAND:
      return vtable &&
             iree_any_bit_set(vtable->vtable_flags,
                              LOOM_OP_VTABLE_VARIADIC_OPERANDS) &&
             index == vtable->fixed_operand_count;
    case LOOM_FIELD_RESULT:
      return vtable &&
             iree_any_bit_set(vtable->vtable_flags,
                              LOOM_OP_VTABLE_VARIADIC_RESULTS) &&
             index == vtable->fixed_result_count;
    default:
      return false;
  }
}

static bool loom_type_propagator_resolve_value_field(
    const loom_op_t* op, const loom_op_vtable_t* vtable,
    loom_field_ref_t field_ref, loom_type_value_span_t* out_span) {
  *out_span = (loom_type_value_span_t){0};
  uint8_t category = LOOM_FIELD_REF_CATEGORY(field_ref);
  uint8_t index = LOOM_FIELD_REF_INDEX(field_ref);
  if (loom_type_propagator_field_is_variadic(vtable, field_ref)) {
    switch (category) {
      case LOOM_FIELD_OPERAND:
        if (index > op->operand_count) return false;
        out_span->values = loom_op_const_operands(op) + index;
        out_span->count = (uint16_t)(op->operand_count - index);
        return true;
      case LOOM_FIELD_RESULT:
        if (index > op->result_count) return false;
        out_span->values = loom_op_const_results(op) + index;
        out_span->count = (uint16_t)(op->result_count - index);
        return true;
      default:
        return false;
    }
  }

  switch (category) {
    case LOOM_FIELD_OPERAND:
      if (index >= op->operand_count) return false;
      out_span->single_value = loom_op_const_operands(op)[index];
      out_span->values = &out_span->single_value;
      out_span->count = 1;
      return true;
    case LOOM_FIELD_RESULT:
      if (index >= op->result_count) return false;
      out_span->single_value = loom_op_const_results(op)[index];
      out_span->values = &out_span->single_value;
      out_span->count = 1;
      return true;
    default:
      return false;
  }
}

static bool loom_type_propagator_region_entry_args(
    loom_op_t* op, loom_field_ref_t field_ref,
    loom_type_value_span_t* out_span) {
  *out_span = (loom_type_value_span_t){0};
  if (LOOM_FIELD_REF_CATEGORY(field_ref) != LOOM_FIELD_REGION) return false;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (region_index >= op->region_count) return false;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return false;
  loom_block_t* entry = loom_region_entry_block(region);
  out_span->values = entry->arg_ids;
  out_span->count = entry->arg_count;
  return true;
}

static bool loom_type_propagator_op_is_terminator(
    const loom_type_propagator_t* propagator, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(propagator->module, op);
  return vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR);
}

static bool loom_type_propagator_terminator_matches(
    const loom_region_descriptor_t* descriptor, const loom_op_t* terminator) {
  if (!terminator || !descriptor ||
      descriptor->terminator == LOOM_OP_KIND_UNKNOWN) {
    return true;
  }
  return terminator->kind == descriptor->terminator ||
         terminator->kind == descriptor->implicit_terminator;
}

static bool loom_type_propagator_region_yield_operands(
    const loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, loom_field_ref_t field_ref,
    loom_type_value_span_t* out_span) {
  *out_span = (loom_type_value_span_t){0};
  if (LOOM_FIELD_REF_CATEGORY(field_ref) != LOOM_FIELD_REGION) return false;
  uint8_t region_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (region_index >= op->region_count) return false;
  const loom_region_descriptor_t* descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  if (!descriptor) return false;
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!region || region->block_count == 0) return false;
  const loom_block_t* entry = loom_region_const_entry_block(region);
  if (entry->op_count == 0) {
    return descriptor->implicit_terminator != LOOM_OP_KIND_UNKNOWN;
  }

  const loom_op_t* terminator = loom_block_const_last_op(entry);
  if (!loom_type_propagator_op_is_terminator(propagator, terminator)) {
    return false;
  }
  if (!loom_type_propagator_terminator_matches(descriptor, terminator)) {
    return false;
  }
  out_span->values = loom_op_const_operands(terminator);
  out_span->count = terminator->operand_count;
  return true;
}

static iree_status_t loom_type_propagator_join_value_span(
    loom_type_propagator_t* propagator, loom_type_value_span_t span,
    loom_constraint_property_t property) {
  loom_value_id_t first_value = LOOM_VALUE_ID_INVALID;
  for (uint16_t i = 0; i < span.count; ++i) {
    if (loom_type_propagator_valid_value_id(propagator, span.values[i])) {
      first_value = span.values[i];
      break;
    }
  }
  if (first_value == LOOM_VALUE_ID_INVALID) return iree_ok_status();

  loom_type_t joined_type =
      loom_type_propagator_value_type(propagator, first_value);
  for (uint16_t i = 0; i < span.count; ++i) {
    loom_value_id_t value_id = span.values[i];
    if (!loom_type_propagator_valid_value_id(propagator, value_id)) continue;
    loom_type_t value_type =
        loom_type_propagator_value_type(propagator, value_id);
    loom_type_t refined_type = joined_type;
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_propagator_refine_property_with_candidate(
        joined_type, value_type, property, propagator->arena, &refined_type,
        &result));
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      propagator->conflict = true;
      return iree_ok_status();
    }
    joined_type = refined_type;
  }

  for (uint16_t i = 0; i < span.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
        propagator, span.values[i], joined_type, property));
    if (propagator->conflict) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_join_pair(
    loom_type_propagator_t* propagator, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_constraint_property_t property) {
  loom_value_id_t values[] = {lhs, rhs};
  return loom_type_propagator_join_value_span(
      propagator,
      (loom_type_value_span_t){.values = values,
                               .count = IREE_ARRAYSIZE(values)},
      property);
}

static iree_status_t loom_type_propagator_relation_pairwise_eq(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return iree_ok_status();
  for (uint8_t i = 0; i < constraint->arg_count; ++i) {
    loom_type_value_span_t span = {0};
    if (!loom_type_propagator_resolve_value_field(op, vtable,
                                                  constraint->args[i], &span)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_value_span(
        propagator, span, constraint->property));
    if (propagator->conflict) return iree_ok_status();
  }

  loom_type_value_span_t first_span = {0};
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[0],
                                                &first_span)) {
    return iree_ok_status();
  }
  if (first_span.count == 0) return iree_ok_status();
  loom_value_id_t reference = first_span.values[0];
  for (uint8_t i = 1; i < constraint->arg_count; ++i) {
    loom_type_value_span_t span = {0};
    if (!loom_type_propagator_resolve_value_field(op, vtable,
                                                  constraint->args[i], &span)) {
      return iree_ok_status();
    }
    for (uint16_t j = 0; j < span.count; ++j) {
      IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
          propagator, reference, span.values[j], constraint->property));
      if (propagator->conflict) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_relation_all_same(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 1) return iree_ok_status();
  loom_type_value_span_t span = {0};
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[0],
                                                &span)) {
    return iree_ok_status();
  }
  return loom_type_propagator_join_value_span(propagator, span,
                                              constraint->property);
}

static iree_status_t loom_type_propagator_relation_region_arg_match(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return iree_ok_status();
  loom_type_value_span_t args = {0};
  loom_type_value_span_t inputs = {0};
  if (!loom_type_propagator_region_entry_args(op, constraint->args[0], &args)) {
    return iree_ok_status();
  }
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[1],
                                                &inputs)) {
    return iree_ok_status();
  }
  uint16_t count = args.count < inputs.count ? args.count : inputs.count;
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
        propagator, args.values[i], inputs.values[i], constraint->property));
    if (propagator->conflict) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_relation_yield_match(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return iree_ok_status();
  if (constraint->property == LOOM_PROPERTY_TYPE) {
    return iree_ok_status();
  }
  loom_type_value_span_t yield_operands = {0};
  loom_type_value_span_t results = {0};
  if (!loom_type_propagator_region_yield_operands(
          propagator, op, vtable, constraint->args[0], &yield_operands)) {
    return iree_ok_status();
  }
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[1],
                                                &results)) {
    return iree_ok_status();
  }
  uint16_t count = yield_operands.count < results.count ? yield_operands.count
                                                        : results.count;
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
        propagator, yield_operands.values[i], results.values[i],
        constraint->property));
    if (propagator->conflict) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_relation_variadic_match(
    loom_type_propagator_t* propagator, loom_op_t* op,
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint) {
  if (constraint->arg_count < 2) return iree_ok_status();
  if (constraint->property == LOOM_PROPERTY_TYPE) {
    return iree_ok_status();
  }
  loom_type_value_span_t lhs = {0};
  loom_type_value_span_t rhs = {0};
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[0],
                                                &lhs)) {
    return iree_ok_status();
  }
  if (!loom_type_propagator_resolve_value_field(op, vtable, constraint->args[1],
                                                &rhs)) {
    return iree_ok_status();
  }
  uint16_t count = lhs.count < rhs.count ? lhs.count : rhs.count;
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_join_pair(
        propagator, lhs.values[i], rhs.values[i], constraint->property));
    if (propagator->conflict) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_seed_op_value_facts(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_op_t* op) {
  if (!rewriter->fact_table.entries) return iree_ok_status();
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_value_id_t value_id = operands[i];
    if (!loom_type_propagator_valid_value_id(propagator, value_id)) continue;
    loom_type_t current_type =
        loom_type_propagator_value_type(propagator, value_id);
    loom_type_t refined_type = current_type;
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_refine_with_value_facts(
        current_type, &rewriter->fact_table, propagator->arena, &refined_type,
        &result));
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      propagator->conflict = true;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
        propagator, value_id, refined_type, LOOM_PROPERTY_TYPE));
    if (propagator->conflict) return iree_ok_status();
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t value_id = results[i];
    if (!loom_type_propagator_valid_value_id(propagator, value_id)) continue;
    loom_type_t current_type =
        loom_type_propagator_value_type(propagator, value_id);
    loom_type_t refined_type = current_type;
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_refine_with_value_facts(
        current_type, &rewriter->fact_table, propagator->arena, &refined_type,
        &result));
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      propagator->conflict = true;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_type_propagator_seed_candidate(
        propagator, value_id, refined_type, LOOM_PROPERTY_TYPE));
    if (propagator->conflict) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_process_op_constraints(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_type_propagator_record_op_region_owners(propagator, op));
  IREE_RETURN_IF_ERROR(
      loom_type_propagator_seed_op_value_facts(propagator, rewriter, op));
  if (propagator->conflict) return iree_ok_status();

  const loom_op_vtable_t* vtable = loom_op_vtable(propagator->module, op);
  if (!vtable) {
    return iree_ok_status();
  }

  if (vtable->constraint_count > 0 && vtable->constraints) {
    for (uint8_t i = 0; i < vtable->constraint_count; ++i) {
      const loom_constraint_t* constraint = &vtable->constraints[i];
      switch ((enum loom_constraint_relation_e)constraint->relation) {
        case LOOM_RELATION_PAIRWISE_EQ: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_pairwise_eq(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_ALL_SAME: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_all_same(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_REGION_ARG_MATCH: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_region_arg_match(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_YIELD_MATCH: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_yield_match(
              propagator, op, vtable, constraint));
          break;
        }
        case LOOM_RELATION_VARIADIC_MATCH: {
          IREE_RETURN_IF_ERROR(loom_type_propagator_relation_variadic_match(
              propagator, op, vtable, constraint));
          break;
        }
        default:
          break;
      }
      if (propagator->conflict) return iree_ok_status();
    }
  }

  if (vtable->type_transfer) {
    loom_type_transfer_context_t context = {.propagator = propagator};
    IREE_RETURN_IF_ERROR(
        vtable->type_transfer(&context, propagator->module, op));
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_process_value_adjacency(
    loom_type_propagator_t* propagator, const loom_rewriter_t* rewriter,
    loom_value_id_t value_id) {
  if (!loom_type_propagator_valid_value_id(propagator, value_id)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_type_propagator_ensure_value_capacity(propagator));

  loom_value_t* value = loom_module_value(propagator->module, value_id);
  if (loom_value_is_block_arg(value)) {
    if ((iree_host_size_t)value_id < propagator->value_capacity &&
        propagator->owner_generations[value_id] ==
            propagator->owner_generation) {
      IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
          propagator, rewriter, propagator->owner_ops[value_id]));
      if (propagator->conflict) return iree_ok_status();
    }
  } else {
    loom_op_t* def_op = loom_value_def_op(value);
    IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
        propagator, rewriter, def_op));
    if (propagator->conflict) return iree_ok_status();
  }

  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
        propagator, rewriter, loom_use_user_op(uses[i])));
    if (propagator->conflict) return iree_ok_status();
  }

  if ((iree_host_size_t)value_id >=
      propagator->module->type_uses.value_capacity) {
    return iree_ok_status();
  }
  loom_type_use_id_t type_use_id =
      propagator->module->type_uses.value_heads[value_id].first_incoming_use_id;
  while (type_use_id != LOOM_TYPE_USE_ID_INVALID) {
    const loom_type_use_t* type_use =
        &propagator->module->type_uses.records[type_use_id];
    loom_value_id_t user_value_id = type_use->user_value_id;
    if (loom_type_propagator_valid_value_id(propagator, user_value_id)) {
      loom_value_t* user_value =
          loom_module_value(propagator->module, user_value_id);
      if (loom_value_is_block_arg(user_value)) {
        if ((iree_host_size_t)user_value_id < propagator->value_capacity &&
            propagator->owner_generations[user_value_id] ==
                propagator->owner_generation) {
          IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
              propagator, rewriter, propagator->owner_ops[user_value_id]));
        }
      } else {
        IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
            propagator, rewriter, loom_value_def_op(user_value)));
      }
      if (propagator->conflict) return iree_ok_status();
      const loom_use_t* user_value_uses = loom_value_uses(user_value);
      for (uint32_t i = 0; i < user_value->use_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_type_propagator_process_op_constraints(
            propagator, rewriter, loom_use_user_op(user_value_uses[i])));
        if (propagator->conflict) return iree_ok_status();
      }
    }
    type_use_id = type_use->next_incoming_use_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_type_propagator_commit(
    loom_type_propagator_t* propagator, loom_rewriter_t* rewriter,
    bool* out_changed) {
  *out_changed = false;
  for (iree_host_size_t i = 0; i < propagator->touched_count; ++i) {
    loom_value_id_t value_id = propagator->touched_values[i];
    loom_type_t current_type =
        loom_module_value_type(propagator->module, value_id);
    loom_type_t candidate_type = propagator->candidate_types[value_id];
    if (loom_type_equal(current_type, candidate_type)) continue;

    loom_type_t committed_type = current_type;
    loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
    IREE_RETURN_IF_ERROR(loom_type_refine_with_candidate(
        current_type, candidate_type, &propagator->module->arena,
        &committed_type, &result));
    if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "accepted type propagation transaction conflicted during commit");
    }
    if (result == LOOM_TYPE_REFINEMENT_UNCHANGED ||
        loom_type_equal(current_type, committed_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_value_type(rewriter, value_id, committed_type));
    *out_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_type_propagator_apply_op(loom_type_propagator_t* propagator,
                                            loom_rewriter_t* rewriter,
                                            loom_op_t* op, bool* out_changed) {
  if (!propagator || !rewriter || !op || !out_changed) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "type propagation requires propagator, rewriter, op, and output");
  }
  *out_changed = false;
  IREE_RETURN_IF_ERROR(loom_type_propagator_ensure_value_capacity(propagator));
  loom_type_propagator_next_transaction(propagator);
  IREE_RETURN_IF_ERROR(
      loom_type_propagator_process_op_constraints(propagator, rewriter, op));

  while (!propagator->conflict && propagator->value_worklist_count > 0) {
    loom_value_id_t value_id =
        propagator->value_worklist[--propagator->value_worklist_count];
    propagator->queued_generations[value_id] = 0;
    IREE_RETURN_IF_ERROR(loom_type_propagator_process_value_adjacency(
        propagator, rewriter, value_id));
  }
  if (propagator->conflict) return iree_ok_status();
  return loom_type_propagator_commit(propagator, rewriter, out_changed);
}
