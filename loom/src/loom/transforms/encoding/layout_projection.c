// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/encoding/layout_projection.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/transforms/boundary/projection_plan.h"

typedef struct loom_layout_projection_slot_plan_t {
  // Static stride or INT64_MIN for each transported dynamic axis.
  int64_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Layout axis supplied by each physical component.
  uint8_t component_axes[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Layout rank.
  uint8_t rank;
  // Number of scalar stride components.
  uint8_t component_count;
  // Encoding role reconstructed from the transported address layout.
  loom_encoding_role_t encoding_role;
  // Signature view whose rank and physical schema define a callable slot.
  loom_value_id_t carrier_value_id;
  // Logical signature index of carrier_value_id in carrier_role.
  uint16_t carrier_index;
  // Exact static storage schema retained for physical storage.
  uint16_t storage_schema_encoding_id;
  // Callable storage-schema equivalence-class parent.
  struct loom_layout_projection_slot_plan_t* storage_schema_parent;
  // Union-by-rank depth bound for storage_schema_parent.
  uint8_t storage_schema_tree_rank;
  // Signature partition containing carrier_value_id.
  loom_boundary_projection_slot_role_t carrier_role;
  // Whether this is the stable callable schema rather than a local CFG schema.
  bool callable;
  // Whether incompatible exact schemas reached this equivalence class.
  bool storage_schema_conflict;
} loom_layout_projection_slot_plan_t;

typedef enum loom_layout_projection_component_kind_e {
  LOOM_LAYOUT_PROJECTION_COMPONENT_VALUE = 0,
  LOOM_LAYOUT_PROJECTION_COMPONENT_CONSTANT = 1,
  LOOM_LAYOUT_PROJECTION_COMPONENT_SLOT = 2,
  LOOM_LAYOUT_PROJECTION_COMPONENT_PRODUCT = 3,
} loom_layout_projection_component_kind_t;

typedef struct loom_layout_projection_component_t {
  // How the scalar component is obtained at the outgoing boundary.
  loom_layout_projection_component_kind_t kind;
  union {
    // Existing index SSA value for VALUE.
    loom_value_id_t value_id;
    // Exact nonnegative element stride for CONSTANT.
    int64_t constant;
    struct {
      // Projected slot supplying the future scalar value.
      iree_host_size_t slot;
      // Physical component within slot.
      uint8_t component;
    } slot;
    struct {
      // Static factor multiplied into every dynamic factor.
      int64_t coefficient;
      // Arena-owned dynamic factors in multiplication order.
      const loom_value_id_t* factors;
      // Number of entries in factors.
      uint8_t factor_count;
    } product;
  } value;
} loom_layout_projection_component_t;

typedef struct loom_layout_projection_source_plan_t {
  // One recipe per destination component.
  loom_layout_projection_component_t* components;
  // Number of entries in components.
  uint8_t component_count;
} loom_layout_projection_source_plan_t;

static loom_layout_projection_slot_plan_t*
loom_layout_projection_storage_schema_root(
    loom_layout_projection_slot_plan_t* slot_plan) {
  IREE_ASSERT(slot_plan->storage_schema_parent != NULL);
  loom_layout_projection_slot_plan_t* root = slot_plan;
  while (root->storage_schema_parent != root) {
    root = root->storage_schema_parent;
  }
  while (slot_plan->storage_schema_parent != slot_plan) {
    loom_layout_projection_slot_plan_t* parent =
        slot_plan->storage_schema_parent;
    slot_plan->storage_schema_parent = root;
    slot_plan = parent;
  }
  return root;
}

static void loom_layout_projection_seed_storage_schema(
    loom_layout_projection_slot_plan_t* slot_plan,
    uint16_t storage_schema_encoding_id) {
  if (storage_schema_encoding_id == 0) {
    return;
  }
  loom_layout_projection_slot_plan_t* root =
      loom_layout_projection_storage_schema_root(slot_plan);
  if (root->storage_schema_encoding_id == 0) {
    root->storage_schema_encoding_id = storage_schema_encoding_id;
  } else if (root->storage_schema_encoding_id != storage_schema_encoding_id) {
    root->storage_schema_conflict = true;
  }
}

static void loom_layout_projection_union_storage_schemas(
    loom_layout_projection_slot_plan_t* lhs,
    loom_layout_projection_slot_plan_t* rhs) {
  loom_layout_projection_slot_plan_t* left =
      loom_layout_projection_storage_schema_root(lhs);
  loom_layout_projection_slot_plan_t* right =
      loom_layout_projection_storage_schema_root(rhs);
  if (left == right) {
    return;
  }
  if (left->storage_schema_tree_rank < right->storage_schema_tree_rank) {
    loom_layout_projection_slot_plan_t* temporary = left;
    left = right;
    right = temporary;
  }
  right->storage_schema_parent = left;
  if (left->storage_schema_tree_rank == right->storage_schema_tree_rank) {
    ++left->storage_schema_tree_rank;
  }
  left->storage_schema_conflict |= right->storage_schema_conflict;
  loom_layout_projection_seed_storage_schema(left,
                                             right->storage_schema_encoding_id);
}

static iree_status_t loom_layout_projection_initialize_schema(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_layout_projection_slot_plan_t* slot_plan,
    loom_boundary_projection_schema_t* out_schema) {
  loom_type_t* component_types = NULL;
  iree_string_view_t* component_name_suffixes = NULL;
  char* suffix_storage = NULL;
  const iree_host_size_t suffix_capacity = 16;
  if (slot_plan->component_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, slot_plan->component_count, sizeof(*component_types),
        (void**)&component_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, slot_plan->component_count,
        sizeof(*component_name_suffixes), (void**)&component_name_suffixes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        plan->arena, slot_plan->component_count * suffix_capacity,
        (void**)&suffix_storage));
  }
  for (uint8_t component = 0; component < slot_plan->component_count;
       ++component) {
    component_types[component] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    char* suffix = suffix_storage + component * suffix_capacity;
    const int length = iree_snprintf(suffix, suffix_capacity, "stride_%" PRIu8,
                                     slot_plan->component_axes[component]);
    IREE_ASSERT(length > 0 && (iree_host_size_t)length < suffix_capacity);
    component_name_suffixes[component] =
        iree_make_string_view(suffix, (iree_host_size_t)length);
  }

  *out_schema = (loom_boundary_projection_schema_t){
      .rule = rule,
      .component_types = component_types,
      .component_name_suffixes = component_name_suffixes,
      .component_count = slot_plan->component_count,
      .rule_plan = slot_plan,
  };
  return iree_ok_status();
}

static bool loom_layout_projection_find_callable_carrier(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t provider,
    loom_value_id_t* out_carrier, uint8_t* out_rank) {
  *out_carrier = LOOM_VALUE_ID_INVALID;
  *out_rank = 0;
  loom_type_use_iterator_t users;
  loom_module_value_type_users(plan->module, provider, &users);
  for (loom_value_id_t carrier = loom_type_users_next(&users);
       carrier != LOOM_VALUE_ID_INVALID;
       carrier = loom_type_users_next(&users)) {
    uint16_t carrier_index = 0;
    if (!loom_boundary_projection_signature_slot_index(
            plan->module, function, role, carrier, &carrier_index)) {
      continue;
    }
    const loom_type_t type = loom_module_value_type(plan->module, carrier);
    if (!loom_type_is_view(type) || !loom_type_has_ssa_encoding(type) ||
        loom_type_encoding_value_id(type) != provider) {
      return false;
    }
    const uint8_t rank = loom_type_rank(type);
    if (*out_carrier != LOOM_VALUE_ID_INVALID && *out_rank != rank) {
      return false;
    }
    *out_carrier = carrier;
    *out_rank = rank;
  }
  return *out_carrier != LOOM_VALUE_ID_INVALID;
}

static iree_status_t loom_layout_projection_plan_callable_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_boundary_projection_schema_t* out_schema, bool* out_claimed) {
  const loom_type_t type = loom_module_value_type(plan->module, value_id);
  if (!loom_type_is_encoding(type)) {
    return iree_ok_status();
  }
  const loom_encoding_role_t encoding_role = loom_type_encoding_role(type);
  if (encoding_role != LOOM_ENCODING_ROLE_ADDRESS_LAYOUT &&
      encoding_role != LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    return iree_ok_status();
  }

  loom_value_id_t carrier_value_id = LOOM_VALUE_ID_INVALID;
  uint8_t rank = 0;
  if (!loom_layout_projection_find_callable_carrier(
          plan, function, role, value_id, &carrier_value_id, &rank)) {
    return iree_ok_status();
  }

  loom_layout_projection_slot_plan_t* slot_plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*slot_plan), (void**)&slot_plan));
  memset(slot_plan, 0, sizeof(*slot_plan));
  slot_plan->rank = rank;
  slot_plan->component_count = rank;
  slot_plan->encoding_role = encoding_role;
  slot_plan->carrier_value_id = carrier_value_id;
  slot_plan->carrier_role = role;
  slot_plan->callable = true;
  const bool has_carrier_index = loom_boundary_projection_signature_slot_index(
      plan->module, function, role, carrier_value_id,
      &slot_plan->carrier_index);
  IREE_ASSERT(has_carrier_index);
  (void)has_carrier_index;
  if (encoding_role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    slot_plan->storage_schema_parent = slot_plan;
  }
  for (uint8_t axis = 0; axis < rank; ++axis) {
    slot_plan->static_strides[axis] = INT64_MIN;
    slot_plan->component_axes[axis] = axis;
  }
  IREE_RETURN_IF_ERROR(loom_layout_projection_initialize_schema(
      rule, plan, slot_plan, out_schema));
  *out_claimed = true;
  return iree_ok_status();
}

static iree_status_t loom_layout_projection_plan_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  if (role == LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT ||
      role == LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT) {
    return loom_layout_projection_plan_callable_slot(
        rule, plan, function, role, value_id, out_schema, out_claimed);
  }
  if (!function->facts) {
    return iree_ok_status();
  }
  if (role == LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT) {
    loom_region_t* body = loom_func_like_body(function->function);
    if (!block || !body || block == loom_region_entry_block(body)) {
      return iree_ok_status();
    }
  } else if (role != LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE) {
    return iree_ok_status();
  }

  const loom_type_t type = loom_module_value_type(plan->module, value_id);
  if (!loom_type_is_encoding(type) ||
      loom_type_encoding_role(type) != LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    return iree_ok_status();
  }
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_value_address_layout(&function->facts->context,
                                                value_id, &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      layout.rank == 0 || layout.rank > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
      !layout.strides) {
    return iree_ok_status();
  }

  loom_layout_projection_slot_plan_t* slot_plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*slot_plan), (void**)&slot_plan));
  memset(slot_plan, 0, sizeof(*slot_plan));
  slot_plan->rank = layout.rank;
  slot_plan->encoding_role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT;
  slot_plan->carrier_value_id = LOOM_VALUE_ID_INVALID;
  for (uint8_t axis = 0; axis < layout.rank; ++axis) {
    int64_t exact_stride = 0;
    if (loom_value_facts_as_exact_i64(layout.strides[axis], &exact_stride)) {
      if (exact_stride < 0) {
        return iree_ok_status();
      }
      slot_plan->static_strides[axis] = exact_stride;
    } else {
      slot_plan->static_strides[axis] = INT64_MIN;
      slot_plan->component_axes[slot_plan->component_count++] = axis;
    }
  }
  if (slot_plan->component_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_layout_projection_initialize_schema(
      rule, plan, slot_plan, out_schema));
  *out_claimed = true;
  return iree_ok_status();
}

static iree_status_t loom_layout_projection_seed_callable_storage_schema(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_schema_t* schema, bool* out_conflict) {
  *out_conflict = false;
  if (schema->rule != rule) {
    return iree_ok_status();
  }
  loom_layout_projection_slot_plan_t* slot_plan =
      (loom_layout_projection_slot_plan_t*)schema->rule_plan;
  if (!slot_plan->callable ||
      slot_plan->encoding_role != LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    return iree_ok_status();
  }
  loom_value_fact_storage_schema_t storage_schema = {0};
  const loom_type_t carrier_type =
      loom_module_value_type(plan->module, slot_plan->carrier_value_id);
  if (!loom_encoding_query_type_storage_schema(
          function->facts ? &function->facts->context : NULL, plan->module,
          carrier_type, &storage_schema) ||
      storage_schema.static_spec_encoding_id == 0) {
    return iree_ok_status();
  }
  loom_layout_projection_seed_storage_schema(
      slot_plan, storage_schema.static_spec_encoding_id);
  *out_conflict = loom_layout_projection_storage_schema_root(slot_plan)
                      ->storage_schema_conflict;
  return iree_ok_status();
}

static iree_status_t loom_layout_projection_prepare_callable_function(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  // Multi-region function-like operations may participate in local boundary
  // projection without exposing their callable signature to this pass.
  if (!function->selected || !function->signature_changes) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    bool conflict = false;
    IREE_RETURN_IF_ERROR(loom_layout_projection_seed_callable_storage_schema(
        rule, plan, function, &function->argument_schemas[i], &conflict));
    if (conflict) {
      function->selected = false;
      return iree_ok_status();
    }
  }
  for (uint16_t i = 0; i < function->result_count; ++i) {
    bool conflict = false;
    IREE_RETURN_IF_ERROR(loom_layout_projection_seed_callable_storage_schema(
        rule, plan, function, &function->result_schemas[i], &conflict));
    if (conflict) {
      function->selected = false;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_layout_projection_schema_is_unresolved_storage(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_schema_t* schema) {
  if (schema->rule != rule) {
    return false;
  }
  const loom_layout_projection_slot_plan_t* slot_plan =
      (const loom_layout_projection_slot_plan_t*)schema->rule_plan;
  if (!slot_plan->callable ||
      slot_plan->encoding_role != LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    return false;
  }
  loom_layout_projection_slot_plan_t* root =
      loom_layout_projection_storage_schema_root(
          (loom_layout_projection_slot_plan_t*)slot_plan);
  return root->storage_schema_conflict || root->storage_schema_encoding_id == 0;
}

static void loom_layout_projection_finalize_callable_plan(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan) {
  for (iree_host_size_t function_index = 0;
       function_index < plan->function_count; ++function_index) {
    loom_boundary_projection_function_t* function =
        &plan->functions[function_index];
    if (!function->selected || !function->signature_changes) {
      continue;
    }
    for (uint16_t i = 0; i < function->argument_count; ++i) {
      if (loom_layout_projection_schema_is_unresolved_storage(
              rule, &function->argument_schemas[i])) {
        function->selected = false;
        break;
      }
    }
    for (uint16_t i = 0; function->selected && i < function->result_count;
         ++i) {
      if (loom_layout_projection_schema_is_unresolved_storage(
              rule, &function->result_schemas[i])) {
        function->selected = false;
        break;
      }
    }
  }
}

static loom_layout_projection_slot_plan_t*
loom_layout_projection_storage_plan_for_schema(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_schema_t* schema) {
  if (schema->rule != rule) {
    return NULL;
  }
  loom_layout_projection_slot_plan_t* slot_plan =
      (loom_layout_projection_slot_plan_t*)schema->rule_plan;
  return slot_plan->callable &&
                 slot_plan->encoding_role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE
             ? slot_plan
             : NULL;
}

static loom_layout_projection_slot_plan_t*
loom_layout_projection_find_callable_storage_plan(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_value_id_t value_id) {
  uint16_t index = 0;
  if (function->signature_changes) {
    if (loom_boundary_projection_signature_slot_index(
            plan->module, function,
            LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT, value_id,
            &index)) {
      return loom_layout_projection_storage_plan_for_schema(
          rule, &function->argument_schemas[index]);
    }
    if (loom_boundary_projection_signature_slot_index(
            plan->module, function,
            LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT, value_id, &index)) {
      return loom_layout_projection_storage_plan_for_schema(
          rule, &function->result_schemas[index]);
    }
  }
  const iree_host_size_t slot_index =
      loom_boundary_projection_slot_index(function, value_id);
  return slot_index != IREE_HOST_SIZE_MAX
             ? loom_layout_projection_storage_plan_for_schema(
                   rule, &function->candidates[slot_index].schema)
             : NULL;
}

static void loom_layout_projection_link_storage_source(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_layout_projection_slot_plan_t* destination_plan,
    loom_value_id_t source_value_id) {
  loom_layout_projection_slot_plan_t* source_plan =
      loom_layout_projection_find_callable_storage_plan(rule, plan, function,
                                                        source_value_id);
  if (source_plan) {
    loom_layout_projection_union_storage_schemas(destination_plan, source_plan);
  }

  const loom_value_fact_layout_origin_t source_origin =
      loom_encoding_query_value_layout_origin(&function->facts->context,
                                              source_value_id);
  if (!source_origin.has_physical_encoding_value) {
    return;
  }
  source_plan = loom_layout_projection_find_callable_storage_plan(
      rule, plan, function, source_origin.physical_encoding_value_id);
  if (source_plan) {
    loom_layout_projection_union_storage_schemas(destination_plan, source_plan);
  }
}

static bool loom_layout_projection_source_carrier(
    const loom_boundary_projection_plan_t* plan,
    const loom_layout_projection_slot_plan_t* destination_plan,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_value_id_t* out_carrier_value_id) {
  *out_carrier_value_id = LOOM_VALUE_ID_INVALID;
  if (!destination_plan->callable) {
    return false;
  }
  if (destination_plan->carrier_role ==
      LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT) {
    const loom_call_like_t call =
        loom_call_like_cast(plan->module, boundary_op);
    if (!loom_call_like_isa(call)) {
      return false;
    }
    const loom_value_slice_t operands = loom_call_like_operands(call);
    if (destination_plan->carrier_index >= operands.count) {
      return false;
    }
    *out_carrier_value_id = operands.values[destination_plan->carrier_index];
  } else if (destination_plan->carrier_role ==
             LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT) {
    if (destination_plan->carrier_index >= boundary_op->operand_count) {
      return false;
    }
    *out_carrier_value_id =
        loom_op_operands(boundary_op)[destination_plan->carrier_index];
  } else {
    return false;
  }

  const loom_type_t carrier_type =
      loom_module_value_type(plan->module, *out_carrier_value_id);
  if (!loom_type_is_view(carrier_type) ||
      loom_type_rank(carrier_type) != destination_plan->rank ||
      !loom_type_has_ssa_encoding(carrier_type) ||
      loom_type_encoding_value_id(carrier_type) != source_value_id) {
    *out_carrier_value_id = LOOM_VALUE_ID_INVALID;
    return false;
  }
  return true;
}

static iree_status_t loom_layout_projection_plan_direct_component(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_value_id_t source_value_id, loom_value_id_t source_carrier_value_id,
    const loom_layout_projection_slot_plan_t* destination_plan, uint8_t axis,
    loom_layout_projection_component_t* out_component, bool* out_planned) {
  *out_planned = false;
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_value_address_layout(&function->facts->context,
                                                source_value_id, &layout)) {
    return iree_ok_status();
  }
  loom_value_fact_layout_origin_t origin = {0};
  loom_value_facts_t numeric_stride_facts = loom_value_facts_unknown();
  uint64_t dense_dimensions[LOOM_TYPE_MAX_RANK];
  if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    if (layout.rank != destination_plan->rank || !layout.strides) {
      return iree_ok_status();
    }
    origin = loom_encoding_query_value_layout_origin(&function->facts->context,
                                                     source_value_id);
    numeric_stride_facts = layout.strides[axis];
  } else if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    if (source_carrier_value_id == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    const loom_type_t carrier_type =
        loom_module_value_type(plan->module, source_carrier_value_id);
    for (uint8_t i = 0; i < destination_plan->rank; ++i) {
      dense_dimensions[i] = loom_type_dim(carrier_type, i);
    }
    origin = (loom_value_fact_layout_origin_t){
        .values.dense_dimensions = dense_dimensions,
        .count = destination_plan->rank,
        .kind = LOOM_VALUE_FACT_LAYOUT_ORIGIN_DENSE_DIMENSIONS,
    };
  } else {
    return iree_ok_status();
  }
  loom_value_fact_layout_axis_t layout_axis;
  loom_value_fact_layout_query_axis(function->facts->context.table, origin,
                                    axis, numeric_stride_facts, &layout_axis);
  if (layout_axis.kind == LOOM_VALUE_FACT_LAYOUT_AXIS_STATIC) {
    *out_component = (loom_layout_projection_component_t){
        .kind = LOOM_LAYOUT_PROJECTION_COMPONENT_CONSTANT,
        .value.constant = layout_axis.static_coefficient,
    };
    *out_planned = true;
    return iree_ok_status();
  }
  if (layout_axis.kind != LOOM_VALUE_FACT_LAYOUT_AXIS_DYNAMIC) {
    return iree_ok_status();
  }
  if (layout_axis.static_coefficient == 1 &&
      layout_axis.dynamic_factor_count == 1) {
    *out_component = (loom_layout_projection_component_t){
        .kind = LOOM_LAYOUT_PROJECTION_COMPONENT_VALUE,
        .value.value_id = layout_axis.dynamic_factors[0],
    };
  } else {
    loom_value_id_t* factors = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, layout_axis.dynamic_factor_count,
                                  sizeof(*factors), (void**)&factors));
    memcpy(factors, layout_axis.dynamic_factors,
           layout_axis.dynamic_factor_count * sizeof(*factors));
    *out_component = (loom_layout_projection_component_t){
        .kind = LOOM_LAYOUT_PROJECTION_COMPONENT_PRODUCT,
        .value.product =
            {
                .coefficient = layout_axis.static_coefficient,
                .factors = factors,
                .factor_count = layout_axis.dynamic_factor_count,
            },
    };
  }
  *out_planned = true;
  return iree_ok_status();
}

static bool loom_layout_projection_slot_component_for_axis(
    const loom_layout_projection_slot_plan_t* slot_plan, uint8_t axis,
    uint8_t* out_component) {
  for (uint8_t component = 0; component < slot_plan->component_count;
       ++component) {
    if (slot_plan->component_axes[component] == axis) {
      *out_component = component;
      return true;
    }
  }
  return false;
}

bool loom_layout_boundary_projection_component_for_axis(
    const loom_boundary_projection_schema_t* schema, uint8_t axis,
    uint8_t* out_component) {
  if (!schema ||
      (schema->rule != loom_cfg_layout_boundary_projection_rule() &&
       schema->rule != loom_loop_layout_boundary_projection_rule() &&
       schema->rule != loom_callable_encoding_boundary_projection_rule())) {
    return false;
  }
  return loom_layout_projection_slot_component_for_axis(
      (const loom_layout_projection_slot_plan_t*)schema->rule_plan, axis,
      out_component);
}

static iree_status_t loom_layout_projection_plan_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  *out_source = (loom_boundary_projection_source_t){0};
  *out_planned = false;
  IREE_ASSERT(schema->rule == rule);
  loom_layout_projection_slot_plan_t* destination_plan =
      (loom_layout_projection_slot_plan_t*)schema->rule_plan;
  IREE_ASSERT(destination_plan != NULL);
  IREE_ASSERT_EQ(destination_plan->component_count, schema->component_count);
  IREE_ASSERT(function->facts != NULL);

  if (destination_plan->encoding_role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    loom_value_fact_encoding_summary_t summary = {0};
    if (loom_value_facts_query_encoding_summary(
            &function->facts->context,
            loom_value_fact_table_lookup(function->facts->context.table,
                                         source_value_id),
            &summary) &&
        summary.role != LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
      return iree_ok_status();
    }
    loom_layout_projection_seed_storage_schema(
        destination_plan, summary.storage_schema.static_spec_encoding_id);
    loom_layout_projection_link_storage_source(
        rule, plan, function, destination_plan, source_value_id);
  }

  loom_value_id_t source_carrier_value_id = LOOM_VALUE_ID_INVALID;
  (void)loom_layout_projection_source_carrier(plan, destination_plan,
                                              source_value_id, boundary_op,
                                              &source_carrier_value_id);
  loom_value_id_t source_slot_value_id = source_value_id;
  if (destination_plan->encoding_role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE &&
      loom_boundary_projection_slot_index(function, source_slot_value_id) ==
          IREE_HOST_SIZE_MAX) {
    const loom_value_fact_layout_origin_t source_origin =
        loom_encoding_query_value_layout_origin(&function->facts->context,
                                                source_value_id);
    if (source_origin.has_physical_encoding_value) {
      source_slot_value_id = source_origin.physical_encoding_value_id;
    }
  }

  loom_layout_projection_source_plan_t* source_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*source_plan),
                                           (void**)&source_plan));
  if (schema->component_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, schema->component_count, sizeof(*source_plan->components),
        (void**)&source_plan->components));
  }
  source_plan->component_count = (uint8_t)schema->component_count;

  const iree_host_size_t destination_index =
      destination
          ? loom_boundary_projection_slot_index(function, destination->value_id)
          : IREE_HOST_SIZE_MAX;
  IREE_ASSERT(destination == NULL || destination_index != IREE_HOST_SIZE_MAX);
  for (uint8_t component = 0; component < source_plan->component_count;
       ++component) {
    const uint8_t axis = destination_plan->component_axes[component];
    bool component_planned = false;
    IREE_RETURN_IF_ERROR(loom_layout_projection_plan_direct_component(
        plan, function, source_value_id, source_carrier_value_id,
        destination_plan, axis, &source_plan->components[component],
        &component_planned));
    if (component_planned) {
      continue;
    }
    const iree_host_size_t source_index =
        loom_boundary_projection_slot_index(function, source_slot_value_id);
    if (source_index == IREE_HOST_SIZE_MAX) {
      return iree_ok_status();
    }
    const loom_boundary_projection_slot_t* source =
        &function->candidates[source_index];
    if (!source->selected || source->schema.rule != rule) {
      return iree_ok_status();
    }
    const loom_layout_projection_slot_plan_t* source_slot_plan =
        (const loom_layout_projection_slot_plan_t*)source->schema.rule_plan;
    uint8_t source_component = 0;
    if (source_slot_plan->rank != destination_plan->rank ||
        !loom_layout_projection_slot_component_for_axis(source_slot_plan, axis,
                                                        &source_component)) {
      return iree_ok_status();
    }
    const loom_layout_projection_component_t projected_component = {
        .kind = LOOM_LAYOUT_PROJECTION_COMPONENT_SLOT,
        .value.slot =
            {
                .slot = source_index,
                .component = source_component,
            },
    };
    source_plan->components[component] = projected_component;
    if (destination) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
          plan, function, source_index, destination_index,
          /*orders_realization=*/false));
    }
  }

  *out_source = (loom_boundary_projection_source_t){
      .rule = rule,
      .rule_plan = source_plan,
      .boundary_op = boundary_op,
  };
  *out_planned = true;
  return iree_ok_status();
}

static iree_status_t loom_layout_projection_materialize_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  IREE_ASSERT(source->rule == rule);
  IREE_ASSERT(source->boundary_op != NULL);
  const loom_layout_projection_source_plan_t* source_plan =
      (const loom_layout_projection_source_plan_t*)source->rule_plan;
  IREE_ASSERT(source_plan != NULL);
  for (uint8_t component = 0; component < source_plan->component_count;
       ++component) {
    const loom_layout_projection_component_t* projection =
        &source_plan->components[component];
    switch (projection->kind) {
      case LOOM_LAYOUT_PROJECTION_COMPONENT_VALUE:
        out_component_values[component] =
            loom_boundary_projection_resolve_value(function,
                                                   projection->value.value_id);
        break;
      case LOOM_LAYOUT_PROJECTION_COMPONENT_CONSTANT: {
        loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter.builder);
        loom_op_t* constant_op = NULL;
        iree_status_t status = loom_index_constant_build(
            &plan->rewriter.builder, loom_attr_i64(projection->value.constant),
            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
            source->boundary_op->location, &constant_op);
        loom_builder_restore(&plan->rewriter.builder, saved_ip);
        IREE_RETURN_IF_ERROR(status);
        out_component_values[component] =
            loom_index_constant_result(constant_op);
        break;
      }
      case LOOM_LAYOUT_PROJECTION_COMPONENT_SLOT: {
        const loom_boundary_projection_slot_t* projected_slot =
            &function->candidates[projection->value.slot.slot];
        out_component_values[component] =
            projected_slot
                ->component_value_ids[projection->value.slot.component];
        IREE_ASSERT(out_component_values[component] != LOOM_VALUE_ID_INVALID);
        break;
      }
      case LOOM_LAYOUT_PROJECTION_COMPONENT_PRODUCT: {
        loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter.builder);
        loom_value_id_t product = LOOM_VALUE_ID_INVALID;
        iree_status_t status = iree_ok_status();
        if (projection->value.product.coefficient != 1) {
          loom_op_t* constant_op = NULL;
          status = loom_index_constant_build(
              &plan->rewriter.builder,
              loom_attr_i64(projection->value.product.coefficient),
              loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
              source->boundary_op->location, &constant_op);
          if (iree_status_is_ok(status)) {
            product = loom_index_constant_result(constant_op);
          }
        }
        for (uint8_t i = 0; i < projection->value.product.factor_count &&
                            iree_status_is_ok(status);
             ++i) {
          const loom_value_id_t factor = loom_boundary_projection_resolve_value(
              function, projection->value.product.factors[i]);
          if (product == LOOM_VALUE_ID_INVALID) {
            product = factor;
            continue;
          }
          loom_op_t* multiply_op = NULL;
          status =
              loom_index_mul_build(&plan->rewriter.builder, product, factor,
                                   source->boundary_op->location, &multiply_op);
          if (iree_status_is_ok(status)) {
            product = loom_index_mul_result(multiply_op);
          }
        }
        loom_builder_restore(&plan->rewriter.builder, saved_ip);
        IREE_RETURN_IF_ERROR(status);
        IREE_ASSERT_NE(product, LOOM_VALUE_ID_INVALID);
        out_component_values[component] = product;
        break;
      }
      default:
        IREE_ASSERT(false);
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_layout_projection_reconstruct(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  (void)function;
  IREE_ASSERT(slot->schema.rule == rule);
  const loom_layout_projection_slot_plan_t* slot_plan =
      (const loom_layout_projection_slot_plan_t*)slot->schema.rule_plan;
  IREE_ASSERT(slot_plan != NULL);
  loom_op_t* layout_op = NULL;
  const loom_type_t layout_type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  IREE_RETURN_IF_ERROR(loom_encoding_layout_strided_build(
      &plan->rewriter.builder, slot->component_value_ids,
      slot_plan->component_count, slot_plan->static_strides, slot_plan->rank,
      slot_plan->encoding_role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT
          ? logical_type
          : layout_type,
      location, &layout_op));
  const loom_value_id_t layout_value =
      loom_encoding_layout_strided_result(layout_op);
  if (slot_plan->encoding_role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    *out_logical_value = layout_value;
  } else {
    IREE_ASSERT_EQ(slot_plan->encoding_role,
                   LOOM_ENCODING_ROLE_PHYSICAL_STORAGE);
    loom_layout_projection_slot_plan_t* storage_schema_root =
        loom_layout_projection_storage_schema_root(
            (loom_layout_projection_slot_plan_t*)slot_plan);
    IREE_ASSERT(!storage_schema_root->storage_schema_conflict);
    IREE_ASSERT_NE(storage_schema_root->storage_schema_encoding_id, 0);
    loom_op_t* storage_op = NULL;
    IREE_RETURN_IF_ERROR(loom_encoding_storage_build(
        &plan->rewriter.builder, layout_value,
        storage_schema_root->storage_schema_encoding_id, logical_type, location,
        &storage_op));
    *out_logical_value = loom_encoding_define_result(storage_op);
  }
  loom_boundary_projection_record(plan, rule, 1, slot_plan->component_count);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kCfgLayoutProjectionRule = {
    .name = IREE_SVL("cfg-dynamic-strided-layout"),
    .type_kind_bits =
        LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_ENCODING),
    .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT),
    .plan_slot = loom_layout_projection_plan_slot,
    .transport =
        {
            .plan_source = loom_layout_projection_plan_source,
            .materialize_source = loom_layout_projection_materialize_source,
            .reconstruct = loom_layout_projection_reconstruct,
        },
};

static const loom_boundary_projection_rule_t kLoopLayoutProjectionRule = {
    .name = IREE_SVL("loop-dynamic-strided-layout"),
    .type_kind_bits =
        LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_ENCODING),
    .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    .plan_slot = loom_layout_projection_plan_slot,
    .transport =
        {
            .plan_source = loom_layout_projection_plan_source,
            .materialize_source = loom_layout_projection_materialize_source,
            .reconstruct = loom_layout_projection_reconstruct,
        },
};

static const loom_boundary_projection_rule_t kCallableEncodingProjectionRule = {
    .name = IREE_SVL("callable-physical-encoding"),
    .type_kind_bits =
        LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_ENCODING),
    .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                          LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT) |
                      LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                          LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT),
    .prepare_function = loom_layout_projection_prepare_callable_function,
    .finalize_plan = loom_layout_projection_finalize_callable_plan,
    .plan_slot = loom_layout_projection_plan_slot,
    .transport =
        {
            .plan_source = loom_layout_projection_plan_source,
            .materialize_source = loom_layout_projection_materialize_source,
            .reconstruct = loom_layout_projection_reconstruct,
        },
};

const loom_boundary_projection_rule_t* loom_cfg_layout_boundary_projection_rule(
    void) {
  return &kCfgLayoutProjectionRule;
}

const loom_boundary_projection_rule_t*
loom_loop_layout_boundary_projection_rule(void) {
  return &kLoopLayoutProjectionRule;
}

const loom_boundary_projection_rule_t*
loom_callable_encoding_boundary_projection_rule(void) {
  return &kCallableEncodingProjectionRule;
}
