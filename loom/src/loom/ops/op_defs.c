// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/op_defs.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

//===----------------------------------------------------------------------===//
// Type constraint names
//===----------------------------------------------------------------------===//

const char* loom_type_constraint_name(loom_type_constraint_t constraint) {
  static const char* const names[] = {
      [LOOM_TYPE_CONSTRAINT_TILE] = "tile",
      [LOOM_TYPE_CONSTRAINT_TENSOR] = "tensor",
      [LOOM_TYPE_CONSTRAINT_INTEGER] = "integer",
      [LOOM_TYPE_CONSTRAINT_FLOAT] = "float",
      [LOOM_TYPE_CONSTRAINT_SCALAR] = "scalar",
      [LOOM_TYPE_CONSTRAINT_INDEX] = "index",
      [LOOM_TYPE_CONSTRAINT_ANY] = "any",
      [LOOM_TYPE_CONSTRAINT_GROUP] = "group",
      [LOOM_TYPE_CONSTRAINT_ENCODING] = "encoding",
      [LOOM_TYPE_CONSTRAINT_POOL] = "pool",
      [LOOM_TYPE_CONSTRAINT_I1] = "i1",
  };
  static_assert(IREE_ARRAYSIZE(names) == LOOM_TYPE_CONSTRAINT_COUNT_,
                "constraint names out of sync with enum");
  if (constraint < LOOM_TYPE_CONSTRAINT_COUNT_) return names[constraint];
  return "unknown";
}

//===----------------------------------------------------------------------===//
// Constraint relation and property names
//===----------------------------------------------------------------------===//

const char* loom_constraint_relation_name(loom_constraint_relation_t relation) {
  static const char* const names[] = {
      [LOOM_RELATION_PAIRWISE_EQ] = "PairwiseEq",
      [LOOM_RELATION_ALL_SAME] = "AllSame",
      [LOOM_RELATION_COUNT_MATCHES_RANK] = "CountMatchesRank",
      [LOOM_RELATION_ATTR_IN_RANGE_RANK] = "AttrInRangeRank",
      [LOOM_RELATION_REGION_ARG_COUNT] = "RegionArgCount",
      [LOOM_RELATION_REGION_ARG_MATCH] = "RegionArgMatch",
      [LOOM_RELATION_YIELD_COUNT] = "YieldCount",
      [LOOM_RELATION_YIELD_MATCH] = "YieldMatch",
  };
  static_assert(IREE_ARRAYSIZE(names) == LOOM_RELATION_COUNT_,
                "relation names out of sync with enum");
  if (relation < LOOM_RELATION_COUNT_) return names[relation];
  return "unknown";
}

const char* loom_constraint_property_name(loom_constraint_property_t property) {
  static const char* const names[] = {
      [LOOM_PROPERTY_TYPE] = "Type",
      [LOOM_PROPERTY_ELEMENT_TYPE] = "ElementType",
      [LOOM_PROPERTY_ENCODING] = "Encoding",
      [LOOM_PROPERTY_SHAPE] = "Shape",
      [LOOM_PROPERTY_RANK] = "Rank",
  };
  static_assert(IREE_ARRAYSIZE(names) == LOOM_PROPERTY_COUNT_,
                "property names out of sync with enum");
  if (property < LOOM_PROPERTY_COUNT_) return names[property];
  return "unknown";
}

//===----------------------------------------------------------------------===//
// Keyword B-string table
//===----------------------------------------------------------------------===//

// Generated from KEYWORD_MAP in c_tables.py — do not edit manually.
const loom_bstring_t loom_keyword_bstrings[LOOM_KW_COUNT_] = {
#include "loom/ops/keyword_table.inc"
};

//===----------------------------------------------------------------------===//
// Effect query helpers
//===----------------------------------------------------------------------===//

loom_trait_flags_t loom_op_effective_traits(const loom_module_t* module,
                                            const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) return 0;
  if (vtable->effective_traits) return vtable->effective_traits(op);
  return vtable->traits;
}

bool loom_op_may_write(const loom_module_t* module, const loom_op_t* op) {
  return loom_traits_may_write(loom_op_effective_traits(module, op));
}

bool loom_op_results_unused(const loom_module_t* module, const loom_op_t* op) {
  loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID &&
        loom_module_value(module, results[i])->use_count > 0) {
      return false;
    }
  }
  return true;
}

bool loom_op_is_trivially_dead(const loom_module_t* module,
                               const loom_op_t* op) {
  if (op->result_count == 0) return false;
  if (loom_op_may_write(module, op)) return false;
  return loom_op_results_unused(module, op);
}

//===----------------------------------------------------------------------===//
// Attribute equality and hashing
//===----------------------------------------------------------------------===//

// FNV-1a hash over a byte range, folded into a running hash.
static uint32_t loom_hash_bytes(const void* data, iree_host_size_t length,
                                uint32_t hash) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

// Maximum nesting depth for recursive DICT attribute comparison and
// hashing. No legitimate op definition creates DICTs nested deeper
// than 2-3 levels. Beyond this limit, equal returns false
// (conservative — prevents CSE but never miscompiles) and hash falls
// back to raw struct bytes (consistent with equal's refusal to match).
#define LOOM_ATTRIBUTE_MAX_DEPTH 16

static bool loom_attribute_equal_impl(const loom_attribute_t* a,
                                      const loom_attribute_t* b, int depth) {
  if (a->kind != b->kind) return false;
  switch ((loom_attr_kind_t)a->kind) {
    case LOOM_ATTR_I64_ARRAY:
      if (a->count != b->count) return false;
      if (a->i64_array == b->i64_array) return true;
      return memcmp(a->i64_array, b->i64_array,
                    (iree_host_size_t)a->count * sizeof(int64_t)) == 0;
    case LOOM_ATTR_PREDICATE_LIST:
      if (a->count != b->count) return false;
      if (a->predicate_list == b->predicate_list) return true;
      return memcmp(a->predicate_list, b->predicate_list,
                    (iree_host_size_t)a->count * sizeof(loom_predicate_t)) == 0;
    case LOOM_ATTR_DICT:
      if (a->count != b->count) return false;
      if (a->dict == b->dict) return true;
      if (depth >= LOOM_ATTRIBUTE_MAX_DEPTH) return false;
      for (uint16_t i = 0; i < a->count; ++i) {
        if (a->dict[i].name_id != b->dict[i].name_id) return false;
        if (!loom_attribute_equal_impl(&a->dict[i].value, &b->dict[i].value,
                                       depth + 1)) {
          return false;
        }
      }
      return true;
    default:
      return memcmp(a, b, sizeof(loom_attribute_t)) == 0;
  }
}

bool loom_attribute_equal(const loom_attribute_t* a,
                          const loom_attribute_t* b) {
  return loom_attribute_equal_impl(a, b, 0);
}

static uint32_t loom_attribute_hash_impl(const loom_attribute_t* attr,
                                         int depth) {
  uint32_t hash = 2166136261u;
  hash = loom_hash_bytes(&attr->kind, 1, hash);
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_I64_ARRAY:
      hash = loom_hash_bytes(&attr->count, sizeof(attr->count), hash);
      hash = loom_hash_bytes(attr->i64_array,
                             (iree_host_size_t)attr->count * sizeof(int64_t),
                             hash);
      break;
    case LOOM_ATTR_PREDICATE_LIST:
      hash = loom_hash_bytes(&attr->count, sizeof(attr->count), hash);
      hash = loom_hash_bytes(
          attr->predicate_list,
          (iree_host_size_t)attr->count * sizeof(loom_predicate_t), hash);
      break;
    case LOOM_ATTR_DICT:
      if (depth >= LOOM_ATTRIBUTE_MAX_DEPTH) {
        hash = loom_hash_bytes(attr, sizeof(loom_attribute_t), hash);
        break;
      }
      hash = loom_hash_bytes(&attr->count, sizeof(attr->count), hash);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash = loom_hash_bytes(&attr->dict[i].name_id, sizeof(loom_string_id_t),
                               hash);
        uint32_t value_hash =
            loom_attribute_hash_impl(&attr->dict[i].value, depth + 1);
        hash = loom_hash_bytes(&value_hash, sizeof(value_hash), hash);
      }
      break;
    default:
      hash = loom_hash_bytes(attr, sizeof(loom_attribute_t), hash);
      break;
  }
  return hash;
}

uint32_t loom_attribute_hash(const loom_attribute_t* attr) {
  return loom_attribute_hash_impl(attr, 0);
}

//===----------------------------------------------------------------------===//
// FuncLike interface
//===----------------------------------------------------------------------===//

loom_func_like_t loom_func_like_cast(const loom_module_t* module,
                                     loom_op_t* op) {
  if (!op) return (loom_func_like_t){.op = NULL, .vtable = NULL};
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->func_like) {
    return (loom_func_like_t){.op = NULL, .vtable = NULL};
  }
  return (loom_func_like_t){.op = op, .vtable = vtable->func_like};
}

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

void loom_builder_initialize(loom_module_t* module,
                             iree_arena_allocator_t* arena, loom_block_t* block,
                             loom_builder_t* out_builder) {
  out_builder->module = module;
  out_builder->arena = arena;
  out_builder->ip.block = block;
  out_builder->ip.parent_op = NULL;
  out_builder->ip.index = UINT16_MAX;  // Append mode.
  out_builder->on_op_finalized.fn = NULL;
  out_builder->on_op_finalized.user_data = NULL;
}

void loom_builder_set_block(loom_builder_t* builder, loom_block_t* block) {
  builder->ip.block = block;
  builder->ip.index = UINT16_MAX;  // Append mode.
}

loom_builder_ip_t loom_builder_enter_region(loom_builder_t* builder,
                                            loom_op_t* parent_op,
                                            loom_region_t* region) {
  loom_builder_ip_t saved = builder->ip;
  builder->ip.block = &region->blocks[0];
  builder->ip.parent_op = parent_op;
  builder->ip.index = UINT16_MAX;  // Append mode.
  return saved;
}

void loom_builder_set_before(loom_builder_t* builder, const loom_op_t* op) {
  builder->ip.block = op->parent_block;
  builder->ip.index = loom_block_find_op(builder->ip.block, op);
}

void loom_builder_set_after(loom_builder_t* builder, const loom_op_t* op) {
  builder->ip.block = op->parent_block;
  uint16_t index = loom_block_find_op(builder->ip.block, op);
  builder->ip.index =
      (index != UINT16_MAX) ? (uint16_t)(index + 1) : UINT16_MAX;
}

loom_builder_ip_t loom_builder_save(const loom_builder_t* builder) {
  return builder->ip;
}

void loom_builder_restore(loom_builder_t* builder, loom_builder_ip_t ip) {
  builder->ip = ip;
}

iree_status_t loom_builder_define_value(loom_builder_t* builder,
                                        loom_type_t type,
                                        loom_value_id_t* out_value_id) {
  return loom_module_define_value(builder->module, type, out_value_id);
}

iree_status_t loom_builder_define_block_arg(loom_builder_t* builder,
                                            loom_block_t* block,
                                            loom_type_t type,
                                            loom_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(loom_builder_define_value(builder, type, out_value_id));
  return loom_block_add_arg(builder->module, block, *out_value_id);
}

iree_status_t loom_builder_intern_string(loom_builder_t* builder,
                                         iree_string_view_t string,
                                         loom_string_id_t* out_string_id) {
  return loom_module_intern_string(builder->module, string, out_string_id);
}

iree_status_t loom_builder_allocate_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  if (!builder->ip.block || !builder->module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "builder has no insertion block or module");
  }

  iree_host_size_t operands_size =
      (iree_host_size_t)operand_count * sizeof(loom_value_id_t);
  iree_host_size_t results_size =
      (iree_host_size_t)result_count * sizeof(loom_value_id_t);
  iree_host_size_t regions_size =
      (iree_host_size_t)region_count * sizeof(loom_region_t*);
  iree_host_size_t tied_size =
      (iree_host_size_t)tied_result_count * sizeof(loom_tied_result_t);

  iree_host_size_t before_attrs = sizeof(loom_op_t) + operands_size +
                                  results_size + regions_size + tied_size;
  iree_host_size_t aligned_before_attrs =
      attribute_count > 0
          ? iree_host_align(before_attrs, iree_alignof(loom_attribute_t))
          : before_attrs;
  iree_host_size_t attrs_size =
      (iree_host_size_t)attribute_count * sizeof(loom_attribute_t);
  iree_host_size_t total_size = aligned_before_attrs + attrs_size;

  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(builder->arena, total_size, &allocation));
  memset(allocation, 0, total_size);

  loom_op_t* op = (loom_op_t*)allocation;
  op->kind = kind;
  op->operand_count = operand_count;
  op->result_count = result_count;
  op->region_count = region_count;
  op->tied_result_count = tied_result_count;
  op->attribute_count = attribute_count;
  op->location = location;
  op->parent_op = builder->ip.parent_op;
  op->parent_block = builder->ip.block;

  if (builder->ip.index == UINT16_MAX) {
    IREE_RETURN_IF_ERROR(
        loom_block_append_op(builder->module, builder->ip.block, op));
  } else {
    IREE_RETURN_IF_ERROR(loom_block_insert_op(
        builder->module, builder->ip.block, builder->ip.index, op));
    ++builder->ip.index;
  }

  *out_op = op;
  return iree_ok_status();
}

iree_status_t loom_op_erase(loom_module_t* module, loom_op_t* op) {
  // Verify that all results are unused (caller must RAUW first).
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID &&
        module->values.entries[results[i]].use_count > 0) {
      iree_string_view_t op_name = loom_op_name(module, op);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot erase %.*s: result %%%u still has %u use(s)",
          (int)op_name.size, op_name.data, (unsigned)results[i],
          (unsigned)module->values.entries[results[i]].use_count);
    }
  }
  // Remove all operand uses from the referenced values.
  loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_value_remove_use(module, operands[i], op, i));
    }
  }
  // Clear def pointers on result values (the op is being erased, so
  // the pointers would dangle).
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      module->values.entries[results[i]].def = loom_value_def_make_none();
    }
  }
  op->flags |= LOOM_OP_FLAG_DEAD;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Use-def list maintenance
//===----------------------------------------------------------------------===//

// Initial overflow capacity when transitioning from inline to overflow.
// 8 covers the common case of values used 4-8 times without further
// reallocation. Values used more than 8 times get geometric growth.
#define LOOM_USE_INITIAL_OVERFLOW_CAPACITY 8

iree_status_t loom_value_add_use(loom_module_t* module,
                                 loom_value_id_t value_id, loom_op_t* user_op,
                                 uint16_t operand_index) {
  loom_value_t* value = &module->values.entries[value_id];
  loom_use_t use = loom_use_make(user_op, operand_index);

  if (!loom_value_has_overflow_uses(value)) {
    if (value->use_count < LOOM_VALUE_INLINE_USE_COUNT) {
      // Common path: store inline.
      value->inline_uses[value->use_count] = use;
      ++value->use_count;
      return iree_ok_status();
    }
    // Transition from inline to overflow: allocate array, copy inline
    // uses, then add the new use.
    uint32_t capacity = LOOM_USE_INITIAL_OVERFLOW_CAPACITY;
    loom_use_t* overflow = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, capacity, sizeof(loom_use_t), (void**)&overflow));
    for (uint16_t i = 0; i < LOOM_VALUE_INLINE_USE_COUNT; ++i) {
      overflow[i] = value->inline_uses[i];
    }
    overflow[LOOM_VALUE_INLINE_USE_COUNT] = use;
    value->overflow_uses = overflow;
    value->overflow_capacity = capacity;
    value->flags |= LOOM_VALUE_FLAG_OVERFLOW_USES;
    ++value->use_count;
    return iree_ok_status();
  }

  // Already in overflow mode.
  if (value->use_count < value->overflow_capacity) {
    // Space available: append.
    value->overflow_uses[value->use_count] = use;
    ++value->use_count;
    return iree_ok_status();
  }

  // Overflow array is full: grow by 2x (floor to initial capacity as
  // a safety net against zero-capacity invariant violations).
  uint32_t new_capacity = iree_max(value->overflow_capacity * 2,
                                   LOOM_USE_INITIAL_OVERFLOW_CAPACITY);
  loom_use_t* new_overflow = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, new_capacity, sizeof(loom_use_t), (void**)&new_overflow));
  memcpy(new_overflow, value->overflow_uses,
         (iree_host_size_t)value->use_count * sizeof(loom_use_t));
  new_overflow[value->use_count] = use;
  value->overflow_uses = new_overflow;
  value->overflow_capacity = new_capacity;
  ++value->use_count;
  return iree_ok_status();
}

iree_status_t loom_value_remove_use(loom_module_t* module,
                                    loom_value_id_t value_id,
                                    loom_op_t* user_op,
                                    uint16_t operand_index) {
  loom_value_t* value = &module->values.entries[value_id];
  loom_use_t* uses = loom_value_uses_mutable(value);
  for (uint16_t i = 0; i < value->use_count; ++i) {
    if (loom_use_user_op(uses[i]) == user_op &&
        loom_use_operand_index(uses[i]) == operand_index) {
      // Swap with last and decrement.
      uses[i] = uses[value->use_count - 1];
      --value->use_count;
      return iree_ok_status();
    }
  }
  iree_string_view_t op_name = loom_op_name(module, user_op);
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "no matching use of value %%%u by %.*s operand %u",
                          (unsigned)value_id, (int)op_name.size, op_name.data,
                          (unsigned)operand_index);
}

void loom_module_link_symbol_defining_op(loom_module_t* module, loom_op_t* op,
                                         const loom_op_vtable_t* vtable) {
  loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    if (vtable->attr_descriptors[i].attr_kind != LOOM_ATTR_SYMBOL) continue;
    loom_symbol_ref_t ref = loom_attr_as_symbol(attrs[i]);
    if (loom_symbol_ref_is_valid(ref) &&
        ref.symbol_id < module->symbols.count) {
      module->symbols.entries[ref.symbol_id].defining_op = op;
      module->symbols.entries[ref.symbol_id].kind = vtable->symbol_kind;
    }
    return;
  }
}

iree_status_t loom_builder_finalize_op(loom_builder_t* builder, loom_op_t* op) {
  // Register operand uses.
  loom_value_id_t* operands = loom_op_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_value_add_use(builder->module, operands[i], op, i));
    }
  }
  // Set the def pointer on each result value.
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      builder->module->values.entries[results[i]].def =
          loom_value_def_make_op(op, i);
    }
  }
  // Wire the symbol table entry for symbol-defining ops so that
  // loom_func_like_cast can find the defining op without a scan.
  const loom_op_vtable_t* vtable = loom_op_vtable(builder->module, op);
  if (vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    loom_module_link_symbol_defining_op(builder->module, op, vtable);
  }
  // Notify the rewriter (or other listener) that a fully-wired op exists.
  if (builder->on_op_finalized.fn) {
    IREE_RETURN_IF_ERROR(
        builder->on_op_finalized.fn(builder->on_op_finalized.user_data, op));
  }
  return iree_ok_status();
}

iree_status_t loom_op_set_operand(loom_module_t* module, loom_op_t* op,
                                  uint16_t operand_index,
                                  loom_value_id_t new_value_id) {
  loom_value_id_t* operands = loom_op_operands(op);
  loom_value_id_t old_value_id = operands[operand_index];
  if (old_value_id == new_value_id) return iree_ok_status();
  if (old_value_id != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_remove_use(module, old_value_id, op, operand_index));
  }
  operands[operand_index] = new_value_id;
  if (new_value_id != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_add_use(module, new_value_id, op, operand_index));
  }
  return iree_ok_status();
}

// Ensures a value's use list has capacity for at least |additional| more
// entries. Used by RAUW to pre-allocate before bulk transfer.
static iree_status_t loom_value_ensure_use_capacity(loom_module_t* module,
                                                    loom_value_t* value,
                                                    uint16_t additional) {
  uint32_t needed = (uint32_t)value->use_count + (uint32_t)additional;
  if (!loom_value_has_overflow_uses(value)) {
    if (needed <= LOOM_VALUE_INLINE_USE_COUNT) return iree_ok_status();
    // Transition to overflow with enough capacity.
    uint32_t capacity = LOOM_USE_INITIAL_OVERFLOW_CAPACITY;
    while (capacity < needed) capacity *= 2;
    loom_use_t* overflow = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, capacity, sizeof(loom_use_t), (void**)&overflow));
    for (uint16_t i = 0; i < value->use_count; ++i) {
      overflow[i] = value->inline_uses[i];
    }
    value->overflow_uses = overflow;
    value->overflow_capacity = capacity;
    value->flags |= LOOM_VALUE_FLAG_OVERFLOW_USES;
    return iree_ok_status();
  }
  if (needed <= value->overflow_capacity) return iree_ok_status();
  // Grow overflow (floor to initial capacity for safety).
  uint32_t new_capacity =
      iree_max(value->overflow_capacity, LOOM_USE_INITIAL_OVERFLOW_CAPACITY);
  while (new_capacity < needed) new_capacity *= 2;
  loom_use_t* new_overflow = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, new_capacity, sizeof(loom_use_t), (void**)&new_overflow));
  memcpy(new_overflow, value->overflow_uses,
         (iree_host_size_t)value->use_count * sizeof(loom_use_t));
  value->overflow_uses = new_overflow;
  value->overflow_capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t loom_value_replace_all_uses_with(loom_module_t* module,
                                               loom_value_id_t old_id,
                                               loom_value_id_t new_id) {
  if (old_id == new_id) return iree_ok_status();
  loom_value_t* old_value = &module->values.entries[old_id];
  if (old_value->use_count == 0) return iree_ok_status();

  // Patch every user op's operand slot.
  const loom_use_t* old_uses = loom_value_uses(old_value);
  for (uint16_t i = 0; i < old_value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(old_uses[i]);
    uint16_t operand_index = loom_use_operand_index(old_uses[i]);
    loom_op_operands(user_op)[operand_index] = new_id;
  }

  // Bulk-transfer use entries from old to new.
  loom_value_t* new_value = &module->values.entries[new_id];
  IREE_RETURN_IF_ERROR(
      loom_value_ensure_use_capacity(module, new_value, old_value->use_count));

  // Append old's entries to new's list. The old_uses pointer (captured
  // above) is still valid: ensure_use_capacity only touches new_value,
  // and old_id != new_id is guarded at entry.
  loom_use_t* new_uses = loom_value_uses_mutable(new_value);
  for (uint16_t i = 0; i < old_value->use_count; ++i) {
    new_uses[new_value->use_count + i] = old_uses[i];
  }
  new_value->use_count += old_value->use_count;

  // Clear old value's use list.
  old_value->use_count = 0;
  old_value->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  return iree_ok_status();
}

iree_status_t loom_value_replace_all_uses_except(loom_module_t* module,
                                                 loom_value_id_t old_id,
                                                 loom_value_id_t new_id,
                                                 const loom_op_t* except_op) {
  if (old_id == new_id) return iree_ok_status();
  loom_value_t* old_value = &module->values.entries[old_id];
  if (old_value->use_count == 0) return iree_ok_status();

  // Count how many uses will be transferred vs kept.
  const loom_use_t* old_uses = loom_value_uses(old_value);
  uint16_t transfer_count = 0;
  for (uint16_t i = 0; i < old_value->use_count; ++i) {
    if (loom_use_user_op(old_uses[i]) != except_op) {
      ++transfer_count;
    }
  }
  if (transfer_count == 0) return iree_ok_status();

  // Ensure new has capacity.
  loom_value_t* new_value = &module->values.entries[new_id];
  IREE_RETURN_IF_ERROR(
      loom_value_ensure_use_capacity(module, new_value, transfer_count));

  // Patch operand slots and transfer use entries.
  // Walk old's list backwards so swap-removal doesn't skip entries.
  loom_use_t* old_uses_mutable = loom_value_uses_mutable(old_value);
  loom_use_t* new_uses = loom_value_uses_mutable(new_value);
  for (int32_t i = (int32_t)old_value->use_count - 1; i >= 0; --i) {
    if (loom_use_user_op(old_uses_mutable[i]) == except_op) continue;
    // Patch the operand slot.
    loom_op_t* user_op = loom_use_user_op(old_uses_mutable[i]);
    uint16_t operand_index = loom_use_operand_index(old_uses_mutable[i]);
    loom_op_operands(user_op)[operand_index] = new_id;
    // Add to new's list.
    new_uses[new_value->use_count] = old_uses_mutable[i];
    ++new_value->use_count;
    // Remove from old's list (swap with last).
    old_uses_mutable[i] = old_uses_mutable[old_value->use_count - 1];
    --old_value->use_count;
  }

  // If old is now empty and was overflow, clear the flag.
  if (old_value->use_count == 0) {
    old_value->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  }
  return iree_ok_status();
}

iree_status_t loom_value_replace_uses_if(loom_module_t* module,
                                         loom_value_id_t old_id,
                                         loom_value_id_t new_id,
                                         loom_use_predicate_fn predicate,
                                         void* user_data) {
  if (old_id == new_id) return iree_ok_status();
  loom_value_t* old_value = &module->values.entries[old_id];
  if (old_value->use_count == 0) return iree_ok_status();

  // Count how many uses will be transferred.
  const loom_use_t* old_uses = loom_value_uses(old_value);
  uint16_t transfer_count = 0;
  for (uint16_t i = 0; i < old_value->use_count; ++i) {
    if (predicate(loom_use_user_op(old_uses[i]), user_data)) {
      ++transfer_count;
    }
  }
  if (transfer_count == 0) return iree_ok_status();

  // Ensure new has capacity.
  loom_value_t* new_value = &module->values.entries[new_id];
  IREE_RETURN_IF_ERROR(
      loom_value_ensure_use_capacity(module, new_value, transfer_count));

  // Patch and transfer (walk backwards for safe swap-removal).
  loom_use_t* old_uses_mutable = loom_value_uses_mutable(old_value);
  loom_use_t* new_uses = loom_value_uses_mutable(new_value);
  for (int32_t i = (int32_t)old_value->use_count - 1; i >= 0; --i) {
    if (!predicate(loom_use_user_op(old_uses_mutable[i]), user_data)) continue;
    loom_op_t* user_op = loom_use_user_op(old_uses_mutable[i]);
    uint16_t operand_index = loom_use_operand_index(old_uses_mutable[i]);
    loom_op_operands(user_op)[operand_index] = new_id;
    new_uses[new_value->use_count] = old_uses_mutable[i];
    ++new_value->use_count;
    old_uses_mutable[i] = old_uses_mutable[old_value->use_count - 1];
    --old_value->use_count;
  }
  if (old_value->use_count == 0) {
    old_value->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
  }
  return iree_ok_status();
}

// Walks all blocks in a region recursively, adding uses, setting
// def pointers, and setting parent pointers for each op.
static iree_status_t loom_region_compute_uses(loom_module_t* module,
                                              loom_region_t* region,
                                              loom_op_t* parent_op) {
  if (!region) return iree_ok_status();
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    // Set def pointers for block arguments.
    for (uint16_t a = 0; a < block->arg_count; ++a) {
      loom_value_id_t arg_id = block->arg_ids[a];
      if (arg_id != LOOM_VALUE_ID_INVALID) {
        module->values.entries[arg_id].def =
            loom_value_def_make_block(block, a);
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      // Set parent pointers.
      op->parent_op = parent_op;
      op->parent_block = block;
      // Register operand uses.
      loom_value_id_t* operands = loom_op_operands(op);
      for (uint16_t i = 0; i < op->operand_count; ++i) {
        if (operands[i] != LOOM_VALUE_ID_INVALID) {
          IREE_RETURN_IF_ERROR(loom_value_add_use(module, operands[i], op, i));
        }
      }
      // Set def pointers on result values.
      loom_value_id_t* results = loom_op_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        if (results[i] != LOOM_VALUE_ID_INVALID) {
          module->values.entries[results[i]].def =
              loom_value_def_make_op(op, i);
        }
      }
      // Link symbol-defining ops at module scope. Nested ops cannot
      // define symbols so the vtable lookup is skipped for inner regions.
      if (!parent_op) {
        const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
        if (vtable &&
            iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
          loom_module_link_symbol_defining_op(module, op, vtable);
        }
      }
      // Recurse into nested regions.
      for (uint8_t r = 0; r < op->region_count; ++r) {
        loom_region_t* nested = loom_op_regions(op)[r];
        IREE_RETURN_IF_ERROR(loom_region_compute_uses(module, nested, op));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_module_compute_uses(loom_module_t* module) {
  // Clear all use and def data on every value.
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    loom_value_t* value = &module->values.entries[i];
    value->use_count = 0;
    value->flags &= ~LOOM_VALUE_FLAG_OVERFLOW_USES;
    value->def = loom_value_def_make_none();
    memset(value->inline_uses, 0,
           LOOM_VALUE_INLINE_USE_COUNT * sizeof(loom_use_t));
  }
  // Walk all ops and re-add uses, def pointers, and parent pointers.
  return loom_region_compute_uses(module, module->body, NULL);
}
