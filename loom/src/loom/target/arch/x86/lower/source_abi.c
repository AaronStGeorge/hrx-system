// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/source_abi.h"

#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/call_abi.h"
#include "loom/target/arch/x86/descriptors/entry.h"
#include "loom/target/arch/x86/register_classes.h"
#include "loom/target/arch/x86/source_abi.h"
#include "loom/target/registers.h"

enum {
  LOOM_X86_SOURCE_ABI_INTEGER_ARGUMENT_REGISTER_COUNT = 6,
  LOOM_X86_SOURCE_ABI_SSE_ARGUMENT_REGISTER_COUNT = 8,
  LOOM_X86_SOURCE_ABI_RESULT_REGISTER_COUNT = 2,
};

static const int64_t kLoomX86SourceAbiIntegerArgumentRegisters[] = {7, 6, 2,
                                                                    1, 8, 9};
static const int64_t kLoomX86SourceAbiIntegerResultRegisters[] = {0, 2};

typedef enum loom_x86_source_abi_leaf_kind_e {
  LOOM_X86_SOURCE_ABI_LEAF_SCALAR = 0,
  LOOM_X86_SOURCE_ABI_LEAF_VECTOR = 1,
  LOOM_X86_SOURCE_ABI_LEAF_POINTER = 2,
} loom_x86_source_abi_leaf_kind_t;

typedef struct loom_x86_source_abi_leaf_t {
  // First flat logical value carrying this source leaf.
  uint16_t logical_offset;
  // Number of flat logical values; pointers consume base and byte offset.
  uint8_t logical_count;
  // loom_x86_source_abi_leaf_kind_t.
  uint8_t kind;
  // Byte offset from the containing root object.
  uint32_t byte_offset;
  // Exact source object width of this leaf.
  uint32_t byte_length;
  // True when a loaded GPR32 value must be reduced to one bit.
  bool normalize_i1;
} loom_x86_source_abi_leaf_t;
static_assert(sizeof(loom_x86_source_abi_leaf_t) == 16,
              "x86 source ABI leaf recipes must be exactly 16 bytes");

typedef struct loom_x86_source_abi_carrier_t {
  // Byte offset from the source object carried in the low bits.
  uint32_t byte_offset;
  // Physical x86 register class used by the Low signature.
  uint16_t register_class;
} loom_x86_source_abi_carrier_t;
static_assert(sizeof(loom_x86_source_abi_carrier_t) == 8,
              "x86 source ABI carrier recipes must be exactly 8 bytes");

typedef struct loom_x86_source_abi_object_t {
  // Complete SysV classification of this root source object.
  loom_x86_source_abi_classification_t classification;
  // First flat logical value carrying the source object.
  uint16_t logical_offset;
  // Number of flat logical values carrying the source object.
  uint16_t logical_count;
  // First leaf recipe in the corresponding argument or result leaf array.
  uint16_t leaf_offset;
  // Number of leaf recipes owned by this source object.
  uint16_t leaf_count;
  // First physical carrier in the corresponding argument or result array.
  uint16_t physical_offset;
  // Number of physical carriers for this object. An indirect result has none.
  uint16_t physical_count;
  // Private byte image required to pack or unpack this object safely.
  uint32_t storage_byte_length;
  // True when the complete argument crosses through stack memory.
  bool is_memory_argument;
} loom_x86_source_abi_object_t;
static_assert(sizeof(loom_x86_source_abi_object_t) == 40,
              "x86 source ABI object recipes must be exactly 40 bytes");

typedef struct loom_x86_source_abi_plan_t {
  // Descriptor vocabulary selected with the immutable representation view.
  const loom_x86_entry_descriptors_t* entry;
  // Flat logical argument and result carrier types retained by common lowering.
  const loom_type_t* logical_argument_types;
  const loom_type_t* logical_result_types;
  // Source object recipes in signature order.
  loom_x86_source_abi_object_t* arguments;
  loom_x86_source_abi_object_t* results;
  uint16_t argument_object_count;
  uint16_t result_object_count;
  // Leaf recipes in flat logical order.
  loom_x86_source_abi_leaf_t* argument_leaves;
  loom_x86_source_abi_leaf_t* result_leaves;
  uint16_t argument_leaf_count;
  uint16_t result_leaf_count;
  // Physical carrier recipes parallel to the emitted Low signature.
  loom_x86_source_abi_carrier_t* argument_carriers;
  loom_x86_source_abi_carrier_t* result_carriers;
  uint16_t argument_carrier_count;
  uint16_t result_carrier_count;
  // A memory-class result is returned through physical argument zero.
  bool has_indirect_result;
} loom_x86_source_abi_plan_t;

typedef struct loom_x86_source_abi_emission_state_t {
  // Private storage reserved by a call for an indirect result.
  loom_value_id_t result_storage;
  // Address passed by a caller or received by a definition for that result.
  loom_value_id_t result_address;
} loom_x86_source_abi_emission_state_t;

typedef struct loom_x86_source_abi_plan_builder_t {
  loom_low_lower_context_t* context;
  const loom_func_c_abi_signature_t* signature;
  iree_arena_allocator_t* arena;
  loom_x86_source_abi_plan_t* plan;
  // Physical Low and source-semantic signatures under construction.
  loom_type_t* argument_types;
  loom_type_t* result_types;
  loom_type_t* semantic_argument_types;
  loom_type_t* semantic_result_types;
  // Exact placement arrays parallel to the physical signature.
  int64_t* argument_registers;
  int64_t* argument_offsets;
  int64_t* result_registers;
  // Current physical signature lengths.
  uint16_t argument_count;
  uint16_t result_count;
  // Current register-bank consumption.
  uint8_t integer_argument_count;
  uint8_t sse_argument_count;
  uint8_t integer_result_count;
  uint8_t sse_result_count;
  // Current stack argument geometry.
  uint32_t stack_bytes;
  uint8_t stack_alignment;
} loom_x86_source_abi_plan_builder_t;

static iree_status_t loom_x86_source_abi_unsupported(
    loom_low_lower_context_t* context, iree_string_view_t reason) {
  return loom_low_lower_emit_source_abi_unsupported(
      context, loom_low_lower_context_source_function(context).op, reason);
}

static uint32_t loom_x86_source_abi_register_byte_length(
    loom_type_t register_type) {
  if (!loom_low_type_is_register(register_type) ||
      loom_low_register_type_unit_count(register_type) != 1) {
    return 0;
  }
  return loom_x86_call_abi_register_byte_length(
      loom_low_register_type_class_id(register_type));
}

static iree_status_t loom_x86_source_abi_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static iree_status_t loom_x86_source_abi_build_leaves(
    const loom_func_c_abi_signature_t* signature, uint32_t type_index,
    uint32_t object_byte_offset, const loom_type_t* logical_types,
    uint16_t logical_count, uint16_t* inout_logical_offset,
    loom_x86_source_abi_leaf_t* leaves, uint16_t leaf_capacity,
    uint16_t* inout_leaf_count, uint32_t* inout_storage_byte_length) {
  IREE_ASSERT_LT(type_index, signature->types.count);
  const loom_attribute_t type = signature->types.values[type_index];
  const loom_func_abi_c_type_kind_t kind = loom_func_abi_c_type_attr_kind(type);
  if (kind == LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY) {
    const uint32_t element_type =
        (uint32_t)loom_func_abi_c_type_attr_element_type(type);
    const uint32_t element_count =
        (uint32_t)loom_func_abi_c_type_attr_element_count(type);
    const uint32_t element_stride =
        (uint32_t)loom_func_abi_c_type_attr_element_stride(type);
    for (uint32_t i = 0; i < element_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_build_leaves(
          signature, element_type, object_byte_offset + i * element_stride,
          logical_types, logical_count, inout_logical_offset, leaves,
          leaf_capacity, inout_leaf_count, inout_storage_byte_length));
    }
    return iree_ok_status();
  }
  if (kind == LOOM_FUNC_ABI_C_TYPE_KIND_RECORD) {
    const loom_parameterized_attr_array_t fields =
        loom_func_abi_c_type_attr_fields(type);
    for (iree_host_size_t i = 0; i < fields.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_build_leaves(
          signature,
          (uint32_t)loom_func_abi_c_field_attr_type(fields.values[i]),
          object_byte_offset + (uint32_t)loom_func_abi_c_field_attr_byte_offset(
                                   fields.values[i]),
          logical_types, logical_count, inout_logical_offset, leaves,
          leaf_capacity, inout_leaf_count, inout_storage_byte_length));
    }
    return iree_ok_status();
  }

  IREE_ASSERT_LT(*inout_leaf_count, leaf_capacity);
  const bool pointer = kind == LOOM_FUNC_ABI_C_TYPE_KIND_POINTER;
  const uint8_t component_count = pointer ? 2 : 1;
  IREE_ASSERT_LE((uint32_t)*inout_logical_offset + component_count,
                 logical_count);
  const uint32_t byte_length =
      (uint32_t)loom_func_abi_c_type_attr_byte_length(type);
  loom_x86_source_abi_leaf_t* leaf = &leaves[(*inout_leaf_count)++];
  *leaf = (loom_x86_source_abi_leaf_t){
      .logical_offset = *inout_logical_offset,
      .logical_count = component_count,
      .kind = pointer ? LOOM_X86_SOURCE_ABI_LEAF_POINTER
              : kind == LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR
                  ? LOOM_X86_SOURCE_ABI_LEAF_VECTOR
                  : LOOM_X86_SOURCE_ABI_LEAF_SCALAR,
      .byte_offset = object_byte_offset,
      .byte_length = byte_length,
      .normalize_i1 = kind == LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN,
  };
  if (leaf->kind == LOOM_X86_SOURCE_ABI_LEAF_VECTOR) {
    const uint32_t carrier_bytes = loom_x86_source_abi_register_byte_length(
        logical_types[*inout_logical_offset]);
    IREE_ASSERT_NE(carrier_bytes, 0);
    *inout_storage_byte_length = iree_max(*inout_storage_byte_length,
                                          object_byte_offset + carrier_bytes);
  }
  *inout_logical_offset += component_count;
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_plan_objects(
    loom_low_lower_context_t* context,
    const loom_func_c_abi_signature_t* signature, loom_i64_array_t roots,
    const loom_type_t* logical_types, uint16_t logical_count,
    uint16_t maximum_vector_bytes, iree_arena_allocator_t* arena,
    loom_x86_source_abi_object_t** out_objects,
    loom_x86_source_abi_leaf_t** out_leaves, uint16_t* out_leaf_count) {
  *out_objects = NULL;
  *out_leaves = NULL;
  *out_leaf_count = 0;
  if (roots.count > UINT16_MAX) {
    return loom_x86_source_abi_unsupported(
        context, IREE_SV("x86 source ABI object count exceeds 65535"));
  }
  loom_x86_source_abi_object_t* objects = NULL;
  loom_x86_source_abi_leaf_t* leaves = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, roots.count, sizeof(*objects), (void**)&objects));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, logical_count, sizeof(*leaves), (void**)&leaves));
  if (roots.count) {
    memset(objects, 0, roots.count * sizeof(*objects));
  }
  if (logical_count) {
    memset(leaves, 0, logical_count * sizeof(*leaves));
  }

  uint16_t logical_offset = 0;
  uint16_t leaf_count = 0;
  for (uint16_t i = 0; i < roots.count; ++i) {
    const uint32_t root_type_index = (uint32_t)roots.values[i];
    loom_x86_source_abi_object_t* object = &objects[i];
    if (!loom_x86_source_abi_classify(signature, root_type_index,
                                      maximum_vector_bytes,
                                      &object->classification)) {
      return loom_x86_source_abi_unsupported(
          context, IREE_SV("source object is outside the x86-64 SysV family"));
    }
    object->logical_offset = logical_offset;
    object->logical_count = object->classification.component_count;
    object->leaf_offset = leaf_count;
    object->storage_byte_length = object->classification.byte_length;
    IREE_RETURN_IF_ERROR(loom_x86_source_abi_build_leaves(
        signature, root_type_index, /*object_byte_offset=*/0, logical_types,
        logical_count, &logical_offset, leaves, logical_count, &leaf_count,
        &object->storage_byte_length));
    object->leaf_count = leaf_count - object->leaf_offset;
  }
  IREE_ASSERT_EQ(logical_offset, logical_count);
  *out_objects = objects;
  *out_leaves = leaves;
  *out_leaf_count = leaf_count;
  return iree_ok_status();
}

static uint16_t loom_x86_source_abi_direct_carrier_count(
    const loom_x86_source_abi_classification_t* classification,
    uint8_t* out_integer_count, uint8_t* out_sse_count) {
  *out_integer_count = 0;
  *out_sse_count = 0;
  uint16_t carrier_count = 0;
  for (uint8_t i = 0; i < classification->eightbyte_count; ++i) {
    switch (classification->eightbytes[i]) {
      case LOOM_X86_SOURCE_ABI_CLASS_INTEGER:
        ++*out_integer_count;
        ++carrier_count;
        break;
      case LOOM_X86_SOURCE_ABI_CLASS_SSE:
        ++*out_sse_count;
        ++carrier_count;
        break;
      case LOOM_X86_SOURCE_ABI_CLASS_SSEUP:
      case LOOM_X86_SOURCE_ABI_CLASS_NO_CLASS:
        break;
      case LOOM_X86_SOURCE_ABI_CLASS_MEMORY:
        IREE_ASSERT_UNREACHABLE("direct source object has memory eightbyte");
        break;
    }
  }
  return carrier_count;
}

static loom_type_t loom_x86_source_abi_vector_semantic_type(
    uint32_t byte_length) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                             loom_dim_pack_static(byte_length), 0);
}

static iree_status_t loom_x86_source_abi_append_carrier(
    loom_x86_source_abi_plan_builder_t* builder, bool argument,
    loom_x86_source_abi_object_t* object, uint32_t byte_offset,
    uint16_t register_class, loom_type_t semantic_type,
    int64_t register_location, int64_t stack_offset) {
  loom_type_t physical_type;
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      builder->context, register_class, 1, &physical_type));
  const uint32_t register_bytes =
      loom_x86_call_abi_register_byte_length(register_class);
  IREE_ASSERT_NE(register_bytes, 0);
  object->storage_byte_length =
      iree_max(object->storage_byte_length, byte_offset + register_bytes);
  if (argument) {
    const uint16_t index = builder->argument_count++;
    builder->argument_types[index] = physical_type;
    builder->semantic_argument_types[index] = semantic_type;
    builder->argument_registers[index] = register_location;
    builder->argument_offsets[index] = stack_offset;
    builder->plan->argument_carriers[index] = (loom_x86_source_abi_carrier_t){
        .byte_offset = byte_offset,
        .register_class = register_class,
    };
  } else {
    const uint16_t index = builder->result_count++;
    builder->result_types[index] = physical_type;
    builder->semantic_result_types[index] = semantic_type;
    builder->result_registers[index] = register_location;
    builder->plan->result_carriers[index] = (loom_x86_source_abi_carrier_t){
        .byte_offset = byte_offset,
        .register_class = register_class,
    };
  }
  ++object->physical_count;
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_append_direct_object(
    loom_x86_source_abi_plan_builder_t* builder, bool argument,
    loom_x86_source_abi_object_t* object) {
  const loom_x86_source_abi_classification_t* classification =
      &object->classification;
  object->physical_offset =
      argument ? builder->argument_count : builder->result_count;
  for (uint8_t i = 0; i < classification->eightbyte_count; ++i) {
    const loom_x86_source_abi_class_t abi_class = classification->eightbytes[i];
    if (abi_class == LOOM_X86_SOURCE_ABI_CLASS_NO_CLASS ||
        abi_class == LOOM_X86_SOURCE_ABI_CLASS_SSEUP) {
      continue;
    }
    const uint32_t byte_offset = (uint32_t)i * 8u;
    if (abi_class == LOOM_X86_SOURCE_ABI_CLASS_INTEGER) {
      const uint8_t bank_index = argument ? builder->integer_argument_count++
                                          : builder->integer_result_count++;
      const int64_t location =
          argument ? kLoomX86SourceAbiIntegerArgumentRegisters[bank_index]
                   : kLoomX86SourceAbiIntegerResultRegisters[bank_index];
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_append_carrier(
          builder, argument, object, byte_offset, LOOM_X86_REGISTER_CLASS_GPR64,
          loom_type_scalar(LOOM_SCALAR_TYPE_I64), location,
          /*stack_offset=*/-1));
      continue;
    }
    IREE_ASSERT_EQ(abi_class, LOOM_X86_SOURCE_ABI_CLASS_SSE);
    uint8_t run_count = 1;
    while (i + run_count < classification->eightbyte_count &&
           classification->eightbytes[i + run_count] ==
               LOOM_X86_SOURCE_ABI_CLASS_SSEUP) {
      ++run_count;
    }
    const uint32_t run_bytes = (uint32_t)run_count * 8u;
    uint16_t register_class = LOOM_X86_REGISTER_CLASS_XMM;
    if (run_bytes == 32) {
      register_class = LOOM_X86_REGISTER_CLASS_YMM;
    }
    if (run_bytes == 64) {
      register_class = LOOM_X86_REGISTER_CLASS_ZMM;
    }
    const uint32_t remaining_bytes = classification->byte_length - byte_offset;
    const loom_type_t semantic_type =
        run_count > 1
            ? loom_x86_source_abi_vector_semantic_type(run_bytes)
            : loom_type_scalar(remaining_bytes <= 4 ? LOOM_SCALAR_TYPE_F32
                                                    : LOOM_SCALAR_TYPE_F64);
    const uint8_t bank_index =
        argument ? builder->sse_argument_count++ : builder->sse_result_count++;
    IREE_RETURN_IF_ERROR(loom_x86_source_abi_append_carrier(
        builder, argument, object, byte_offset, register_class, semantic_type,
        bank_index, /*stack_offset=*/-1));
    i += run_count - 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_append_memory_argument(
    loom_x86_source_abi_plan_builder_t* builder,
    loom_x86_source_abi_object_t* object) {
  const uint32_t alignment =
      iree_max(8u, object->classification.byte_alignment);
  if (alignment > UINT8_MAX) {
    return loom_x86_source_abi_unsupported(
        builder->context,
        IREE_SV("x86 source ABI stack alignment exceeds 255 bytes"));
  }
  const uint64_t aligned_offset =
      iree_host_align(builder->stack_bytes, alignment);
  const uint64_t object_bytes =
      iree_host_align(object->classification.byte_length, 8u);
  if (aligned_offset + object_bytes > UINT32_MAX) {
    return loom_x86_source_abi_unsupported(
        builder->context,
        IREE_SV("x86 source ABI stack argument area exceeds 32 bits"));
  }
  object->physical_offset = builder->argument_count;
  object->is_memory_argument = true;
  const uint16_t carrier_count =
      (uint16_t)((object->classification.byte_length + 7u) / 8u);
  for (uint16_t i = 0; i < carrier_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_x86_source_abi_append_carrier(
        builder, /*argument=*/true, object, (uint32_t)i * 8u,
        LOOM_X86_REGISTER_CLASS_GPR64, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        /*register_location=*/-1,
        (int64_t)(aligned_offset + (uint64_t)i * 8u)));
  }
  builder->stack_bytes = (uint32_t)(aligned_offset + object_bytes);
  builder->stack_alignment = (uint8_t)iree_max(
      (uint32_t)builder->stack_alignment, iree_max(16u, alignment));
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_allocate_physical_arrays(
    loom_x86_source_abi_plan_builder_t* builder, uint16_t argument_capacity,
    uint16_t result_capacity) {
  iree_arena_allocator_t* arena = builder->arena;
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, argument_capacity, sizeof(*builder->argument_types),
      (void**)&builder->argument_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, result_capacity, sizeof(*builder->result_types),
      (void**)&builder->result_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, argument_capacity, sizeof(*builder->semantic_argument_types),
      (void**)&builder->semantic_argument_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, result_capacity, sizeof(*builder->semantic_result_types),
      (void**)&builder->semantic_result_types));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, argument_capacity, sizeof(*builder->argument_registers),
      (void**)&builder->argument_registers));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, argument_capacity, sizeof(*builder->argument_offsets),
      (void**)&builder->argument_offsets));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, result_capacity, sizeof(*builder->result_registers),
      (void**)&builder->result_registers));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, argument_capacity, sizeof(*builder->plan->argument_carriers),
      (void**)&builder->plan->argument_carriers));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_array(
      arena, result_capacity, sizeof(*builder->plan->result_carriers),
      (void**)&builder->plan->result_carriers));
  return iree_ok_status();
}

iree_status_t loom_x86_plan_source_abi(
    void* user_data, loom_low_lower_context_t* context,
    const loom_func_c_abi_signature_t* signature,
    const loom_type_t* logical_argument_types, uint16_t logical_argument_count,
    const loom_type_t* logical_result_types, uint16_t logical_result_count,
    iree_arena_allocator_t* arena,
    loom_low_lower_source_abi_target_plan_t* out_plan) {
  (void)user_data;
  *out_plan = (loom_low_lower_source_abi_target_plan_t){0};
  const loom_x86_entry_descriptors_t* entry = loom_x86_entry_descriptors(
      loom_low_lower_context_descriptor_set(context));
  if (!entry) {
    return loom_x86_source_abi_unsupported(
        context, IREE_SV("x86 representation has no source ABI vocabulary"));
  }
  if (signature->results.count > 1) {
    return loom_x86_source_abi_unsupported(
        context, IREE_SV("C source ABI requires at most one result object"));
  }

  loom_x86_source_abi_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->entry = entry;
  plan->logical_argument_types = logical_argument_types;
  plan->logical_result_types = logical_result_types;
  plan->argument_object_count = (uint16_t)signature->arguments.count;
  plan->result_object_count = (uint16_t)signature->results.count;
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_plan_objects(
      context, signature, signature->arguments, logical_argument_types,
      logical_argument_count, entry->maximum_simd_byte_length, arena,
      &plan->arguments, &plan->argument_leaves, &plan->argument_leaf_count));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_plan_objects(
      context, signature, signature->results, logical_result_types,
      logical_result_count, entry->maximum_simd_byte_length, arena,
      &plan->results, &plan->result_leaves, &plan->result_leaf_count));

  plan->has_indirect_result = plan->result_object_count != 0 &&
                              plan->results[0].classification.is_memory;
  uint64_t argument_capacity = plan->has_indirect_result ? 1u : 0u;
  for (uint16_t i = 0; i < plan->argument_object_count; ++i) {
    argument_capacity +=
        ((uint64_t)plan->arguments[i].classification.byte_length + 7u) / 8u;
  }
  uint64_t result_capacity = plan->has_indirect_result ? 1u : 0u;
  if (plan->result_object_count != 0 && !plan->has_indirect_result) {
    result_capacity =
        ((uint64_t)plan->results[0].classification.byte_length + 7u) / 8u;
  }
  if (argument_capacity > UINT16_MAX || result_capacity > UINT16_MAX) {
    return loom_x86_source_abi_unsupported(
        context, IREE_SV("x86 source ABI physical signature exceeds 65535"));
  }

  loom_x86_source_abi_plan_builder_t builder = {
      .context = context,
      .signature = signature,
      .arena = arena,
      .plan = plan,
  };
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_allocate_physical_arrays(
      &builder, (uint16_t)argument_capacity, (uint16_t)result_capacity));

  if (plan->has_indirect_result) {
    loom_x86_source_abi_object_t hidden_object = {
        .storage_byte_length = 8,
    };
    IREE_RETURN_IF_ERROR(loom_x86_source_abi_append_carrier(
        &builder, /*argument=*/true, &hidden_object, /*byte_offset=*/0,
        LOOM_X86_REGISTER_CLASS_GPR64, loom_type_buffer(),
        kLoomX86SourceAbiIntegerArgumentRegisters[0], /*stack_offset=*/-1));
    builder.integer_argument_count = 1;
  }

  for (uint16_t i = 0; i < plan->argument_object_count; ++i) {
    loom_x86_source_abi_object_t* object = &plan->arguments[i];
    uint8_t required_integer = 0;
    uint8_t required_sse = 0;
    loom_x86_source_abi_direct_carrier_count(&object->classification,
                                             &required_integer, &required_sse);
    const bool registers_available =
        !object->classification.is_memory &&
        builder.integer_argument_count + required_integer <=
            LOOM_X86_SOURCE_ABI_INTEGER_ARGUMENT_REGISTER_COUNT &&
        builder.sse_argument_count + required_sse <=
            LOOM_X86_SOURCE_ABI_SSE_ARGUMENT_REGISTER_COUNT;
    if (registers_available) {
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_append_direct_object(
          &builder, /*argument=*/true, object));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_x86_source_abi_append_memory_argument(&builder, object));
    }
  }

  if (plan->result_object_count != 0) {
    loom_x86_source_abi_object_t* object = &plan->results[0];
    if (plan->has_indirect_result) {
      object->physical_offset = 0;
      loom_x86_source_abi_object_t hidden_object = {
          .storage_byte_length = 8,
      };
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_append_carrier(
          &builder, /*argument=*/false, &hidden_object, /*byte_offset=*/0,
          LOOM_X86_REGISTER_CLASS_GPR64, loom_type_buffer(),
          kLoomX86SourceAbiIntegerResultRegisters[0], /*stack_offset=*/-1));
    } else {
      uint8_t required_integer = 0;
      uint8_t required_sse = 0;
      loom_x86_source_abi_direct_carrier_count(
          &object->classification, &required_integer, &required_sse);
      IREE_ASSERT_LE(required_integer,
                     LOOM_X86_SOURCE_ABI_RESULT_REGISTER_COUNT);
      IREE_ASSERT_LE(required_sse, LOOM_X86_SOURCE_ABI_RESULT_REGISTER_COUNT);
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_append_direct_object(
          &builder, /*argument=*/false, object));
    }
  }

  if (builder.stack_bytes != 0) {
    builder.stack_bytes =
        (uint32_t)iree_host_align(builder.stack_bytes, builder.stack_alignment);
  }
  loom_named_attr_slice_t abi_layout;
  IREE_RETURN_IF_ERROR(loom_x86_call_abi_layout_make(
      loom_low_lower_context_module(context), builder.semantic_argument_types,
      builder.argument_count, builder.semantic_result_types,
      builder.result_count, builder.argument_registers,
      builder.argument_offsets, builder.result_registers, builder.stack_bytes,
      builder.stack_alignment, &abi_layout));

  const loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  if (plan->has_indirect_result) {
    loom_tied_result_t* tie = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*tie), (void**)&tie));
    *tie = (loom_tied_result_t){
        .result_index = 0,
        .operand_index = 0,
        .has_type_change = false,
    };
    tied_results = tie;
    tied_result_count = 1;
  }
  plan->argument_carrier_count = builder.argument_count;
  plan->result_carrier_count = builder.result_count;
  *out_plan = (loom_low_lower_source_abi_target_plan_t){
      .argument_types = builder.argument_types,
      .argument_count = builder.argument_count,
      .result_types = builder.result_types,
      .result_count = builder.result_count,
      .tied_results = tied_results,
      .tied_result_count = tied_result_count,
      .abi_layout = abi_layout,
      .target_data = plan,
  };
  return iree_ok_status();
}

static loom_location_id_t loom_x86_source_abi_location(
    const loom_op_t* source_op) {
  return source_op ? source_op->location : LOOM_LOCATION_UNKNOWN;
}

static iree_status_t loom_x86_source_abi_emit_packet(
    loom_low_lower_context_t* context, uint32_t descriptor_ref,
    const loom_value_id_t* operands, uint16_t operand_count,
    iree_string_view_t immediate_name, loom_attribute_t immediate,
    loom_type_t result_type, bool tie_first_operand,
    loom_location_id_t location, loom_value_id_t* out_value) {
  if (out_value) {
    *out_value = LOOM_VALUE_ID_INVALID;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ref);
  loom_named_attr_t attr = {.value = immediate};
  const uint16_t attr_count = iree_string_view_is_empty(immediate_name) ? 0 : 1;
  if (attr_count) {
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        loom_low_lower_context_module(context), immediate_name, &attr.name_id));
  }
  const uint16_t result_count =
      loom_type_kind(result_type) == LOOM_TYPE_NONE ? 0 : 1;
  const loom_tied_result_t tie = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* op = NULL;
  if (loom_low_descriptor_layout(descriptor_set, descriptor)->op_kind ==
      LOOM_LOW_DESCRIPTOR_OP_KIND_CONST) {
    IREE_ASSERT_EQ(result_count, 1);
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
        loom_low_lower_context_builder(context), descriptor_set, descriptor,
        loom_make_named_attr_slice(&attr, attr_count), result_type, location,
        &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
        loom_low_lower_context_builder(context), descriptor_set, descriptor,
        /*access_flags=*/0, operands, operand_count,
        loom_make_named_attr_slice(&attr, attr_count),
        result_count ? &result_type : NULL, result_count,
        tie_first_operand ? &tie : NULL, tie_first_operand ? 1 : 0, location,
        &op));
  }
  if (out_value) {
    *out_value = loom_op_results(op)[0];
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_make_register_type(
    loom_low_lower_context_t* context, uint16_t register_class,
    loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, register_class, 1,
                                           out_type);
}

static iree_status_t loom_x86_source_abi_emit_load(
    loom_low_lower_context_t* context,
    const loom_x86_entry_descriptors_t* entry, loom_value_id_t base,
    uint32_t byte_offset, uint32_t byte_length, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const uint32_t descriptor_ref = byte_length == 1   ? entry->load_u8
                                  : byte_length == 2 ? entry->load_u16
                                  : byte_length == 4 ? entry->load_u32
                                                     : entry->load_u64;
  loom_type_t result_type;
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_make_register_type(
      context,
      byte_length <= 4 ? LOOM_X86_REGISTER_CLASS_GPR32
                       : LOOM_X86_REGISTER_CLASS_GPR64,
      &result_type));
  return loom_x86_source_abi_emit_packet(
      context, descriptor_ref, &base, 1, IREE_SV("disp32"),
      loom_attr_i64(byte_offset), result_type, /*tie_first_operand=*/false,
      location, out_value);
}

static iree_status_t loom_x86_source_abi_emit_store(
    loom_low_lower_context_t* context,
    const loom_x86_entry_descriptors_t* entry, loom_value_id_t value,
    loom_value_id_t base, uint32_t byte_offset, uint32_t byte_length,
    loom_location_id_t location) {
  const uint32_t descriptor_ref = byte_length == 1   ? entry->store_u8
                                  : byte_length == 2 ? entry->store_u16
                                  : byte_length == 4 ? entry->store_u32
                                                     : entry->store_u64;
  const loom_value_id_t operands[] = {value, base};
  return loom_x86_source_abi_emit_packet(
      context, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      IREE_SV("disp32"), loom_attr_i64(byte_offset), loom_type_none(),
      /*tie_first_operand=*/false, location, /*out_value=*/NULL);
}

static iree_status_t loom_x86_source_abi_emit_move(
    loom_low_lower_context_t* context,
    const loom_x86_entry_descriptors_t* entry, loom_value_id_t source,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_type_t source_type =
      loom_module_value_type(loom_low_lower_context_module(context), source);
  const uint16_t source_class = loom_low_register_type_class_id(source_type);
  const uint16_t result_class = loom_low_register_type_class_id(result_type);
  if (source_class == result_class) {
    *out_value = source;
    return iree_ok_status();
  }
  uint32_t descriptor_ref = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (source_class == LOOM_X86_REGISTER_CLASS_XMM &&
      result_class == LOOM_X86_REGISTER_CLASS_GPR32) {
    descriptor_ref = entry->move_xmm_u32;
  } else if (source_class == LOOM_X86_REGISTER_CLASS_XMM &&
             result_class == LOOM_X86_REGISTER_CLASS_GPR64) {
    descriptor_ref = entry->move_xmm_u64;
  } else if (source_class == LOOM_X86_REGISTER_CLASS_GPR32 &&
             result_class == LOOM_X86_REGISTER_CLASS_XMM) {
    descriptor_ref = entry->move_u32_xmm;
  } else if (source_class == LOOM_X86_REGISTER_CLASS_GPR64 &&
             result_class == LOOM_X86_REGISTER_CLASS_XMM) {
    descriptor_ref = entry->move_u64_xmm;
  } else {
    IREE_ASSERT_UNREACHABLE("unsupported x86 source ABI register move");
  }
  return loom_x86_source_abi_emit_packet(
      context, descriptor_ref, &source, 1, iree_string_view_empty(),
      loom_attr_absent(), result_type, /*tie_first_operand=*/false, location,
      out_value);
}

static iree_status_t loom_x86_source_abi_emit_pointer_address(
    loom_low_lower_context_t* context,
    const loom_x86_entry_descriptors_t* entry, loom_value_id_t base,
    loom_value_id_t byte_offset, loom_location_id_t location,
    loom_value_id_t* out_address) {
  const loom_type_t base_type =
      loom_module_value_type(loom_low_lower_context_module(context), base);
  loom_op_t* copy = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_copy_build(loom_low_lower_context_builder(context), base,
                          /*detached=*/false, base_type, location, &copy));
  const loom_value_id_t operands[] = {loom_low_copy_result(copy), byte_offset};
  return loom_x86_source_abi_emit_packet(
      context, entry->add_u64, operands, IREE_ARRAYSIZE(operands),
      iree_string_view_empty(), loom_attr_absent(), base_type,
      /*tie_first_operand=*/true, location, out_address);
}

static iree_status_t loom_x86_source_abi_emit_zero_offset(
    loom_low_lower_context_t* context,
    const loom_x86_entry_descriptors_t* entry, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  return loom_x86_source_abi_emit_packet(
      context, entry->constant_u64, /*operands=*/NULL, /*operand_count=*/0,
      IREE_SV("imm64"), loom_attr_i64(0), result_type,
      /*tie_first_operand=*/false, location, out_value);
}

static iree_status_t loom_x86_source_abi_normalize_i1(
    loom_low_lower_context_t* context,
    const loom_x86_entry_descriptors_t* entry, loom_value_id_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_value_id_t operands[] = {value};
  return loom_x86_source_abi_emit_packet(
      context, entry->mask_i1, operands, IREE_ARRAYSIZE(operands),
      IREE_SV("imm32"), loom_attr_i64(1), result_type,
      /*tie_first_operand=*/true, location, out_value);
}

static iree_status_t loom_x86_source_abi_reserve_storage(
    loom_low_lower_context_t* context,
    const loom_x86_source_abi_object_t* object, loom_location_id_t location,
    loom_value_id_t* out_storage, loom_value_id_t* out_address) {
  loom_builder_t* ir = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      ir, object->storage_byte_length, object->classification.byte_alignment,
      loom_type_storage(LOOM_STORAGE_SPACE_PRIVATE), location, &storage_op));
  *out_storage = loom_low_storage_reserve_storage(storage_op);
  loom_type_t address_type;
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_GPR64, &address_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      ir, *out_storage, /*offset=*/0, address_type, location, &address_op));
  *out_address = loom_low_storage_address_result(address_op);
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_emit_vector_store(
    loom_low_lower_context_t* context,
    const loom_x86_entry_descriptors_t* entry, loom_value_id_t value,
    loom_value_id_t base, uint32_t byte_offset, loom_location_id_t location) {
  const loom_type_t type =
      loom_module_value_type(loom_low_lower_context_module(context), value);
  const uint16_t register_class = loom_low_register_type_class_id(type);
  const uint32_t descriptor_ref =
      register_class == LOOM_X86_REGISTER_CLASS_XMM   ? entry->store_128_xmm
      : register_class == LOOM_X86_REGISTER_CLASS_YMM ? entry->store_256_ymm
                                                      : entry->store_512_zmm;
  const loom_value_id_t operands[] = {value, base};
  return loom_x86_source_abi_emit_packet(
      context, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      IREE_SV("disp32"), loom_attr_i64(byte_offset), loom_type_none(),
      /*tie_first_operand=*/false, location, /*out_value=*/NULL);
}

static iree_status_t loom_x86_source_abi_pack_leaves(
    loom_low_lower_context_t* context, const loom_x86_source_abi_plan_t* plan,
    const loom_x86_source_abi_object_t* object,
    const loom_x86_source_abi_leaf_t* leaves,
    const loom_value_id_t* logical_values, loom_value_id_t address,
    loom_location_id_t location) {
  for (uint16_t i = 0; i < object->leaf_count; ++i) {
    const loom_x86_source_abi_leaf_t* leaf = &leaves[object->leaf_offset + i];
    if (leaf->kind == LOOM_X86_SOURCE_ABI_LEAF_VECTOR) {
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_emit_vector_store(
          context, plan->entry, logical_values[leaf->logical_offset], address,
          leaf->byte_offset, location));
      continue;
    }
    loom_value_id_t bits = logical_values[leaf->logical_offset];
    uint32_t byte_length = leaf->byte_length;
    if (leaf->kind == LOOM_X86_SOURCE_ABI_LEAF_POINTER) {
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_emit_pointer_address(
          context, plan->entry, bits, logical_values[leaf->logical_offset + 1],
          location, &bits));
      byte_length = 8;
    } else {
      loom_type_t gpr_type;
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_make_register_type(
          context,
          byte_length <= 4 ? LOOM_X86_REGISTER_CLASS_GPR32
                           : LOOM_X86_REGISTER_CLASS_GPR64,
          &gpr_type));
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_emit_move(
          context, plan->entry, bits, gpr_type, location, &bits));
    }
    IREE_RETURN_IF_ERROR(loom_x86_source_abi_emit_store(
        context, plan->entry, bits, address, leaf->byte_offset, byte_length,
        location));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_unpack_leaves(
    loom_low_lower_context_t* context, const loom_x86_source_abi_plan_t* plan,
    const loom_x86_source_abi_object_t* object,
    const loom_x86_source_abi_leaf_t* leaves, const loom_type_t* logical_types,
    loom_value_id_t storage, loom_value_id_t address,
    loom_value_id_t* logical_values, loom_location_id_t location) {
  for (uint16_t i = 0; i < object->leaf_count; ++i) {
    const loom_x86_source_abi_leaf_t* leaf = &leaves[object->leaf_offset + i];
    if (leaf->kind == LOOM_X86_SOURCE_ABI_LEAF_VECTOR) {
      loom_op_t* reload = NULL;
      IREE_RETURN_IF_ERROR(loom_low_reload_build(
          loom_low_lower_context_builder(context), storage, leaf->byte_offset,
          logical_types[leaf->logical_offset], location, &reload));
      logical_values[leaf->logical_offset] = loom_low_reload_result(reload);
      continue;
    }
    const uint32_t byte_length =
        leaf->kind == LOOM_X86_SOURCE_ABI_LEAF_POINTER ? 8 : leaf->byte_length;
    loom_value_id_t bits;
    IREE_RETURN_IF_ERROR(loom_x86_source_abi_emit_load(
        context, plan->entry, address, leaf->byte_offset, byte_length, location,
        &bits));
    if (leaf->kind == LOOM_X86_SOURCE_ABI_LEAF_POINTER) {
      logical_values[leaf->logical_offset] = bits;
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_emit_zero_offset(
          context, plan->entry, logical_types[leaf->logical_offset + 1],
          location, &logical_values[leaf->logical_offset + 1]));
      continue;
    }
    if (leaf->normalize_i1) {
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_normalize_i1(
          context, plan->entry, bits, logical_types[leaf->logical_offset],
          location, &bits));
    } else {
      IREE_RETURN_IF_ERROR(loom_x86_source_abi_emit_move(
          context, plan->entry, bits, logical_types[leaf->logical_offset],
          location, &bits));
    }
    logical_values[leaf->logical_offset] = bits;
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_pack_object(
    loom_low_lower_context_t* context, const loom_x86_source_abi_plan_t* plan,
    const loom_x86_source_abi_object_t* object,
    const loom_x86_source_abi_leaf_t* leaves,
    const loom_value_id_t* logical_values,
    const loom_x86_source_abi_carrier_t* carriers,
    loom_value_id_t* physical_values, loom_location_id_t location) {
  loom_value_id_t storage;
  loom_value_id_t address;
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_reserve_storage(
      context, object, location, &storage, &address));
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_pack_leaves(
      context, plan, object, leaves, logical_values, address, location));
  for (uint16_t i = 0; i < object->physical_count; ++i) {
    const uint16_t physical_index = object->physical_offset + i;
    loom_type_t physical_type;
    IREE_RETURN_IF_ERROR(loom_x86_source_abi_make_register_type(
        context, carriers[physical_index].register_class, &physical_type));
    loom_op_t* reload = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_reload_build(loom_low_lower_context_builder(context), storage,
                              carriers[physical_index].byte_offset,
                              physical_type, location, &reload));
    physical_values[physical_index] = loom_low_reload_result(reload);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_source_abi_unpack_object(
    loom_low_lower_context_t* context, const loom_x86_source_abi_plan_t* plan,
    const loom_x86_source_abi_object_t* object,
    const loom_x86_source_abi_leaf_t* leaves, const loom_type_t* logical_types,
    const loom_x86_source_abi_carrier_t* carriers,
    const loom_value_id_t* physical_values, loom_value_id_t* logical_values,
    loom_location_id_t location) {
  loom_value_id_t storage;
  loom_value_id_t address;
  IREE_RETURN_IF_ERROR(loom_x86_source_abi_reserve_storage(
      context, object, location, &storage, &address));
  for (uint16_t i = 0; i < object->physical_count; ++i) {
    const uint16_t physical_index = object->physical_offset + i;
    loom_op_t* spill = NULL;
    IREE_RETURN_IF_ERROR(loom_low_spill_build(
        loom_low_lower_context_builder(context),
        physical_values[physical_index], storage,
        carriers[physical_index].byte_offset, location, &spill));
  }
  return loom_x86_source_abi_unpack_leaves(context, plan, object, leaves,
                                           logical_types, storage, address,
                                           logical_values, location);
}

static iree_status_t loom_x86_source_abi_state_allocate(
    iree_arena_allocator_t* arena,
    loom_x86_source_abi_emission_state_t** out_state) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(**out_state), (void**)out_state));
  **out_state = (loom_x86_source_abi_emission_state_t){
      .result_storage = LOOM_VALUE_ID_INVALID,
      .result_address = LOOM_VALUE_ID_INVALID,
  };
  return iree_ok_status();
}

iree_status_t loom_x86_emit_source_abi(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_source_abi_emission_phase_t phase,
    const loom_op_t* source_op, const void* target_plan,
    const loom_value_id_t* input_values, uint16_t input_count,
    loom_value_id_t* output_values, uint16_t output_count,
    iree_arena_allocator_t* state_arena, void** inout_state) {
  (void)user_data;
  const loom_x86_source_abi_plan_t* plan = target_plan;
  const loom_location_id_t location = loom_x86_source_abi_location(source_op);
  switch (phase) {
    case LOOM_LOW_LOWER_SOURCE_ABI_EMISSION_DEFINITION_ENTRY: {
      IREE_ASSERT_EQ(input_count, plan->argument_carrier_count);
      uint16_t logical_count = 0;
      for (uint16_t i = 0; i < plan->argument_object_count; ++i) {
        const loom_x86_source_abi_object_t* object = &plan->arguments[i];
        IREE_RETURN_IF_ERROR(loom_x86_source_abi_unpack_object(
            context, plan, object, plan->argument_leaves,
            plan->logical_argument_types, plan->argument_carriers, input_values,
            output_values, location));
        logical_count += object->logical_count;
      }
      IREE_ASSERT_EQ(output_count, logical_count);
      if (plan->has_indirect_result) {
        loom_x86_source_abi_emission_state_t* state = NULL;
        IREE_RETURN_IF_ERROR(
            loom_x86_source_abi_state_allocate(state_arena, &state));
        state->result_address = input_values[0];
        *inout_state = state;
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_SOURCE_ABI_EMISSION_DEFINITION_EXIT: {
      IREE_ASSERT_EQ(input_count, plan->result_object_count == 0
                                      ? 0
                                      : plan->results[0].logical_count);
      IREE_ASSERT_EQ(output_count, plan->result_carrier_count);
      if (plan->result_object_count == 0) {
        return iree_ok_status();
      }
      const loom_x86_source_abi_object_t* object = &plan->results[0];
      if (plan->has_indirect_result) {
        const loom_x86_source_abi_emission_state_t* state = *inout_state;
        IREE_ASSERT_NE(state, NULL);
        IREE_RETURN_IF_ERROR(loom_x86_source_abi_pack_leaves(
            context, plan, object, plan->result_leaves, input_values,
            state->result_address, location));
        output_values[0] = state->result_address;
        return iree_ok_status();
      }
      return loom_x86_source_abi_pack_object(
          context, plan, object, plan->result_leaves, input_values,
          plan->result_carriers, output_values, location);
    }
    case LOOM_LOW_LOWER_SOURCE_ABI_EMISSION_CALL_ARGUMENTS: {
      IREE_ASSERT_EQ(input_count,
                     plan->argument_object_count == 0
                         ? 0
                         : plan->arguments[plan->argument_object_count - 1]
                                   .logical_offset +
                               plan->arguments[plan->argument_object_count - 1]
                                   .logical_count);
      IREE_ASSERT_EQ(output_count, plan->argument_carrier_count);
      if (plan->has_indirect_result) {
        loom_x86_source_abi_emission_state_t* state = NULL;
        IREE_RETURN_IF_ERROR(
            loom_x86_source_abi_state_allocate(state_arena, &state));
        IREE_RETURN_IF_ERROR(loom_x86_source_abi_reserve_storage(
            context, &plan->results[0], location, &state->result_storage,
            &state->result_address));
        output_values[0] = state->result_address;
        *inout_state = state;
      }
      for (uint16_t i = 0; i < plan->argument_object_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_x86_source_abi_pack_object(
            context, plan, &plan->arguments[i], plan->argument_leaves,
            input_values, plan->argument_carriers, output_values, location));
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_SOURCE_ABI_EMISSION_CALL_RESULTS: {
      IREE_ASSERT_EQ(input_count, plan->result_carrier_count);
      IREE_ASSERT_EQ(output_count, plan->result_object_count == 0
                                       ? 0
                                       : plan->results[0].logical_count);
      if (plan->result_object_count == 0) {
        return iree_ok_status();
      }
      const loom_x86_source_abi_object_t* object = &plan->results[0];
      if (plan->has_indirect_result) {
        loom_x86_source_abi_emission_state_t* state = *inout_state;
        IREE_ASSERT_NE(state, NULL);
        state->result_address = input_values[0];
        return loom_x86_source_abi_unpack_leaves(
            context, plan, object, plan->result_leaves,
            plan->logical_result_types, state->result_storage,
            state->result_address, output_values, location);
      }
      return loom_x86_source_abi_unpack_object(
          context, plan, object, plan->result_leaves,
          plan->logical_result_types, plan->result_carriers, input_values,
          output_values, location);
    }
  }
  IREE_ASSERT_UNREACHABLE("unknown x86 source ABI emission phase");
}
