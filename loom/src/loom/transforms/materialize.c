// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/materialize.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"

static iree_status_t loom_ir_clone_value_name(loom_ir_remap_t* remap,
                                              loom_value_id_t source_value_id,
                                              loom_value_id_t target_value_id) {
  const loom_value_t* source_value =
      loom_module_value(remap->source_module, source_value_id);
  loom_value_t* target_value =
      loom_module_value(remap->target_module, target_value_id);
  return loom_ir_remap_string_id(remap, source_value->name_id,
                                 /*allow_invalid=*/true,
                                 &target_value->name_id);
}

static iree_status_t loom_ir_clone_block_label(loom_ir_remap_t* remap,
                                               const loom_block_t* source_block,
                                               loom_block_t* target_block) {
  return loom_ir_remap_string_id(remap, source_block->label_id,
                                 /*allow_invalid=*/true,
                                 &target_block->label_id);
}

static iree_status_t loom_ir_clone_block_args(loom_ir_remap_t* remap,
                                              const loom_block_t* source_block,
                                              loom_block_t* target_block) {
  if (source_block->arg_count == 0) return iree_ok_status();

  loom_value_id_t* target_args = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(remap->arena, source_block->arg_count,
                                sizeof(loom_value_id_t), (void**)&target_args));
  for (uint16_t i = 0; i < source_block->arg_count; ++i) {
    loom_value_id_t target_arg = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        remap->target_module, loom_type_none(), &target_arg));
    IREE_RETURN_IF_ERROR(
        loom_block_add_arg(remap->target_module, target_block, target_arg));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, loom_block_arg_id(source_block, i), target_arg));
    IREE_RETURN_IF_ERROR(loom_ir_clone_value_name(
        remap, loom_block_arg_id(source_block, i), target_arg));
    target_args[i] = target_arg;
  }
  for (uint16_t i = 0; i < source_block->arg_count; ++i) {
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap,
        loom_module_value_type(remap->source_module,
                               loom_block_arg_id(source_block, i)),
        &target_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        remap->target_module, target_args[i], target_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_results(loom_ir_remap_t* remap,
                                              const loom_op_t* source_op,
                                              loom_value_id_t* target_results) {
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    target_results[i] = LOOM_VALUE_ID_INVALID;
    if (source_results[i] == LOOM_VALUE_ID_INVALID) continue;
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        remap->target_module, loom_type_none(), &target_results[i]));
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(remap, source_results[i], target_results[i]));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_value_name(remap, source_results[i], target_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_operands(
    loom_ir_remap_t* remap, const loom_op_t* source_op,
    loom_value_id_t* target_operands) {
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    if (source_operands[i] == LOOM_VALUE_ID_INVALID) {
      target_operands[i] = LOOM_VALUE_ID_INVALID;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(remap, source_operands[i],
                                                     &target_operands[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_result_types(
    loom_ir_remap_t* remap, const loom_op_t* source_op,
    loom_value_id_t* target_results, loom_type_t* target_result_types) {
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    if (source_results[i] == LOOM_VALUE_ID_INVALID) {
      target_result_types[i] = loom_type_none();
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap, loom_module_value_type(remap->source_module, source_results[i]),
        &target_result_types[i]));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        remap->target_module, target_results[i], target_result_types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_attrs(loom_ir_remap_t* remap,
                                            const loom_op_t* source_op,
                                            loom_attribute_t* target_attrs) {
  const loom_attribute_t* source_attrs = loom_op_const_attrs(source_op);
  for (uint8_t i = 0; i < source_op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_attribute(remap, source_attrs[i], &target_attrs[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_ir_clone_op_regions(loom_builder_t* builder,
                                              const loom_op_t* source_op,
                                              loom_ir_remap_t* remap,
                                              loom_op_t* target_op) {
  loom_region_t** target_regions = loom_op_regions(target_op);
  loom_region_t* const* source_regions = loom_op_regions(source_op);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0; i < source_op->region_count && iree_status_is_ok(status);
       ++i) {
    target_regions[i] = NULL;
    if (!source_regions[i]) continue;
    builder->ip.parent_op = target_op;
    status = loom_ir_clone_region(builder, source_regions[i], remap,
                                  &target_regions[i]);
    if (!iree_status_is_ok(status)) continue;
    target_regions[i]->flags = source_regions[i]->flags;
    for (uint16_t block_index = 0; block_index < target_regions[i]->block_count;
         ++block_index) {
      target_regions[i]->blocks[block_index]->parent_region = target_regions[i];
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

iree_status_t loom_ir_clone_op(loom_builder_t* builder,
                               const loom_op_t* source_op,
                               loom_ir_remap_t* remap,
                               loom_op_t** out_cloned_op) {
  if (!builder || !source_op || !remap || !out_cloned_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "builder, source op, remap, and cloned op output are required");
  }
  *out_cloned_op = NULL;
  if (builder->module != remap->target_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "builder module must be the remap target module");
  }
  if (iree_any_bit_set(source_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot clone a dead op");
  }
  if (!loom_op_vtable(remap->target_module, source_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target module has no vtable for source op kind");
  }

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_location_id(remap, source_op->location, &target_location));

  loom_value_id_t* target_operands = NULL;
  if (source_op->operand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->operand_count, sizeof(loom_value_id_t),
        (void**)&target_operands));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op_operands(remap, source_op, target_operands));
  }

  loom_value_id_t* target_results = NULL;
  loom_type_t* target_result_types = NULL;
  if (source_op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->result_count, sizeof(loom_value_id_t),
        (void**)&target_results));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->result_count, sizeof(loom_type_t),
        (void**)&target_result_types));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op_results(remap, source_op, target_results));
    IREE_RETURN_IF_ERROR(loom_ir_clone_op_result_types(
        remap, source_op, target_results, target_result_types));
  }

  loom_attribute_t* target_attrs = NULL;
  if (source_op->attribute_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        remap->arena, source_op->attribute_count, sizeof(loom_attribute_t),
        (void**)&target_attrs));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op_attrs(remap, source_op, target_attrs));
  }

  loom_op_t* target_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, source_op->kind, source_op->operand_count,
      source_op->result_count, source_op->region_count,
      source_op->tied_result_count, source_op->attribute_count, target_location,
      &target_op));
  target_op->instance_flags = source_op->instance_flags;
  if (source_op->operand_count > 0) {
    memcpy(
        loom_op_operands(target_op), target_operands,
        (iree_host_size_t)source_op->operand_count * sizeof(loom_value_id_t));
  }
  if (source_op->result_count > 0) {
    memcpy(loom_op_results(target_op), target_results,
           (iree_host_size_t)source_op->result_count * sizeof(loom_value_id_t));
  }
  if (source_op->tied_result_count > 0) {
    memcpy(loom_op_tied_results(target_op), loom_op_tied_results(source_op),
           (iree_host_size_t)source_op->tied_result_count *
               sizeof(loom_tied_result_t));
  }
  if (source_op->attribute_count > 0) {
    memcpy(loom_op_attrs(target_op), target_attrs,
           (iree_host_size_t)source_op->attribute_count *
               sizeof(loom_attribute_t));
  }

  IREE_RETURN_IF_ERROR(
      loom_ir_clone_op_regions(builder, source_op, remap, target_op));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, target_op));
  *out_cloned_op = target_op;
  return iree_ok_status();
}

iree_status_t loom_ir_clone_block_ops(
    loom_builder_t* builder, const loom_block_t* source_block,
    loom_ir_remap_t* remap, const loom_ir_clone_block_options_t* options) {
  if (!builder || !source_block || !remap) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "builder, source block, and remap are required");
  }
  bool omit_terminators = options ? options->omit_terminators : false;
  for (const loom_op_t* source_op = source_block->first_op; source_op;
       source_op = source_op->next_op) {
    const loom_op_vtable_t* vtable =
        loom_op_vtable(remap->source_module, source_op);
    if (omit_terminators && vtable &&
        iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR)) {
      continue;
    }
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op(builder, source_op, remap, &cloned_op));
  }
  return iree_ok_status();
}

iree_status_t loom_ir_clone_region(loom_builder_t* builder,
                                   const loom_region_t* source_region,
                                   loom_ir_remap_t* remap,
                                   loom_region_t** out_target_region) {
  if (!builder || !source_region || !remap || !out_target_region) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "builder, source region, remap, and target region output are required");
  }
  *out_target_region = NULL;
  loom_region_t* target_region = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate_region(
      remap->target_module, source_region->block_count, &target_region));
  target_region->flags = source_region->flags;

  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_region, block_index);
    loom_block_t* target_block = loom_region_block(target_region, block_index);
    target_block->flags = source_block->flags;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_block_label(remap, source_block, target_block));
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_block_args(remap, source_block, target_block));
  }

  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_op_t* parent_op = builder->ip.parent_op;
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < source_region->block_count && iree_status_is_ok(status);
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_region, block_index);
    loom_block_t* target_block = loom_region_block(target_region, block_index);
    builder->ip.block = target_block;
    builder->ip.parent_op = parent_op;
    builder->ip.before_op = NULL;
    status = loom_ir_clone_block_ops(builder, source_block, remap,
                                     /*options=*/NULL);
  }
  loom_builder_restore(builder, saved_ip);

  IREE_RETURN_IF_ERROR(status);
  *out_target_region = target_region;
  return iree_ok_status();
}
