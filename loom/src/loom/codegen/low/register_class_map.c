// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/register_class_map.h"

#include <inttypes.h>

#include "loom/ir/module.h"

static iree_status_t loom_low_register_class_check_descriptor_count(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set->reg_class_count > LOOM_LOW_REG_CLASS_NONE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low descriptor set has %" PRIu32
        " register classes but descriptor register-class IDs are 16-bit",
        descriptor_set->reg_class_count);
  }
  return iree_ok_status();
}

iree_status_t loom_low_register_class_map_initialize(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, loom_low_register_class_map_t* out_map) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_map);
  *out_map = (loom_low_register_class_map_t){
      .module = module,
      .descriptor_set = descriptor_set,
      .descriptor_register_class_id_count = module->strings.count,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_register_class_check_descriptor_count(descriptor_set));
  if (module->strings.count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->strings.count,
      sizeof(*out_map->descriptor_register_class_ids),
      (void**)&out_map->descriptor_register_class_ids));
  for (iree_host_size_t i = 0; i < module->strings.count; ++i) {
    out_map->descriptor_register_class_ids[i] = LOOM_LOW_REG_CLASS_NONE;
  }
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* register_class =
        &descriptor_set->reg_classes[i];
    iree_string_view_t descriptor_register_class_name =
        loom_low_descriptor_set_string(descriptor_set,
                                       register_class->name_string_offset);
    for (iree_host_size_t string_id = 0; string_id < module->strings.count;
         ++string_id) {
      if (!iree_string_view_equal(module->strings.entries[string_id],
                                  descriptor_register_class_name)) {
        continue;
      }
      out_map->descriptor_register_class_ids[string_id] = (uint16_t)i;
      break;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_register_class_map_try_resolve_string_id(
    const loom_low_register_class_map_t* map,
    loom_string_id_t register_class_string_id,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class,
    bool* out_found) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(out_descriptor_register_class_id);
  IREE_ASSERT_ARGUMENT(out_found);
  *out_descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (out_descriptor_register_class) {
    *out_descriptor_register_class = NULL;
  }
  *out_found = false;
  if (register_class_string_id == LOOM_STRING_ID_INVALID ||
      register_class_string_id >= map->descriptor_register_class_id_count) {
    return iree_ok_status();
  }
  const uint16_t descriptor_register_class_id =
      map->descriptor_register_class_ids[register_class_string_id];
  if (descriptor_register_class_id == LOOM_LOW_REG_CLASS_NONE ||
      descriptor_register_class_id >= map->descriptor_set->reg_class_count) {
    return iree_ok_status();
  }
  *out_descriptor_register_class_id = descriptor_register_class_id;
  if (out_descriptor_register_class) {
    *out_descriptor_register_class =
        &map->descriptor_set->reg_classes[descriptor_register_class_id];
  }
  *out_found = true;
  return iree_ok_status();
}

iree_status_t loom_low_register_class_map_try_resolve_type(
    const loom_low_register_class_map_t* map, loom_type_t type,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class,
    bool* out_found) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(out_descriptor_register_class_id);
  IREE_ASSERT_ARGUMENT(out_found);
  *out_descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (out_descriptor_register_class) {
    *out_descriptor_register_class = NULL;
  }
  *out_found = false;
  if (!loom_type_is_register(type)) {
    return iree_ok_status();
  }
  return loom_low_register_class_map_try_resolve_string_id(
      map, loom_type_register_class_id(type), out_descriptor_register_class_id,
      out_descriptor_register_class, out_found);
}

iree_status_t loom_low_register_class_try_lookup_name(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t register_class_name,
    uint16_t* out_descriptor_register_class_id,
    const loom_low_reg_class_t** out_descriptor_register_class,
    bool* out_found) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(out_descriptor_register_class_id);
  IREE_ASSERT_ARGUMENT(out_found);
  *out_descriptor_register_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (out_descriptor_register_class) {
    *out_descriptor_register_class = NULL;
  }
  *out_found = false;
  IREE_RETURN_IF_ERROR(
      loom_low_register_class_check_descriptor_count(descriptor_set));
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* register_class =
        &descriptor_set->reg_classes[i];
    iree_string_view_t descriptor_register_class_name =
        loom_low_descriptor_set_string(descriptor_set,
                                       register_class->name_string_offset);
    if (!iree_string_view_equal(register_class_name,
                                descriptor_register_class_name)) {
      continue;
    }
    *out_descriptor_register_class_id = (uint16_t)i;
    if (out_descriptor_register_class) {
      *out_descriptor_register_class = register_class;
    }
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}
