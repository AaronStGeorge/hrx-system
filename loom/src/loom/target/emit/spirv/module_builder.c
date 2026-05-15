// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_builder.h"

#include "loom/target/emit/spirv/binary_format.h"

void loom_spirv_module_binary_deinitialize(loom_spirv_module_binary_t* module,
                                           iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(module);
  iree_allocator_free(allocator, module->words);
  *module = (loom_spirv_module_binary_t){0};
}

static iree_status_t loom_spirv_module_builder_validate_target(
    const loom_target_bundle_t* target) {
  if (target->snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_SPIRV) {
    iree_string_view_t actual =
        loom_target_codegen_format_name(target->snapshot->codegen_format);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target '%.*s' uses codegen format '%.*s', "
                            "expected 'spirv'",
                            (int)target->name.size, target->name.data,
                            (int)actual.size, actual.data);
  }
  if (target->snapshot->artifact_format !=
      LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY) {
    iree_string_view_t actual =
        loom_target_artifact_format_name(target->snapshot->artifact_format);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target '%.*s' emits artifact format '%.*s', "
                            "expected 'spirv_binary'",
                            (int)target->name.size, target->name.data,
                            (int)actual.size, actual.data);
  }
  if (target->export_plan->abi_kind != LOOM_TARGET_ABI_SHADER_ENTRY_POINT) {
    iree_string_view_t actual =
        loom_target_abi_kind_name(target->export_plan->abi_kind);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target '%.*s' uses export ABI '%.*s', expected "
                            "'shader_entry_point'",
                            (int)target->name.size, target->name.data,
                            (int)actual.size, actual.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_module_builder_emit_feature_preamble(
    loom_spirv_module_builder_t* builder) {
  if (!loom_spirv_feature_set_has_atom(&builder->feature_set,
                                       LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V shader module emission requires the Vulkan shader feature");
  }

  loom_spirv_binary_writer_t* capability_writer =
      &builder->sections[LOOM_SPIRV_MODULE_SECTION_CAPABILITY];
  for (uint8_t i = 0; i < builder->feature_set.capability_count; ++i) {
    const uint32_t operands[] = {builder->feature_set.capabilities[i]};
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        capability_writer, LOOM_SPIRV_OP_CAPABILITY, operands,
        IREE_ARRAYSIZE(operands)));
  }

  loom_spirv_binary_writer_t* extension_writer =
      &builder->sections[LOOM_SPIRV_MODULE_SECTION_EXTENSION];
  for (uint8_t i = 0; i < builder->feature_set.extension_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_instruction(
        extension_writer, LOOM_SPIRV_OP_EXTENSION, NULL, 0,
        builder->feature_set.extension_names[i], NULL, 0));
  }

  loom_spirv_binary_writer_t* memory_model_writer =
      &builder->sections[LOOM_SPIRV_MODULE_SECTION_MEMORY_MODEL];
  const uint32_t memory_model_operands[] = {
      builder->feature_set.addressing_model,
      builder->feature_set.memory_model,
  };
  return loom_spirv_binary_write_instruction(
      memory_model_writer, LOOM_SPIRV_OP_MEMORY_MODEL, memory_model_operands,
      IREE_ARRAYSIZE(memory_model_operands));
}

iree_status_t loom_spirv_module_builder_initialize(
    const loom_target_bundle_t* target, iree_allocator_t allocator,
    loom_spirv_module_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_builder);

  IREE_RETURN_IF_ERROR(loom_spirv_module_builder_validate_target(target));

  *out_builder = (loom_spirv_module_builder_t){
      .allocator = allocator,
      .id_bound = 1,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(out_builder->sections); ++i) {
    loom_spirv_binary_writer_initialize(allocator, &out_builder->sections[i]);
  }

  const loom_spirv_feature_bits_t feature_bits =
      (loom_spirv_feature_bits_t)target->config->contract_feature_bits;
  iree_status_t status = loom_spirv_feature_set_prepare(
      target->name, feature_bits, &out_builder->feature_set);
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_builder_emit_feature_preamble(out_builder);
  }
  if (!iree_status_is_ok(status)) {
    loom_spirv_module_builder_deinitialize(out_builder);
  }
  return status;
}

void loom_spirv_module_builder_deinitialize(
    loom_spirv_module_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(builder->sections); ++i) {
    loom_spirv_binary_writer_deinitialize(&builder->sections[i]);
  }
  *builder = (loom_spirv_module_builder_t){0};
}

loom_spirv_binary_writer_t* loom_spirv_module_builder_section(
    loom_spirv_module_builder_t* builder, loom_spirv_module_section_t section) {
  IREE_ASSERT_ARGUMENT(builder);
  const uint32_t section_index = (uint32_t)section;
  IREE_ASSERT(section_index < LOOM_SPIRV_MODULE_SECTION_COUNT);
  return &builder->sections[section_index];
}

uint32_t loom_spirv_module_builder_allocate_id(
    loom_spirv_module_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT(builder->id_bound != UINT32_MAX);
  return builder->id_bound++;
}

void loom_spirv_module_builder_require_id_bound(
    loom_spirv_module_builder_t* builder, uint32_t id_bound) {
  IREE_ASSERT_ARGUMENT(builder);
  builder->id_bound = iree_max(builder->id_bound, id_bound);
}

static iree_status_t loom_spirv_module_builder_write_header(
    const loom_spirv_module_builder_t* builder,
    loom_spirv_binary_writer_t* module_writer) {
  const uint32_t header[] = {
      LOOM_SPIRV_MAGIC_NUMBER,    builder->feature_set.minimum_spirv_version,
      LOOM_SPIRV_GENERATOR_LOOM,  builder->id_bound,
      LOOM_SPIRV_SCHEMA_RESERVED,
  };
  return loom_spirv_binary_write_words(module_writer, header,
                                       IREE_ARRAYSIZE(header));
}

iree_status_t loom_spirv_module_builder_finalize(
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_module);

  *out_module = (loom_spirv_module_binary_t){0};
  loom_spirv_binary_writer_t module_writer;
  loom_spirv_binary_writer_initialize(builder->allocator, &module_writer);
  iree_status_t status =
      loom_spirv_module_builder_write_header(builder, &module_writer);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(builder->sections) && iree_status_is_ok(status);
       ++i) {
    status = loom_spirv_binary_write_words(&module_writer,
                                           builder->sections[i].words,
                                           builder->sections[i].word_count);
  }
  if (iree_status_is_ok(status)) {
    loom_spirv_binary_writer_steal_words(&module_writer, &out_module->words,
                                         &out_module->word_count);
  }
  loom_spirv_binary_writer_deinitialize(&module_writer);
  return status;
}
