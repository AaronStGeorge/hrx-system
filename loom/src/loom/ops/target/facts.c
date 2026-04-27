// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"

const uint8_t loom_target_profile_preset_registry_resource_key = 0;

static iree_status_t loom_target_symbol_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target symbol %.*s string id %u is invalid",
                            (int)field_name.size, field_name.data,
                            (uint32_t)string_id);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_target_profile_string_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  if (attr->kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile override %.*s must be a string",
                            (int)field_name.size, field_name.data);
  }
  return loom_target_symbol_string_from_id(
      module, loom_attr_as_string_id(*attr), field_name, out_string);
}

static iree_status_t loom_target_profile_u32_attr(const loom_attribute_t* attr,
                                                  iree_string_view_t field_name,
                                                  uint32_t minimum_value,
                                                  uint32_t maximum_value,
                                                  uint32_t* out_value) {
  if (attr->kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile override %.*s must be an integer",
                            (int)field_name.size, field_name.data);
  }
  int64_t value = loom_attr_as_i64(*attr);
  if (value < minimum_value || value > maximum_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile override %.*s value %" PRIi64
                            " is outside the valid u32 range",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_target_profile_u64_attr(const loom_attribute_t* attr,
                                                  iree_string_view_t field_name,
                                                  uint64_t* out_value) {
  if (attr->kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile override %.*s must be an integer",
                            (int)field_name.size, field_name.data);
  }
  int64_t value = loom_attr_as_i64(*attr);
  if (value < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile override %.*s value %" PRIi64
                            " must be non-negative",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
}

static iree_status_t loom_target_profile_codegen_format_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_target_codegen_format_t* out_format) {
  if (attr->kind == LOOM_ATTR_ENUM) {
    *out_format = (loom_target_codegen_format_t)loom_attr_as_enum(*attr);
    return iree_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_target_profile_string_attr(
      module, attr, IREE_SV("codegen_format"), &value));
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_format = LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN;
  } else if (iree_string_view_equal(value, IREE_SV("llvmir"))) {
    *out_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR;
  } else if (iree_string_view_equal(value, IREE_SV("spirv"))) {
    *out_format = LOOM_TARGET_CODEGEN_FORMAT_SPIRV;
  } else if (iree_string_view_equal(value, IREE_SV("vm"))) {
    *out_format = LOOM_TARGET_CODEGEN_FORMAT_VM;
  } else if (iree_string_view_equal(value, IREE_SV("low_native"))) {
    *out_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE;
  } else if (iree_string_view_equal(value, IREE_SV("wasm"))) {
    *out_format = LOOM_TARGET_CODEGEN_FORMAT_WASM;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target.profile override codegen_format '%.*s' is unknown",
        (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_profile_artifact_format_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_target_artifact_format_t* out_format) {
  if (attr->kind == LOOM_ATTR_ENUM) {
    *out_format = (loom_target_artifact_format_t)loom_attr_as_enum(*attr);
    return iree_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_target_profile_string_attr(
      module, attr, IREE_SV("artifact_format"), &value));
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_format = LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN;
  } else if (iree_string_view_equal(value, IREE_SV("elf"))) {
    *out_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
  } else if (iree_string_view_equal(value, IREE_SV("coff"))) {
    *out_format = LOOM_TARGET_ARTIFACT_FORMAT_COFF;
  } else if (iree_string_view_equal(value, IREE_SV("macho"))) {
    *out_format = LOOM_TARGET_ARTIFACT_FORMAT_MACHO;
  } else if (iree_string_view_equal(value, IREE_SV("spirv_binary"))) {
    *out_format = LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
  } else if (iree_string_view_equal(value, IREE_SV("vm_bytecode"))) {
    *out_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE;
  } else if (iree_string_view_equal(value, IREE_SV("wasm_binary"))) {
    *out_format = LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target.profile override artifact_format '%.*s' is unknown",
        (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_profile_abi_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_target_abi_kind_t* out_abi) {
  if (attr->kind == LOOM_ATTR_ENUM) {
    *out_abi = (loom_target_abi_kind_t)loom_attr_as_enum(*attr);
    return iree_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_target_profile_string_attr(module, attr, IREE_SV("abi"), &value));
  if (iree_string_view_equal(value, IREE_SV("unknown"))) {
    *out_abi = LOOM_TARGET_ABI_UNKNOWN;
  } else if (iree_string_view_equal(value, IREE_SV("object_function"))) {
    *out_abi = LOOM_TARGET_ABI_OBJECT_FUNCTION;
  } else if (iree_string_view_equal(value, IREE_SV("hal_kernel"))) {
    *out_abi = LOOM_TARGET_ABI_HAL_KERNEL;
  } else if (iree_string_view_equal(value, IREE_SV("vm_module_function"))) {
    *out_abi = LOOM_TARGET_ABI_VM_MODULE_FUNCTION;
  } else if (iree_string_view_equal(value, IREE_SV("shader_entry_point"))) {
    *out_abi = LOOM_TARGET_ABI_SHADER_ENTRY_POINT;
  } else if (iree_string_view_equal(value, IREE_SV("wasm_function"))) {
    *out_abi = LOOM_TARGET_ABI_WASM_FUNCTION;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile override abi '%.*s' is unknown",
                            (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_profile_linkage_attr(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_target_linkage_t* out_linkage) {
  if (attr->kind == LOOM_ATTR_ENUM) {
    *out_linkage = (loom_target_linkage_t)loom_attr_as_enum(*attr);
    return iree_ok_status();
  }
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_target_profile_string_attr(
      module, attr, IREE_SV("linkage"), &value));
  if (iree_string_view_equal(value, IREE_SV("default"))) {
    *out_linkage = LOOM_TARGET_LINKAGE_DEFAULT;
  } else if (iree_string_view_equal(value, IREE_SV("dso_local"))) {
    *out_linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile override linkage '%.*s' is unknown",
                            (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_profile_apply_override(
    const loom_module_t* module, iree_string_view_t name,
    const loom_attribute_t* value, loom_target_profile_symbol_facts_t* facts) {
  if (iree_string_view_equal(name, IREE_SV("codegen_format"))) {
    return loom_target_profile_codegen_format_attr(
        module, value, &facts->snapshot.codegen_format);
  } else if (iree_string_view_equal(name, IREE_SV("target_triple"))) {
    return loom_target_profile_string_attr(module, value, name,
                                           &facts->snapshot.target_triple);
  } else if (iree_string_view_equal(name, IREE_SV("data_layout"))) {
    return loom_target_profile_string_attr(module, value, name,
                                           &facts->snapshot.data_layout);
  } else if (iree_string_view_equal(name, IREE_SV("artifact_format"))) {
    return loom_target_profile_artifact_format_attr(
        module, value, &facts->snapshot.artifact_format);
  } else if (iree_string_view_equal(name, IREE_SV("target_cpu"))) {
    return loom_target_profile_string_attr(module, value, name,
                                           &facts->snapshot.target_cpu);
  } else if (iree_string_view_equal(name, IREE_SV("target_features"))) {
    return loom_target_profile_string_attr(module, value, name,
                                           &facts->snapshot.target_features);
  } else if (iree_string_view_equal(name,
                                    IREE_SV("default_pointer_bitwidth"))) {
    return loom_target_profile_u32_attr(
        value, name, 1, 64, &facts->snapshot.default_pointer_bitwidth);
  } else if (iree_string_view_equal(name, IREE_SV("index_bitwidth"))) {
    return loom_target_profile_u32_attr(value, name, 1, 64,
                                        &facts->snapshot.index_bitwidth);
  } else if (iree_string_view_equal(name, IREE_SV("offset_bitwidth"))) {
    return loom_target_profile_u32_attr(value, name, 1, 64,
                                        &facts->snapshot.offset_bitwidth);
  } else if (iree_string_view_equal(name, IREE_SV("memory_space_generic"))) {
    return loom_target_profile_u32_attr(value, name, 0, UINT32_MAX,
                                        &facts->snapshot.memory_spaces.generic);
  } else if (iree_string_view_equal(name, IREE_SV("memory_space_global"))) {
    return loom_target_profile_u32_attr(value, name, 0, UINT32_MAX,
                                        &facts->snapshot.memory_spaces.global);
  } else if (iree_string_view_equal(name, IREE_SV("memory_space_workgroup"))) {
    return loom_target_profile_u32_attr(
        value, name, 0, UINT32_MAX, &facts->snapshot.memory_spaces.workgroup);
  } else if (iree_string_view_equal(name, IREE_SV("memory_space_constant"))) {
    return loom_target_profile_u32_attr(
        value, name, 0, UINT32_MAX, &facts->snapshot.memory_spaces.constant);
  } else if (iree_string_view_equal(name, IREE_SV("memory_space_private"))) {
    return loom_target_profile_u32_attr(
        value, name, 0, UINT32_MAX,
        &facts->snapshot.memory_spaces.private_memory);
  } else if (iree_string_view_equal(name, IREE_SV("memory_space_host"))) {
    return loom_target_profile_u32_attr(value, name, 0, UINT32_MAX,
                                        &facts->snapshot.memory_spaces.host);
  } else if (iree_string_view_equal(name, IREE_SV("memory_space_descriptor"))) {
    return loom_target_profile_u32_attr(
        value, name, 0, UINT32_MAX, &facts->snapshot.memory_spaces.descriptor);
  } else if (iree_string_view_equal(name, IREE_SV("abi"))) {
    return loom_target_profile_abi_attr(module, value,
                                        &facts->export_plan.abi_kind);
  } else if (iree_string_view_equal(name, IREE_SV("linkage"))) {
    return loom_target_profile_linkage_attr(module, value,
                                            &facts->export_plan.linkage);
  } else if (iree_string_view_equal(name, IREE_SV("hal_binding_alignment"))) {
    return loom_target_profile_u32_attr(
        value, name, 0, UINT32_MAX,
        &facts->export_plan.hal_kernel.binding_alignment);
  } else if (iree_string_view_equal(name,
                                    IREE_SV("hal_buffer_resource_flags"))) {
    return loom_target_profile_u32_attr(
        value, name, 0, UINT32_MAX,
        &facts->export_plan.hal_kernel.buffer_resource_flags);
  } else if (iree_string_view_equal(name, IREE_SV("contract_set_key"))) {
    return loom_target_profile_string_attr(module, value, name,
                                           &facts->config.contract_set_key);
  } else if (iree_string_view_equal(name, IREE_SV("contract_feature_bits"))) {
    return loom_target_profile_u64_attr(value, name,
                                        &facts->config.contract_feature_bits);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "target.profile override '%.*s' is not supported",
                          (int)name.size, name.data);
}

static iree_status_t loom_target_profile_apply_overrides(
    const loom_module_t* module, loom_named_attr_slice_t overrides,
    loom_target_profile_symbol_facts_t* facts) {
  for (iree_host_size_t i = 0; i < overrides.count; ++i) {
    const loom_named_attr_t* entry = &overrides.entries[i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_target_symbol_string_from_id(
        module, entry->name_id, IREE_SV("override name"), &name));
    IREE_RETURN_IF_ERROR(
        loom_target_profile_apply_override(module, name, &entry->value, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_profile_validate_preset_bundle(
    iree_string_view_t preset_key, const loom_target_bundle_t* bundle) {
  if (!bundle->snapshot || !bundle->export_plan || !bundle->config) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile preset '%.*s' must provide snapshot, export plan, and "
        "config records",
        (int)preset_key.size, preset_key.data);
  }
  return iree_ok_status();
}

static loom_target_artifact_abi_kind_t loom_target_artifact_default_abi(
    loom_target_artifact_format_t format) {
  switch (format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      return LOOM_TARGET_ARTIFACT_ABI_KIND_OBJECT_FILE;
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
      return LOOM_TARGET_ARTIFACT_ABI_KIND_SPIRV_MODULE;
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
      return LOOM_TARGET_ARTIFACT_ABI_KIND_VM_MODULE;
    case LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY:
      return LOOM_TARGET_ARTIFACT_ABI_KIND_WASM_MODULE;
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
    default:
      return LOOM_TARGET_ARTIFACT_ABI_KIND_UNKNOWN;
  }
}

static bool loom_target_artifact_abi_matches_format(
    loom_target_artifact_format_t format,
    loom_target_artifact_abi_kind_t abi_kind) {
  switch (abi_kind) {
    case LOOM_TARGET_ARTIFACT_ABI_KIND_OBJECT_FILE:
      return format == LOOM_TARGET_ARTIFACT_FORMAT_ELF ||
             format == LOOM_TARGET_ARTIFACT_FORMAT_COFF ||
             format == LOOM_TARGET_ARTIFACT_FORMAT_MACHO;
    case LOOM_TARGET_ARTIFACT_ABI_KIND_HAL_EXECUTABLE:
      return format == LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    case LOOM_TARGET_ARTIFACT_ABI_KIND_VM_MODULE:
      return format == LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE;
    case LOOM_TARGET_ARTIFACT_ABI_KIND_WASM_MODULE:
      return format == LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY;
    case LOOM_TARGET_ARTIFACT_ABI_KIND_SPIRV_MODULE:
      return format == LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
    case LOOM_TARGET_ARTIFACT_ABI_KIND_UNKNOWN:
    default:
      return false;
  }
}

static iree_status_t loom_target_artifact_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  IREE_ASSERT_ARGUMENT(domain);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;

  loom_target_artifact_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->artifact_op = symbol->defining_op;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  IREE_RETURN_IF_ERROR(loom_target_symbol_string_from_id(
      module, symbol->name_id, IREE_SV("symbol"), &facts->name));

  facts->target_symbol = loom_target_artifact_target(symbol->defining_op);
  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_lookup_ref(
      context, facts->target_symbol, &target_base_facts));
  facts->target_profile =
      loom_target_profile_symbol_facts_cast(target_base_facts);
  if (!facts->target_profile) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target.artifact target must resolve to target.profile symbol facts");
  }

  facts->format =
      (loom_target_artifact_format_t)loom_target_artifact_artifact_format(
          symbol->defining_op);
  if (facts->format == LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    facts->format = facts->target_profile->snapshot.artifact_format;
  }
  if (facts->format == LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.artifact must resolve an artifact format");
  }

  facts->abi_kind = (loom_target_artifact_abi_kind_t)loom_target_artifact_abi(
      symbol->defining_op);
  if (facts->abi_kind == LOOM_TARGET_ARTIFACT_ABI_KIND_UNKNOWN) {
    facts->abi_kind = loom_target_artifact_default_abi(facts->format);
  }
  if (!loom_target_artifact_abi_matches_format(facts->format,
                                               facts->abi_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.artifact ABI does not match its format");
  }

  *out_facts = &facts->base;
  return iree_ok_status();
}

static iree_status_t loom_target_profile_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  IREE_ASSERT_ARGUMENT(domain);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;

  loom_target_profile_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  IREE_RETURN_IF_ERROR(loom_target_symbol_string_from_id(
      module, symbol->name_id, IREE_SV("symbol"), &facts->name));
  IREE_RETURN_IF_ERROR(loom_target_symbol_string_from_id(
      module, loom_target_profile_preset(symbol->defining_op),
      IREE_SV("preset"), &facts->preset_key));
  facts->preset_key = iree_string_view_trim(facts->preset_key);
  if (iree_string_view_is_empty(facts->preset_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target.profile preset key must be non-empty");
  }

  const loom_target_preset_registry_t* preset_registry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_profile_symbol_fact_context_lookup_preset_registry(
          context, &preset_registry));
  IREE_RETURN_IF_ERROR(loom_target_preset_registry_lookup_bundle(
      preset_registry, facts->preset_key, &facts->preset_bundle));
  IREE_RETURN_IF_ERROR(loom_target_profile_validate_preset_bundle(
      facts->preset_key, facts->preset_bundle));

  facts->snapshot = *facts->preset_bundle->snapshot;
  facts->export_plan = *facts->preset_bundle->export_plan;
  facts->config = *facts->preset_bundle->config;
  facts->snapshot.name = facts->name;
  facts->export_plan.name = facts->name;
  facts->export_plan.export_symbol = iree_string_view_empty();
  facts->config.name = facts->name;
  facts->bundle = (loom_target_bundle_t){
      .name = facts->name,
      .snapshot = &facts->snapshot,
      .export_plan = &facts->export_plan,
      .config = &facts->config,
  };

  IREE_RETURN_IF_ERROR(loom_target_profile_apply_overrides(
      module, loom_target_profile_overrides(symbol->defining_op), facts));

  *out_facts = &facts->base;
  return iree_ok_status();
}

const loom_symbol_fact_domain_t loom_target_profile_symbol_fact_domain = {
    .compute = loom_target_profile_symbol_fact_compute,
};

const loom_symbol_fact_domain_t loom_target_artifact_symbol_fact_domain = {
    .compute = loom_target_artifact_symbol_fact_compute,
};

iree_status_t loom_target_profile_symbol_fact_context_lookup_preset_registry(
    loom_symbol_fact_context_t* context,
    const loom_target_preset_registry_t** out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = NULL;
  const void* resource = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_lookup_resource(
      context, &loom_target_profile_preset_registry_resource_key, &resource));
  *out_registry = (const loom_target_preset_registry_t*)resource;
  return iree_ok_status();
}

const loom_target_profile_symbol_facts_t* loom_target_profile_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts) {
  if (!facts || facts->domain != &loom_target_profile_symbol_fact_domain) {
    return NULL;
  }
  return (const loom_target_profile_symbol_facts_t*)facts;
}

const loom_target_artifact_symbol_facts_t*
loom_target_artifact_symbol_facts_cast(const loom_symbol_facts_base_t* facts) {
  if (!facts || facts->domain != &loom_target_artifact_symbol_fact_domain) {
    return NULL;
  }
  return (const loom_target_artifact_symbol_facts_t*)facts;
}
