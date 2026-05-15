// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/features.h"

#include <inttypes.h>
#include <string.h>

#define LOOM_SPIRV_VERSION_1_0 UINT32_C(0x00010000)

static const iree_string_view_t kSpirvVulkanShaderExtensions[] = {
    IREE_SVL("SPV_KHR_vulkan_memory_model"),
};

static const uint32_t kSpirvVulkanShaderCapabilities[] = {
    LOOM_SPIRV_CAPABILITY_SHADER,
    LOOM_SPIRV_CAPABILITY_VULKAN_MEMORY_MODEL,
};

static const iree_string_view_t kSpirvPhysicalStorageBufferExtensions[] = {
    IREE_SVL("SPV_KHR_physical_storage_buffer"),
};

static const uint32_t kSpirvPhysicalStorageBufferCapabilities[] = {
    LOOM_SPIRV_CAPABILITY_PHYSICAL_STORAGE_BUFFER_ADDRESSES,
};

static const uint32_t kSpirvPhysicalStorageBufferStorageClasses[] = {
    LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
};

static const uint32_t kSpirvPhysicalStorageBufferDecorations[] = {
    LOOM_SPIRV_DECORATION_RESTRICT_POINTER,
    LOOM_SPIRV_DECORATION_ALIASED_POINTER,
};

static const iree_string_view_t kSpirvCooperativeVectorNvExtensions[] = {
    IREE_SVL("SPV_NV_cooperative_vector"),
};

static const uint32_t kSpirvCooperativeVectorNvCapabilities[] = {
    LOOM_SPIRV_CAPABILITY_COOPERATIVE_VECTOR_NV,
};

static const uint32_t kSpirvCooperativeVectorNvOpcodes[] = {
    LOOM_SPIRV_OPCODE_TYPE_COOPERATIVE_VECTOR_NV,
    LOOM_SPIRV_OPCODE_COOPERATIVE_VECTOR_MATRIX_MUL_NV,
    LOOM_SPIRV_OPCODE_COOPERATIVE_VECTOR_MATRIX_MUL_ADD_NV,
    LOOM_SPIRV_OPCODE_COOPERATIVE_VECTOR_LOAD_NV,
    LOOM_SPIRV_OPCODE_COOPERATIVE_VECTOR_STORE_NV,
};

static const uint32_t kSpirvCooperativeVectorTrainingNvCapabilities[] = {
    LOOM_SPIRV_CAPABILITY_COOPERATIVE_VECTOR_TRAINING_NV,
};

static const uint32_t kSpirvCooperativeVectorTrainingNvOpcodes[] = {
    LOOM_SPIRV_OPCODE_COOPERATIVE_VECTOR_OUTER_PRODUCT_ACCUMULATE_NV,
    LOOM_SPIRV_OPCODE_COOPERATIVE_VECTOR_REDUCE_SUM_ACCUMULATE_NV,
};

static const iree_string_view_t kSpirvCooperativeMatrixKhrExtensions[] = {
    IREE_SVL("SPV_KHR_cooperative_matrix"),
};

static const uint32_t kSpirvCooperativeMatrixKhrCapabilities[] = {
    LOOM_SPIRV_CAPABILITY_COOPERATIVE_MATRIX_KHR,
};

static const uint32_t kSpirvCooperativeMatrixKhrOpcodes[] = {
    LOOM_SPIRV_OPCODE_TYPE_COOPERATIVE_MATRIX_KHR,
    LOOM_SPIRV_OPCODE_COOPERATIVE_MATRIX_LOAD_KHR,
    LOOM_SPIRV_OPCODE_COOPERATIVE_MATRIX_STORE_KHR,
    LOOM_SPIRV_OPCODE_COOPERATIVE_MATRIX_MUL_ADD_KHR,
    LOOM_SPIRV_OPCODE_COOPERATIVE_MATRIX_LENGTH_KHR,
};

static const loom_spirv_feature_atom_descriptor_t kSpirvFeatureAtoms[] = {
    [LOOM_SPIRV_FEATURE_ATOM_UNKNOWN] = {0},
    [LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER] =
        {
            .atom = LOOM_SPIRV_FEATURE_ATOM_VULKAN_SHADER,
            .name = IREE_SVL("spirv.vulkan.shader"),
            .required_atom_bits = 0,
            .minimum_spirv_version = LOOM_SPIRV_VERSION_1_0,
            .addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED,
            .memory_model = LOOM_SPIRV_MEMORY_MODEL_VULKAN,
            .extension_names = kSpirvVulkanShaderExtensions,
            .extension_count = IREE_ARRAYSIZE(kSpirvVulkanShaderExtensions),
            .capabilities = kSpirvVulkanShaderCapabilities,
            .capability_count = IREE_ARRAYSIZE(kSpirvVulkanShaderCapabilities),
        },
    [LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER] =
        {
            .atom = LOOM_SPIRV_FEATURE_ATOM_PHYSICAL_STORAGE_BUFFER,
            .name = IREE_SVL("spirv.physical_storage_buffer"),
            .required_atom_bits = LOOM_SPIRV_FEATURE_VULKAN_SHADER,
            .minimum_spirv_version = LOOM_SPIRV_VERSION_1_0,
            .addressing_model =
                LOOM_SPIRV_ADDRESSING_MODEL_PHYSICAL_STORAGE_BUFFER64,
            .memory_model = LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED,
            .extension_names = kSpirvPhysicalStorageBufferExtensions,
            .extension_count =
                IREE_ARRAYSIZE(kSpirvPhysicalStorageBufferExtensions),
            .capabilities = kSpirvPhysicalStorageBufferCapabilities,
            .capability_count =
                IREE_ARRAYSIZE(kSpirvPhysicalStorageBufferCapabilities),
            .storage_classes = kSpirvPhysicalStorageBufferStorageClasses,
            .storage_class_count =
                IREE_ARRAYSIZE(kSpirvPhysicalStorageBufferStorageClasses),
            .decorations = kSpirvPhysicalStorageBufferDecorations,
            .decoration_count =
                IREE_ARRAYSIZE(kSpirvPhysicalStorageBufferDecorations),
        },
    [LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV] =
        {
            .atom = LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_NV,
            .name = IREE_SVL("spirv.cooperative_vector.nv"),
            .required_atom_bits = LOOM_SPIRV_FEATURE_VULKAN_SHADER,
            .minimum_spirv_version = LOOM_SPIRV_VERSION_1_0,
            .addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED,
            .memory_model = LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED,
            .extension_names = kSpirvCooperativeVectorNvExtensions,
            .extension_count =
                IREE_ARRAYSIZE(kSpirvCooperativeVectorNvExtensions),
            .capabilities = kSpirvCooperativeVectorNvCapabilities,
            .capability_count =
                IREE_ARRAYSIZE(kSpirvCooperativeVectorNvCapabilities),
            .opcodes = kSpirvCooperativeVectorNvOpcodes,
            .opcode_count = IREE_ARRAYSIZE(kSpirvCooperativeVectorNvOpcodes),
        },
    [LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV] =
        {
            .atom = LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV,
            .name = IREE_SVL("spirv.cooperative_vector.training.nv"),
            .required_atom_bits = LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                                  LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
            .minimum_spirv_version = LOOM_SPIRV_VERSION_1_0,
            .addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED,
            .memory_model = LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED,
            .extension_names = kSpirvCooperativeVectorNvExtensions,
            .extension_count =
                IREE_ARRAYSIZE(kSpirvCooperativeVectorNvExtensions),
            .capabilities = kSpirvCooperativeVectorTrainingNvCapabilities,
            .capability_count =
                IREE_ARRAYSIZE(kSpirvCooperativeVectorTrainingNvCapabilities),
            .opcodes = kSpirvCooperativeVectorTrainingNvOpcodes,
            .opcode_count =
                IREE_ARRAYSIZE(kSpirvCooperativeVectorTrainingNvOpcodes),
        },
    [LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR] =
        {
            .atom = LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR,
            .name = IREE_SVL("spirv.cooperative_matrix.khr"),
            .required_atom_bits = LOOM_SPIRV_FEATURE_VULKAN_SHADER,
            .minimum_spirv_version = LOOM_SPIRV_VERSION_1_0,
            .addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED,
            .memory_model = LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED,
            .extension_names = kSpirvCooperativeMatrixKhrExtensions,
            .extension_count =
                IREE_ARRAYSIZE(kSpirvCooperativeMatrixKhrExtensions),
            .capabilities = kSpirvCooperativeMatrixKhrCapabilities,
            .capability_count =
                IREE_ARRAYSIZE(kSpirvCooperativeMatrixKhrCapabilities),
            .opcodes = kSpirvCooperativeMatrixKhrOpcodes,
            .opcode_count = IREE_ARRAYSIZE(kSpirvCooperativeMatrixKhrOpcodes),
        },
};

loom_spirv_feature_bits_t loom_spirv_feature_atom_bit(
    loom_spirv_feature_atom_t atom) {
  if (atom <= LOOM_SPIRV_FEATURE_ATOM_UNKNOWN ||
      atom >= LOOM_SPIRV_FEATURE_ATOM_COUNT) {
    return 0;
  }
  return UINT64_C(1) << atom;
}

const loom_spirv_feature_atom_descriptor_t* loom_spirv_feature_atom_descriptor(
    loom_spirv_feature_atom_t atom) {
  if (atom <= LOOM_SPIRV_FEATURE_ATOM_UNKNOWN ||
      atom >= LOOM_SPIRV_FEATURE_ATOM_COUNT) {
    return NULL;
  }
  return &kSpirvFeatureAtoms[atom];
}

iree_string_view_t loom_spirv_feature_atom_name(
    loom_spirv_feature_atom_t atom) {
  const loom_spirv_feature_atom_descriptor_t* descriptor =
      loom_spirv_feature_atom_descriptor(atom);
  return descriptor ? descriptor->name : IREE_SV("unknown");
}

loom_spirv_feature_bits_t loom_spirv_known_feature_bits(void) {
  loom_spirv_feature_bits_t known_bits = 0;
  for (uint32_t i = LOOM_SPIRV_FEATURE_ATOM_UNKNOWN + 1;
       i < LOOM_SPIRV_FEATURE_ATOM_COUNT; ++i) {
    known_bits |= loom_spirv_feature_atom_bit((loom_spirv_feature_atom_t)i);
  }
  return known_bits;
}

bool loom_spirv_feature_set_has_atom(
    const loom_spirv_feature_set_t* feature_set,
    loom_spirv_feature_atom_t atom) {
  IREE_ASSERT_ARGUMENT(feature_set);
  return iree_any_bit_set(feature_set->atom_bits,
                          loom_spirv_feature_atom_bit(atom));
}

static bool loom_spirv_feature_set_has_extension(
    const loom_spirv_feature_set_t* feature_set, iree_string_view_t name) {
  for (uint8_t i = 0; i < feature_set->extension_count; ++i) {
    if (iree_string_view_equal(feature_set->extension_names[i], name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_spirv_feature_set_append_extension(
    loom_spirv_feature_set_t* feature_set, iree_string_view_t name) {
  if (loom_spirv_feature_set_has_extension(feature_set, name)) {
    return iree_ok_status();
  }
  if (feature_set->extension_count >= LOOM_SPIRV_FEATURE_MAX_EXTENSION_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SPIR-V feature extension list overflow");
  }
  feature_set->extension_names[feature_set->extension_count++] = name;
  return iree_ok_status();
}

static bool loom_spirv_feature_set_has_uint32(const uint32_t* values,
                                              uint8_t count, uint32_t value) {
  for (uint8_t i = 0; i < count; ++i) {
    if (values[i] == value) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_spirv_feature_set_append_uint32(
    uint32_t* values, uint8_t* count, uint8_t capacity, uint32_t value,
    const char* row_name) {
  if (loom_spirv_feature_set_has_uint32(values, *count, value)) {
    return iree_ok_status();
  }
  if (*count >= capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SPIR-V feature %s list overflow", row_name);
  }
  values[(*count)++] = value;
  return iree_ok_status();
}

static iree_status_t loom_spirv_feature_set_apply_models(
    const loom_spirv_feature_atom_descriptor_t* descriptor,
    iree_string_view_t target_name, loom_spirv_feature_set_t* feature_set) {
  if (descriptor->addressing_model != LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED) {
    if (feature_set->addressing_model !=
            LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED &&
        feature_set->addressing_model != descriptor->addressing_model) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "SPIR-V target '%.*s' feature atom '%.*s' conflicts with the "
          "selected addressing model",
          (int)target_name.size, target_name.data, (int)descriptor->name.size,
          descriptor->name.data);
    }
    feature_set->addressing_model = descriptor->addressing_model;
  }
  if (descriptor->memory_model != LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED) {
    if (feature_set->memory_model != LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED &&
        feature_set->memory_model != descriptor->memory_model) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "SPIR-V target '%.*s' feature atom '%.*s' conflicts with the "
          "selected memory model",
          (int)target_name.size, target_name.data, (int)descriptor->name.size,
          descriptor->name.data);
    }
    feature_set->memory_model = descriptor->memory_model;
  }
  return iree_ok_status();
}

static const loom_spirv_feature_atom_descriptor_t*
loom_spirv_feature_set_find_missing_dependency(
    loom_spirv_feature_bits_t missing_atom_bits) {
  for (uint32_t i = LOOM_SPIRV_FEATURE_ATOM_UNKNOWN + 1;
       i < LOOM_SPIRV_FEATURE_ATOM_COUNT; ++i) {
    const loom_spirv_feature_atom_t atom = (loom_spirv_feature_atom_t)i;
    if (iree_any_bit_set(missing_atom_bits,
                         loom_spirv_feature_atom_bit(atom))) {
      return &kSpirvFeatureAtoms[atom];
    }
  }
  return NULL;
}

static iree_string_view_t loom_spirv_feature_atom_primary_extension(
    const loom_spirv_feature_atom_descriptor_t* descriptor) {
  return descriptor->extension_count != 0 ? descriptor->extension_names[0]
                                          : IREE_SV("<core>");
}

static uint32_t loom_spirv_feature_atom_primary_capability(
    const loom_spirv_feature_atom_descriptor_t* descriptor) {
  return descriptor->capability_count != 0 ? descriptor->capabilities[0]
                                           : UINT32_MAX;
}

static iree_status_t loom_spirv_feature_set_apply_atom(
    const loom_spirv_feature_atom_descriptor_t* descriptor,
    iree_string_view_t target_name,
    loom_spirv_feature_bits_t requested_atom_bits,
    loom_spirv_feature_set_t* feature_set) {
  const loom_spirv_feature_bits_t missing_bits =
      descriptor->required_atom_bits & ~requested_atom_bits;
  if (missing_bits != 0) {
    const loom_spirv_feature_atom_descriptor_t* missing_descriptor =
        loom_spirv_feature_set_find_missing_dependency(missing_bits);
    const iree_string_view_t missing_name = missing_descriptor != NULL
                                                ? missing_descriptor->name
                                                : IREE_SV("<unknown>");
    const iree_string_view_t extension_name =
        loom_spirv_feature_atom_primary_extension(descriptor);
    const uint32_t capability =
        loom_spirv_feature_atom_primary_capability(descriptor);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V target '%.*s' feature atom '%.*s' from extension '%.*s' and "
        "capability %" PRIu32
        " is missing dependency '%.*s' (bits 0x%016" PRIx64 ")",
        (int)target_name.size, target_name.data, (int)descriptor->name.size,
        descriptor->name.data, (int)extension_name.size, extension_name.data,
        capability, (int)missing_name.size, missing_name.data, missing_bits);
  }
  feature_set->minimum_spirv_version = iree_max(
      feature_set->minimum_spirv_version, descriptor->minimum_spirv_version);
  IREE_RETURN_IF_ERROR(loom_spirv_feature_set_apply_models(
      descriptor, target_name, feature_set));
  for (uint8_t i = 0; i < descriptor->extension_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_feature_set_append_extension(
        feature_set, descriptor->extension_names[i]));
  }
  for (uint8_t i = 0; i < descriptor->capability_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_feature_set_append_uint32(
        feature_set->capabilities, &feature_set->capability_count,
        LOOM_SPIRV_FEATURE_MAX_CAPABILITY_COUNT, descriptor->capabilities[i],
        "capability"));
  }
  for (uint8_t i = 0; i < descriptor->opcode_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_feature_set_append_uint32(
        feature_set->opcodes, &feature_set->opcode_count,
        LOOM_SPIRV_FEATURE_MAX_OPCODE_COUNT, descriptor->opcodes[i], "opcode"));
  }
  for (uint8_t i = 0; i < descriptor->storage_class_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_feature_set_append_uint32(
        feature_set->storage_classes, &feature_set->storage_class_count,
        LOOM_SPIRV_FEATURE_MAX_STORAGE_CLASS_COUNT,
        descriptor->storage_classes[i], "storage-class"));
  }
  for (uint8_t i = 0; i < descriptor->decoration_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_feature_set_append_uint32(
        feature_set->decorations, &feature_set->decoration_count,
        LOOM_SPIRV_FEATURE_MAX_DECORATION_COUNT, descriptor->decorations[i],
        "decoration"));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_feature_set_prepare(
    iree_string_view_t target_name,
    loom_spirv_feature_bits_t requested_atom_bits,
    loom_spirv_feature_set_t* out_feature_set) {
  IREE_ASSERT_ARGUMENT(out_feature_set);

  const loom_spirv_feature_bits_t unknown_bits =
      requested_atom_bits & ~loom_spirv_known_feature_bits();
  if (unknown_bits != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SPIR-V target '%.*s' requested unknown feature "
                            "bits 0x%016" PRIx64,
                            (int)target_name.size, target_name.data,
                            unknown_bits);
  }

  memset(out_feature_set, 0, sizeof(*out_feature_set));
  out_feature_set->atom_bits = requested_atom_bits;
  out_feature_set->addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED;
  out_feature_set->memory_model = LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED;
  for (uint32_t i = LOOM_SPIRV_FEATURE_ATOM_UNKNOWN + 1;
       i < LOOM_SPIRV_FEATURE_ATOM_COUNT; ++i) {
    const loom_spirv_feature_atom_t atom = (loom_spirv_feature_atom_t)i;
    if (!iree_any_bit_set(requested_atom_bits,
                          loom_spirv_feature_atom_bit(atom))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_feature_set_apply_atom(
        &kSpirvFeatureAtoms[atom], target_name, requested_atom_bits,
        out_feature_set));
  }
  if (out_feature_set->addressing_model ==
      LOOM_SPIRV_ADDRESSING_MODEL_UNSPECIFIED) {
    out_feature_set->addressing_model = LOOM_SPIRV_ADDRESSING_MODEL_LOGICAL;
  }
  if (out_feature_set->memory_model == LOOM_SPIRV_MEMORY_MODEL_UNSPECIFIED) {
    out_feature_set->memory_model = LOOM_SPIRV_MEMORY_MODEL_GLSL450;
  }
  return iree_ok_status();
}
