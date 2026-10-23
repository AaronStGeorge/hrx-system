// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/source_signature.h"

#include <string.h>

#include "loom/ops/func/ops.h"

typedef struct loom_x86_source_signature_classifier_t {
  const loom_func_c_abi_signature_t* signature;
  uint64_t root_byte_length;
  uint16_t maximum_vector_bytes;
  bool unsupported;
  bool memory;
  loom_x86_source_signature_class_t
      eightbytes[LOOM_X86_SOURCE_SIGNATURE_MAX_EIGHTBYTE_COUNT];
} loom_x86_source_signature_classifier_t;

static loom_x86_source_signature_class_t loom_x86_source_signature_merge_class(
    loom_x86_source_signature_class_t lhs,
    loom_x86_source_signature_class_t rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS) {
    return rhs;
  }
  if (rhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS) {
    return lhs;
  }
  if (lhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY ||
      rhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY) {
    return LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY;
  }
  if (lhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER ||
      rhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER) {
    return LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER;
  }
  return LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE;
}

static void loom_x86_source_signature_mark(
    loom_x86_source_signature_classifier_t* classifier, uint64_t byte_offset,
    uint64_t byte_length, loom_x86_source_signature_class_t first_class,
    loom_x86_source_signature_class_t remaining_class) {
  if (classifier->memory || byte_length == 0) {
    return;
  }
  const uint64_t first = byte_offset / 8u;
  const uint64_t last = (byte_offset + byte_length - 1u) / 8u;
  if (last >= LOOM_X86_SOURCE_SIGNATURE_MAX_EIGHTBYTE_COUNT) {
    classifier->memory = true;
    return;
  }
  for (uint64_t i = first; i <= last; ++i) {
    const loom_x86_source_signature_class_t value =
        i == first ? first_class : remaining_class;
    classifier->eightbytes[i] =
        loom_x86_source_signature_merge_class(classifier->eightbytes[i], value);
  }
}

static bool loom_x86_source_signature_is_aligned(uint64_t byte_offset,
                                                 uint64_t byte_alignment) {
  return byte_alignment != 0 && (byte_offset & (byte_alignment - 1u)) == 0;
}

static void loom_x86_source_signature_classify_type(
    loom_x86_source_signature_classifier_t* classifier, uint32_t type_index,
    uint64_t byte_offset, uint32_t depth) {
  if (classifier->unsupported || classifier->memory) {
    return;
  }
  IREE_ASSERT_LT(type_index, classifier->signature->types.count);
  IREE_ASSERT_LT(depth, 64u);
  const loom_attribute_t type = classifier->signature->types.values[type_index];
  const uint64_t byte_length =
      (uint64_t)loom_func_abi_c_type_attr_byte_length(type);
  const uint64_t byte_alignment =
      (uint64_t)loom_func_abi_c_type_attr_byte_alignment(type);
  if (byte_offset > classifier->root_byte_length ||
      byte_length > classifier->root_byte_length - byte_offset) {
    IREE_ASSERT_UNREACHABLE(
        "verified source signature child exceeds root object");
    return;
  }

  switch (loom_func_abi_c_type_attr_kind(type)) {
    case LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN:
    case LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_POINTER:
      loom_x86_source_signature_mark(classifier, byte_offset, byte_length,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER);
      return;
    case LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING: {
      const loom_func_abi_c_type_format_t format =
          loom_func_abi_c_type_attr_format(type);
      const loom_x86_source_signature_class_t abi_class =
          format == LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E4M3FN ||
                  format == LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E5M2
              ? LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER
              : LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE;
      loom_x86_source_signature_mark(classifier, byte_offset, byte_length,
                                     abi_class, abi_class);
      return;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR: {
      const uint64_t carrier_byte_length =
          byte_length <= 16u ? 16u : byte_length;
      if ((carrier_byte_length != 16u && carrier_byte_length != 32u &&
           carrier_byte_length != 64u) ||
          carrier_byte_length > classifier->maximum_vector_bytes) {
        classifier->memory = true;
        return;
      }
      loom_x86_source_signature_mark(classifier, byte_offset, byte_length,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP);
      return;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY: {
      const uint32_t element_type =
          (uint32_t)loom_func_abi_c_type_attr_element_type(type);
      const uint64_t element_count =
          (uint64_t)loom_func_abi_c_type_attr_element_count(type);
      const uint64_t element_stride =
          (uint64_t)loom_func_abi_c_type_attr_element_stride(type);
      const loom_attribute_t element =
          classifier->signature->types.values[element_type];
      const uint64_t element_alignment =
          (uint64_t)loom_func_abi_c_type_attr_byte_alignment(element);
      if (byte_alignment < element_alignment ||
          !loom_x86_source_signature_is_aligned(element_stride,
                                                element_alignment)) {
        classifier->memory = true;
        return;
      }
      for (uint64_t i = 0; i < element_count; ++i) {
        loom_x86_source_signature_classify_type(
            classifier, element_type, byte_offset + i * element_stride,
            depth + 1u);
      }
      return;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_RECORD: {
      const loom_parameterized_attr_array_t fields =
          loom_func_abi_c_type_attr_fields(type);
      for (iree_host_size_t i = 0; i < fields.count; ++i) {
        const loom_attribute_t field = fields.values[i];
        const uint32_t field_type =
            (uint32_t)loom_func_abi_c_field_attr_type(field);
        const uint64_t field_offset =
            (uint64_t)loom_func_abi_c_field_attr_byte_offset(field);
        const loom_attribute_t field_type_attr =
            classifier->signature->types.values[field_type];
        const uint64_t field_alignment =
            (uint64_t)loom_func_abi_c_type_attr_byte_alignment(field_type_attr);
        if (byte_alignment < field_alignment ||
            !loom_x86_source_signature_is_aligned(byte_offset + field_offset,
                                                  field_alignment)) {
          classifier->memory = true;
          return;
        }
        loom_x86_source_signature_classify_type(
            classifier, field_type, byte_offset + field_offset, depth + 1u);
      }
      return;
    }
    default:
      classifier->unsupported = true;
      return;
  }
}

bool loom_x86_source_signature_classify(
    const loom_func_c_abi_signature_t* signature, uint32_t root_type_index,
    uint16_t maximum_vector_bytes,
    loom_x86_source_signature_classification_t* out_classification) {
  IREE_ASSERT_ARGUMENT(signature);
  IREE_ASSERT_ARGUMENT(out_classification);
  *out_classification = (loom_x86_source_signature_classification_t){0};
  IREE_ASSERT_LT(root_type_index, signature->types.count);
  const loom_attribute_t root = signature->types.values[root_type_index];
  const int64_t raw_byte_length = loom_func_abi_c_type_attr_byte_length(root);
  const int64_t raw_byte_alignment =
      loom_func_abi_c_type_attr_byte_alignment(root);
  const int64_t raw_component_count =
      loom_func_abi_c_type_attr_component_count(root);
  if (raw_byte_length <= 0 || raw_byte_length > UINT32_MAX ||
      raw_byte_alignment <= 0 || raw_byte_alignment > UINT32_MAX ||
      raw_component_count <= 0 || raw_component_count > UINT16_MAX ||
      (maximum_vector_bytes != 0 && maximum_vector_bytes != 16 &&
       maximum_vector_bytes != 32 && maximum_vector_bytes != 64)) {
    return false;
  }

  loom_x86_source_signature_classifier_t classifier = {
      .signature = signature,
      .root_byte_length = (uint64_t)raw_byte_length,
      .maximum_vector_bytes = maximum_vector_bytes,
      .memory = raw_byte_length > 64,
  };
  loom_x86_source_signature_classify_type(&classifier, root_type_index,
                                          /*byte_offset=*/0, /*depth=*/0);
  if (classifier.unsupported) {
    return false;
  }

  const uint8_t eightbyte_count =
      classifier.memory ? 0 : (uint8_t)((raw_byte_length + 7) / 8);
  for (uint8_t i = 0; i < eightbyte_count; ++i) {
    if (classifier.eightbytes[i] == LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP &&
        (i == 0 ||
         (classifier.eightbytes[i - 1] != LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE &&
          classifier.eightbytes[i - 1] !=
              LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP))) {
      classifier.eightbytes[i] = LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE;
    }
  }
  if (eightbyte_count > 2) {
    if (classifier.eightbytes[0] != LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE) {
      classifier.memory = true;
    }
    for (uint8_t i = 1; i < eightbyte_count; ++i) {
      if (classifier.eightbytes[i] != LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP) {
        classifier.memory = true;
      }
    }
  }
  *out_classification = (loom_x86_source_signature_classification_t){
      .byte_length = (uint32_t)raw_byte_length,
      .byte_alignment = (uint32_t)raw_byte_alignment,
      .component_count = (uint16_t)raw_component_count,
      .eightbyte_count = classifier.memory ? 0 : eightbyte_count,
      .is_memory = classifier.memory,
  };
  if (!classifier.memory) {
    memcpy(out_classification->eightbytes, classifier.eightbytes,
           sizeof(classifier.eightbytes));
  }
  return true;
}
