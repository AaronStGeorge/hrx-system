// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include <inttypes.h>

#include "loom/util/stable_id.h"

static iree_status_t loom_low_descriptor_set_string_impl(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, bool allow_none,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    if (allow_none) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required low descriptor string offset is NONE");
  }
  if (descriptor_set->string_table.data == NULL ||
      descriptor_set->string_table.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor string table is empty");
  }
  if (string_offset >= descriptor_set->string_table.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor string offset %" PRIu32
                            " exceeds string table length %" PRIu32,
                            string_offset,
                            descriptor_set->string_table.data_length);
  }
  loom_bstring_t bstring = NULL;
  if (!loom_bstring_table_try_get(&descriptor_set->string_table, string_offset,
                                  &bstring)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor string offset %" PRIu32
                            " does not name a complete B-string",
                            string_offset);
  }
  *out_string = loom_bstring_view(bstring);
  return iree_ok_status();
}
uint64_t loom_low_descriptor_stable_id_from_key(iree_string_view_t key) {
  return loom_stable_id_from_string(key);
}
static bool loom_low_descriptor_constraint_ties_operands(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_operand_index,
    uint16_t packet_operand_index) {
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_TIED &&
        constraint->lhs_operand_index == result_operand_index &&
        constraint->rhs_operand_index == packet_operand_index) {
      return true;
    }
  }
  return false;
}

bool loom_low_descriptor_operands_are_tied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t lhs_operand_index,
    uint16_t rhs_operand_index) {
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  const loom_low_operand_t* lhs = &operands[lhs_operand_index];
  const loom_low_operand_t* rhs = &operands[rhs_operand_index];
  if (lhs->role == LOOM_LOW_OPERAND_ROLE_RESULT &&
      loom_low_operand_role_is_packet_operand(rhs->role)) {
    return loom_low_descriptor_constraint_ties_operands(
        descriptor_set, descriptor, lhs_operand_index, rhs_operand_index);
  }
  if (rhs->role == LOOM_LOW_OPERAND_ROLE_RESULT &&
      loom_low_operand_role_is_packet_operand(lhs->role)) {
    return loom_low_descriptor_constraint_ties_operands(
        descriptor_set, descriptor, rhs_operand_index, lhs_operand_index);
  }
  return false;
}
static iree_status_t loom_low_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, out_key));
  if (iree_string_view_is_empty(*out_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set key is empty");
  }
  return iree_ok_status();
}

iree_host_size_t loom_low_descriptor_registry_descriptor_set_count(
    const loom_low_descriptor_registry_t* registry) {
  if (registry == NULL) {
    return 0;
  }
  return registry->descriptor_set_count +
         registry->descriptor_set_provider_count;
}

const loom_low_descriptor_set_t* loom_low_descriptor_registry_descriptor_set_at(
    const loom_low_descriptor_registry_t* registry, iree_host_size_t index) {
  if (registry == NULL) {
    return NULL;
  }
  if (index < registry->descriptor_set_count) {
    if (registry->descriptor_sets == NULL) {
      return NULL;
    }
    return registry->descriptor_sets[index];
  }
  index -= registry->descriptor_set_count;
  if (index < registry->descriptor_set_provider_count) {
    if (registry->descriptor_set_providers == NULL) {
      return NULL;
    }
    loom_low_descriptor_set_provider_t provider =
        registry->descriptor_set_providers[index];
    return provider ? provider() : NULL;
  }
  return NULL;
}

iree_status_t loom_low_descriptor_registry_lookup(
    const loom_low_descriptor_registry_t* registry, iree_string_view_t key,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  if (out_descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor lookup output is required");
  }
  *out_descriptor_set = NULL;
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
  if (registry->descriptor_set_count != 0 &&
      registry->descriptor_sets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry entries are required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry providers are required");
  }
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(registry, i);
    iree_string_view_t candidate_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_low_descriptor_set_key(descriptor_set, &candidate_key));
    if (!iree_string_view_equal(candidate_key, key)) {
      continue;
    }
    *out_descriptor_set = descriptor_set;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_registry_lookup_by_id(
    const loom_low_descriptor_registry_t* registry, uint64_t stable_id,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  IREE_ASSERT_ARGUMENT(out_descriptor_set);
  *out_descriptor_set = NULL;
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
  if (stable_id == LOOM_LOW_STABLE_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set stable ID is required");
  }
  if (registry->descriptor_set_count != 0 &&
      registry->descriptor_sets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry entries are required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry providers are required");
  }
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(registry, i);
    if (descriptor_set == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor registry entry is required");
    }
    if (descriptor_set->stable_id != stable_id) {
      continue;
    }
    *out_descriptor_set = descriptor_set;
    return iree_ok_status();
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_set_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string) {
  if (out_string == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor string output is required");
  }
  *out_string = iree_string_view_empty();
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  return loom_low_descriptor_set_string_impl(descriptor_set, string_offset,
                                             /*allow_none=*/true, out_string);
}

const loom_low_descriptor_t* loom_low_descriptor_set_descriptor_at(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  if (descriptor_set == NULL ||
      descriptor_ordinal >= descriptor_set->descriptor_count) {
    return NULL;
  }
  return &descriptor_set->descriptors[descriptor_ordinal];
}

const loom_low_asm_form_t* loom_low_descriptor_set_asm_form_at(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t asm_form_ordinal) {
  if (descriptor_set == NULL ||
      asm_form_ordinal >= descriptor_set->asm_form_count) {
    return NULL;
  }
  return &descriptor_set->asm_forms[asm_form_ordinal];
}

iree_status_t loom_low_descriptor_set_lookup_canonical_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint32_t* out_asm_form_ordinal) {
  if (out_asm_form_ordinal == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low canonical asm form output is required");
  }
  *out_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor ordinal %" PRIu32
                            " is out of range",
                            descriptor_ordinal);
  }
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor->key_string_offset, /*allow_none=*/false,
        &key));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low descriptor '%.*s' has no unique canonical "
                            "asm form",
                            (int)key.size, key.data);
  }
  const loom_low_asm_form_t* asm_form = loom_low_descriptor_set_asm_form_at(
      descriptor_set, descriptor->canonical_asm_form_ordinal);
  if (asm_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor canonical asm form ordinal is out "
                            "of range");
  }
  if (asm_form->descriptor_ordinal != descriptor_ordinal) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor canonical asm form references a "
                            "different descriptor");
  }
  *out_asm_form_ordinal = descriptor->canonical_asm_form_ordinal;
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_set_lookup_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint32_t* out_descriptor_ordinal) {
  if (out_descriptor_ordinal == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor lookup output is required");
  }
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  if (descriptor_set->descriptor_ref_count !=
      descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor reference map count %" PRIu32
                            " does not match descriptor count %" PRIu32,
                            descriptor_set->descriptor_ref_count,
                            descriptor_set->descriptor_count);
  }
  if (descriptor_set->descriptor_ref_count != 0 &&
      descriptor_set->descriptor_refs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor reference map is required");
  }
  uint32_t low = 0;
  uint32_t high = descriptor_set->descriptor_ref_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_descriptor_ref_t* descriptor_ref =
        &descriptor_set->descriptor_refs[mid];
    iree_string_view_t descriptor_ref_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor_ref->key_string_offset, /*allow_none=*/false,
        &descriptor_ref_key));
    const int comparison = iree_string_view_compare(descriptor_ref_key, key);
    if (comparison == 0) {
      if (descriptor_ref->descriptor_ordinal >=
          descriptor_set->descriptor_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low descriptor reference '%.*s' points at descriptor ordinal "
            "%" PRIu32 " but only %" PRIu32 " descriptors exist",
            (int)descriptor_ref_key.size, descriptor_ref_key.data,
            descriptor_ref->descriptor_ordinal,
            descriptor_set->descriptor_count);
      }
      *out_descriptor_ordinal = descriptor_ref->descriptor_ordinal;
      return iree_ok_status();
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "low descriptor key '%.*s' was not found",
                          (int)key.size, key.data);
}

uint32_t loom_low_descriptor_set_lookup_descriptor_by_id(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t stable_id) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT(stable_id != LOOM_LOW_DESCRIPTOR_ID_NONE);

  uint32_t low = 0;
  uint32_t high = descriptor_set->descriptor_id_ref_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_descriptor_id_ref_t* descriptor_id_ref =
        &descriptor_set->descriptor_id_refs[mid];
    if (descriptor_id_ref->stable_id == stable_id) {
      IREE_ASSERT(descriptor_id_ref->descriptor_ordinal <
                  descriptor_set->descriptor_count);
      return descriptor_id_ref->descriptor_ordinal;
    }
    if (descriptor_id_ref->stable_id < stable_id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

iree_status_t loom_low_descriptor_set_lookup_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic, uint32_t* out_asm_form_ordinal) {
  if (out_asm_form_ordinal == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm form lookup output is required");
  }
  *out_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  if (descriptor_set->asm_form_count != 0 &&
      descriptor_set->asm_forms == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm form table is required");
  }
  uint32_t low = 0;
  uint32_t high = descriptor_set->asm_form_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_asm_form_t* asm_form = &descriptor_set->asm_forms[mid];
    iree_string_view_t asm_mnemonic = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, asm_form->mnemonic_string_offset, /*allow_none=*/false,
        &asm_mnemonic));
    const int comparison = iree_string_view_compare(asm_mnemonic, mnemonic);
    if (comparison == 0) {
      if (asm_form->descriptor_ordinal >= descriptor_set->descriptor_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low asm form '%.*s' points at descriptor "
            "ordinal %" PRIu32 " but only %" PRIu32 " descriptors exist",
            (int)asm_mnemonic.size, asm_mnemonic.data,
            asm_form->descriptor_ordinal, descriptor_set->descriptor_count);
      }
      *out_asm_form_ordinal = mid;
      return iree_ok_status();
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return iree_ok_status();
}
