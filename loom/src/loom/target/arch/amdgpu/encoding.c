// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/encoding.h"

#include <inttypes.h>

#include "loom/target/arch/amdgpu/cdna4_encoding_tables.h"
#include "loom/target/arch/amdgpu/rdna3_encoding_tables.h"
#include "loom/target/arch/amdgpu/rdna4_encoding_tables.h"
#include "loom/target/arch/amdgpu/rdna4_gfx125x_encoding_tables.h"
#include "loom/target/arch/amdgpu/target_info.h"

static uint64_t loom_amdgpu_encoding_u64_mask(uint8_t bit_count) {
  if (bit_count == 0) {
    return 0;
  }
  if (bit_count >= 64) {
    return UINT64_MAX;
  }
  return (UINT64_C(1) << bit_count) - 1;
}

static uint32_t loom_amdgpu_encoding_u32_mask(uint8_t bit_count) {
  if (bit_count == 0) {
    return 0;
  }
  if (bit_count >= 32) {
    return UINT32_MAX;
  }
  return (UINT32_C(1) << bit_count) - 1;
}

static const loom_amdgpu_encoding_format_layout_t*
loom_amdgpu_encoding_find_format(const loom_amdgpu_encoding_table_t* table,
                                 uint16_t encoding_format) {
  uint32_t low = 0;
  uint32_t high = table->format_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_amdgpu_encoding_format_layout_t* format = &table->formats[mid];
    if (format->format_id == encoding_format) {
      return format;
    }
    if (format->format_id < encoding_format) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return NULL;
}

static const loom_amdgpu_encoding_field_layout_t*
loom_amdgpu_encoding_find_field(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_format_layout_t* format, uint16_t field_id) {
  for (uint16_t i = 0; i < format->field_count; ++i) {
    const loom_amdgpu_encoding_field_layout_t* field =
        &table->fields[format->field_start + i];
    if (field->field_id == field_id) {
      return field;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_encoding_write_bits(
    loom_amdgpu_encoding_packet_t* packet, uint8_t bit_offset,
    uint8_t bit_count, uint64_t value) {
  uint8_t remaining_bits = bit_count;
  uint8_t current_bit_offset = bit_offset;
  uint64_t current_value = value;
  while (remaining_bits > 0) {
    const uint8_t word_index = current_bit_offset / 32;
    if (word_index >= packet->word_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU encoding bit range starting at bit %u exceeds %u-word packet",
          bit_offset, packet->word_count);
    }
    const uint8_t word_bit_offset = current_bit_offset % 32;
    const uint8_t chunk_bit_count =
        (uint8_t)iree_min(remaining_bits, (uint8_t)(32 - word_bit_offset));
    const uint32_t chunk_mask = loom_amdgpu_encoding_u32_mask(chunk_bit_count);
    packet->words[word_index] =
        (packet->words[word_index] & ~(chunk_mask << word_bit_offset)) |
        ((uint32_t)(current_value & chunk_mask) << word_bit_offset);
    current_value >>= chunk_bit_count;
    current_bit_offset += chunk_bit_count;
    remaining_bits -= chunk_bit_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encoding_set_field(
    const loom_amdgpu_encoding_table_t* table,
    const loom_amdgpu_encoding_format_layout_t* format,
    loom_amdgpu_encoding_packet_t* packet, uint16_t field_id, uint64_t value) {
  const loom_amdgpu_encoding_field_layout_t* field =
      loom_amdgpu_encoding_find_field(table, format, field_id);
  if (field == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU encoding format %" PRIu16
                            " has no field id %" PRIu16,
                            format->format_id, field_id);
  }
  if (field->value_bit_count < 64 && (value >> field->value_bit_count) != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU encoding field %" PRIu16 " value 0x%" PRIx64
                            " does not fit %u source bits",
                            field_id, value, field->value_bit_count);
  }
  for (uint8_t i = 0; i < field->range_count; ++i) {
    const loom_amdgpu_encoding_bit_range_t* bit_range =
        &table->bit_ranges[field->range_start + i];
    if (bit_range->padding_bit_count != 0) {
      const uint64_t padding_mask =
          loom_amdgpu_encoding_u64_mask(bit_range->padding_bit_count);
      const uint64_t padding_value =
          (value >> bit_range->source_bit_offset) & padding_mask;
      if (padding_value != bit_range->padding_value) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU encoding field %" PRIu16
                                " requires padding value 0x%" PRIx64
                                " at source bit %u but found 0x%" PRIx64,
                                field_id, bit_range->padding_value,
                                bit_range->source_bit_offset, padding_value);
      }
    }
    const uint8_t value_bit_offset =
        bit_range->source_bit_offset + bit_range->padding_bit_count;
    const uint64_t range_value =
        (value >> value_bit_offset) &
        loom_amdgpu_encoding_u64_mask(bit_range->bit_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_write_bits(
        packet, bit_range->bit_offset, bit_range->bit_count, range_value));
  }
  return iree_ok_status();
}

iree_string_view_t loom_amdgpu_encoding_format_name(uint16_t encoding_format) {
  switch (encoding_format) {
    case LOOM_AMDGPU_ENCODING_FORMAT_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP1:
      return IREE_SV("sop1");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP1_LITERAL:
      return IREE_SV("sop1_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP2:
      return IREE_SV("sop2");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPP:
      return IREE_SV("sopp");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPC:
      return IREE_SV("sopc");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2:
      return IREE_SV("vop2");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL:
      return IREE_SV("vop2_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3:
      return IREE_SV("vop3");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P:
      return IREE_SV("vop3p");
    case LOOM_AMDGPU_ENCODING_FORMAT_SMEM:
      return IREE_SV("smem");
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
      return IREE_SV("mubuf");
    case LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER:
      return IREE_SV("vbuffer");
    case LOOM_AMDGPU_ENCODING_FORMAT_DS:
      return IREE_SV("ds");
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT:
      return IREE_SV("flat");
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL:
      return IREE_SV("flat_glbl");
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL:
      return IREE_SV("flat_global");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPK:
      return IREE_SV("sopk");
    case LOOM_AMDGPU_ENCODING_FORMAT_VDS:
      return IREE_SV("vds");
    case LOOM_AMDGPU_ENCODING_FORMAT_VFLAT:
      return IREE_SV("vflat");
    case LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL:
      return IREE_SV("vglobal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1:
      return IREE_SV("vop1");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL:
      return IREE_SV("vop1_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA:
      return IREE_SV("vop3p_mfma");
    default:
      return IREE_SV("unknown");
  }
}

bool loom_amdgpu_encoding_field_uses_unified_source(uint16_t field_id) {
  switch (field_id) {
    case LOOM_AMDGPU_ENCODING_FIELD_SRC0:
    case LOOM_AMDGPU_ENCODING_FIELD_SRC1:
    case LOOM_AMDGPU_ENCODING_FIELD_SRC2:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_encoding_sisrc_inline_u32(
    const loom_amdgpu_encoding_table_t* table, uint32_t value,
    uint16_t* out_sisrc) {
  if (value < table->scalar_inline_u32_count) {
    *out_sisrc = (uint16_t)(table->scalar_inline_u32_zero + value);
    return true;
  }
  return false;
}

const loom_amdgpu_encoding_table_t*
loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
    uint16_t descriptor_set_ordinal) {
  const loom_amdgpu_encoding_table_t* const
      tables[LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT] = {
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4] =
              loom_amdgpu_cdna4_encoding_table(),
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3] =
              loom_amdgpu_rdna3_encoding_table(),
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4] =
              loom_amdgpu_rdna4_encoding_table(),
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X] =
              loom_amdgpu_rdna4_gfx125x_encoding_table(),
      };
  if (descriptor_set_ordinal >= IREE_ARRAYSIZE(tables)) {
    return NULL;
  }
  return tables[descriptor_set_ordinal];
}

iree_status_t loom_amdgpu_encoding_pack(
    const loom_amdgpu_encoding_table_t* table, uint16_t encoding_format,
    uint16_t opcode, const loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count,
    loom_amdgpu_encoding_packet_t* out_packet) {
  *out_packet = (loom_amdgpu_encoding_packet_t){0};

  const loom_amdgpu_encoding_format_layout_t* format =
      loom_amdgpu_encoding_find_format(table, encoding_format);
  if (format == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU encoding table '%.*s' has no format id %" PRIu16,
        (int)table->descriptor_set_key.size, table->descriptor_set_key.data,
        encoding_format);
  }
  if (format->word_count > LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU encoding format %" PRIu16
                            " has unsupported word count %" PRIu16,
                            encoding_format, format->word_count);
  }

  out_packet->bit_count = format->bit_count;
  out_packet->word_count = format->word_count;
  for (uint16_t i = 0; i < format->word_count; ++i) {
    out_packet->words[i] = format->identifier_words[i];
  }

  for (iree_host_size_t i = 0; i < field_value_count; ++i) {
    if (opcode != LOOM_AMDGPU_ENCODING_OPCODE_NONE &&
        field_values[i].field_id == LOOM_AMDGPU_ENCODING_FIELD_OP) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU encoding callers must not pass OP as a field value when a "
          "separate opcode is supplied");
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (field_values[j].field_id == field_values[i].field_id) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU encoding repeats field id %" PRIu16,
                                field_values[i].field_id);
      }
    }
  }

  if (opcode != LOOM_AMDGPU_ENCODING_OPCODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_set_field(
        table, format, out_packet, LOOM_AMDGPU_ENCODING_FIELD_OP, opcode));
  }
  for (iree_host_size_t i = 0; i < field_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_set_field(
        table, format, out_packet, field_values[i].field_id,
        field_values[i].value));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_encoding_pack_sopp_simm16(
    const loom_amdgpu_encoding_table_t* table, uint16_t opcode,
    uint16_t immediate, loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU SOPP immediate encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SIMM16,
          .value = immediate,
      },
  };
  return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_SOPP,
                                   opcode, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_sgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint16_t ssrc0,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU s_mov_b32 encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SDST,
          .value = sdst,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SSRC0,
          .value = ssrc0,
      },
  };
  return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_SOP1,
                                   table->s_mov_b32_opcode, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU s_mov_b32 encoding requires an encoding table");
  }
  uint16_t ssrc0 = 0;
  if (loom_amdgpu_encoding_sisrc_inline_u32(table, imm32, &ssrc0)) {
    loom_amdgpu_encoding_field_value_t field_values[] = {
        {
            .field_id = LOOM_AMDGPU_ENCODING_FIELD_SDST,
            .value = sdst,
        },
        {
            .field_id = LOOM_AMDGPU_ENCODING_FIELD_SSRC0,
            .value = ssrc0,
        },
    };
    return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_SOP1,
                                     table->s_mov_b32_opcode, field_values,
                                     IREE_ARRAYSIZE(field_values), out_packet);
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SDST,
          .value = sdst,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SSRC0,
          .value = table->source_literal,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_LITERAL,
          .value = imm32,
      },
  };
  return loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_SOP1_LITERAL, table->s_mov_b32_opcode,
      field_values, IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_vgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint16_t vsrc0,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU v_mov_b32 encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VDST,
          .value = vdst,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SRC0,
          .value = (uint64_t)table->vector_source_vgpr0 + vsrc0,
      },
  };
  return loom_amdgpu_encoding_pack(table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1,
                                   table->v_mov_b32_opcode, field_values,
                                   IREE_ARRAYSIZE(field_values), out_packet);
}

iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet) {
  if (table == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU v_mov_b32 encoding requires an encoding table");
  }
  loom_amdgpu_encoding_field_value_t field_values[] = {
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_VDST,
          .value = vdst,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_SRC0,
          .value = table->source_literal,
      },
      {
          .field_id = LOOM_AMDGPU_ENCODING_FIELD_LITERAL,
          .value = imm32,
      },
  };
  return loom_amdgpu_encoding_pack(
      table, LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL, table->v_mov_b32_opcode,
      field_values, IREE_ARRAYSIZE(field_values), out_packet);
}
