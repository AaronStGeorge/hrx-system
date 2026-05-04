// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-owned low descriptor encoding format identifiers.
//
// These values are written into loom_low_descriptor_t::encoding_format_id by
// AMDGPU descriptor overlays. They are intentionally compact and stable within
// Loom; vendor XML encoding names stay in the generator layer instead of being
// linked into native emitters as strings.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ENCODING_H_
#define LOOM_TARGET_ARCH_AMDGPU_ENCODING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT 4u
// Sentinel passed to loom_amdgpu_encoding_pack when the descriptor supplies all
// opcode-like fields through explicit encoding field values.
#define LOOM_AMDGPU_ENCODING_OPCODE_NONE UINT16_MAX

typedef enum loom_amdgpu_encoding_format_e {
  // Descriptor does not carry an AMDGPU machine encoding format.
  LOOM_AMDGPU_ENCODING_FORMAT_NONE = 0,
  // Scalar one-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP1 = 1,
  // Scalar one-source instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP1_LITERAL = 38,
  // Scalar two-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP2 = 2,
  // Scalar program-flow or wait instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPP = 3,
  // Vector two-source 32-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP2 = 4,
  // VOP2 form with a mandatory literal payload.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL = 5,
  // Vector three-source 64-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3 = 6,
  // Packed vector three-source 64-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3P = 7,
  // Scalar memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SMEM = 8,
  // Memory buffer instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_MUBUF = 9,
  // RDNA4 VBUFFER instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER = 10,
  // Local data share instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_DS = 11,
  // Flat generic-address memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_FLAT = 13,
  // CDNA global/flat memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL = 14,
  // RDNA3 global/flat memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL = 15,
  // Scalar compare instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPC = 20,
  // Scalar one-destination immediate instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPK = 21,
  // RDNA4 local data share instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VDS = 22,
  // RDNA4 flat generic-address memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VFLAT = 25,
  // RDNA4 global memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL = 26,
  // Vector one-source 32-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP1 = 30,
  // Vector one-source 32-bit instruction format with mandatory literal.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL = 42,
  // CDNA MFMA packed vector three-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA = 53,
} loom_amdgpu_encoding_format_t;

typedef struct loom_amdgpu_encoding_bit_range_t {
  // Target bit offset populated by this range.
  uint8_t bit_offset;
  // Number of encoded bits copied into the target packet.
  uint8_t bit_count;
  // Source value bit offset before optional padding for this range.
  uint8_t source_bit_offset;
  // Number of source padding bits that must match padding_value.
  uint8_t padding_bit_count;
  // Required value for the low padding bits in this range.
  uint64_t padding_value;
} loom_amdgpu_encoding_bit_range_t;

typedef struct loom_amdgpu_encoding_field_layout_t {
  // Stable target-owned encoding field identifier.
  uint16_t field_id;
  // First bit-range row for this field.
  uint16_t range_start;
  // Number of bit-range rows for this field.
  uint8_t range_count;
  // Number of low source bits consumed by this field including padding.
  uint8_t value_bit_count;
  // Target-owned field flags; bit 0 marks a conditional XML field.
  uint16_t flags;
} loom_amdgpu_encoding_field_layout_t;

typedef struct loom_amdgpu_encoding_format_layout_t {
  // Stable target-owned encoding format identifier.
  uint16_t format_id;
  // Total encoded packet width in bits.
  uint16_t bit_count;
  // Number of 32-bit words occupied by this packet.
  uint16_t word_count;
  // First field-layout row for this format.
  uint16_t field_start;
  // Number of field-layout rows for this format.
  uint16_t field_count;
  // Base identifier words loaded before fields are overlaid.
  uint32_t identifier_words[LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT];
  // Identifier-mask words from the vendor XML.
  uint32_t identifier_mask_words[LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT];
} loom_amdgpu_encoding_format_layout_t;

typedef struct loom_amdgpu_encoding_table_t {
  // Descriptor-set key this table can encode.
  iree_string_view_t descriptor_set_key;
  // Dense generated descriptor-set ordinal this table can encode.
  uint16_t descriptor_set_ordinal;
  // SOP1 opcode used for target-inserted s_mov_b32 register/literal moves.
  uint16_t s_mov_b32_opcode;
  // VOP1 opcode used for target-inserted v_mov_b32 register moves.
  uint16_t v_mov_b32_opcode;
  // Unified source field value selecting the literal word after a packet.
  uint16_t source_literal;
  // Scalar source field value selecting unsigned inline integer zero.
  uint16_t scalar_inline_u32_zero;
  // Number of contiguous unsigned inline integer scalar source values.
  uint16_t scalar_inline_u32_count;
  // Unified source field value selecting VGPR zero.
  uint16_t vector_source_vgpr0;
  // Number of contiguous VGPR unified-source values.
  uint16_t vector_source_vgpr_count;
  // Sorted format-layout rows.
  const loom_amdgpu_encoding_format_layout_t* formats;
  // Number of format-layout rows.
  uint32_t format_count;
  // Dense field-layout rows referenced by format rows.
  const loom_amdgpu_encoding_field_layout_t* fields;
  // Number of field-layout rows.
  uint32_t field_count;
  // Dense bit-range rows referenced by field rows.
  const loom_amdgpu_encoding_bit_range_t* bit_ranges;
  // Number of bit-range rows.
  uint32_t bit_range_count;
} loom_amdgpu_encoding_table_t;

typedef struct loom_amdgpu_encoding_field_value_t {
  // Stable target-owned encoding field identifier.
  uint16_t field_id;
  // Reserved for alignment and future flags.
  uint16_t reserved;
  // Unsigned field value before XML padding/range extraction.
  uint64_t value;
} loom_amdgpu_encoding_field_value_t;

typedef struct loom_amdgpu_encoding_packet_t {
  // Packed little-endian 32-bit instruction words.
  uint32_t words[LOOM_AMDGPU_ENCODING_MAX_WORD_COUNT];
  // Total encoded packet width in bits.
  uint16_t bit_count;
  // Number of valid words in words.
  uint16_t word_count;
} loom_amdgpu_encoding_packet_t;

// Returns a short stable diagnostic name for |encoding_format|.
iree_string_view_t loom_amdgpu_encoding_format_name(uint16_t encoding_format);

// Returns true when |field_id| uses AMDGPU's unified scalar/vector source
// register encoding, where VGPR operands are biased by 0x100.
bool loom_amdgpu_encoding_field_uses_unified_source(uint16_t field_id);

// Returns the generated encoding table for |descriptor_set_ordinal|, or NULL
// when this binary was not linked with a matching table.
const loom_amdgpu_encoding_table_t*
loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
    uint16_t descriptor_set_ordinal);

// Packs a native AMDGPU packet by applying |opcode| and target field values to
// the XML-derived encoding layout. When |opcode| is
// LOOM_AMDGPU_ENCODING_OPCODE_NONE, no implicit OP field is written and callers
// must supply any opcode-like fields explicitly. Field values are architectural
// numbers; XML padding bits are validated before encoded ranges are extracted.
iree_status_t loom_amdgpu_encoding_pack(
    const loom_amdgpu_encoding_table_t* table, uint16_t encoding_format,
    uint16_t opcode, const loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a SOPP packet with its 16-bit immediate field.
iree_status_t loom_amdgpu_encoding_pack_sopp_simm16(
    const loom_amdgpu_encoding_table_t* table, uint16_t opcode,
    uint16_t immediate, loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit SGPR-to-SGPR s_mov_b32 packet using the target table's SOP1
// layout.
iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_sgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint16_t ssrc0,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit immediate-to-SGPR s_mov_b32 packet using an inline scalar
// source when possible and a literal SOP1 form otherwise.
iree_status_t loom_amdgpu_encoding_pack_s_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t sdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit VGPR-to-VGPR v_mov_b32 packet using the target table's VOP1
// layout.
iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_vgpr(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint16_t vsrc0,
    loom_amdgpu_encoding_packet_t* out_packet);

// Packs a 32-bit literal-to-VGPR v_mov_b32 packet using the target table's
// VOP1 literal layout.
iree_status_t loom_amdgpu_encoding_pack_v_mov_b32_u32(
    const loom_amdgpu_encoding_table_t* table, uint16_t vdst, uint32_t imm32,
    loom_amdgpu_encoding_packet_t* out_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ENCODING_H_
