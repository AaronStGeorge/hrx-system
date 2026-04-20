// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/descriptor.h"

#include <inttypes.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// AMDHSA descriptor bit fields
//===----------------------------------------------------------------------===//

#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_SHIFT 0u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH 6u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_SHIFT 6u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH 4u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DENORM_16_64_SHIFT 18u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT 21u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT 23u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT 29u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_MEM_ORDERED_SHIFT 30u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT 31u

#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_PRIVATE_SEGMENT_SHIFT 0u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT 1u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_WIDTH 5u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_X_SHIFT 7u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Y_SHIFT 8u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Z_SHIFT 9u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_INFO_SHIFT 10u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_SHIFT 11u
#define LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_WIDTH 2u

#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_BUFFER_SHIFT 0u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_PTR_SHIFT 1u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_QUEUE_PTR_SHIFT 2u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_KERNARG_SEGMENT_PTR_SHIFT 3u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_ID_SHIFT 4u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_FLAT_SCRATCH_INIT_SHIFT 5u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_SIZE_SHIFT 6u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_WAVEFRONT_SIZE32_SHIFT 10u
#define LOOM_AMDGPU_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK_SHIFT 11u

typedef struct loom_amdgpu_kernel_descriptor_target_t {
  // VGPR encoding granule when wavefront-size-32 mode is enabled.
  uint32_t vgpr_encoding_granule_wave32;
  // VGPR encoding granule when wavefront-size-64 mode is enabled.
  uint32_t vgpr_encoding_granule_wave64;
  // True when flat scratch is architected and legacy user SGPRs are invalid.
  bool has_architected_flat_scratch;
  // True when the target uses the GFX10+ SGPR resource encoding rule.
  bool uses_gfx10_sgpr_encoding;
  // True when DX10 clamp and IEEE mode defaults are supported.
  bool has_dx10_clamp_and_ieee_mode;
} loom_amdgpu_kernel_descriptor_target_t;

static bool loom_amdgpu_kernel_descriptor_is_gfx11_cpu(
    iree_string_view_t target_cpu) {
  static const iree_string_view_t gfx11_processors[] = {
      IREE_SVL("gfx1100"), IREE_SVL("gfx1101"), IREE_SVL("gfx1102"),
      IREE_SVL("gfx1103"), IREE_SVL("gfx1150"), IREE_SVL("gfx1151"),
      IREE_SVL("gfx1152"), IREE_SVL("gfx1153"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(gfx11_processors); ++i) {
    if (iree_string_view_equal(target_cpu, gfx11_processors[i])) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_amdgpu_kernel_descriptor_resolve_target(
    iree_string_view_t target_cpu,
    loom_amdgpu_kernel_descriptor_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_amdgpu_kernel_descriptor_target_t){0};
  if (loom_amdgpu_kernel_descriptor_is_gfx11_cpu(target_cpu)) {
    *out_target = (loom_amdgpu_kernel_descriptor_target_t){
        .vgpr_encoding_granule_wave32 = 8,
        .vgpr_encoding_granule_wave64 = 4,
        .has_architected_flat_scratch = true,
        .uses_gfx10_sgpr_encoding = true,
        .has_dx10_clamp_and_ieee_mode = true,
    };
    return iree_ok_status();
  }
  if (iree_string_view_is_empty(target_cpu)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel descriptor target CPU is required");
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU kernel descriptor target CPU '%.*s' is not supported yet",
      (int)target_cpu.size, target_cpu.data);
}

static uint32_t loom_amdgpu_kernel_descriptor_bit_mask(uint32_t width) {
  return (UINT32_C(1) << width) - 1u;
}

static void loom_amdgpu_kernel_descriptor_set_bits_u32(uint32_t* inout_value,
                                                       uint32_t shift,
                                                       uint32_t width,
                                                       uint32_t value) {
  const uint32_t mask = loom_amdgpu_kernel_descriptor_bit_mask(width);
  *inout_value &= ~(mask << shift);
  *inout_value |= (value & mask) << shift;
}

static void loom_amdgpu_kernel_descriptor_set_bit_u32(uint32_t* inout_value,
                                                      uint32_t shift,
                                                      bool value) {
  if (value) {
    *inout_value |= UINT32_C(1) << shift;
  } else {
    *inout_value &= ~(UINT32_C(1) << shift);
  }
}

static void loom_amdgpu_kernel_descriptor_set_bit_u16(uint16_t* inout_value,
                                                      uint32_t shift,
                                                      bool value) {
  if (value) {
    *inout_value |= (uint16_t)(UINT16_C(1) << shift);
  } else {
    *inout_value &= (uint16_t)~(UINT16_C(1) << shift);
  }
}

static iree_status_t loom_amdgpu_kernel_descriptor_granulated_blocks(
    uint32_t register_count, uint32_t granule, uint32_t field_width,
    uint32_t* out_block_count) {
  IREE_ASSERT_ARGUMENT(out_block_count);
  *out_block_count = 0;
  uint64_t normalized_count = register_count == 0 ? 1u : register_count;
  if (normalized_count > UINT64_MAX - (granule - 1u)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel descriptor register count overflows");
  }
  uint64_t granulated_count = normalized_count + granule - 1u;
  granulated_count /= granule;
  const uint64_t encoded_count = granulated_count - 1u;
  const uint32_t max_value =
      loom_amdgpu_kernel_descriptor_bit_mask(field_width);
  if (encoded_count > max_value) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel descriptor register block count %" PRIu64
        " exceeds field capacity %" PRIu32,
        encoded_count, max_value);
  }
  *out_block_count = (uint32_t)encoded_count;
  return iree_ok_status();
}

static uint32_t loom_amdgpu_kernel_descriptor_implied_user_sgpr_count(
    const loom_amdgpu_kernel_descriptor_t* descriptor) {
  uint32_t count = 0;
  if (descriptor->enable_sgpr_private_segment_buffer) {
    count += 4;
  }
  if (descriptor->enable_sgpr_dispatch_ptr) {
    count += 2;
  }
  if (descriptor->enable_sgpr_queue_ptr) {
    count += 2;
  }
  if (descriptor->enable_sgpr_kernarg_segment_ptr) {
    count += 2;
  }
  if (descriptor->enable_sgpr_dispatch_id) {
    count += 2;
  }
  if (descriptor->enable_sgpr_flat_scratch_init) {
    count += 2;
  }
  if (descriptor->enable_sgpr_private_segment_size) {
    count += 1;
  }
  return count;
}

static iree_status_t loom_amdgpu_kernel_descriptor_validate(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    loom_amdgpu_kernel_descriptor_target_t* out_target) {
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel descriptor is required");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_resolve_target(
      descriptor->target_cpu, out_target));
  if (descriptor->system_vgpr_workitem_id >
      loom_amdgpu_kernel_descriptor_bit_mask(
          LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_WIDTH)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor workitem ID mode must fit in 2 bits");
  }
  const uint32_t implied_user_sgpr_count =
      loom_amdgpu_kernel_descriptor_implied_user_sgpr_count(descriptor);
  if (descriptor->user_sgpr_count < implied_user_sgpr_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel descriptor user SGPR count %" PRIu32
                            " is smaller than implied count %" PRIu32,
                            descriptor->user_sgpr_count,
                            implied_user_sgpr_count);
  }
  if (descriptor->user_sgpr_count >
      loom_amdgpu_kernel_descriptor_bit_mask(
          LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_WIDTH)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel descriptor user SGPR count exceeds target capacity");
  }
  if (out_target->has_architected_flat_scratch &&
      descriptor->enable_sgpr_private_segment_buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor private segment buffer user SGPR is invalid "
        "with architected flat scratch");
  }
  if (out_target->has_architected_flat_scratch &&
      descriptor->enable_sgpr_flat_scratch_init) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor flat scratch init user SGPR is invalid with "
        "architected flat scratch");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_descriptor_validate_metadata_kernel(
    const loom_amdgpu_metadata_kernel_t* metadata_kernel) {
  if (metadata_kernel == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU metadata kernel is required");
  }
  if (metadata_kernel->wavefront_size != 32 &&
      metadata_kernel->wavefront_size != 64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU metadata wavefront size must be either 32 or 64");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_amdgpu_kernel_descriptor_initialize_from_metadata(
    iree_string_view_t target_cpu,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel,
    int64_t kernel_code_entry_byte_offset,
    loom_amdgpu_kernel_descriptor_t* out_descriptor) {
  IREE_ASSERT_ARGUMENT(out_descriptor);
  *out_descriptor = (loom_amdgpu_kernel_descriptor_t){0};
  loom_amdgpu_kernel_descriptor_target_t target = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_resolve_target(target_cpu, &target));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_validate_metadata_kernel(metadata_kernel));
  *out_descriptor = (loom_amdgpu_kernel_descriptor_t){
      .target_cpu = target_cpu,
      .group_segment_fixed_size = metadata_kernel->group_segment_fixed_size,
      .private_segment_fixed_size = metadata_kernel->private_segment_fixed_size,
      .kernarg_size = metadata_kernel->kernarg_segment_size,
      .kernel_code_entry_byte_offset = kernel_code_entry_byte_offset,
      .next_free_sgpr = metadata_kernel->sgpr_count,
      .next_free_vgpr = metadata_kernel->vgpr_count,
      .system_vgpr_workitem_id = 0,
      .enable_wavefront_size32 = metadata_kernel->wavefront_size == 32,
  };
  (void)target;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_kernel_descriptor_validate_metadata(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    const loom_amdgpu_metadata_kernel_t* metadata_kernel) {
  loom_amdgpu_kernel_descriptor_target_t target = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_validate(descriptor, &target));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_validate_metadata_kernel(metadata_kernel));
  if (descriptor->group_segment_fixed_size !=
      metadata_kernel->group_segment_fixed_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor group segment size does not "
                            "match metadata");
  }
  if (descriptor->private_segment_fixed_size !=
      metadata_kernel->private_segment_fixed_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor private segment size does not "
                            "match metadata");
  }
  if (descriptor->kernarg_size != metadata_kernel->kernarg_segment_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor kernarg size does not match "
                            "metadata");
  }
  if (descriptor->next_free_sgpr != metadata_kernel->sgpr_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor SGPR count does not match "
                            "metadata");
  }
  if (descriptor->next_free_vgpr != metadata_kernel->vgpr_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor VGPR count does not match "
                            "metadata");
  }
  if (descriptor->enable_wavefront_size32 !=
      (metadata_kernel->wavefront_size == 32)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor wavefront size does not match "
                            "metadata");
  }
  return iree_ok_status();
}

static void loom_amdgpu_kernel_descriptor_store_le_u16(uint8_t* target,
                                                       uint16_t value) {
  target[0] = (uint8_t)value;
  target[1] = (uint8_t)(value >> 8);
}

static void loom_amdgpu_kernel_descriptor_store_le_u32(uint8_t* target,
                                                       uint32_t value) {
  target[0] = (uint8_t)value;
  target[1] = (uint8_t)(value >> 8);
  target[2] = (uint8_t)(value >> 16);
  target[3] = (uint8_t)(value >> 24);
}

static void loom_amdgpu_kernel_descriptor_store_le_i64(uint8_t* target,
                                                       int64_t value) {
  uint64_t unsigned_value = (uint64_t)value;
  for (iree_host_size_t i = 0; i < 8; ++i) {
    target[i] = (uint8_t)(unsigned_value >> (i * 8));
  }
}

iree_status_t loom_amdgpu_kernel_descriptor_write(
    const loom_amdgpu_kernel_descriptor_t* descriptor,
    iree_byte_span_t target_bytes) {
  loom_amdgpu_kernel_descriptor_target_t target = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_validate(descriptor, &target));
  if (target_bytes.data == NULL ||
      target_bytes.data_length < LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel descriptor output requires at least %u bytes",
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  }

  const uint32_t vgpr_granule = descriptor->enable_wavefront_size32
                                    ? target.vgpr_encoding_granule_wave32
                                    : target.vgpr_encoding_granule_wave64;
  uint32_t vgpr_block_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_granulated_blocks(
      descriptor->next_free_vgpr, vgpr_granule,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH, &vgpr_block_count));

  uint32_t sgpr_block_count = 0;
  if (!target.uses_gfx10_sgpr_encoding) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_descriptor_granulated_blocks(
        descriptor->next_free_sgpr, 16,
        LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH, &sgpr_block_count));
  }

  uint32_t compute_pgm_rsrc1 = 0;
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_VGPR_COUNT_WIDTH, vgpr_block_count);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC1_SGPR_COUNT_WIDTH, sgpr_block_count);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DENORM_16_64_SHIFT, 2,
      3);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT,
      target.has_dx10_clamp_and_ieee_mode);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT,
      target.has_dx10_clamp_and_ieee_mode);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT, true);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_MEM_ORDERED_SHIFT,
      true);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc1, LOOM_AMDGPU_COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT,
      true);

  uint32_t compute_pgm_rsrc2 = 0;
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_PRIVATE_SEGMENT_SHIFT,
      descriptor->enable_private_segment);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC2_USER_SGPR_COUNT_WIDTH,
      descriptor->user_sgpr_count);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_X_SHIFT,
      descriptor->enable_sgpr_workgroup_id_x);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Y_SHIFT,
      descriptor->enable_sgpr_workgroup_id_y);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_ID_Z_SHIFT,
      descriptor->enable_sgpr_workgroup_id_z);
  loom_amdgpu_kernel_descriptor_set_bit_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKGROUP_INFO_SHIFT,
      descriptor->enable_sgpr_workgroup_info);
  loom_amdgpu_kernel_descriptor_set_bits_u32(
      &compute_pgm_rsrc2, LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_SHIFT,
      LOOM_AMDGPU_COMPUTE_PGM_RSRC2_WORKITEM_ID_WIDTH,
      descriptor->system_vgpr_workitem_id);

  uint16_t kernel_code_properties = 0;
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_BUFFER_SHIFT,
      descriptor->enable_sgpr_private_segment_buffer);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_PTR_SHIFT,
      descriptor->enable_sgpr_dispatch_ptr);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties, LOOM_AMDGPU_KERNEL_CODE_PROPERTY_QUEUE_PTR_SHIFT,
      descriptor->enable_sgpr_queue_ptr);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_KERNARG_SEGMENT_PTR_SHIFT,
      descriptor->enable_sgpr_kernarg_segment_ptr);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_DISPATCH_ID_SHIFT,
      descriptor->enable_sgpr_dispatch_id);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_FLAT_SCRATCH_INIT_SHIFT,
      descriptor->enable_sgpr_flat_scratch_init);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_PRIVATE_SEGMENT_SIZE_SHIFT,
      descriptor->enable_sgpr_private_segment_size);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_WAVEFRONT_SIZE32_SHIFT,
      descriptor->enable_wavefront_size32);
  loom_amdgpu_kernel_descriptor_set_bit_u16(
      &kernel_code_properties,
      LOOM_AMDGPU_KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK_SHIFT,
      descriptor->uses_dynamic_stack);

  memset(target_bytes.data, 0, LOOM_AMDGPU_KERNEL_DESCRIPTOR_LENGTH);
  loom_amdgpu_kernel_descriptor_store_le_u32(
      target_bytes.data + 0, descriptor->group_segment_fixed_size);
  loom_amdgpu_kernel_descriptor_store_le_u32(
      target_bytes.data + 4, descriptor->private_segment_fixed_size);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 8,
                                             descriptor->kernarg_size);
  loom_amdgpu_kernel_descriptor_store_le_i64(
      target_bytes.data + 16, descriptor->kernel_code_entry_byte_offset);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 44, 0);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 48,
                                             compute_pgm_rsrc1);
  loom_amdgpu_kernel_descriptor_store_le_u32(target_bytes.data + 52,
                                             compute_pgm_rsrc2);
  loom_amdgpu_kernel_descriptor_store_le_u16(target_bytes.data + 56,
                                             kernel_code_properties);
  loom_amdgpu_kernel_descriptor_store_le_u16(target_bytes.data + 58, 0);
  return iree_ok_status();
}
