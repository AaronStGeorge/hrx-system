// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/bitstream_writer.h"

#include <string.h>

static uint64_t loom_llvmir_low_bit_mask(uint32_t bit_count) {
  return bit_count == 64 ? UINT64_MAX : ((UINT64_C(1) << bit_count) - 1);
}

static iree_status_t loom_llvmir_bitstream_writer_append_bytes(
    loom_llvmir_bitstream_writer_t* writer, const uint8_t* data,
    iree_host_size_t length) {
  const uint8_t* source = data;
  while (length > 0) {
    iree_host_size_t page_remaining =
        LOOM_LLVMIR_BITSTREAM_PAGE_SIZE - writer->page_position;
    if (page_remaining == 0) {
      IREE_RETURN_IF_ERROR(loom_llvmir_bitstream_writer_flush(writer));
      page_remaining = LOOM_LLVMIR_BITSTREAM_PAGE_SIZE;
    }
    iree_host_size_t chunk = length < page_remaining ? length : page_remaining;
    memcpy(writer->page + writer->page_position, source, chunk);
    writer->page_position += chunk;
    source += chunk;
    length -= chunk;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_bitstream_writer_emit_complete_bytes(
    loom_llvmir_bitstream_writer_t* writer) {
  while (writer->pending_bit_count >= 8) {
    uint8_t byte = (uint8_t)(writer->pending_bits & 0xFF);
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitstream_writer_append_bytes(writer, &byte, 1));
    writer->pending_bits >>= 8;
    writer->pending_bit_count -= 8;
  }
  return iree_ok_status();
}

void loom_llvmir_bitstream_writer_initialize(
    loom_output_stream_t* stream, loom_llvmir_bitstream_writer_t* out_writer) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(out_writer);
  memset(out_writer, 0, sizeof(*out_writer));
  out_writer->stream = stream;
}

uint64_t loom_llvmir_bitstream_writer_bit_offset(
    const loom_llvmir_bitstream_writer_t* writer) {
  IREE_ASSERT_ARGUMENT(writer);
  return writer->bit_offset;
}

iree_status_t loom_llvmir_bitstream_writer_write_bits(
    loom_llvmir_bitstream_writer_t* writer, uint64_t value,
    uint32_t bit_count) {
  IREE_ASSERT_ARGUMENT(writer);
  if (bit_count > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bit field width exceeds 64 bits");
  }
  if (bit_count < 64 && (value >> bit_count) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM bit field value exceeds its width");
  }
  if (bit_count == 0) return iree_ok_status();
  if (UINT64_MAX - writer->bit_offset < bit_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitstream offset overflow");
  }

  uint64_t remaining_value = value;
  uint32_t remaining_bit_count = bit_count;
  while (remaining_bit_count > 0) {
    uint32_t available_bit_count = 64 - writer->pending_bit_count;
    uint32_t chunk_bit_count = remaining_bit_count < available_bit_count
                                   ? remaining_bit_count
                                   : available_bit_count;
    uint64_t chunk =
        remaining_value & loom_llvmir_low_bit_mask(chunk_bit_count);
    if (writer->pending_bit_count == 0) {
      writer->pending_bits = chunk;
    } else {
      writer->pending_bits |= chunk << writer->pending_bit_count;
    }
    writer->pending_bit_count += chunk_bit_count;
    remaining_bit_count -= chunk_bit_count;
    if (chunk_bit_count == 64) {
      remaining_value = 0;
    } else {
      remaining_value >>= chunk_bit_count;
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitstream_writer_emit_complete_bytes(writer));
  }

  writer->bit_offset += bit_count;
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitstream_writer_write_vbr(
    loom_llvmir_bitstream_writer_t* writer, uint64_t value, uint32_t width) {
  IREE_ASSERT_ARGUMENT(writer);
  if (width < 2 || width > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM VBR width must be in [2, 64]");
  }

  const uint32_t payload_width = width - 1;
  const uint64_t payload_mask = loom_llvmir_low_bit_mask(payload_width);
  uint64_t remaining_value = value;
  do {
    uint64_t chunk = remaining_value & payload_mask;
    remaining_value >>= payload_width;
    if (remaining_value != 0) {
      chunk |= UINT64_C(1) << payload_width;
    }
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitstream_writer_write_bits(writer, chunk, width));
  } while (remaining_value != 0);
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitstream_writer_align32(
    loom_llvmir_bitstream_writer_t* writer) {
  IREE_ASSERT_ARGUMENT(writer);
  uint32_t misalignment = (uint32_t)(writer->bit_offset & 31);
  if (misalignment == 0) return iree_ok_status();
  return loom_llvmir_bitstream_writer_write_bits(writer, 0, 32 - misalignment);
}

iree_status_t loom_llvmir_bitstream_writer_write_bytes(
    loom_llvmir_bitstream_writer_t* writer, const void* data,
    iree_host_size_t length) {
  IREE_ASSERT_ARGUMENT(writer);
  if (length == 0) return iree_ok_status();
  if (data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty LLVM bitstream byte payload is null");
  }
  if (length > (UINT64_MAX - writer->bit_offset) / 8) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM bitstream offset overflow");
  }

  const uint8_t* bytes = (const uint8_t*)data;
  if (writer->pending_bit_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitstream_writer_append_bytes(writer, bytes, length));
    writer->bit_offset += (uint64_t)length * 8;
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < length; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitstream_writer_write_bits(writer, bytes[i], 8));
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitstream_writer_flush(
    loom_llvmir_bitstream_writer_t* writer) {
  IREE_ASSERT_ARGUMENT(writer);
  if (writer->page_position == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_output_stream_write(
      writer->stream,
      iree_make_string_view((const char*)writer->page, writer->page_position)));
  writer->page_position = 0;
  return iree_ok_status();
}

iree_status_t loom_llvmir_bitstream_writer_finish(
    loom_llvmir_bitstream_writer_t* writer) {
  IREE_ASSERT_ARGUMENT(writer);
  if (writer->pending_bit_count != 0) {
    uint32_t padding_bit_count = 8 - writer->pending_bit_count;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitstream_writer_write_bits(writer, 0, padding_bit_count));
  }
  return loom_llvmir_bitstream_writer_flush(writer);
}
