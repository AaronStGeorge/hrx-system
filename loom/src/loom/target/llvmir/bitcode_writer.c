// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/bitcode_writer.h"

#include "loom/target/llvmir/bitcode_format.h"
#include "loom/target/llvmir/bitcode_record_writer.h"
#include "loom/target/llvmir/types.h"

static iree_status_t loom_llvmir_bitcode_write_record_u64(
    loom_llvmir_bitcode_record_writer_t* writer, uint64_t code,
    uint64_t operand) {
  return loom_llvmir_bitcode_record_writer_write_unabbrev_record(writer, code,
                                                                 &operand, 1);
}

static iree_status_t loom_llvmir_bitcode_write_type_block(
    const loom_llvmir_module_t* module,
    loom_llvmir_bitcode_record_writer_t* writer) {
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      writer, LOOM_LLVMIR_BITCODE_TYPE_BLOCK,
      LOOM_LLVMIR_BITCODE_TYPE_ABBREV_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
      writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_NUMENTRY, module->type_count));

  for (iree_host_size_t i = 0; i < module->type_count; ++i) {
    const loom_llvmir_type_t* type = &module->types[i];
    switch (type->kind) {
      case LOOM_LLVMIR_TYPE_VOID: {
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_VOID, NULL, 0));
        break;
      }
      case LOOM_LLVMIR_TYPE_INTEGER: {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
            writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_INTEGER, type->bit_width));
        break;
      }
      case LOOM_LLVMIR_TYPE_FLOAT: {
        switch (type->float_kind) {
          case LOOM_LLVMIR_FLOAT_F16: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_HALF, NULL, 0));
            break;
          }
          case LOOM_LLVMIR_FLOAT_F32: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_FLOAT, NULL, 0));
            break;
          }
          case LOOM_LLVMIR_FLOAT_F64: {
            IREE_RETURN_IF_ERROR(
                loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                    writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_DOUBLE, NULL, 0));
            break;
          }
        }
        break;
      }
      case LOOM_LLVMIR_TYPE_POINTER: {
        IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
            writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_OPAQUE_POINTER,
            type->address_space));
        break;
      }
      case LOOM_LLVMIR_TYPE_VECTOR: {
        uint64_t operands[] = {type->element_count, type->element_type};
        IREE_RETURN_IF_ERROR(
            loom_llvmir_bitcode_record_writer_write_unabbrev_record(
                writer, LOOM_LLVMIR_BITCODE_TYPE_CODE_VECTOR, operands,
                IREE_ARRAYSIZE(operands)));
        break;
      }
    }
  }

  return loom_llvmir_bitcode_record_writer_exit_block(writer);
}

iree_status_t loom_llvmir_bitcode_write_module(
    const loom_llvmir_module_t* module, iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(stream);
  if (module->function_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVM bitcode function emission is not implemented");
  }
  if (module->value_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVM bitcode constant emission is not implemented");
  }
  if (module->attr_group_count != 0 || module->metadata_node_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVM bitcode attribute/metadata emission is not implemented");
  }

  static const uint8_t magic[] = {
      LOOM_LLVMIR_BITCODE_MAGIC_0,
      LOOM_LLVMIR_BITCODE_MAGIC_1,
      LOOM_LLVMIR_BITCODE_MAGIC_2,
      LOOM_LLVMIR_BITCODE_MAGIC_3,
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(magic), magic));

  loom_llvmir_bitstream_writer_t bitstream;
  loom_llvmir_bitstream_writer_initialize(stream, &bitstream);
  loom_llvmir_bitcode_record_writer_t record_writer;
  loom_llvmir_bitcode_record_writer_initialize(&bitstream, &record_writer);

  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_enter_subblock(
      &record_writer, LOOM_LLVMIR_BITCODE_MODULE_BLOCK,
      LOOM_LLVMIR_BITCODE_MODULE_ABBREV_WIDTH));
  IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_write_record_u64(
      &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_VERSION,
      LOOM_LLVMIR_BITCODE_MODULE_VERSION));
  if (!iree_string_view_is_empty(module->target_config.source_name)) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_SOURCE_FILENAME,
        module->target_config.source_name));
  }
  if (!iree_string_view_is_empty(module->target_config.target_triple)) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_TRIPLE,
        module->target_config.target_triple));
  }
  if (!iree_string_view_is_empty(module->target_config.data_layout)) {
    IREE_RETURN_IF_ERROR(loom_llvmir_bitcode_record_writer_write_string_record(
        &record_writer, LOOM_LLVMIR_BITCODE_MODULE_CODE_DATALAYOUT,
        module->target_config.data_layout));
  }
  if (module->type_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_bitcode_write_type_block(module, &record_writer));
  }
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_record_writer_exit_block(&record_writer));
  return loom_llvmir_bitstream_writer_finish(&bitstream);
}
