// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/text_asm_roundtrip_test_util.h"

#include "iree/io/vec_stream.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/registry.h"

namespace loom::testing {

static iree_status_t RegisterDialect(
    loom_context_t* context, uint8_t dialect_id,
    const loom_op_vtable_t* const* (*dialect_vtables_fn)(iree_host_size_t*)) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

LowTextAsmRoundTripHarness::~LowTextAsmRoundTripHarness() { Deinitialize(); }

iree_status_t LowTextAsmRoundTripHarness::Initialize(
    loom_low_descriptor_set_provider_t descriptor_set_provider) {
  if (context_initialized_ || block_pool_initialized_) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low asm round-trip harness is already "
                            "initialized");
  }
  if (!descriptor_set_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "descriptor set provider must not be NULL");
  }

  descriptor_set_provider_ = descriptor_set_provider;
  descriptor_registry_ = {};
  descriptor_registry_.descriptor_set_providers = &descriptor_set_provider_;
  descriptor_registry_.descriptor_set_provider_count = 1;

  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool_);
  block_pool_initialized_ = true;
  loom_context_initialize(iree_allocator_system(), &context_);
  context_initialized_ = true;

  iree_status_t status = loom_test_dialect_register(&context_);
  if (iree_status_is_ok(status)) {
    status =
        RegisterDialect(&context_, LOOM_DIALECT_LOW, loom_low_dialect_vtables);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&context_);
  }
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_asm_environment_initialize(&descriptor_registry_,
                                                        &environment_);
  } else {
    Deinitialize();
  }
  return status;
}

void LowTextAsmRoundTripHarness::Deinitialize() {
  environment_ = {};
  descriptor_registry_ = {};
  descriptor_set_provider_ = nullptr;
  if (context_initialized_) {
    loom_context_deinitialize(&context_);
    context_ = {};
    context_initialized_ = false;
  }
  if (block_pool_initialized_) {
    iree_arena_block_pool_deinitialize(&block_pool_);
    block_pool_ = {};
    block_pool_initialized_ = false;
  }
}

iree_status_t LowTextAsmRoundTripHarness::RoundTrip(
    iree_string_view_t source, iree_string_view_t descriptor_set_key,
    std::string* out_text) {
  if (!out_text) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "round-trip output string is required");
  }
  out_text->clear();

  loom_text_parse_options_t parse_options = {
      .max_errors = 100,
      .low_asm_environment = environment_,
  };
  loom_module_t* module = nullptr;
  IREE_RETURN_IF_ERROR(loom_text_parse(source, IREE_SV("test.loom"), &context_,
                                       &block_pool_, &parse_options, &module));
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm source failed to parse");
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_environment = environment_,
      .low_asm_descriptor_set_key = descriptor_set_key,
  };
  iree_status_t status = loom_text_print_module_to_builder_with_options(
      module, &builder, &print_options);
  loom_module_free(module);

  if (iree_status_is_ok(status)) {
    *out_text = std::string(iree_string_builder_buffer(&builder),
                            iree_string_builder_size(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  IREE_RETURN_IF_ERROR(status);

  loom_module_t* reparsed_module = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_text_parse(iree_make_string_view(out_text->data(), out_text->size()),
                      IREE_SV("printed.loom"), &context_, &block_pool_,
                      &parse_options, &reparsed_module));
  if (!reparsed_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "printed low asm failed to parse");
  }
  loom_module_free(reparsed_module);
  return iree_ok_status();
}

LowFuncAsmRoundTripHarness::~LowFuncAsmRoundTripHarness() { Deinitialize(); }

iree_status_t LowFuncAsmRoundTripHarness::Initialize(
    loom_low_descriptor_set_provider_t descriptor_set_provider,
    const loom_target_bundle_t* target_bundle) {
  if (context_initialized_ || block_pool_initialized_) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low func asm round-trip harness is already "
                            "initialized");
  }
  if (!descriptor_set_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "descriptor set provider must not be NULL");
  }
  if (!target_bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle must not be NULL");
  }

  descriptor_set_provider_ = descriptor_set_provider;
  target_bundle_ = target_bundle;
  descriptor_registry_ = {};
  descriptor_registry_.descriptor_set_providers = &descriptor_set_provider_;
  descriptor_registry_.descriptor_set_provider_count = 1;
  descriptor_registry_.target_bundles = &target_bundle_;
  descriptor_registry_.target_bundle_count = 1;

  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool_);
  block_pool_initialized_ = true;
  loom_context_initialize(iree_allocator_system(), &context_);
  context_initialized_ = true;
  iree_status_t status = RegisterDialect(&context_, LOOM_DIALECT_TARGET,
                                         loom_target_dialect_vtables);
  if (iree_status_is_ok(status)) {
    status =
        RegisterDialect(&context_, LOOM_DIALECT_LOW, loom_low_dialect_vtables);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&context_);
  }
  if (iree_status_is_ok(status)) {
    loom_low_descriptor_text_asm_environment_initialize(&descriptor_registry_,
                                                        &environment_);
  } else {
    Deinitialize();
  }
  return status;
}

void LowFuncAsmRoundTripHarness::Deinitialize() {
  environment_ = {};
  descriptor_registry_ = {};
  descriptor_set_provider_ = nullptr;
  target_bundle_ = nullptr;
  if (context_initialized_) {
    loom_context_deinitialize(&context_);
    context_ = {};
    context_initialized_ = false;
  }
  if (block_pool_initialized_) {
    iree_arena_block_pool_deinitialize(&block_pool_);
    block_pool_ = {};
    block_pool_initialized_ = false;
  }
}

iree_status_t LowFuncAsmRoundTripHarness::ParseModule(
    iree_string_view_t source, iree_string_view_t filename,
    loom_module_t** out_module) {
  *out_module = nullptr;
  loom_text_parse_options_t parse_options = {
      .max_errors = 100,
      .low_asm_environment = environment_,
  };
  IREE_RETURN_IF_ERROR(loom_text_parse(
      source, filename, &context_, &block_pool_, &parse_options, out_module));
  if (*out_module == nullptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low func asm source failed to parse");
  }
  return iree_ok_status();
}

iree_status_t LowFuncAsmRoundTripHarness::VerifyModule(
    const loom_module_t* module) {
  loom_low_verify_options_t verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = &descriptor_registry_,
      .max_errors = 20,
  };
  loom_low_verify_result_t verify_result = {};
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &verify_options, &verify_result));
  if (verify_result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor verification emitted %u errors",
                            verify_result.error_count);
  }
  return iree_ok_status();
}

iree_status_t LowFuncAsmRoundTripHarness::PrintAsmModule(
    const loom_module_t* module, iree_string_view_t descriptor_set_key,
    std::string* out_text) {
  out_text->clear();
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_environment = environment_,
      .low_asm_descriptor_set_key = descriptor_set_key,
  };
  iree_status_t status = loom_text_print_module_to_builder_with_options(
      module, &builder, &print_options);
  if (iree_status_is_ok(status)) {
    *out_text = std::string(iree_string_builder_buffer(&builder),
                            iree_string_builder_size(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

iree_status_t LowFuncAsmRoundTripHarness::WriteModule(
    const loom_module_t* module, std::vector<uint8_t>* out_bytes) {
  out_bytes->clear();
  iree_io_stream_t* stream = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, iree_allocator_system(), &stream));
  iree_status_t status =
      loom_bytecode_write_module(module, stream, nullptr, &block_pool_);
  if (iree_status_is_ok(status)) {
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    out_bytes->resize((size_t)length);
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_read(stream, out_bytes->size(), out_bytes->data(),
                                 nullptr);
  }
  iree_io_stream_release(stream);
  return status;
}

iree_status_t LowFuncAsmRoundTripHarness::ReadModule(
    const std::vector<uint8_t>& bytes, loom_module_t** out_module) {
  *out_module = nullptr;
  loom_bytecode_read_options_t options = {
      .verify_module = false,
      .verify_max_errors = 20,
  };
  loom_bytecode_read_result_t result = {};
  IREE_RETURN_IF_ERROR(loom_bytecode_read_module(
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("low_func_asm_roundtrip.loombc"), &context_, &block_pool_,
      &options, &result, out_module, iree_allocator_system()));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode read emitted %u errors",
                            result.error_count);
  }
  if (*out_module == nullptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode read produced no module");
  }
  return iree_ok_status();
}

iree_status_t LowFuncAsmRoundTripHarness::RoundTripAndVerify(
    iree_string_view_t source, iree_string_view_t descriptor_set_key,
    std::string* out_text) {
  if (!out_text) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "round-trip output string is required");
  }
  out_text->clear();

  loom_module_t* module = nullptr;
  loom_module_t* reparsed_module = nullptr;
  loom_module_t* bytecode_module = nullptr;
  std::vector<uint8_t> bytecode;
  std::string printed_text;

  iree_status_t status =
      ParseModule(source, IREE_SV("low_func_asm_input.loom"), &module);
  if (iree_status_is_ok(status)) {
    status = VerifyModule(module);
  }
  if (iree_status_is_ok(status)) {
    status = PrintAsmModule(module, descriptor_set_key, &printed_text);
  }
  if (iree_status_is_ok(status)) {
    status = ParseModule(
        iree_make_string_view(printed_text.data(), printed_text.size()),
        IREE_SV("low_func_asm_printed.loom"), &reparsed_module);
  }
  if (iree_status_is_ok(status)) {
    status = VerifyModule(reparsed_module);
  }
  if (iree_status_is_ok(status)) {
    status = WriteModule(reparsed_module, &bytecode);
  }
  if (iree_status_is_ok(status)) {
    status = ReadModule(bytecode, &bytecode_module);
  }
  if (iree_status_is_ok(status)) {
    status = VerifyModule(bytecode_module);
  }
  if (iree_status_is_ok(status)) {
    status = PrintAsmModule(bytecode_module, descriptor_set_key, out_text);
  }

  if (bytecode_module) loom_module_free(bytecode_module);
  if (reparsed_module) loom_module_free(reparsed_module);
  if (module) loom_module_free(module);
  return status;
}

}  // namespace loom::testing
