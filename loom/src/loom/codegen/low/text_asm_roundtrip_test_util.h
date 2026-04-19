// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test helper for descriptor-backed low asm parse/print round-trips.

#ifndef LOOM_CODEGEN_LOW_TEXT_ASM_ROUNDTRIP_TEST_UTIL_H_
#define LOOM_CODEGEN_LOW_TEXT_ASM_ROUNDTRIP_TEST_UTIL_H_

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/format/text/low_asm.h"
#include "loom/ir/context.h"

namespace loom::testing {

class LowTextAsmRoundTripHarness {
 public:
  LowTextAsmRoundTripHarness() = default;
  LowTextAsmRoundTripHarness(const LowTextAsmRoundTripHarness&) = delete;
  LowTextAsmRoundTripHarness& operator=(const LowTextAsmRoundTripHarness&) =
      delete;

  ~LowTextAsmRoundTripHarness();

  // Initializes a parser/printer context using one descriptor-set provider.
  iree_status_t Initialize(
      loom_low_descriptor_set_provider_t descriptor_set_provider);

  // Releases all context and arena state. Safe to call repeatedly.
  void Deinitialize();

  // Parses |source|, prints it as low asm using |descriptor_set_key|, reparses
  // the printed text, and returns the printed text in |out_text|.
  iree_status_t RoundTrip(iree_string_view_t source,
                          iree_string_view_t descriptor_set_key,
                          std::string* out_text);

 private:
  // Arena block pool backing parsed modules.
  iree_arena_block_pool_t block_pool_ = {};
  // Parser/printer context with test and low dialects registered.
  loom_context_t context_ = {};
  // Single descriptor-set provider borrowed from the target package.
  loom_low_descriptor_set_provider_t descriptor_set_provider_ = nullptr;
  // Registry view exposing |descriptor_set_provider_|.
  loom_low_descriptor_registry_t descriptor_registry_ = {};
  // Descriptor-backed low asm parser/printer environment.
  loom_text_low_asm_environment_t environment_ = {};
  // True after |block_pool_| has been initialized.
  bool block_pool_initialized_ = false;
  // True after |context_| has been initialized.
  bool context_initialized_ = false;
};

class LowFuncAsmRoundTripHarness {
 public:
  LowFuncAsmRoundTripHarness() = default;
  LowFuncAsmRoundTripHarness(const LowFuncAsmRoundTripHarness&) = delete;
  LowFuncAsmRoundTripHarness& operator=(const LowFuncAsmRoundTripHarness&) =
      delete;

  ~LowFuncAsmRoundTripHarness();

  // Initializes a production parser/printer context using one descriptor-set
  // provider so low.func.def inputs can carry target records without every
  // target test re-registering dialects by hand.
  iree_status_t Initialize(
      loom_low_descriptor_set_provider_t descriptor_set_provider);

  // Releases all context and arena state. Safe to call repeatedly.
  void Deinitialize();

  // Parses |source|, verifies descriptor-local low legality, prints the low
  // function body as asm using |descriptor_set_key|, reparses that text, writes
  // and reads canonical bytecode, verifies the read module, and returns the
  // bytecode-read module's asm text in |out_text|.
  iree_status_t RoundTripAndVerify(iree_string_view_t source,
                                   iree_string_view_t descriptor_set_key,
                                   std::string* out_text);

 private:
  iree_status_t ParseModule(iree_string_view_t source,
                            iree_string_view_t filename,
                            loom_module_t** out_module);
  iree_status_t VerifyModule(const loom_module_t* module);
  iree_status_t PrintAsmModule(const loom_module_t* module,
                               iree_string_view_t descriptor_set_key,
                               std::string* out_text);
  iree_status_t WriteModule(const loom_module_t* module,
                            std::vector<uint8_t>* out_bytes);
  iree_status_t ReadModule(const std::vector<uint8_t>& bytes,
                           loom_module_t** out_module);

  // Arena block pool backing parsed modules and bytecode scratch state.
  iree_arena_block_pool_t block_pool_ = {};
  // Parser/printer context with the production op registry registered.
  loom_context_t context_ = {};
  // Single descriptor-set provider borrowed from the target package.
  loom_low_descriptor_set_provider_t descriptor_set_provider_ = nullptr;
  // Registry view exposing |descriptor_set_provider_|.
  loom_low_descriptor_registry_t descriptor_registry_ = {};
  // Descriptor-backed low asm parser/printer environment.
  loom_text_low_asm_environment_t environment_ = {};
  // True after |block_pool_| has been initialized.
  bool block_pool_initialized_ = false;
  // True after |context_| has been initialized.
  bool context_initialized_ = false;
};

}  // namespace loom::testing

#endif  // LOOM_CODEGEN_LOW_TEXT_ASM_ROUNDTRIP_TEST_UTIL_H_
