// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzzes generated source-to-low lowering through the synthetic test-low
// target.
//
// The first byte selects a bounded workload scale. Remaining bytes drive the
// structured source generator. Every generated module is expected to verify,
// lower, low-verify, packetize, schedule, and allocate. A failure means either
// the workload generator produced IR outside its claimed contract or the low
// pipeline mishandled a valid source program.

#include <cstddef>
#include <cstdint>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/source_workload.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/context.h"
#include "loom/verify/verify.h"

static loom_context_t g_context;
static loom_target_low_descriptor_registry_t g_descriptor_registry;
static loom_low_lower_policy_registry_t g_policy_registry;
static bool g_initialized = false;

static void trap_with_status(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_abort(status);
}

static void ensure_context(void) {
  if (g_initialized) {
    return;
  }
  trap_with_status(
      loom_testing_context_initialize_all(iree_allocator_system(), &g_context));
  loom_test_low_descriptor_registry_initialize(&g_descriptor_registry);
  loom_test_low_lower_policy_registry_initialize(&g_policy_registry);
  g_initialized = true;
}

static iree_status_t verify_source_module(loom_module_t* module) {
  loom_verify_options_t options = {
      .sink = {loom_diagnostic_stderr_sink, NULL},
  };
  loom_verify_result_t result = {};
  IREE_RETURN_IF_ERROR(loom_verify_module(module, &options, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "generated source failed verification");
  }
  return iree_ok_status();
}

static iree_status_t verify_low_module(loom_module_t* module) {
  const loom_low_verify_options_t options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = &g_descriptor_registry.registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .max_errors = 20,
  };
  loom_low_verify_result_t result = {};
  IREE_RETURN_IF_ERROR(loom_low_verify_module(module, &options, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "generated low function failed verification");
  }
  return iree_ok_status();
}

static iree_status_t lower_and_packetize_module(
    loom_module_t* module, iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t lowering_arena;
  iree_arena_initialize(block_pool, &lowering_arena);

  loom_low_source_selection_t selection = {};
  loom_low_lower_result_t lower_result = {};
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    const loom_low_source_selection_options_t selection_options = {
        .descriptor_registry = &g_descriptor_registry.registry,
        .policy_registry = &g_policy_registry,
    };
    status = loom_low_select_source_func(module, &selection_options,
                                         &lowering_arena, &selection);
  }
  if (iree_status_is_ok(status)) {
    const loom_low_lower_options_t lower_options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &g_descriptor_registry.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = selection.policy,
        .max_errors = 20,
    };
    status = loom_low_lower_function(module, selection.func, &lower_options,
                                     &lower_result);
  }
  if (iree_status_is_ok(status) && lower_result.error_count != 0) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "generated source lowering produced errors");
  }
  if (iree_status_is_ok(status)) {
    status = verify_low_module(module);
  }

  iree_arena_allocator_t packet_arena;
  bool packet_arena_initialized = false;
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &packet_arena);
    packet_arena_initialized = true;
    const loom_low_packetization_options_t packet_options = {
        .descriptor_registry = &g_descriptor_registry.registry,
        .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
    };
    loom_low_packetization_t packetization = {};
    status = loom_low_packetize_function(module, lower_result.low_func_op,
                                         &packet_options, &packet_arena,
                                         &packetization);
    if (iree_status_is_ok(status)) {
      status = loom_low_packet_validate_sidecars(&packetization.schedule,
                                                 &packetization.allocation);
    }
  }
  if (packet_arena_initialized) {
    iree_arena_deinitialize(&packet_arena);
  }
  iree_arena_deinitialize(&lowering_arena);
  return status;
}

static iree_status_t fuzz_one_input(const uint8_t* data, size_t size) {
  uint32_t scale = 1;
  if (size > 0) {
    scale = (uint32_t)(data[0] % 4) + 1;
    ++data;
    --size;
  }

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  loom_test_gen_t gen;
  loom_test_gen_initialize_fuzz(data, size, &gen);
  loom_low_source_workload_config_t workload_config =
      loom_low_source_workload_config_make(IREE_SV("test-low"), scale);

  loom_module_t* module = NULL;
  loom_symbol_ref_t func_ref = loom_symbol_ref_null();
  iree_status_t status = loom_low_source_workload_generate_module(
      &gen, &workload_config, &g_context, &block_pool, &module, &func_ref);
  (void)func_ref;
  if (iree_status_is_ok(status)) {
    status = verify_source_module(module);
  }
  if (iree_status_is_ok(status)) {
    status = lower_and_packetize_module(module, &block_pool);
  }
  if (module) {
    loom_module_free(module);
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  ensure_context();
  trap_with_status(fuzz_one_input(data, size));
  return 0;
}
