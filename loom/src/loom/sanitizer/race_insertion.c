// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/race_insertion.h"

#include <string.h>

#include "loom/ops/atomic.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/sanitizer/options.h"
#include "loom/sanitizer/options_cli.h"
#include "loom/sanitizer/site_location.h"
#include "loom/sanitizer/site_payload.h"
#include "loom/util/fact_table.h"

typedef struct loom_sanitizer_insert_race_observations_state_t {
  // Check classes enabled for this pass invocation.
  loom_sanitizer_checks_t checks;
  // True when checks were supplied explicitly.
  bool has_checks_option;
} loom_sanitizer_insert_race_observations_state_t;

#define LOOM_SANITIZER_INSERT_RACE_OBSERVATIONS_STATISTICS(V, statistics_type) \
  V(statistics_type, access_observations_inserted,                             \
    "access-observations-inserted",                                            \
    "Number of sanitizer.race.access ops inserted.")                           \
  V(statistics_type, sync_observations_inserted, "sync-observations-inserted", \
    "Number of sanitizer.race.sync ops inserted.")

LOOM_PASS_STATISTICS_DEFINE(
    loom_sanitizer_insert_race_observations_statistics,
    loom_sanitizer_insert_race_observations_statistics_t,
    LOOM_SANITIZER_INSERT_RACE_OBSERVATIONS_STATISTICS)

static const loom_pass_option_def_t kSanitizerInsertRaceObservationsOptions[] =
    {
        {IREE_SVL("checks"),
         IREE_SVL("Sanitizer checks to insert: none, all, race, or tsan.")},
};

static const loom_pass_info_t
    loom_sanitizer_insert_race_observations_pass_info_storage = {
        .name = IREE_SVL("sanitizer-insert-race-observations"),
        .description =
            IREE_SVL("Insert sanitizer race observations for enabled checks."),
        .kind = LOOM_PASS_FUNCTION,
        .option_defs = kSanitizerInsertRaceObservationsOptions,
        .option_count = IREE_ARRAYSIZE(kSanitizerInsertRaceObservationsOptions),
        .statistic_layout =
            &loom_sanitizer_insert_race_observations_statistics_layout,
};

const loom_pass_info_t* loom_sanitizer_insert_race_observations_pass_info(
    void) {
  return &loom_sanitizer_insert_race_observations_pass_info_storage;
}

static iree_status_t loom_sanitizer_insert_race_observations_parse_option(
    void* user_data, iree_string_view_t name, iree_string_view_t value) {
  loom_sanitizer_insert_race_observations_state_t* state =
      (loom_sanitizer_insert_race_observations_state_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("checks"))) {
    if (state->has_checks_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate option 'checks' for pass "
                              "'sanitizer-insert-race-observations'");
    }
    IREE_RETURN_IF_ERROR(loom_sanitizer_checks_parse(
        value, IREE_SV("sanitizer-insert-race-observations option 'checks'"),
        &state->checks));
    state->has_checks_option = true;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown option '%.*s' for pass 'sanitizer-insert-race-observations'",
      (int)name.size, name.data);
}

iree_status_t loom_sanitizer_insert_race_observations_create(
    loom_pass_t* pass, iree_string_view_t options) {
  loom_sanitizer_insert_race_observations_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  memset(state, 0, sizeof(*state));
  state->checks = LOOM_SANITIZER_CHECK_RACE;
  if (pass->decoded_options) {
    for (uint16_t i = 0; i < pass->decoded_options->option_count; ++i) {
      const loom_pass_decoded_option_t* option =
          &pass->decoded_options->options[i];
      if (!option->present) continue;
      if (iree_string_view_equal(option->schema->name, IREE_SV("checks"))) {
        IREE_RETURN_IF_ERROR(loom_sanitizer_checks_parse(
            option->string_value,
            IREE_SV("sanitizer-insert-race-observations option 'checks'"),
            &state->checks));
        state->has_checks_option = true;
        continue;
      }
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown decoded option '%.*s' for pass "
                              "'sanitizer-insert-race-observations'",
                              (int)option->schema->name.size,
                              option->schema->name.data);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_pass_options_parse(
        pass->info->name, options,
        (loom_pass_option_parse_callback_t){
            .fn = loom_sanitizer_insert_race_observations_parse_option,
            .user_data = state,
        }));
  }
  pass->state = state;
  return iree_ok_status();
}

static bool loom_sanitizer_race_check_enabled(const loom_pass_t* pass) {
  const loom_sanitizer_insert_race_observations_state_t* state =
      (const loom_sanitizer_insert_race_observations_state_t*)pass->state;
  const loom_sanitizer_checks_t enabled =
      state ? state->checks : LOOM_SANITIZER_CHECK_RACE;
  return iree_any_bit_set(enabled, LOOM_SANITIZER_CHECK_RACE);
}

static bool loom_sanitizer_i64_arrays_equal(loom_attribute_t lhs,
                                            loom_attribute_t rhs) {
  if (lhs.kind != LOOM_ATTR_I64_ARRAY || rhs.kind != LOOM_ATTR_I64_ARRAY ||
      lhs.count != rhs.count) {
    return false;
  }
  for (uint16_t i = 0; i < lhs.count; ++i) {
    if (lhs.i64_array[i] != rhs.i64_array[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_sanitizer_value_slices_equal(loom_value_slice_t lhs,
                                              loom_value_slice_t rhs) {
  if (lhs.count != rhs.count) return false;
  for (uint16_t i = 0; i < lhs.count; ++i) {
    if (lhs.values[i] != rhs.values[i]) return false;
  }
  return true;
}

static bool loom_sanitizer_preceded_by_matching_race_access(
    loom_op_t* op, loom_sanitizer_race_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices, bool atomic,
    loom_atomic_ordering_t ordering, loom_atomic_scope_t scope) {
  loom_op_t* previous_op = op->prev_op;
  if (!previous_op || !loom_sanitizer_race_access_isa(previous_op) ||
      loom_sanitizer_race_access_kind(previous_op) != kind ||
      loom_sanitizer_race_access_view(previous_op) != view ||
      loom_sanitizer_race_access_atomic(previous_op) != atomic) {
    return false;
  }
  if (atomic && (loom_sanitizer_race_access_ordering(previous_op) != ordering ||
                 loom_sanitizer_race_access_scope(previous_op) != scope)) {
    return false;
  }
  return loom_sanitizer_value_slices_equal(
             loom_sanitizer_race_access_indices(previous_op), indices) &&
         loom_sanitizer_i64_arrays_equal(
             loom_sanitizer_race_access_static_indices(previous_op),
             static_indices);
}

static bool loom_sanitizer_followed_by_matching_race_sync(
    loom_op_t* op, loom_value_fact_memory_space_t memory_space,
    loom_atomic_ordering_t ordering, loom_atomic_scope_t scope) {
  loom_op_t* next_op = op->next_op;
  return next_op && loom_sanitizer_race_sync_isa(next_op) &&
         loom_sanitizer_race_sync_memory_space(next_op) == memory_space &&
         loom_sanitizer_race_sync_ordering(next_op) == ordering &&
         loom_sanitizer_race_sync_scope(next_op) == scope;
}

static bool loom_sanitizer_view_memory_space(
    const loom_rewriter_t* rewriter, loom_value_id_t view,
    loom_value_fact_memory_space_t* out_memory_space) {
  *out_memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!rewriter->fact_table) return false;
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(
          &rewriter->fact_table->context,
          loom_rewriter_value_facts(rewriter, view), &reference)) {
    return false;
  }
  *out_memory_space = reference.memory_space;
  return true;
}

static iree_status_t loom_sanitizer_build_race_access(
    loom_module_t* module, loom_rewriter_t* rewriter,
    loom_sanitizer_race_access_kind_t kind, loom_value_id_t view,
    loom_value_slice_t indices, loom_attribute_t static_indices, bool atomic,
    loom_atomic_ordering_t ordering, loom_atomic_scope_t scope,
    loom_location_id_t source_location, loom_op_t** out_op) {
  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_RACE,
      .check_kind = LOOM_SANITIZER_CHECK_KIND_DATA_RACE,
      .provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      .lane_policy = LOOM_SANITIZER_LANE_POLICY_PER_LANE,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, source_location, &payload, &site_location));
  const loom_sanitizer_race_access_build_flags_t build_flags =
      atomic ? LOOM_SANITIZER_RACE_ACCESS_BUILD_FLAG_HAS_ORDERING |
                   LOOM_SANITIZER_RACE_ACCESS_BUILD_FLAG_HAS_SCOPE
             : 0;
  return loom_sanitizer_race_access_build(
      &rewriter->builder, build_flags, kind, view, indices.values,
      indices.count, static_indices.i64_array, static_indices.count, atomic,
      ordering, scope, site_location, out_op);
}

static iree_status_t loom_sanitizer_try_instrument_race_access_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  loom_sanitizer_race_access_kind_t kind = 0;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  loom_value_slice_t indices = {0};
  loom_attribute_t static_indices = loom_attr_absent();
  bool atomic = false;
  loom_atomic_ordering_t ordering = 0;
  loom_atomic_scope_t scope = 0;
  if (loom_view_load_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ;
    view = loom_view_load_view(op);
    indices = loom_view_load_indices(op);
    static_indices = loom_view_load_static_indices(op);
  } else if (loom_view_store_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE;
    view = loom_view_store_view(op);
    indices = loom_view_store_indices(op);
    static_indices = loom_view_store_static_indices(op);
  } else if (loom_view_atomic_reduce_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE;
    view = loom_view_atomic_reduce_view(op);
    indices = loom_view_atomic_reduce_indices(op);
    static_indices = loom_view_atomic_reduce_static_indices(op);
    atomic = true;
    ordering = loom_view_atomic_reduce_ordering(op);
    scope = loom_view_atomic_reduce_scope(op);
  } else if (loom_view_atomic_rmw_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE;
    view = loom_view_atomic_rmw_view(op);
    indices = loom_view_atomic_rmw_indices(op);
    static_indices = loom_view_atomic_rmw_static_indices(op);
    atomic = true;
    ordering = loom_view_atomic_rmw_ordering(op);
    scope = loom_view_atomic_rmw_scope(op);
  } else if (loom_view_atomic_cmpxchg_isa(op)) {
    kind = LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE;
    view = loom_view_atomic_cmpxchg_view(op);
    indices = loom_view_atomic_cmpxchg_indices(op);
    static_indices = loom_view_atomic_cmpxchg_static_indices(op);
    atomic = true;
    ordering = loom_view_atomic_cmpxchg_success_ordering(op);
    scope = loom_view_atomic_cmpxchg_scope(op);
  } else {
    return iree_ok_status();
  }

  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_sanitizer_view_memory_space(rewriter, view, &memory_space) ||
      memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  if (loom_sanitizer_preceded_by_matching_race_access(
          op, kind, view, indices, static_indices, atomic, ordering, scope)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* observe_op = NULL;
  IREE_RETURN_IF_ERROR(loom_sanitizer_build_race_access(
      module, rewriter, kind, view, indices, static_indices, atomic, ordering,
      scope, op->location, &observe_op));
  (void)observe_op;
  loom_sanitizer_insert_race_observations_statistics_t* statistics =
      loom_sanitizer_insert_race_observations_statistics(pass);
  ++statistics->access_observations_inserted;
  return iree_ok_status();
}

static iree_status_t loom_sanitizer_try_instrument_race_sync_op(
    loom_pass_t* pass, loom_module_t* module, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  if (!loom_kernel_barrier_isa(op)) return iree_ok_status();
  const loom_value_fact_memory_space_t memory_space =
      loom_kernel_barrier_memory_space(op);
  const loom_atomic_ordering_t ordering = loom_kernel_barrier_ordering(op);
  const loom_atomic_scope_t scope = loom_kernel_barrier_scope(op);
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }
  if (scope != LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sanitizer race observation does not yet support non-workgroup "
        "kernel.barrier scope");
  }
  if (loom_sanitizer_followed_by_matching_race_sync(op, memory_space, ordering,
                                                    scope)) {
    return iree_ok_status();
  }

  const loom_sanitizer_site_payload_t payload = {
      .site_kind = LOOM_SANITIZER_SITE_KIND_RACE,
      .check_kind = LOOM_SANITIZER_CHECK_KIND_DATA_RACE,
      .provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_COMPILER_CONTRACT,
      .lane_policy = LOOM_SANITIZER_LANE_POLICY_PER_LANE,
      .lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL,
      .flags = 0,
      .extension_data = iree_const_byte_span_empty(),
  };
  loom_location_id_t site_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_sanitizer_make_site_location(
      module, op->location, &payload, &site_location));
  loom_builder_set_after(&rewriter->builder, op);
  loom_op_t* observe_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_sanitizer_race_sync_build(&rewriter->builder, memory_space, ordering,
                                     scope, site_location, &observe_op));
  (void)observe_op;
  loom_sanitizer_insert_race_observations_statistics_t* statistics =
      loom_sanitizer_insert_race_observations_statistics(pass);
  ++statistics->sync_observations_inserted;
  return iree_ok_status();
}

iree_status_t loom_sanitizer_insert_race_observations_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function) {
  if (!loom_sanitizer_race_check_enabled(pass)) return iree_ok_status();
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_value_fact_table_t* value_facts = NULL;
  iree_status_t status = loom_pass_value_facts_prepare(
      pass, module, loom_pass_value_fact_scope_function(function),
      &value_facts);
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_enable_analysis(&rewriter, function, value_facts);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_seed_function(&rewriter, function);
  }
  while (iree_status_is_ok(status)) {
    loom_op_t* op = loom_rewriter_pop(&rewriter);
    if (!op) break;
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) continue;
    status = loom_sanitizer_try_instrument_race_access_op(pass, module,
                                                          &rewriter, op);
    if (!iree_status_is_ok(status)) continue;
    status =
        loom_sanitizer_try_instrument_race_sync_op(pass, module, &rewriter, op);
  }

  if (iree_status_is_ok(status) &&
      (rewriter.created_op_count != 0 || rewriter.erased_op_count != 0 ||
       iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED))) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status) && pass->value_facts) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  return status;
}
