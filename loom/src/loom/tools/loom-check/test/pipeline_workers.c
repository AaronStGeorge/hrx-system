// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/test/pipeline_workers.h"

#include <inttypes.h>

#include "loom/analysis/channel_plan.h"
#include "loom/analysis/pipeline_resources.h"
#include "loom/analysis/pipeline_workers.h"
#include "loom/analysis/source_storage_packing.h"
#include "loom/ir/module.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/util/fact_table.h"

// Exact source names make record identity and operand wiring observable without
// exposing allocation addresses, module ordinals, or physical target plans.
static iree_status_t loom_check_pipeline_worker_format(
    const loom_module_t* module, const loom_pipeline_worker_t* worker,
    iree_string_builder_t* output) {
  const loom_pipeline_transport_t* transport = &worker->transport;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "execution=%s block_counts=%s streams=%" PRIhsz "\n",
      transport->count ? "transport" : "worker",
      worker->block_execution_counts ? "exact" : "unknown",
      transport->stream_count));
  for (iree_host_size_t i = 0; i < transport->batch_count; ++i) {
    const loom_pipeline_transport_batch_t* batch = &transport->batches[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "batch %" PRIhsz ": records=%" PRIhsz " replays=%" PRIu64 "\n",
        i, batch->record_count, batch->replays));
    for (iree_host_size_t j = 0; j < batch->record_count; ++j) {
      const loom_pipeline_transport_cohort_t* record = &batch->records[j];
      const iree_string_view_t admission = loom_module_value_name(
          module, loom_op_const_results(record->admission->op)[0]);
      const iree_string_view_t source =
          loom_module_value_name(module, record->admission->channel->value_id);
      const iree_string_view_t release =
          record->release
              ? loom_module_value_name(
                    module, loom_channel_plan_admission(
                                &worker->channels,
                                loom_channel_release_read(record->release)))
              : IREE_SV("none");
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output,
          "  record %.*s from %.*s: admissions=%" PRIu64 " transfers=%" PRIu64
          " release=%.*s\n",
          (int)admission.size, admission.data, (int)source.size, source.data,
          record->admissions, record->transfers, (int)release.size,
          release.data));
      for (iree_host_size_t k = 0; k < record->destination_count; ++k) {
        const loom_pipeline_transport_destination_t* destination =
            &record->destinations[k];
        const iree_string_view_t channel =
            loom_module_value_name(module, destination->channel->value_id);
        const iree_string_view_t reservation = loom_module_value_name(
            module, loom_channel_reserve_write(destination->reservation));
        const iree_string_view_t copy_source = loom_module_value_name(
            module, loom_channel_plan_admission(
                        &worker->channels,
                        loom_channel_copy_source(destination->copy)));
        const iree_string_view_t copy_destination = loom_module_value_name(
            module, loom_channel_plan_admission(
                        &worker->channels,
                        loom_channel_copy_destination(destination->copy)));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            output, "    to %.*s reservation=%.*s copy=%.*s -> %.*s\n",
            (int)channel.size, channel.data, (int)reservation.size,
            reservation.data, (int)copy_source.size, copy_source.data,
            (int)copy_destination.size, copy_destination.data));
      }
    }
  }
  return iree_ok_status();
}

static bool loom_check_pipeline_workers_matches(
    const loom_check_emit_provider_t* provider, iree_string_view_t name) {
  (void)provider;
  return iree_string_view_equal(name, IREE_SV("pipeline-workers"));
}

static iree_status_t loom_check_pipeline_workers_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module) {
  (void)provider;
  loom_module_t* module = native_module->module;
  iree_string_view_t symbol_name =
      iree_string_view_trim(request->target_options);
  if (!iree_string_view_consume_prefix(&symbol_name, IREE_SV("@"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline-workers requires @pipeline");
  }
  const loom_symbol_id_t symbol = loom_module_find_symbol(
      module, loom_module_lookup_string(module, symbol_name));
  if (symbol == LOOM_SYMBOL_ID_INVALID ||
      !loom_pipeline_def_isa(module->symbols.entries[symbol].defining_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline-workers requires a pipeline definition");
  }
  const loom_func_like_t function =
      loom_func_like_cast(module, module->symbols.entries[symbol].defining_op);
  const loom_block_t* entry =
      loom_region_entry_block(loom_func_like_body(function));
  if (!entry->arg_count ||
      !loom_type_is_pool(loom_module_value_type(module, entry->arg_ids[0]))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline-workers requires a pool as first argument");
  }
  iree_arena_allocator_t* arena = request->case_arena;
  loom_value_fact_table_t facts;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&facts, arena, 0));
  loom_type_registry_configure_fact_context(&facts.context);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(&facts, module, function));
  loom_pipeline_resource_pool_t pool = {
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
      .byte_capacity = 1024,
  };
  IREE_RETURN_IF_ERROR(loom_source_storage_packing_allocate(
      (loom_source_storage_packing_interference_callback_t){0}, NULL, 0, arena,
      &pool.packing));
  const loom_pipeline_resource_pool_binding_t binding = {entry->arg_ids[0], 0};
  loom_check_diagnostic_emitter_capture_t capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = module,
      .source_resolver = native_module->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &capture,
  };
  loom_pipeline_resources_t resources;
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_resources_build(
      module, function, &facts, &pool, 1, NULL, 0, &binding, 1, emitter, arena,
      &resources, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  loom_pipeline_worker_t* workers = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_workers_build(module, &resources, &facts,
                                                   NULL, 0, NULL, emitter,
                                                   arena, &workers, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < resources.strand_count && iree_status_is_ok(status); ++i) {
    loom_local_value_domain_restore(&workers[i].value_domain);
    status = loom_check_pipeline_worker_format(module, &workers[i],
                                               &request->result->actual_output);
    loom_local_value_domain_release(&workers[i].value_domain);
  }
  return status;
}

static iree_status_t loom_check_pipeline_workers_append_names(
    const loom_check_emit_provider_t* provider, iree_string_builder_t* output) {
  (void)provider;
  return iree_string_builder_append_cstring(output, "pipeline-workers");
}

const loom_check_emit_provider_t loom_check_test_pipeline_workers_provider = {
    .name = IREE_SVL("pipeline-workers"),
    .match = loom_check_pipeline_workers_matches,
    .execute_native = loom_check_pipeline_workers_execute,
    .append_names = loom_check_pipeline_workers_append_names,
};
