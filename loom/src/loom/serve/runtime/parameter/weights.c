// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/weights.h"

#include <string.h>

#include "iree/async/semaphore.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/parameter/inputs.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/json.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/program.h"
#include "loomc/iree.h"

enum { WEIGHT_LANES = 4 };

typedef struct loom_serve_weight_span_t {
  // Final placement with a key borrowed from the plan's owned key storage.
  loom_cmd_program_parameter_t parameter;
  // Unique final storage index, resolved once before any queue work.
  iree_host_size_t storage_index;
  // Owned cached preparation command, or null for unchanged bytes.
  iree_hal_command_buffer_t* prepare;
  // Indexed source ranges for out-of-place preparation; empty for direct reads.
  loom_serve_weight_input_recipe_t inputs;
} loom_serve_weight_span_t;

typedef struct loom_serve_weight_group_t {
  // Borrowed consecutive spans in one final allocation. Prepared groups own
  // exactly one tensor; consecutive canonical spans share one readiness edge.
  loom_serve_weight_span_t* spans;
  // Transaction-local view retained by all reads and preparations of this
  // group.
  iree_hal_buffer_t* buffer;
  // Number of spans exposed to the parameter provider.
  iree_host_size_t count;
  // Previous preparation frontier on this group's lane.
  uint64_t wait_value;
  // Read and preparation frontier established by this group.
  uint64_t signal_value;
  // Final input read frontier consumed by this group's preparation command.
  uint64_t read_value;
} loom_serve_weight_group_t;

typedef struct loom_serve_weight_read_t {
  // Captured source range, or NULL for a canonical/in-place gather group.
  const loom_serve_weight_input_t* input;
  // This lane's byte origin in the shared staging allocation.
  iree_device_size_t staging_offset;
  // Previous input read or previous preparation on the same lane.
  uint64_t wait_value;
  // Completed input read, or completed canonical group without preparation.
  uint64_t signal_value;
} loom_serve_weight_read_t;

typedef struct loom_serve_weight_source_batch_t {
  // First gather in the source's contiguous batch.
  iree_host_size_t first;
  // Number of populated gathers for this source.
  iree_host_size_t count;
  // Payload bytes accepted if this provider publishes its complete batch.
  uint64_t byte_length;
} loom_serve_weight_source_batch_t;

typedef struct loom_serve_weight_storage_t {
  // Owned unique root retained independently of the caller's output views.
  iree_hal_buffer_t* buffer;
  // Owned virtual reservation, or NULL for explicitly fixed backing.
  loom_serve_virtual_buffer_t* reservation;
} loom_serve_weight_storage_t;

struct loom_serve_weights_t {
  // Allocator owning the plan and cold activation metadata.
  iree_allocator_t host_allocator;
  // Borrowed device enclosing the complete plan lifetime.
  iree_hal_device_t* device;
  // Borrowed native pool for fixed roots; outlives all resolved storage.
  iree_hal_pool_t* storage_pool;
  // Borrowed exact file transfer queue.
  iree_hal_queue_t* transfer;
  // Borrowed exact in-place preparation queue.
  iree_hal_queue_t* dispatch;
  // Borrowed shared physical budget, or NULL for fixed backing.
  loom_serve_memory_pool_t* pool;
  // Borrowed shared transient backing, separate from retained parameter budget.
  loom_serve_workspace_t* workspace;
  // Parameter-only accounting across all unique roots.
  loom_serve_memory_statistics_t statistics;
  // Accepted streaming/preparation work, never reconstructed from events.
  loom_serve_weight_activity_t activity;
  // Final consumer ownership before virtual roots can be unmapped at teardown.
  loom_serve_retirement_t retirement;
  // Activation lanes retained independently of their submitting host turn.
  struct {
    // Native backing work, joined independently of queue readiness on failure.
    loom_serve_memory_operation_t* backing;
    // Owned native mapping readiness imported by the file-loading lanes.
    iree_hal_semaphore_t* backing_ready;
    // Private fixed-allocation completion, independent of streaming failures.
    iree_async_semaphore_timepoint_t backing_timepoint;
    // Accepted I/O/preparation views, including after failed readiness.
    loom_serve_retirement_t retirement;
    // Owned lane readiness handles imported into the consuming model timeline.
    iree_hal_semaphore_t* semaphores[WEIGHT_LANES];
    // Last submitted preparation or canonical read on each initialized lane.
    uint64_t values[WEIGHT_LANES];
    // Initialized handles, including a partially failed activation.
    iree_host_size_t count;
  } activation;
  // Native release borrower, joined before reactivation or destruction.
  loom_serve_memory_operation_t* deactivation;
  // Owned checkpoint domains, imported-file caches, and captured input ranges.
  loom_serve_weight_inputs_t* inputs;
  // Owned placement/preparation records, with keys in the owned key storage.
  loom_serve_weight_span_t* spans;
  // Owned parameter key bytes, independent of source/JIT lifetime.
  char* keys;
  // Number of initialized placement records, including partial creation.
  iree_host_size_t span_count;
  // Owned unique root records; aliasing output slots do not duplicate these.
  loom_serve_weight_storage_t* storages;
  // Number of initialized unique roots, including partial creation.
  iree_host_size_t storage_count;
  // Backing and all file/preparation work were submitted for this activation.
  bool active;
  // Owned terminal producer failure, replayed to later consumers without I/O.
  iree_status_t activation_status;
};

static const loom_serve_weight_span_t* loom_serve_weight_find(
    iree_host_size_t count, const loom_serve_weight_span_t* spans,
    iree_string_view_t key) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (iree_string_view_equal(spans[i].parameter.key, key)) {
      return &spans[i];
    }
  }
  return NULL;
}

typedef struct loom_serve_weight_root_binding_t {
  // Unique storage allocation selected by parameter identity and placement.
  iree_host_size_t storage_index;
  // Byte range of this root within the selected storage allocation.
  iree_hal_buffer_range_t range;
} loom_serve_weight_root_binding_t;

// Resolves root sharing before submitting any I/O. Only newly allocated roots
// append spans, so tensors shared by multiple stages are read and prepared
// exactly once.
static iree_status_t loom_serve_weight_resolve_root(
    loom_serve_weights_t* weights, const loom_cmd_program_t* program,
    loom_cmd_program_parameter_root_t root, iree_host_size_t* span_count,
    loom_serve_weight_span_t* spans,
    iree_hal_pool_reservation_request_t* requests,
    loom_serve_weight_root_binding_t* out_binding) {
  const iree_host_size_t prior_count = *span_count;
  const loom_serve_weight_span_t* shared = NULL;
  iree_device_size_t origin = 0;
  iree_host_size_t count = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    const loom_serve_weight_span_t* existing =
        loom_serve_weight_find(prior_count, spans, parameter.key);
    if (!count++) {
      shared = existing;
      if (shared) {
        if (shared->parameter.byte_offset < parameter.byte_offset) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "shared parameter root begins before storage");
        }
        origin = shared->parameter.byte_offset - parameter.byte_offset;
      }
    }
    if ((existing != NULL) != (shared != NULL) ||
        (existing &&
         (existing->storage_index != shared->storage_index ||
          existing->parameter.byte_offset != origin + parameter.byte_offset ||
          existing->parameter.byte_length != parameter.byte_length))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter root cannot share existing placement");
    }
  }
  if (!count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty model parameter root");
  }
  if (shared) {
    *out_binding = (loom_serve_weight_root_binding_t){
        shared->storage_index, {origin, root.required_byte_length}};
    return iree_ok_status();
  }
  const iree_host_size_t storage_index = weights->storage_count++;
  requests[storage_index] = iree_hal_make_pool_reservation_request(
      root.required_byte_length, root.minimum_alignment);
  *out_binding = (loom_serve_weight_root_binding_t){
      storage_index, {0, root.required_byte_length}};
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != root.fixed_buffer_index) {
      continue;
    }
    spans[(*span_count)++] = (loom_serve_weight_span_t){
        .parameter = parameter,
        .storage_index = storage_index,
    };
  }
  return iree_ok_status();
}

typedef struct loom_serve_weight_preparer_t {
  // Owned source declaration, independent of the policy invocation.
  loom_serve_preparation_t* declaration;
  // Owned recorded command retaining its executables independently of the JIT.
  iree_hal_command_buffer_t* command;
  // Final output alone, or final output followed by captured input staging.
  uint32_t binding_count;
} loom_serve_weight_preparer_t;

typedef struct loom_serve_weight_preparation_t {
  // Initial CPU task; its destruction transfers ownership to the JIT chain.
  loomc_task_t task;
  // Exclusively borrowed unactivated plan through terminal completion.
  loom_serve_weights_t* weights;
  // Exclusively borrowed source catalog through terminal completion.
  loom_serve_jit_t* jit;
  // Recording policy applied to each source-selected preparation command.
  iree_hal_command_buffer_mode_t command_mode;
  // Borrowed checkpoint sources and policy, retained through completion.
  const loom_serve_preparation_parameter_t* parameter;
  // Copied namespace descriptor borrowing its strings and package through
  // completion. Checkpoints use its filesystem base; policy uses its package.
  loom_serve_sources_t sources;
  // Owned distinct preparers, queried once before native compilation.
  loom_serve_weight_preparer_t* preparers;
  // Number of owned preparers, including partial policy failure.
  iree_host_size_t preparer_count;
  // Current native compilation ordinal; equals count at successful completion.
  iree_host_size_t preparer_index;
  // Owned per-span preparer ordinal; SIZE_MAX preserves canonical file bytes.
  iree_host_size_t* span_preparers;
  // CPU policy/indexing outcome, consumed by the initial task destructor.
  iree_status_t status;
  // Joins the already-queued fixed root allocation before publication.
  iree_async_semaphore_timepoint_t backing_timepoint;
  // Terminal ownership transfer, invoked after all job storage is released.
  loom_serve_weights_completion_fn_t completion;
  // Borrowed terminal callback payload.
  void* user_data;
} loom_serve_weight_preparation_t;

static iree_status_t loom_serve_weight_query_preparer(
    loom_serve_program_t* program, const iree_vm_ref_types_t* types,
    iree_vm_function_t function, iree_vm_module_t* library,
    iree_vm_module_t* input_library, loom_cmd_program_parameter_t parameter,
    iree_allocator_t host_allocator, loom_serve_preparation_t** out_preparation,
    loom_serve_weight_input_recipe_t* out_inputs) {
  *out_preparation = NULL;
  iree_vm_buffer_t* key = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)parameter.key.data, parameter.key.size),
      iree_vm_buffer_release_callback_null(), host_allocator, &key));
  iree_vm_variant_t arguments[] = {
      iree_vm_buffer_variant_from_ptr_move(types, &key)};
  iree_vm_variant_t results[1] = {0};
  loom_serve_preparation_module_begin(library, 0, 0);
  loom_serve_weight_inputs_module_begin(input_library);
  iree_status_t status =
      iree_vm_invoke(loom_serve_program_invocation(program), function,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results));
  loom_serve_preparation_t* preparation =
      loom_serve_preparation_module_take(library);
  *out_inputs = loom_serve_weight_inputs_module_take(input_library);
  const iree_host_size_t stage_count =
      preparation ? loom_serve_preparation_stage_count(preparation) : 0;
  int64_t byte_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[0], &byte_length);
  }
  if (iree_status_is_ok(status)) {
    if (stage_count > 1 ||
        (stage_count && loom_serve_preparation_stage(preparation, 0)->kind !=
                            LOOM_SERVE_PREPARATION_STAGE_COMMAND)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "weight policy must declare at most one command");
    } else if (!stage_count && (byte_length != 0 || out_inputs->count)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unchanged weight policy must return zero byte length and no inputs");
    } else if (stage_count && (byte_length <= 0 || (uint64_t)byte_length !=
                                                       parameter.byte_length)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "weight policy for '%.*s' requires %" PRId64
                                " bytes, reflected parameter has %" PRIu64,
                                (int)parameter.key.size, parameter.key.data,
                                byte_length, (uint64_t)parameter.byte_length);
    }
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    *out_preparation = preparation;
  } else {
    loom_serve_preparation_free(preparation);
  }
  return status;
}

static iree_status_t loom_serve_weight_prepare_policy(
    loom_serve_weight_preparation_t* job) {
  loom_serve_weights_t* weights = job->weights;
  const iree_allocator_t host_allocator = weights->host_allocator;
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(host_allocator, &environment));
  loom_serve_program_t* program = NULL;
  iree_vm_module_t* libraries[4] = {NULL, NULL, NULL, NULL};
  const iree_string_view_t roots[] = {IREE_SVL("prepare_weight")};
  iree_status_t status = loom_serve_preparation_module_create(
      environment, host_allocator, &libraries[0]);
  if (iree_status_is_ok(status)) {
    status = loom_serve_weight_inputs_module_create(
        weights->inputs, environment, host_allocator, &libraries[1]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(environment, host_allocator,
                                           &libraries[2]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(environment, NULL, host_allocator,
                                            &libraries[3]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_allocate(
        loom_serve_jit_target_environment(job->jit), environment, &job->sources,
        job->parameter->policy, IREE_ARRAYSIZE(roots), roots,
        (iree_vm_module_span_t){libraries, IREE_ARRAYSIZE(libraries)},
        loom_serve_jit_report(job->jit), host_allocator, &program);
  }
  iree_vm_ref_types_t types = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types);
  }
  iree_vm_function_t function = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_lookup_function(
        loom_serve_program_process(program), IREE_SV("model"),
        IREE_SV("prepare_weight"), &function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, weights->span_count,
                                         sizeof(*job->preparers),
                                         (void**)&job->preparers);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, weights->span_count,
                                         sizeof(*job->span_preparers),
                                         (void**)&job->span_preparers);
  }
  for (iree_host_size_t i = 0;
       i < weights->span_count && iree_status_is_ok(status); ++i) {
    loom_serve_preparation_t* declaration = NULL;
    status = loom_serve_weight_query_preparer(
        program, &types, function, libraries[0], libraries[1],
        weights->spans[i].parameter, host_allocator, &declaration,
        &weights->spans[i].inputs);
    job->span_preparers[i] = SIZE_MAX;
    if (iree_status_is_ok(status) && declaration) {
      iree_host_size_t p = 0;
      const uint32_t binding_count = weights->spans[i].inputs.count ? 2 : 1;
      while (p < job->preparer_count &&
             (binding_count != job->preparers[p].binding_count ||
              !loom_serve_preparation_equal(declaration,
                                            job->preparers[p].declaration))) {
        ++p;
      }
      if (p == job->preparer_count) {
        loom_serve_weight_preparer_t* preparer =
            &job->preparers[job->preparer_count++];
        preparer->declaration = declaration;
        preparer->binding_count = binding_count;
        declaration = NULL;
      }
      job->span_preparers[i] = p;
    }
    loom_serve_preparation_free(declaration);
  }
  loom_serve_program_free(program);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(libraries); ++i) {
    iree_vm_module_release(libraries[i]);
  }
  iree_vm_environment_free(environment);
  return status;
}

static void loom_serve_weight_preparation_finish(
    loom_serve_weight_preparation_t* job, iree_status_t status) {
  loom_serve_weights_t* weights = job->weights;
  const iree_allocator_t host_allocator = weights->host_allocator;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      const iree_host_size_t p = job->span_preparers[i];
      if (p != SIZE_MAX) {
        weights->spans[i].prepare = job->preparers[p].command;
        iree_hal_command_buffer_retain(weights->spans[i].prepare);
      }
    }
  }
  for (iree_host_size_t i = 0; i < job->preparer_count; ++i) {
    loom_serve_preparation_free(job->preparers[i].declaration);
    iree_hal_command_buffer_release(job->preparers[i].command);
  }
  iree_allocator_free(host_allocator, job->preparers);
  iree_allocator_free(host_allocator, job->span_preparers);
  const loom_serve_weights_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(host_allocator, job);
  completion(user_data, status);
}

static void loom_serve_weight_preparation_backing_ready(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_serve_weight_preparation_finish(user_data, status);
}

// Each native completion submits the next preparer without a worker join.
static void loom_serve_weight_preparation_compiled(
    void* user_data, iree_status_t status, loom_serve_jit_stage_t* compiled);

static void loom_serve_weight_preparation_next(
    loom_serve_weight_preparation_t* job) {
  if (job->preparer_index == job->preparer_count) {
    iree_hal_semaphore_t* backing = job->weights->activation.backing_ready;
    if (!backing) {
      loom_serve_weight_preparation_finish(job, iree_ok_status());
      return;
    }
    job->backing_timepoint.callback =
        loom_serve_weight_preparation_backing_ready;
    job->backing_timepoint.user_data = job;
    // Completion may run inline and release job before acceptance returns.
    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)backing, 1, &job->backing_timepoint);
    if (!iree_status_is_ok(status)) {
      loom_serve_weight_preparation_finish(job, status);
    }
    return;
  }
  const loom_serve_preparation_stage_t* stage = loom_serve_preparation_stage(
      job->preparers[job->preparer_index].declaration, 0);
  iree_status_t status =
      loom_serve_jit_compile_async(job->jit, stage->root, &stage->config,
                                   loom_serve_weight_preparation_compiled, job);
  // An accepted completion may already have destroyed job. Rejection owes no
  // callback and returns the still-exclusive job to this terminal path.
  if (!iree_status_is_ok(status)) {
    loom_serve_weight_preparation_finish(job, status);
  }
}

static void loom_serve_weight_preparation_compiled(
    void* user_data, iree_status_t status, loom_serve_jit_stage_t* compiled) {
  loom_serve_weight_preparation_t* job = user_data;
  loom_serve_weight_preparer_t* preparer = &job->preparers[job->preparer_index];
  if (iree_status_is_ok(status)) {
    const loom_cmd_program_requirements_t* requirements =
        &loom_serve_jit_stage_program(compiled)->requirements;
    if (requirements->fixed_buffer_count ||
        requirements->rebindable_binding_count != preparer->binding_count ||
        requirements->transient.required_byte_length ||
        requirements->launch_counts.required_byte_length) {
      const iree_string_view_t name =
          loom_serve_preparation_stage(preparer->declaration, 0)->root;
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "weight preparer '%.*s' requires %u mutable bindings "
          "and no global scratch",
          (int)name.size, name.data, preparer->binding_count);
    } else {
      status = loom_serve_jit_stage_record(
          compiled, iree_hal_queue_family(job->weights->dispatch),
          job->command_mode, NULL, NULL, &preparer->command);
    }
  }
  loom_serve_jit_stage_free(compiled);
  if (iree_status_is_ok(status)) {
    ++job->preparer_index;
    loom_serve_weight_preparation_next(job);
  } else {
    loom_serve_weight_preparation_finish(job, status);
  }
}

static void loom_serve_weight_preparation_execute(
    loomc_task_t* task, loomc_host_size_t worker_ordinal) {
  loom_serve_weight_preparation_t* job = (loom_serve_weight_preparation_t*)task;
  const iree_allocator_t host_allocator = job->weights->host_allocator;
  iree_status_t status = loom_serve_weight_inputs_allocate(
      job->parameter, &job->sources, WEIGHT_LANES, host_allocator,
      &job->weights->inputs);
  if (iree_status_is_ok(status)) {
    status = loom_serve_weight_prepare_policy(job);
  }
  if (iree_status_is_ok(status) &&
      loom_serve_weight_inputs_staging_length(job->weights->inputs) >
          UINT64_MAX / WEIGHT_LANES) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "weight input lanes exceed device address space");
  }
  job->status = status;
}

static void loom_serve_weight_preparation_destroy(loomc_task_t* task) {
  loom_serve_weight_preparation_t* job = (loom_serve_weight_preparation_t*)task;
  if (iree_status_is_ok(job->status)) {
    loom_serve_weight_preparation_next(job);
  } else {
    loom_serve_weight_preparation_finish(job, job->status);
  }
}

static const loomc_task_vtable_t weight_preparation_vtable = {
    .execute = loom_serve_weight_preparation_execute,
    .destroy = loom_serve_weight_preparation_destroy,
};

iree_status_t loom_serve_weights_prepare_async(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources,
    const loom_serve_preparation_parameter_t* parameter,
    loom_serve_weights_completion_fn_t completion, void* user_data) {
  loom_serve_weight_preparation_t* job = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(weights->host_allocator,
                                             sizeof(*job), (void**)&job));
  loomc_task_initialize(&weight_preparation_vtable, &job->task);
  job->weights = weights;
  job->jit = jit;
  job->command_mode = command_mode;
  job->sources = *sources;
  job->parameter = parameter;
  job->completion = completion;
  job->user_data = user_data;
  iree_status_t status = iree_status_from_loomc(
      loomc_task_sink_submit(loom_serve_host_compiler_sink(host), &job->task));
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(weights->host_allocator, job);
  }
  return status;
}

typedef struct loom_serve_weight_preparation_join_t {
  // Serializes callback publication with the outer owner's final observation.
  iree_slim_mutex_t mutex;
  // Outer application notification, never waited on by a shared worker.
  iree_notification_t notification;
  // Final preparation result, owned by the outer caller after publication.
  iree_status_t status;
  // Terminal callback has published all of its state.
  bool ready;
} loom_serve_weight_preparation_join_t;

static void loom_serve_weight_preparation_ready(void* user_data,
                                                iree_status_t status) {
  loom_serve_weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  join->status = status;
  join->ready = true;
  iree_notification_post(&join->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&join->mutex);
}

static bool loom_serve_weight_preparation_is_ready(void* user_data) {
  loom_serve_weight_preparation_join_t* join = user_data;
  iree_slim_mutex_lock(&join->mutex);
  const bool ready = join->ready;
  iree_slim_mutex_unlock(&join->mutex);
  return ready;
}

iree_status_t loom_serve_weights_prepare(
    loom_serve_weights_t* weights, loom_serve_host_t* host,
    loom_serve_jit_t* jit, iree_hal_command_buffer_mode_t command_mode,
    const loom_serve_sources_t* sources,
    const loom_serve_preparation_parameter_t* parameter) {
  loom_serve_weight_preparation_join_t join = {0};
  iree_slim_mutex_initialize(&join.mutex);
  iree_notification_initialize(&join.notification);
  iree_status_t status = loom_serve_weights_prepare_async(
      weights, host, jit, command_mode, sources, parameter,
      loom_serve_weight_preparation_ready, &join);
  if (iree_status_is_ok(status)) {
    iree_notification_await(&join.notification,
                            loom_serve_weight_preparation_is_ready, &join,
                            iree_infinite_timeout());
    status = join.status;
  }
  iree_notification_deinitialize(&join.notification);
  iree_slim_mutex_deinitialize(&join.mutex);
  return status;
}

static iree_status_t loom_serve_weight_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const loom_serve_weight_group_t* group = user_data;
  const loom_cmd_program_parameter_t parameter = group->spans[i].parameter;
  *out_key = parameter.key;
  *out_span = (iree_io_parameter_span_t){
      .buffer_offset = parameter.byte_offset,
      .length = parameter.byte_length,
  };
  return iree_ok_status();
}

static iree_status_t loom_serve_weight_input_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const loom_serve_weight_read_t* read = user_data;
  *out_key = read->input->entry->key;
  *out_span = read->input->span;
  out_span->buffer_offset += read->staging_offset;
  return iree_ok_status();
}

// The shared scope owns raw allocation/pool lifetime independently of the
// plan's readiness. Tracked views cover every payload read and pack dispatch.
static iree_status_t loom_serve_weight_allocate_staging(
    loom_serve_weights_t* weights, iree_device_size_t stride,
    loom_serve_workspace_scope_t** out_scope, iree_hal_buffer_t** out_buffer,
    iree_hal_semaphore_t** out_ready) {
  const iree_device_size_t length = stride * WEIGHT_LANES;
  const iree_device_size_t alignment =
      loom_serve_weight_inputs_staging_alignment(weights->inputs);
  IREE_RETURN_IF_ERROR(
      loom_serve_workspace_reserve(weights->workspace, length, alignment));
  IREE_RETURN_IF_ERROR(
      loom_serve_workspace_scope_allocate(weights->workspace, true, out_scope));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_ready));
  iree_hal_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_workspace_scope_acquire(*out_scope, &pool));
  uint64_t value = 1;
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(length, alignment);
  iree_status_t status =
      iree_hal_queue_alloca(weights->dispatch, iree_hal_semaphore_list_empty(),
                            (iree_hal_semaphore_list_t){1, out_ready, &value},
                            pool, 1, &request, out_buffer);
  if (iree_status_is_ok(status)) {
    loom_serve_workspace_scope_capture(*out_scope, pool, *out_buffer);
    status = loom_serve_retirement_track(&weights->activation.retirement,
                                         out_buffer, weights->host_allocator);
  } else {
    iree_hal_pool_release(pool);
  }
  return status;
}

static iree_status_t loom_serve_weight_stream(loom_serve_weights_t* weights) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_allocator_t host_allocator = weights->host_allocator;
  const iree_host_size_t span_count = weights->span_count;
  loom_serve_weight_span_t* spans = weights->spans;
  iree_hal_device_t* device = weights->device;
  loom_serve_weight_group_t* groups = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc_array(host_allocator, span_count,
                                      sizeof(*groups), (void**)&groups));
  const uint32_t source_count =
      loom_serve_weight_inputs_source_count(weights->inputs);
  loom_serve_weight_source_batch_t* batches = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, source_count, sizeof(*batches), (void**)&batches);
  iree_hal_buffer_t** views = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, weights->storage_count,
                                         sizeof(*views), (void**)&views);
  }
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    views[i] = weights->storages[i].buffer;
    iree_hal_buffer_retain(views[i]);
    status = loom_serve_retirement_track(&weights->activation.retirement,
                                         &views[i], host_allocator);
  }
  const iree_device_size_t staging_stride =
      loom_serve_weight_inputs_staging_length(weights->inputs);
  loom_serve_workspace_scope_t* scope = NULL;
  iree_hal_buffer_t* staging = NULL;
  iree_hal_semaphore_t* staging_ready = NULL;
  if (staging_stride && iree_status_is_ok(status)) {
    status = loom_serve_weight_allocate_staging(weights, staging_stride, &scope,
                                                &staging, &staging_ready);
  }
  iree_hal_semaphore_t* initial_semaphores[2];
  uint64_t initial_values[] = {1, 1};
  iree_host_size_t initial_count = 0;
  if (weights->activation.backing_ready) {
    initial_semaphores[initial_count++] = weights->activation.backing_ready;
  }
  if (staging_ready) {
    initial_semaphores[initial_count++] = staging_ready;
  }
  iree_hal_semaphore_t* reads[WEIGHT_LANES] = {0};
  iree_hal_semaphore_t** ready = weights->activation.semaphores;
  uint64_t* frontiers = weights->activation.values;
  for (iree_host_size_t i = 0; i < WEIGHT_LANES && iree_status_is_ok(status);
       ++i) {
    status =
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &reads[i]);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready[i]);
      if (iree_status_is_ok(status)) {
        ++weights->activation.count;
        if (initial_count) {
          frontiers[i] = 1;
          status = iree_hal_queue_barrier(
              weights->transfer,
              (iree_hal_semaphore_list_t){initial_count, initial_semaphores,
                                          initial_values},
              (iree_hal_semaphore_list_t){1, &ready[i], &frontiers[i]}, NULL,
              IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
        }
      }
    }
  }
  iree_host_size_t group_count = 0;
  iree_host_size_t read_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < span_count; ++i) {
      const loom_serve_weight_span_t* span = &spans[i];
      if (!i || span->prepare || spans[i - 1].prepare ||
          span->storage_index != spans[i - 1].storage_index) {
        loom_serve_weight_group_t* group = &groups[group_count++];
        group->spans = &spans[i];
        group->buffer = views[span->storage_index];
      }
      ++groups[group_count - 1].count;
    }
    // Count by source once; prefix offsets let us publish one contiguous batch
    // per provider without rescanning tensors for each input checkpoint.
    for (iree_host_size_t i = 0; i < group_count; ++i) {
      const loom_serve_weight_group_t* group = &groups[i];
      const loom_serve_weight_input_recipe_t recipe = group->spans[0].inputs;
      if (recipe.count) {
        for (uint32_t p = 0; p < recipe.count; ++p) {
          const loom_serve_weight_input_t* input =
              loom_serve_weight_inputs_at(weights->inputs, recipe.first + p);
          ++batches[input->source].count;
          batches[input->source].byte_length += input->span.length;
        }
      } else {
        ++batches[0].count;
        for (iree_host_size_t p = 0; p < group->count; ++p) {
          batches[0].byte_length += group->spans[p].parameter.byte_length;
        }
      }
    }
    for (uint32_t s = 0; s < source_count; ++s) {
      batches[s].first = read_count;
      read_count += batches[s].count;
      batches[s].count = 0;
    }
  }
  iree_io_parameter_gather_t* gathers = NULL;
  loom_serve_weight_read_t* read_records = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, read_count,
                                         sizeof(*gathers), (void**)&gathers);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, read_count,
                                         sizeof(*read_records),
                                         (void**)&read_records);
  }
  if (iree_status_is_ok(status)) {
    uint64_t read_values[WEIGHT_LANES] = {0};
    for (iree_host_size_t i = 0; i < group_count; ++i) {
      loom_serve_weight_group_t* group = &groups[i];
      const iree_host_size_t lane = i % WEIGHT_LANES;
      group->wait_value = frontiers[lane];
      group->signal_value = ++frontiers[lane];
      const loom_serve_weight_input_recipe_t recipe = group->spans[0].inputs;
      const uint32_t count = recipe.count ? recipe.count : 1;
      for (uint32_t p = 0; p < count; ++p) {
        const loom_serve_weight_input_t* input =
            recipe.count
                ? loom_serve_weight_inputs_at(weights->inputs, recipe.first + p)
                : NULL;
        loom_serve_weight_source_batch_t* batch =
            &batches[input ? input->source : 0];
        const iree_host_size_t ordinal = batch->first + batch->count++;
        loom_serve_weight_read_t* read = &read_records[ordinal];
        read->input = input;
        read->staging_offset = lane * staging_stride;
        read->wait_value = p ? read_values[lane] : group->wait_value;
        const bool prepared = group->spans[0].prepare != NULL;
        read->signal_value =
            prepared ? ++read_values[lane] : group->signal_value;
        gathers[ordinal] = (iree_io_parameter_gather_t){
            .target_buffer = input ? staging : group->buffer,
            .count = input ? 1 : group->count,
            .enumerator =
                input
                    ? (iree_io_parameter_enumerator_t){loom_serve_weight_input_enumerate,
                                                       read}
                    : (iree_io_parameter_enumerator_t){loom_serve_weight_enumerate,
                                                       group},
            .wait_semaphore_list = {1, p ? &reads[lane] : &ready[lane],
                                    &read->wait_value},
            .signal_semaphore_list = {1, prepared ? &reads[lane] : &ready[lane],
                                      &read->signal_value},
        };
      }
      group->read_value = read_values[lane];
    }
    for (uint32_t s = 0; s < source_count && iree_status_is_ok(status); ++s) {
      if (batches[s].count) {
        status = iree_io_parameter_provider_gather_batch(
            loom_serve_weight_inputs_provider(weights->inputs, s), device,
            weights->transfer, batches[s].count, gathers + batches[s].first);
        if (iree_status_is_ok(status)) {
          weights->activity.read_bytes += batches[s].byte_length;
          weights->activity.groups += batches[s].count;
        }
      }
    }
  }
  for (iree_host_size_t i = 0; i < group_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_weight_group_t* group = &groups[i];
    const loom_serve_weight_span_t* span = group->spans;
    if (!span->prepare) {
      continue;
    }
    const iree_host_size_t lane = i % WEIGHT_LANES;
    const iree_hal_buffer_binding_t bindings[] = {
        {group->buffer, span->parameter.byte_offset,
         span->parameter.byte_length},
        {staging, lane * staging_stride, span->inputs.byte_length}};
    status = iree_hal_queue_execute(
        weights->dispatch,
        (iree_hal_semaphore_list_t){1, &reads[lane], &group->read_value},
        (iree_hal_semaphore_list_t){1, &ready[lane], &group->signal_value},
        span->prepare,
        (iree_hal_buffer_binding_table_t){span->inputs.count ? 2 : 1, bindings},
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      weights->activity.prepared_bytes += span->parameter.byte_length;
    }
  }
  if (staging && iree_status_is_ok(status)) {
    iree_hal_buffer_t* root = iree_hal_buffer_allocated_buffer(staging);
    uint64_t value = frontiers[0] + 1;
    status = iree_hal_queue_dealloca(
        weights->dispatch,
        (iree_hal_semaphore_list_t){WEIGHT_LANES, ready, frontiers},
        (iree_hal_semaphore_list_t){1, &ready[0], &value}, 1, &root);
    if (iree_status_is_ok(status)) {
      frontiers[0] = value;
    }
  }
  if (!iree_status_is_ok(status)) {
    // Release waits for preparations that were never submitted. Queue-owned
    // file, command and buffer references survive until those operations
    // retire.
    for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
      if (reads[i]) {
        iree_hal_semaphore_fail(reads[i], iree_status_clone(status));
      }
      if (ready[i]) {
        iree_hal_semaphore_fail(ready[i], iree_status_clone(status));
      }
    }
  }
  for (iree_host_size_t i = 0; i < WEIGHT_LANES; ++i) {
    iree_hal_semaphore_release(reads[i]);
  }
  iree_hal_semaphore_release(staging_ready);
  iree_hal_buffer_release(staging);
  loom_serve_workspace_scope_retire(scope);
  if (views) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      iree_hal_buffer_release(views[i]);
    }
  }
  iree_allocator_free(host_allocator, views);
  iree_allocator_free(host_allocator, gathers);
  iree_allocator_free(host_allocator, read_records);
  iree_allocator_free(host_allocator, batches);
  iree_allocator_free(host_allocator, groups);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void loom_serve_weights_backing_completed(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_serve_weights_t* weights = user_data;
  // The private semaphore preserves the allocation result for streaming waits.
  iree_status_free(status);
  loom_serve_retirement_release(&weights->activation.retirement);
}

iree_status_t loom_serve_weights_allocate(
    iree_hal_device_t* device, iree_hal_queue_t* transfer,
    iree_hal_queue_t* dispatch, iree_hal_pool_t* storage_pool,
    loom_serve_memory_pool_t* pool, loom_serve_workspace_t* workspace,
    iree_host_size_t shared_root_count, iree_host_size_t root_count,
    const loom_serve_weight_root_t* roots, iree_allocator_t host_allocator,
    loom_serve_weights_t** out_weights) {
  *out_weights = NULL;
  loom_serve_weights_t* weights = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*weights),
                                             (void**)&weights));
  weights->host_allocator = host_allocator;
  weights->device = device;
  weights->transfer = transfer;
  weights->dispatch = dispatch;
  weights->storage_pool = storage_pool;
  weights->pool = pool;
  weights->workspace = workspace;
  loom_serve_retirement_initialize(&weights->retirement);
  loom_serve_retirement_initialize(&weights->activation.retirement);
  *out_weights = weights;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t capacity =
      shared_root_count ? roots[0].program->parameters.count : 0;
  for (iree_host_size_t i = shared_root_count; i < root_count; ++i) {
    capacity += roots[i].program->parameters.count;
  }
  iree_status_t status = iree_allocator_malloc_array(host_allocator, capacity,
                                                     sizeof(*weights->spans),
                                                     (void**)&weights->spans);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, root_count,
                                         sizeof(*weights->storages),
                                         (void**)&weights->storages);
  }
  loom_serve_weight_root_binding_t* bindings = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, root_count,
        sizeof(*bindings) + sizeof(iree_hal_pool_reservation_request_t) +
            sizeof(iree_hal_buffer_t*),
        (void**)&bindings);
  }
  iree_hal_pool_reservation_request_t* requests =
      bindings ? (iree_hal_pool_reservation_request_t*)(bindings + root_count)
               : NULL;
  iree_hal_buffer_t** buffers =
      requests ? (iree_hal_buffer_t**)(requests + root_count) : NULL;
  if (shared_root_count && iree_status_is_ok(status)) {
    loom_cmd_program_parameter_root_t root = roots[0].root;
    for (iree_host_size_t i = 1; i < shared_root_count; ++i) {
      const loom_cmd_program_parameter_root_t other = roots[i].root;
      root.required_byte_length =
          iree_max(root.required_byte_length, other.required_byte_length);
      root.minimum_alignment =
          iree_max(root.minimum_alignment, other.minimum_alignment);
    }
    status = loom_serve_weight_resolve_root(
        weights, roots[0].program, root, &weights->span_count, weights->spans,
        requests, &bindings[0]);
  }
  for (iree_host_size_t i = shared_root_count;
       i < root_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_weight_resolve_root(
        weights, roots[i].program, roots[i].root, &weights->span_count,
        weights->spans, requests, &bindings[i]);
  }
  if (iree_status_is_ok(status) && !pool && weights->storage_count) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &weights->activation.backing_ready);
    if (iree_status_is_ok(status)) {
      weights->activation.backing_timepoint.callback =
          loom_serve_weights_backing_completed;
      weights->activation.backing_timepoint.user_data = weights;
      loom_serve_retirement_acquire(&weights->activation.retirement);
      status = iree_async_semaphore_acquire_timepoint(
          (iree_async_semaphore_t*)weights->activation.backing_ready, 1,
          &weights->activation.backing_timepoint);
      if (!iree_status_is_ok(status)) {
        loom_serve_retirement_release(&weights->activation.retirement);
      }
    }
    if (iree_status_is_ok(status)) {
      uint64_t value = 1;
      status = iree_hal_queue_alloca(
          transfer, iree_hal_semaphore_list_empty(),
          (iree_hal_semaphore_list_t){1, &weights->activation.backing_ready,
                                      &value},
          storage_pool, weights->storage_count, requests, buffers);
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_fail(weights->activation.backing_ready,
                                iree_status_clone(status));
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    loom_serve_weight_storage_t* storage = &weights->storages[i];
    if (pool) {
      status = loom_serve_virtual_buffer_allocate(
          pool, requests[i].allocation_size, requests[i].min_alignment,
          &weights->statistics, &storage->reservation);
      if (iree_status_is_ok(status)) {
        buffers[i] = loom_serve_virtual_buffer_handle(storage->reservation);
        iree_hal_buffer_retain(buffers[i]);
      }
    }
    storage->buffer = buffers[i];
    buffers[i] = NULL;
  }
  for (iree_host_size_t i = 0; i < root_count && iree_status_is_ok(status);
       ++i) {
    if (i && i < shared_root_count) {
      *roots[i].buffer = *roots[0].buffer;
      iree_hal_buffer_retain(*roots[i].buffer);
    } else {
      const loom_serve_weight_root_binding_t binding = bindings[i];
      status = iree_hal_buffer_subspan(
          weights->storages[binding.storage_index].buffer,
          binding.range.byte_offset, binding.range.byte_length, host_allocator,
          roots[i].buffer);
      if (iree_status_is_ok(status)) {
        status = loom_serve_retirement_track(&weights->retirement,
                                             roots[i].buffer, host_allocator);
      }
    }
  }
  if (bindings) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      iree_hal_buffer_release(buffers[i]);
    }
  }
  iree_allocator_free(host_allocator, bindings);
  if (iree_status_is_ok(status)) {
    iree_host_size_t key_bytes = 0;
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      key_bytes += weights->spans[i].parameter.key.size;
    }
    status = iree_allocator_malloc(host_allocator, key_bytes,
                                   (void**)&weights->keys);
    if (iree_status_is_ok(status)) {
      char* target = weights->keys;
      for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
        iree_string_view_t* key = &weights->spans[i].parameter.key;
        memcpy(target, key->data, key->size);
        key->data = target;
        target += key->size;
      }
    }
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_host_size_t loom_serve_weights_parameter_count(
    const loom_serve_weights_t* weights) {
  return weights->span_count;
}

loom_cmd_program_parameter_t loom_serve_weights_parameter_at(
    const loom_serve_weights_t* weights, iree_host_size_t index) {
  return weights->spans[index].parameter;
}

static iree_status_t loom_serve_weights_activate_storage(
    loom_serve_weights_t* weights) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0,
                                   (int64_t)weights->statistics.reserved_bytes);
  iree_status_t status = iree_ok_status();
  if (weights->pool && weights->storage_count) {
    loom_serve_memory_range_t* ranges = NULL;
    status = iree_allocator_malloc_array(weights->host_allocator,
                                         weights->storage_count,
                                         sizeof(*ranges), (void**)&ranges);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(
          weights->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &weights->activation.backing_ready);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
        ranges[i] = (loom_serve_memory_range_t){
            weights->storages[i].reservation, 0,
            iree_hal_buffer_byte_length(weights->storages[i].buffer)};
      }
      bool admitted = false;
      status = loom_serve_memory_pool_commit_async(
          weights->pool, weights->storage_count, ranges,
          weights->activation.backing_ready, 1, &admitted,
          &weights->activation.backing);
      if (iree_status_is_ok(status) && !admitted) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "physical pool cannot back model parameters");
      }
    }
    iree_allocator_free(weights->host_allocator, ranges);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_weight_stream(weights);
  }
  if (iree_status_is_ok(status)) {
    weights->active = true;
    ++weights->activity.activations;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_serve_weights_activate(loom_serve_weights_t* weights) {
  IREE_ASSERT(!weights->deactivation);
  if (!iree_status_is_ok(weights->activation_status)) {
    return iree_status_clone(weights->activation_status);
  }
  if (!weights->active) {
    weights->activation_status = loom_serve_weights_activate_storage(weights);
  }
  return iree_status_clone(weights->activation_status);
}

iree_status_t loom_serve_weights_import_readiness(
    loom_serve_weights_t* weights, loom_serve_execution_t* execution) {
  return loom_serve_execution_barrier(
      execution, (iree_hal_semaphore_list_t){weights->activation.count,
                                             weights->activation.semaphores,
                                             weights->activation.values});
}

iree_status_t loom_serve_weights_await(loom_serve_weights_t* weights) {
  iree_status_t status = iree_hal_semaphore_list_wait(
      (iree_hal_semaphore_list_t){weights->activation.count,
                                  weights->activation.semaphores,
                                  weights->activation.values},
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  loom_serve_retirement_await(&weights->activation.retirement);
  loom_serve_memory_operation_await(weights->activation.backing);
  return iree_status_join(status, loom_serve_memory_operation_query_status(
                                      weights->activation.backing));
}

iree_status_t loom_serve_weights_deactivate(loom_serve_weights_t* weights,
                                            bool* out_complete) {
  *out_complete = false;
  if (!weights->pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fixed parameter backing cannot deactivate");
  }
  if (weights->deactivation) {
    if (loom_serve_memory_operation_is_complete(weights->deactivation)) {
      iree_status_t status =
          loom_serve_memory_operation_query_status(weights->deactivation);
      loom_serve_memory_operation_free(weights->deactivation);
      weights->deactivation = NULL;
      *out_complete = true;
      return status;
    }
    return iree_ok_status();
  }
  if (!weights->active) {
    *out_complete = true;
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)weights->statistics.committed_bytes);
  const bool backing_complete =
      loom_serve_memory_operation_is_complete(weights->activation.backing);
  IREE_ASSERT(backing_complete);
  loom_serve_virtual_buffer_t** reservations = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      weights->host_allocator, weights->storage_count, sizeof(*reservations),
      (void**)&reservations);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < weights->storage_count; ++i) {
      reservations[i] = weights->storages[i].reservation;
      loom_serve_virtual_buffer_begin_trim(reservations[i]);
    }
    status = loom_serve_memory_pool_trim_async(
        weights->pool, weights->storage_count, reservations, NULL, 0,
        &weights->deactivation);
  }
  iree_allocator_free(weights->host_allocator, reservations);
  if (iree_status_is_ok(status)) {
    weights->activity.deactivations += weights->active;
    weights->active = false;
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_release(weights->activation.semaphores[i]);
      weights->activation.semaphores[i] = NULL;
      weights->activation.values[i] = 0;
    }
    weights->activation.count = 0;
    loom_serve_memory_operation_free(weights->activation.backing);
    weights->activation.backing = NULL;
    iree_hal_semaphore_release(weights->activation.backing_ready);
    weights->activation.backing_ready = NULL;
    *out_complete = !weights->deactivation;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

loom_serve_memory_statistics_t loom_serve_weights_statistics(
    const loom_serve_weights_t* weights) {
  // Accounting is published by the serialized owner, never the cold executor.
  loom_serve_memory_operation_is_complete(weights->activation.backing);
  loom_serve_memory_operation_is_complete(weights->deactivation);
  return weights->statistics;
}

loom_serve_weight_activity_t loom_serve_weights_activity(
    const loom_serve_weights_t* weights) {
  return weights->activity;
}

void loom_serve_weights_set_retirement_observer(loom_serve_weights_t* weights,
                                                void (*wake)(void* user_data),
                                                void* user_data) {
  if (weights) {
    loom_serve_memory_operation_set_observer(weights->activation.backing, wake,
                                             user_data);
    loom_serve_memory_operation_set_observer(weights->deactivation, wake,
                                             user_data);
    loom_serve_retirement_set_observer(&weights->retirement, wake, user_data);
    loom_serve_retirement_set_observer(&weights->activation.retirement, wake,
                                       user_data);
  }
}

bool loom_serve_weights_is_retired(loom_serve_weights_t* weights) {
  return !weights ||
         (loom_serve_memory_operation_is_complete(
              weights->activation.backing) &&
          loom_serve_memory_operation_is_complete(weights->deactivation) &&
          loom_serve_retirement_is_idle(&weights->retirement) &&
          loom_serve_retirement_is_idle(&weights->activation.retirement));
}

void loom_serve_weights_await_retirement(loom_serve_weights_t* weights) {
  if (weights) {
    loom_serve_memory_operation_await(weights->activation.backing);
    loom_serve_memory_operation_await(weights->deactivation);
    loom_serve_retirement_await(&weights->retirement);
    loom_serve_retirement_await(&weights->activation.retirement);
  }
}

iree_status_t loom_serve_weights_free(loom_serve_weights_t* weights) {
  if (!weights) {
    return iree_ok_status();
  }
  const bool retired = loom_serve_weights_is_retired(weights);
  IREE_ASSERT(retired);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < weights->storage_count && iree_status_is_ok(status); ++i) {
    status = loom_serve_virtual_buffer_free(weights->storages[i].reservation);
    if (iree_status_is_ok(status)) {
      weights->storages[i].reservation = NULL;
      iree_hal_buffer_release(weights->storages[i].buffer);
      weights->storages[i].buffer = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_serve_memory_operation_free(weights->activation.backing);
    loom_serve_memory_operation_free(weights->deactivation);
    iree_hal_semaphore_release(weights->activation.backing_ready);
    loom_serve_retirement_deinitialize(&weights->retirement);
    loom_serve_retirement_deinitialize(&weights->activation.retirement);
    for (iree_host_size_t i = 0; i < weights->activation.count; ++i) {
      iree_hal_semaphore_release(weights->activation.semaphores[i]);
    }
    for (iree_host_size_t i = 0; i < weights->span_count; ++i) {
      iree_hal_command_buffer_release(weights->spans[i].prepare);
    }
    loom_serve_weight_inputs_free(weights->inputs);
    iree_status_ignore(weights->activation_status);
    iree_allocator_free(weights->host_allocator, weights->storages);
    iree_allocator_free(weights->host_allocator, weights->keys);
    iree_allocator_free(weights->host_allocator, weights->spans);
    iree_allocator_free(weights->host_allocator, weights);
  }
  return status;
}
