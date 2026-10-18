// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/executables.h"

#include <string.h>

#include "loom/serve/runtime/program/stage_set.h"
#include "loomc/iree.h"

typedef struct loom_serve_generation_t {
  // Management, preferred selection, captured batch, and bound runner owners.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this generation and its execution tables.
  iree_allocator_t host_allocator;
  // Retained immutable source snapshot and exact-input native reuse.
  loom_serve_jit_t* jit;
  // Original model evidence or owned trailing storage for a new generation.
  loom_serve_compilation_report_t* report;
  // Cold declaration count, including partial construction on failure.
  iree_host_size_t count;
  // Owned replacement stages; NULL for the model-owned original stages.
  loom_serve_jit_stage_t** compiled;
  // Borrowed original table or owned recordings paired with compiled stages.
  const loom_serve_stage_table_t* table;
  // Native payload retained by the immutable cold stages, excluding index
  // space.
  iree_host_size_t cold_native_bytes;
  // Owner-side request-cache accounting for this source generation.
  struct {
    // Request choices retaining this generation, including construction.
    iree_host_size_t count;
    // Additional native keys/images charged once across those choices.
    iree_host_size_t native_bytes;
  } choices;
} loom_serve_generation_t;

typedef struct loom_serve_choice_code_t {
  // Serialized choice owners; compilation completes before a second borrower.
  iree_host_size_t users;
  // Initialized products, including partial construction on failure.
  iree_host_size_t count;
  // Immutable compiler products shared independently of checkpoint bindings.
  loom_serve_jit_stage_t** compiled;
  // Requested product and array storage, excluding shared native images.
  iree_host_size_t storage_size;
  // Owner collection has charged this bundle once, not once per choice.
  bool accounted;
} loom_serve_choice_code_t;

typedef struct loom_serve_choice_t {
  // Borrowed serialized owner, alive through producer and cache retirement.
  loom_serve_executables_t* owner;
  // Retained source snapshot and cold descriptors preceding request stages.
  loom_serve_generation_t* generation;
  // Owned exact request key, borrowed by the compiler until completion.
  loom_serve_preparation_t* preparation;
  // Pool entry receiving the terminal producer result.
  loom_serve_variant_entry_t* entry;
  // Shared immutable code, independent of this choice's checkpoint selection.
  loom_serve_choice_code_t* code;
  // Owned recordings with a borrowed cold prefix from the retained generation.
  loom_serve_stage_table_t* table;
  // Requested owned payload bytes, finalized before terminal publication.
  iree_host_size_t storage_size;
  // Owner collection has added storage_size to the retained payload total.
  bool accounted;
} loom_serve_choice_t;

struct loom_serve_executables_t {
  // Allocator owning this serialized model's code policy.
  iree_allocator_t host_allocator;
  // Immutable borrowed model resources, live through all compiler callbacks.
  loom_serve_executable_options_t options;
  // Original source snapshot and descriptors, available for restoration.
  loom_serve_generation_t* original;
  // Retained source generation selected for subsequent batches.
  loom_serve_generation_t* selected;
  // Retained source generation captured by the current batch.
  loom_serve_generation_t* captured;
  // Retained owner of the runner's current copied execution table.
  loom_serve_generation_t* bound;
  // Sole invocation's request declaration and compilation ownership.
  struct {
    // Unresolved declarations transferred from the source capability.
    loom_serve_preparation_t* declarations;
    // Accepted producer, or retained ready request choice after collection.
    loom_serve_variant_entry_t* entry;
    // Accepted producer still owns model/compiler borrows.
    bool pending;
  } invocation;
  // Lazy bounded reuse; cold-only models allocate no pool or entry storage.
  struct {
    // Existing variant publication and consumer-borrow mechanism.
    loom_serve_variant_pool_t* pool;
    // Ready provenance anchor, with no additional executable ownership.
    loom_serve_variant_entry_t* original;
    // Runner's independent consumer borrow, retained between invocations.
    loom_serve_variant_entry_t* bound;
    // Nonblocking scheduling callback installed before any producer starts.
    loom_serve_wake_callback_t observer;
    // Fixed pool and provenance metadata, excluded from idle payload limits.
    iree_host_size_t fixed_bytes;
    // Sum of collected, independently owned request payloads.
    iree_host_size_t payload_bytes;
    // Shared native keys/images, charged once per source generation.
    iree_host_size_t native_bytes;
  } cache;
};

static iree_host_size_t loom_serve_native_bytes(loom_serve_jit_t* jit) {
  const loom_serve_jit_cache_storage_t storage =
      loom_serve_jit_cache_storage(jit);
  return storage.host_bytes - storage.index_bytes + storage.code_bytes;
}

static void loom_serve_choice_free(void* value);
static void loom_serve_executables_trim(loom_serve_executables_t* executables);

static void loom_serve_generation_retain(loom_serve_generation_t* generation) {
  if (generation) {
    iree_atomic_ref_count_inc(&generation->ref_count);
  }
}

static void loom_serve_generation_release(void* value) {
  loom_serve_generation_t* generation = value;
  if (!generation) {
    return;
  }
  if (iree_atomic_ref_count_dec(&generation->ref_count) != 1) {
    return;
  }
  if (generation->compiled) {
    loom_serve_stage_table_free((loom_serve_stage_table_t*)generation->table);
    for (iree_host_size_t i = 0; i < generation->count; ++i) {
      loom_serve_jit_stage_free(generation->compiled[i]);
    }
  }
  loom_serve_jit_release(generation->jit);
  iree_allocator_free(generation->host_allocator, generation->compiled);
  iree_allocator_free(generation->host_allocator, generation);
}

static iree_status_t loom_serve_generation_create(
    loom_serve_jit_t* original, iree_allocator_t host_allocator,
    loom_serve_generation_t** out_generation) {
  *out_generation = NULL;
  loom_serve_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      iree_sizeof_struct(*generation) +
          (original ? 0 : sizeof(loom_serve_compilation_report_t)),
      (void**)&generation));
  iree_atomic_ref_count_init(&generation->ref_count);
  generation->host_allocator = host_allocator;
  generation->jit = original;
  loom_serve_jit_retain(original);
  generation->report =
      original
          ? loom_serve_jit_report(original)
          : (loom_serve_compilation_report_t*)((uint8_t*)generation +
                                               iree_sizeof_struct(*generation));
  *out_generation = generation;
  return iree_ok_status();
}

iree_status_t loom_serve_executables_allocate(
    const loom_serve_executable_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_executables_t** out_executables) {
  *out_executables = NULL;
  if (options->cache.capacity > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "request cache capacity is not representable");
  }
  loom_serve_executables_t* executables = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executables), (void**)&executables));
  executables->host_allocator = host_allocator;
  executables->options = *options;
  iree_status_t status = loom_serve_generation_create(
      options->jit, host_allocator, &executables->original);
  if (iree_status_is_ok(status)) {
    loom_serve_generation_t* original = executables->original;
    original->count = loom_serve_preparation_stage_count(options->preparation);
    original->table = options->stages;
    original->cold_native_bytes = loom_serve_native_bytes(original->jit);
    executables->selected = original;
    loom_serve_generation_retain(original);
    executables->bound = original;
    loom_serve_generation_retain(original);
  }
  if (iree_status_is_ok(status)) {
    *out_executables = executables;
  } else {
    loom_serve_executables_free(executables);
  }
  return status;
}

void loom_serve_executables_free(loom_serve_executables_t* executables) {
  if (!executables) {
    return;
  }
  loom_serve_preparation_free(executables->invocation.declarations);
  loom_serve_module_clear_stages(executables->options.runner);
  loom_serve_variant_entry_release(executables->invocation.entry);
  loom_serve_variant_entry_release(executables->cache.bound);
  loom_serve_variant_pool_free(executables->cache.pool);
  loom_serve_generation_release(executables->captured);
  loom_serve_generation_release(executables->bound);
  loom_serve_generation_release(executables->selected);
  loom_serve_generation_release(executables->original);
  iree_allocator_free(executables->host_allocator, executables);
}

void loom_serve_executables_capture(loom_serve_executables_t* executables) {
  executables->captured = executables->selected;
  loom_serve_generation_retain(executables->captured);
}

void loom_serve_executables_declare(loom_serve_executables_t* executables,
                                    loom_serve_preparation_t* preparation) {
  executables->invocation.declarations = preparation;
}

void loom_serve_executables_set_observer(loom_serve_executables_t* executables,
                                         loom_serve_wake_callback_t observer) {
  executables->cache.observer = observer;
  if (executables->cache.pool) {
    loom_serve_variant_pool_set_observer(executables->cache.pool, observer);
  }
}

static iree_status_t loom_serve_executables_bind(
    loom_serve_executables_t* executables) {
  loom_serve_generation_t* generation = executables->captured;
  loom_serve_variant_entry_t* entry = executables->invocation.entry;
  if (executables->bound == generation && executables->cache.bound == entry) {
    return iree_ok_status();
  }
  const loom_serve_choice_t* choice =
      entry ? loom_serve_variant_entry_value(entry) : NULL;
  const loom_serve_stage_table_t* table =
      choice ? choice->table : generation->table;
  IREE_RETURN_IF_ERROR(loom_serve_module_bind_stages(
      executables->options.runner, table->part_count, table->parts));
  loom_serve_variant_entry_retain(entry);
  loom_serve_variant_entry_release(executables->cache.bound);
  executables->cache.bound = entry;
  loom_serve_generation_retain(generation);
  loom_serve_generation_release(executables->bound);
  executables->bound = generation;
  return iree_ok_status();
}

void loom_serve_executables_release(loom_serve_executables_t* executables) {
  loom_serve_preparation_free(executables->invocation.declarations);
  loom_serve_variant_entry_release(executables->invocation.entry);
  memset(&executables->invocation, 0, sizeof(executables->invocation));
  loom_serve_generation_release(executables->captured);
  executables->captured = NULL;
  loom_serve_executables_trim(executables);
}

typedef struct loom_serve_generation_job_t {
  // Finite source-indexing task, retired before asynchronous stage compilation.
  loomc_task_t task;
  // Borrowed immutable model resources through terminal completion.
  loom_serve_executables_t* executables;
  // Owned generation transferred at terminal completion, including failure.
  loom_serve_generation_t* generation;
  // Provider namespace borrowed through source indexing only.
  loom_serve_sources_t sources;
  // Indexing outcome consumed after the task relinquishes compiler access.
  iree_status_t status;
  // Terminal ownership transfer after all model and source accesses retire.
  loom_serve_variant_completion_fn_t completion;
  // Borrowed terminal consumer state.
  void* user_data;
} loom_serve_generation_job_t;

static iree_status_t loom_serve_executables_record(
    const loom_serve_executable_options_t* options,
    const loom_serve_stage_table_t* prefix, iree_host_size_t count,
    loom_serve_jit_stage_t* const* stages, iree_allocator_t host_allocator,
    loom_serve_stage_table_t** out_table) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_serve_stage_record_check(&options->recording, stages[i]);
    const loom_cmd_program_t* program = loom_serve_jit_stage_program(stages[i]);
    if (iree_status_is_ok(status) && program) {
      status =
          loom_serve_parameter_bank_check_layout(options->parameters, program);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_stage_table_allocate(&options->recording, prefix, count,
                                             stages, host_allocator, out_table);
  }
  return status;
}

static void loom_serve_account_natives(loom_serve_executables_t* executables,
                                       loom_serve_generation_t* generation) {
  const iree_host_size_t bytes =
      generation->choices.count ? loom_serve_native_bytes(generation->jit) -
                                      generation->cold_native_bytes
                                : 0;
  executables->cache.native_bytes -= generation->choices.native_bytes;
  executables->cache.native_bytes += bytes;
  generation->choices.native_bytes = bytes;
}

static void loom_serve_choice_free(void* value) {
  loom_serve_choice_t* choice = value;
  if (!choice) {
    return;
  }
  loom_serve_executables_t* executables = choice->owner;
  loom_serve_generation_t* generation = choice->generation;
  loom_serve_stage_table_free(choice->table);
  loom_serve_choice_code_t* code = choice->code;
  if (code && --code->users == 0) {
    for (iree_host_size_t i = 0; i < code->count; ++i) {
      loom_serve_jit_stage_free(code->compiled[i]);
    }
    if (code->accounted) {
      executables->cache.payload_bytes -= code->storage_size;
    }
    iree_allocator_free(executables->host_allocator, code->compiled);
    iree_allocator_free(executables->host_allocator, code);
  }
  loom_serve_preparation_free(choice->preparation);
  if (choice->accounted) {
    executables->cache.payload_bytes -= choice->storage_size;
  }
  --generation->choices.count;
  loom_serve_jit_trim(generation->jit);
  loom_serve_account_natives(executables, generation);
  loom_serve_generation_release(generation);
  iree_allocator_free(executables->host_allocator, choice);
}

static void loom_serve_executables_trim(loom_serve_executables_t* executables) {
  loom_serve_variant_pool_t* pool = executables->cache.pool;
  if (!pool) {
    return;
  }
  while (loom_serve_variant_pool_count(pool) > 1 &&
         (loom_serve_variant_pool_count(pool) - 1 >
              executables->options.cache.capacity ||
          executables->cache.payload_bytes + executables->cache.native_bytes +
                  loom_serve_variant_pool_storage_size(pool) -
                  executables->cache.fixed_bytes >
              executables->options.cache.max_bytes)) {
    loom_serve_variant_entry_t* entry =
        loom_serve_variant_entry_next(executables->cache.original);
    if (entry == executables->cache.bound) {
      loom_serve_module_clear_stages(executables->options.runner);
      loom_serve_variant_entry_release(executables->cache.bound);
      executables->cache.bound = NULL;
      loom_serve_generation_release(executables->bound);
      executables->bound = NULL;
    }
    loom_serve_variant_entry_discard(entry);
  }
}

// Resolves source identity before checking placement. Equal tensor names and
// shapes in two checkpoint domains never establish interchangeable contents.
static iree_status_t loom_serve_choice_bind_parameters(
    loom_serve_choice_t* choice, iree_host_size_t count,
    loom_serve_parameter_binding_t* bindings, iree_hal_buffer_t** buffers) {
  const loom_serve_executable_options_t* options = &choice->owner->options;
  const loom_serve_preparation_stage_t* selected =
      loom_serve_preparation_stage(choice->preparation, 0);
  const loom_serve_preparation_stage_t* retained =
      loom_serve_preparation_stage(options->preparation, 0);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_host_size_t domain = 0;
    for (; domain < retained->parameter_count; ++domain) {
      if (loom_serve_preparation_parameter_equal(
              &selected->parameters[i], &retained->parameters[domain])) {
        break;
      }
    }
    if (domain == retained->parameter_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "request checkpoint domain %" PRIhsz
                                " is not retained by the model",
                                i);
    } else {
      bindings[i] = (loom_serve_parameter_binding_t){options->parameters,
                                                     (uint32_t)domain};
      buffers[i] =
          loom_serve_parameter_bank_buffers(options->parameters)[domain];
    }
  }
  return status;
}

static iree_status_t loom_serve_choice_record(loom_serve_choice_t* choice) {
  loom_serve_executables_t* executables = choice->owner;
  const loom_serve_choice_code_t* code = choice->code;
  const loom_serve_preparation_stage_t* canonical =
      loom_serve_preparation_stage(choice->preparation, 0);
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(code->compiled[0]);
  const iree_host_size_t count = canonical->parameter_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < code->count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_preparation_stage_t* declaration =
        loom_serve_preparation_stage(choice->preparation, i);
    const loom_cmd_program_t* candidate =
        loom_serve_jit_stage_program(code->compiled[i]);
    if (count) {
      status = loom_serve_stage_record_check(&executables->options.recording,
                                             code->compiled[i]);
    }
    if (iree_status_is_ok(status) && candidate) {
      status = loom_serve_preparation_stage_check_workspace(
          declaration, candidate->requirements.transient.required_byte_length);
    }
    if (iree_status_is_ok(status) && count) {
      status = loom_serve_preparation_stage_check_parameters(
          declaration, candidate, canonical, program);
    } else if (iree_status_is_ok(status) && declaration->parameter_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "request stages must share the first stage's "
                                "checkpoint declarations");
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (!count) {
    return loom_serve_executables_record(
        &executables->options, choice->generation->table, code->count,
        code->compiled, executables->host_allocator, &choice->table);
  }

  loom_serve_parameter_binding_t stack_bindings[8];
  iree_hal_buffer_t* stack_buffers[8];
  loom_serve_parameter_binding_t* bindings = stack_bindings;
  iree_hal_buffer_t** buffers = stack_buffers;
  void* allocation = NULL;
  if (count > IREE_ARRAYSIZE(stack_bindings)) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        executables->host_allocator, count,
        sizeof(*bindings) + sizeof(*buffers), &allocation));
    bindings = allocation;
    buffers = (iree_hal_buffer_t**)(bindings + count);
  }
  status = loom_serve_choice_bind_parameters(choice, count, bindings, buffers);
  if (iree_status_is_ok(status)) {
    status = loom_serve_parameter_bindings_check_layout(
        count, bindings, program, executables->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_stage_record_options_t recording =
        executables->options.recording;
    recording.parameters = buffers;
    status = loom_serve_stage_table_allocate(
        &recording, choice->generation->table, code->count, code->compiled,
        executables->host_allocator, &choice->table);
  }
  iree_allocator_free(executables->host_allocator, allocation);
  return status;
}

static void loom_serve_choice_complete(loom_serve_choice_t* choice,
                                       iree_status_t status) {
  if (iree_status_is_ok(status)) {
    choice->storage_size =
        sizeof(*choice) +
        loom_serve_preparation_storage_size(choice->preparation) +
        loom_serve_stage_table_storage_size(choice->table);
  }
  loom_serve_variant_entry_complete(choice->entry, status, choice,
                                    choice->generation->report);
}

static void loom_serve_choice_compiled(void* user_data, iree_status_t status,
                                       loom_serve_stage_set_t* stages) {
  loom_serve_choice_t* choice = user_data;
  loom_serve_choice_code_t* code = choice->code;
  code->count = stages->count;
  code->compiled = stages->values;
  stages->count = 0;
  stages->values = NULL;
  loom_serve_stage_set_free(stages);
  if (iree_status_is_ok(status)) {
    code->storage_size = sizeof(*code) + code->count * sizeof(*code->compiled);
    for (iree_host_size_t i = 0; i < code->count; ++i) {
      code->storage_size +=
          loom_serve_jit_stage_storage_size(code->compiled[i]);
    }
    status = loom_serve_choice_record(choice);
  }
  loom_serve_choice_complete(choice, status);
}

typedef struct loom_serve_choice_record_job_t {
  // Finite recording task; no compiler work is required on a code-cache hit.
  loomc_task_t task;
  // Borrowed unpublished choice, retained through terminal publication.
  loom_serve_choice_t* choice;
  // Recording result transferred after the task relinquishes its resources.
  iree_status_t status;
} loom_serve_choice_record_job_t;

static void loom_serve_choice_record_execute(loomc_task_t* task,
                                             loomc_host_size_t worker) {
  loom_serve_choice_record_job_t* job = (loom_serve_choice_record_job_t*)task;
  job->status = loom_serve_choice_record(job->choice);
}

static void loom_serve_choice_record_complete(loomc_task_t* task) {
  loom_serve_choice_record_job_t* job = (loom_serve_choice_record_job_t*)task;
  loom_serve_choice_t* choice = job->choice;
  iree_status_t status = job->status;
  iree_allocator_free(choice->owner->host_allocator, job);
  loom_serve_choice_complete(choice, status);
}

static const loomc_task_vtable_t loom_serve_choice_record_vtable = {
    .execute = loom_serve_choice_record_execute,
    .destroy = loom_serve_choice_record_complete,
};

static iree_status_t loom_serve_choice_record_async(
    loom_serve_choice_t* choice) {
  loom_serve_executables_t* executables = choice->owner;
  loom_serve_choice_record_job_t* job = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executables->host_allocator,
                                             sizeof(*job), (void**)&job));
  job->choice = choice;
  loomc_task_initialize(&loom_serve_choice_record_vtable, &job->task);
  iree_status_t status = iree_status_from_loomc(loomc_task_sink_submit(
      loom_serve_host_compiler_sink(
          loom_serve_device_host(executables->options.device)),
      &job->task));
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(executables->host_allocator, job);
  }
  return status;
}

static iree_status_t loom_serve_choice_begin(
    loom_serve_executables_t* executables) {
  if (!executables->cache.pool) {
    IREE_RETURN_IF_ERROR(loom_serve_variant_pool_allocate(
        executables->options.cache.capacity + 2, NULL, loom_serve_choice_free,
        executables->host_allocator, &executables->cache.original,
        &executables->cache.pool));
    executables->cache.fixed_bytes =
        loom_serve_variant_pool_storage_size(executables->cache.pool);
    loom_serve_variant_pool_set_observer(executables->cache.pool,
                                         executables->cache.observer);
  }
  loom_serve_choice_code_t* shared = NULL;
  for (loom_serve_variant_entry_t* entry =
           loom_serve_variant_entry_next(executables->cache.original);
       entry; entry = loom_serve_variant_entry_next(entry)) {
    const loom_serve_choice_t* choice = loom_serve_variant_entry_value(entry);
    if (choice->generation == executables->captured &&
        loom_serve_preparation_equal(choice->preparation,
                                     executables->invocation.declarations)) {
      loom_serve_variant_entry_retain(entry);
      loom_serve_variant_entry_touch(entry);
      executables->invocation.entry = entry;
      loom_serve_preparation_free(executables->invocation.declarations);
      executables->invocation.declarations = NULL;
      return iree_ok_status();
    }
    if (choice->generation == executables->captured &&
        loom_serve_preparation_code_equal(
            choice->preparation, executables->invocation.declarations)) {
      shared = choice->code;
    }
  }
  loom_serve_choice_t* choice = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executables->host_allocator,
                                             sizeof(*choice), (void**)&choice));
  choice->owner = executables;
  choice->generation = executables->captured;
  loom_serve_generation_retain(choice->generation);
  ++choice->generation->choices.count;
  choice->code = shared;
  if (shared) {
    ++shared->users;
  }
  iree_status_t status = loom_serve_variant_pool_begin(
      executables->cache.pool,
      loom_serve_variant_entry_id(executables->cache.original), &choice->entry);
  if (!iree_status_is_ok(status)) {
    loom_serve_choice_free(choice);
    return status;
  }
  choice->preparation = executables->invocation.declarations;
  executables->invocation.declarations = NULL;
  executables->invocation.entry = choice->entry;
  executables->invocation.pending = true;
  if (shared) {
    status = loom_serve_choice_record_async(choice);
  } else {
    status =
        iree_allocator_malloc(executables->host_allocator,
                              sizeof(*choice->code), (void**)&choice->code);
    if (iree_status_is_ok(status)) {
      choice->code->users = 1;
      status = loom_serve_stage_set_compile_catalog_async(
          choice->generation->jit, choice->preparation,
          loom_serve_choice_compiled, choice, executables->host_allocator);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_variant_entry_complete(choice->entry, status, choice, NULL);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_executables_advance(
    loom_serve_executables_t* executables, bool cancel, bool* out_complete) {
  *out_complete = true;
  if (!cancel && executables->invocation.declarations) {
    IREE_RETURN_IF_ERROR(loom_serve_choice_begin(executables));
  }
  if (executables->invocation.pending) {
    loom_serve_variant_pool_collect(executables->cache.pool);
    loom_serve_variant_entry_t* entry = executables->invocation.entry;
    const loom_serve_variant_snapshot_t snapshot =
        loom_serve_variant_entry_snapshot(entry);
    if (snapshot.phase == LOOM_SERVE_VARIANT_COMPILING) {
      *out_complete = false;
      return iree_ok_status();
    }
    executables->invocation.pending = false;
    if (snapshot.phase == LOOM_SERVE_VARIANT_FAILED) {
      iree_status_t status = loom_serve_variant_entry_status(entry);
      loom_serve_variant_entry_discard(entry);
      executables->invocation.entry = NULL;
      return status;
    }
    loom_serve_choice_t* choice = loom_serve_variant_entry_value(entry);
    choice->accounted = true;
    executables->cache.payload_bytes += choice->storage_size;
    if (!choice->code->accounted) {
      choice->code->accounted = true;
      executables->cache.payload_bytes += choice->code->storage_size;
    }
    loom_serve_account_natives(executables, choice->generation);
    loom_serve_variant_entry_retain(entry);
  }
  return cancel ? iree_ok_status() : loom_serve_executables_bind(executables);
}

const loom_serve_stage_launch_t* loom_serve_executables_stage(
    const loom_serve_executables_t* executables, uint32_t ordinal) {
  const loom_serve_choice_t* choice =
      executables->invocation.entry
          ? loom_serve_variant_entry_value(executables->invocation.entry)
          : NULL;
  const loom_serve_stage_table_t* table =
      choice ? choice->table : executables->captured->table;
  return &table->launches[ordinal];
}

static void loom_serve_generation_complete(loom_serve_generation_job_t* job,
                                           iree_status_t status) {
  loom_serve_generation_t* generation = job->generation;
  const loom_serve_variant_completion_fn_t completion = job->completion;
  void* user_data = job->user_data;
  iree_allocator_free(generation->host_allocator, job);
  completion(user_data, status, generation, generation->report);
}

static void loom_serve_generation_compiled(void* user_data,
                                           iree_status_t status,
                                           loom_serve_stage_set_t* stages) {
  loom_serve_generation_job_t* job = user_data;
  job->generation->count = stages->count;
  job->generation->compiled = stages->values;
  stages->count = 0;
  stages->values = NULL;
  loom_serve_stage_set_free(stages);
  if (iree_status_is_ok(status)) {
    loom_serve_stage_table_t* table = NULL;
    status = loom_serve_executables_record(
        &job->executables->options, NULL, job->generation->count,
        job->generation->compiled, job->generation->host_allocator, &table);
    job->generation->table = table;
  }
  if (iree_status_is_ok(status)) {
    job->generation->cold_native_bytes =
        loom_serve_native_bytes(job->generation->jit);
  }
  loom_serve_generation_complete(job, status);
}

static void loom_serve_generation_index(loomc_task_t* task,
                                        loomc_host_size_t worker) {
  loom_serve_generation_job_t* job =
      (loom_serve_generation_job_t*)((uint8_t*)task -
                                     offsetof(loom_serve_generation_job_t,
                                              task));
  loom_serve_generation_t* generation = job->generation;
  loom_serve_device_t* device = job->executables->options.device;
  job->status = loom_serve_jit_create(
      loom_serve_device_handle(device),
      loom_serve_device_dispatch_queue(device), loom_serve_device_host(device),
      &job->sources, NULL, generation->report, generation->host_allocator,
      &generation->jit);
}

static void loom_serve_generation_indexed(loomc_task_t* task) {
  loom_serve_generation_job_t* job =
      (loom_serve_generation_job_t*)((uint8_t*)task -
                                     offsetof(loom_serve_generation_job_t,
                                              task));
  iree_status_t status = job->status;
  if (iree_status_is_ok(status)) {
    status = loom_serve_stage_set_compile_catalog_async(
        job->generation->jit, job->executables->options.preparation,
        loom_serve_generation_compiled, job, job->generation->host_allocator);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_generation_complete(job, status);
  }
}

static const loomc_task_vtable_t executable_generation_task_vtable = {
    .execute = loom_serve_generation_index,
    .destroy = loom_serve_generation_indexed,
};

static iree_status_t loom_serve_generation_compile(
    void* self, const loom_serve_sources_t* sources,
    loom_serve_variant_completion_fn_t completion, void* user_data) {
  loom_serve_executables_t* executables = self;
  loom_serve_generation_t* generation = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_generation_create(
      NULL, executables->host_allocator, &generation));
  loom_serve_generation_job_t* job = NULL;
  iree_status_t status = iree_allocator_malloc(executables->host_allocator,
                                               sizeof(*job), (void**)&job);
  if (iree_status_is_ok(status)) {
    job->executables = executables;
    job->generation = generation;
    job->sources = *sources;
    job->completion = completion;
    job->user_data = user_data;
    loomc_task_initialize(&executable_generation_task_vtable, &job->task);
    status = iree_status_from_loomc(loomc_task_sink_submit(
        loom_serve_host_compiler_sink(
            loom_serve_device_host(executables->options.device)),
        &job->task));
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(executables->host_allocator, job);
    loom_serve_generation_release(generation);
  }
  return status;
}

static iree_status_t loom_serve_generation_select(void* self, void* value) {
  loom_serve_executables_t* executables = self;
  loom_serve_generation_t* generation = value ? value : executables->original;
  loom_serve_generation_retain(generation);
  loom_serve_generation_release(executables->selected);
  executables->selected = generation;
  return iree_ok_status();
}

loom_serve_execution_update_t loom_serve_executables_update(
    loom_serve_executables_t* executables) {
  return (loom_serve_execution_update_t){
      .self = executables,
      .compile = loom_serve_generation_compile,
      .select = loom_serve_generation_select,
      .dispose = loom_serve_generation_release};
}
