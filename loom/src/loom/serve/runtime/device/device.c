// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/device/device.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/tooling/device_profiling.h"
#include "loom/serve/runtime/execution/queues.h"
#include "loom/serve/runtime/program/jit_cache.h"
#include "loom/serve/runtime/program/tokenizer_cache.h"

struct loom_serve_device_t {
  // Borrowed enclosing application group, owning this record's storage.
  loom_serve_device_group_t* owner;
  // Borrowed key in the enclosing group's single allocation.
  iree_string_view_t name;
  // Owned device providing memory and queue resources.
  iree_hal_device_t* handle;
  // Exact dispatch queue borrowed from the device.
  iree_hal_queue_t* dispatch;
  // Exact transfer queue borrowed from the device.
  iree_hal_queue_t* transfer;
  // Fixed inventory shared by independently retained model executions.
  loom_serve_execution_queues_t* execution_queues;
  // Borrowed default storage binding, owned by the group's pool inventory.
  loom_serve_pool_t* pool;
  // Exact immutable source products shared by independent model instances.
  loom_serve_jit_cache_t* jit_cache;
};

struct loom_serve_device_group_t {
  // Allocation policy for the group and its per-device resources.
  iree_allocator_t host_allocator;
  // Retained workers and I/O services outliving every HAL device.
  loom_serve_host_t* host;
  // Exact immutable tokenizers shared across model modalities and devices.
  loom_serve_tokenizer_cache_t* tokenizer_cache;
  // Completion registry outliving the shared HAL group.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Owned immutable topology and semaphore/memory namespace.
  iree_hal_device_group_t* handle;
  // One profiling session and sink enclosing every model on all devices.
  iree_hal_profiling_from_flags_t* profiling;
  // Number of initialized records, including nullable partial resources.
  iree_host_size_t device_count;
  // Application key views into trailing string storage in this allocation.
  iree_string_view_t names[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  // Number of pool owners published during application construction.
  iree_host_size_t pool_count;
  // Owned pool inventory, independent of device membership and model bindings.
  loom_serve_pool_t** pools;
  // Owned execution records, borrowed by model owners.
  loom_serve_device_t devices[];
};

static iree_hal_queue_t* loom_serve_device_select_queue(
    iree_hal_device_t* device, iree_hal_queue_family_role_flags_t role) {
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    if (queues->families[i].provisioned_queue_count &&
        iree_all_bits_set(queues->families[i].role_flags, role)) {
      return iree_hal_device_queue(device, i, 0);
    }
  }
  return NULL;
}

static iree_status_t loom_serve_device_group_initialize(
    loom_serve_device_group_t* group,
    const loom_serve_device_options_t* options) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < group->device_count && iree_status_is_ok(status); ++i) {
    iree_hal_device_create_params_t params =
        iree_hal_device_create_params_default();
    params.proactor_pool = loom_serve_host_proactor_pool(group->host);
    params.event_sink = iree_hal_device_event_sink_stderr();
    status = iree_hal_driver_create_device_by_uri(
        options[i].driver, options[i].uri, &params, group->host_allocator,
        &group->devices[i].handle);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "creating serving device %" PRIhsz " (%.*s)", i,
          (int)options[i].uri.size, options[i].uri.data);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), group->host_allocator,
      &group->frontier_tracker));
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, group->frontier_tracker);
  for (iree_host_size_t i = 0;
       i < group->device_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_group_builder_add_device(&builder,
                                                      group->devices[i].handle);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(
        &builder, group->host_allocator, &group->handle);
  }
  iree_hal_device_group_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_serve_device_initialize(
    loom_serve_device_t* device, const loom_serve_device_options_t* options) {
  loom_serve_device_group_t* group = device->owner;
  device->dispatch = loom_serve_device_select_queue(
      device->handle, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH);
  device->transfer = loom_serve_device_select_queue(
      device->handle, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER);
  if (!device->dispatch || !device->transfer) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device needs dispatch and transfer queues");
  }
  const iree_hal_device_sanitizer_spec_t* sanitizer =
      iree_hal_device_spec_sanitizer(iree_hal_device_spec(device->handle));
  if (options->backing == LOOM_SERVE_DEVICE_BACKING_ELASTIC &&
      iree_any_bit_set(sanitizer->flags, IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "address-sanitized serving devices require fixed backing");
  }
  const iree_hal_queue_family_t* dispatch_family =
      iree_hal_queue_family(device->dispatch);
  const iree_hal_queue_family_t* transfer_family =
      iree_hal_queue_family(device->transfer);
  IREE_RETURN_IF_ERROR(loom_serve_execution_queues_create(
      dispatch_family, transfer_family, group->host_allocator,
      &device->execution_queues));
  // Source programs may carry indirect launch counts in retained buffers or
  // execution workspace, both allocated from this dispatch-visible domain.
  const iree_hal_pool_family_access_t families[] = {
      {.family = dispatch_family,
       .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER |
                IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS},
      {.family = transfer_family, .usage = IREE_HAL_BUFFER_USAGE_TRANSFER},
  };
  const iree_hal_pool_scope_t scope = {
      .family_count = dispatch_family == transfer_family ? 1 : 2,
      .families = families,
  };
  loom_serve_pool_options_t pool_options = {
      .elastic = options->backing == LOOM_SERVE_DEVICE_BACKING_ELASTIC,
      .weight_slab_size = options->weight_slab_size,
      .state_slab_size = options->state_slab_size,
      .memory_limit = options->memory_limit,
  };
  if (iree_any_bit_set(sanitizer->flags, IREE_HAL_DEVICE_SANITIZER_FLAG_ASAN)) {
    pool_options.asan = sanitizer->asan.pool_options;
  }
  IREE_RETURN_IF_ERROR(loom_serve_device_group_create_pool(
      group, scope, device->transfer, &pool_options, &device->pool));
  return loom_serve_jit_cache_allocate(device->handle, device->dispatch,
                                       group->host_allocator,
                                       &device->jit_cache);
}

iree_status_t loom_serve_device_group_allocate(
    loom_serve_host_t* host, iree_host_size_t device_count,
    const loom_serve_device_options_t* options, iree_allocator_t host_allocator,
    loom_serve_device_group_t** out_group) {
  *out_group = NULL;
  if (!device_count || device_count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "serving group needs between 1 and %d devices",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }
  iree_string_view_t names[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT];
  iree_host_size_t name_bytes = 0;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    names[i] = !options[i].name.size && device_count == 1 ? IREE_SV("device")
                                                          : options[i].name;
    if (!names[i].size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "each device in a multi-device group needs a name");
    }
    for (iree_host_size_t c = 0; c < names[i].size; ++c) {
      if ((uint8_t)names[i].data[c] < 0x21 ||
          (uint8_t)names[i].data[c] > 0x7e) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "device names must be visible ASCII");
      }
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(names[j], names[i])) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate device name '%.*s'",
                                (int)names[i].size, names[i].data);
      }
    }
    if (!iree_host_size_checked_add(name_bytes, names[i].size, &name_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "device names exceed addressable storage");
    }
    if (options[i].backing != LOOM_SERVE_DEVICE_BACKING_FIXED &&
        options[i].backing != LOOM_SERVE_DEVICE_BACKING_ELASTIC) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported physical backing strategy");
    }
    if (options[i].backing == LOOM_SERVE_DEVICE_BACKING_FIXED &&
        options[i].memory_limit) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "a physical budget requires elastic backing");
    }
  }
  iree_host_size_t total_size = 0, names_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_device_group_t) +
          device_count * sizeof(loom_serve_device_t),
      &total_size, IREE_STRUCT_FIELD(name_bytes, char, &names_offset)));
  loom_serve_device_group_t* group = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&group));
  group->host_allocator = host_allocator;
  group->host = host;
  group->device_count = device_count;
  loom_serve_host_retain(host);
  char* name_storage = (char*)group + names_offset;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    group->devices[i].owner = group;
    memcpy(name_storage, names[i].data, names[i].size);
    group->names[i] = iree_make_string_view(name_storage, names[i].size);
    group->devices[i].name = group->names[i];
    name_storage += names[i].size;
  }
  iree_status_t status = loom_serve_tokenizer_cache_allocate(
      host_allocator, &group->tokenizer_cache);
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_group_initialize(group, options);
  }
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_device_initialize(&group->devices[i], &options[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "initializing serving device %" PRIhsz " (%.*s)", i,
          (int)options[i].uri.size, options[i].uri.data);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        group->handle, host_allocator, &group->profiling);
  }
  if (iree_status_is_ok(status)) {
    *out_group = group;
  } else {
    status = iree_status_join(status, loom_serve_device_group_free(group));
  }
  return status;
}

iree_status_t loom_serve_device_group_free(loom_serve_device_group_t* group) {
  if (!group) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < group->pool_count; ++i) {
    if (loom_serve_pool_is_in_use(group->pools[i])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "serving pool %" PRIhsz " still has live consumers", i);
    }
  }
  iree_status_t status = iree_hal_end_profiling_from_flags(group->profiling);
  group->profiling = NULL;
  loom_serve_tokenizer_cache_free(group->tokenizer_cache);
  group->tokenizer_cache = NULL;
  iree_status_t cleanup_status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < group->pool_count && iree_status_is_ok(cleanup_status); ++i) {
    cleanup_status = loom_serve_pool_free(group->pools[i]);
    if (iree_status_is_ok(cleanup_status)) {
      group->pools[i] = NULL;
    }
  }
  if (!iree_status_is_ok(cleanup_status)) {
    return iree_status_join(status, cleanup_status);
  }
  iree_allocator_free(group->host_allocator, group->pools);
  for (iree_host_size_t i = 0; i < group->device_count; ++i) {
    loom_serve_device_t* device = &group->devices[i];
    loom_serve_jit_cache_free(device->jit_cache);
    loom_serve_execution_queues_release(device->execution_queues);
  }
  for (iree_host_size_t i = 0; i < group->device_count; ++i) {
    iree_hal_device_release(group->devices[i].handle);
  }
  iree_hal_device_group_release(group->handle);
  iree_async_frontier_tracker_release(group->frontier_tracker);
  loom_serve_host_release(group->host);
  iree_allocator_free(group->host_allocator, group);
  return status;
}

iree_status_t loom_serve_device_group_create_pool(
    loom_serve_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_queue_t* allocation_queue,
    const loom_serve_pool_options_t* options, loom_serve_pool_t** out_pool) {
  *out_pool = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      group->host_allocator, group->pool_count + 1, sizeof(*group->pools),
      (void**)&group->pools));
  loom_serve_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_pool_allocate(group->handle, scope,
                                                allocation_queue, options,
                                                group->host_allocator, &pool));
  group->pools[group->pool_count++] = pool;
  *out_pool = pool;
  return iree_ok_status();
}

iree_host_size_t loom_serve_device_group_pool_count(
    const loom_serve_device_group_t* group) {
  return group->pool_count;
}

loom_serve_pool_t* loom_serve_device_group_pool_at(
    const loom_serve_device_group_t* group, iree_host_size_t pool_index) {
  return group->pools[pool_index];
}

loom_serve_pool_t* loom_serve_device_pool(const loom_serve_device_t* device) {
  return device->pool;
}

iree_host_size_t loom_serve_device_group_count(
    const loom_serve_device_group_t* group) {
  return group->device_count;
}

iree_string_view_list_t loom_serve_device_group_names(
    const loom_serve_device_group_t* group) {
  return (iree_string_view_list_t){group->device_count, group->names};
}

iree_string_view_t loom_serve_device_name(const loom_serve_device_t* device) {
  return device->name;
}

loom_serve_device_t* loom_serve_device_group_at(
    loom_serve_device_group_t* group, iree_host_size_t device_index) {
  return &group->devices[device_index];
}

loom_serve_device_memory_statistics_t loom_serve_device_memory_statistics(
    const loom_serve_device_t* device) {
  loom_serve_device_memory_statistics_t statistics = {
      .workspace = loom_serve_workspace_statistics(
          loom_serve_pool_workspace(device->pool))};
  iree_hal_pool_query_stats(loom_serve_pool_storage(device->pool),
                            &statistics.storage);
  if (loom_serve_pool_memory(device->pool)) {
    statistics.retained =
        loom_serve_memory_pool_statistics(loom_serve_pool_memory(device->pool));
  }
  return statistics;
}

iree_status_t loom_serve_device_trim(loom_serve_device_t* device,
                                     uint64_t target_bytes) {
  loom_serve_tokenizer_cache_trim(device->owner->tokenizer_cache);
  loom_serve_jit_cache_trim(device->jit_cache);
  const loom_serve_device_memory_statistics_t memory =
      loom_serve_device_memory_statistics(device);
  const uint64_t committed = loom_serve_pool_memory(device->pool)
                                 ? memory.retained.committed_bytes
                                 : memory.workspace.bytes_committed;
  if (committed <= target_bytes) {
    return iree_ok_status();
  }
  iree_device_size_t workspace_bytes = 0;
  IREE_RETURN_IF_ERROR(loom_serve_workspace_trim(
      loom_serve_pool_workspace(device->pool), &workspace_bytes));
  if (loom_serve_pool_memory(device->pool)) {
    IREE_RETURN_IF_ERROR(loom_serve_residency_cache_trim(
        loom_serve_pool_residency_cache(device->pool), target_bytes));
    loom_serve_memory_pool_trim(loom_serve_pool_memory(device->pool),
                                target_bytes);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_device_trim_idle(
    loom_serve_device_t* device, loom_serve_device_idle_policy_t policy,
    iree_time_t now, bool* out_pending) {
  *out_pending = false;
  if (policy.workspace_timeout) {
    *out_pending = loom_serve_workspace_trim_idle(
        loom_serve_pool_workspace(device->pool), now, policy.workspace_timeout);
  }
  if (policy.parameter_timeout) {
    bool pending = false;
    IREE_RETURN_IF_ERROR(loom_serve_residency_cache_trim_idle(
        loom_serve_pool_residency_cache(device->pool), now,
        policy.parameter_timeout, &pending));
    *out_pending |= pending;
  }
  return iree_ok_status();
}

iree_hal_device_t* loom_serve_device_handle(const loom_serve_device_t* device) {
  return device->handle;
}

iree_hal_pool_t* loom_serve_device_storage_pool(
    const loom_serve_device_t* device) {
  return loom_serve_pool_storage(device->pool);
}

loom_serve_host_t* loom_serve_device_host(const loom_serve_device_t* device) {
  return device->owner->host;
}

loom_serve_tokenizer_cache_t* loom_serve_device_tokenizer_cache(
    const loom_serve_device_t* device) {
  return device->owner->tokenizer_cache;
}

loom_serve_workspace_t* loom_serve_device_workspace(
    const loom_serve_device_t* device) {
  return loom_serve_pool_workspace(device->pool);
}

iree_hal_device_group_t* loom_serve_device_group(
    const loom_serve_device_t* device) {
  return device->owner->handle;
}

iree_hal_queue_t* loom_serve_device_dispatch_queue(
    const loom_serve_device_t* device) {
  return device->dispatch;
}

iree_hal_queue_t* loom_serve_device_transfer_queue(
    const loom_serve_device_t* device) {
  return device->transfer;
}

iree_status_t loom_serve_device_create_execution(
    loom_serve_device_t* device, loom_serve_pool_t* pool,
    loom_serve_execution_t** out_execution) {
  return loom_serve_execution_create(
      device->execution_queues, loom_serve_pool_workspace(pool),
      device->owner->host_allocator, out_execution);
}

loom_serve_memory_pool_t* loom_serve_device_memory_pool(
    const loom_serve_device_t* device) {
  return loom_serve_pool_memory(device->pool);
}

loom_serve_memory_geometry_t loom_serve_device_weight_geometry(
    const loom_serve_device_t* device) {
  return loom_serve_pool_weight_geometry(device->pool);
}

loom_serve_memory_geometry_t loom_serve_device_state_geometry(
    const loom_serve_device_t* device) {
  return loom_serve_pool_state_geometry(device->pool);
}

loom_serve_residency_cache_t* loom_serve_device_residency_cache(
    const loom_serve_device_t* device) {
  return loom_serve_pool_residency_cache(device->pool);
}

iree_status_t loom_serve_device_create_jit(
    loom_serve_device_t* device, const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_compilation_report_t* report, loom_serve_jit_t** out_jit) {
  return loom_serve_jit_create_cached(device->jit_cache, device->owner->host,
                                      sources, sanitizer, report, out_jit);
}
