// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_spec.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/atomics.h"

//===----------------------------------------------------------------------===//
// Serialized device spec format
//===----------------------------------------------------------------------===//

#define IREE_HAL_DEVICE_SPEC_MAGIC 0x43505344u  // DSPC
#define IREE_HAL_DEVICE_SPEC_VERSION 2u
#define IREE_HAL_DEVICE_SPEC_FNV1A64_OFFSET_BASIS 0xcbf29ce484222325ull
#define IREE_HAL_DEVICE_SPEC_FNV1A64_PRIME 0x100000001b3ull

typedef struct iree_hal_device_spec_serialized_range_t {
  // Byte offset from the start of the serialized image.
  uint64_t offset;
  // Number of bytes or elements in the range.
  uint64_t length;
} iree_hal_device_spec_serialized_range_t;

typedef struct iree_hal_device_spec_serialized_section_t {
  // Byte offset from the start of the serialized image.
  uint64_t offset;
  // Number of elements in the section.
  uint64_t count;
  // Byte size of each element in the section.
  uint64_t element_size;
} iree_hal_device_spec_serialized_section_t;

typedef struct iree_hal_device_spec_serialized_header_t {
  // Magic value identifying the serialized device spec image.
  uint32_t magic;
  // Serialized format version.
  uint32_t version;
  // Header byte length.
  uint64_t header_length;
  // Total serialized image byte length.
  uint64_t total_length;
  // Serialized physical device records.
  iree_hal_device_spec_serialized_section_t physical_devices;
  // Serialized memory heap records.
  iree_hal_device_spec_serialized_section_t memory_heaps;
  // Serialized memory type records.
  iree_hal_device_spec_serialized_section_t memory_types;
  // Serialized external buffer handle records.
  iree_hal_device_spec_serialized_section_t external_buffer_handles;
  // Serialized virtual memory class records.
  iree_hal_device_spec_serialized_section_t virtual_memory_classes;
  // Serialized queue family records.
  iree_hal_device_spec_serialized_section_t queue_families;
  // Serialized external timepoint handle records.
  iree_hal_device_spec_serialized_section_t external_timepoint_handles;
  // Serialized executable format records.
  iree_hal_device_spec_serialized_section_t executable_formats;
  // Serialized executable target records.
  iree_hal_device_spec_serialized_section_t executable_targets;
  // Serialized driver-local facet records.
  iree_hal_device_spec_serialized_section_t facets;
  // Serialized string table bytes.
  iree_hal_device_spec_serialized_range_t string_table;
  // Serialized facet payload bytes.
  iree_hal_device_spec_serialized_range_t facet_payloads;
  // Serialized logical device identity record.
  uint64_t identity_offset;
  // Serialized memory record.
  uint64_t memory_offset;
  // Serialized virtual memory record.
  uint64_t virtual_memory_offset;
  // Serialized queue record.
  uint64_t queues_offset;
  // Serialized dispatch record.
  uint64_t dispatch_offset;
  // Serialized timing record.
  uint64_t timing_offset;
  // Serialized executable record.
  uint64_t executables_offset;
} iree_hal_device_spec_serialized_header_t;

typedef struct iree_hal_device_spec_serialized_string_t {
  // Byte offset into the serialized string table.
  uint64_t offset;
  // String byte length.
  uint64_t length;
} iree_hal_device_spec_serialized_string_t;

typedef struct iree_hal_device_spec_serialized_bytes_t {
  // Byte offset into the serialized facet payload table.
  uint64_t offset;
  // Payload byte length.
  uint64_t length;
} iree_hal_device_spec_serialized_bytes_t;

typedef struct iree_hal_serialized_physical_device_identity_t {
  // Human-readable physical device name.
  iree_hal_device_spec_serialized_string_t display_name;
  // Backend-native device path or locator string.
  iree_hal_device_spec_serialized_string_t backend_path;
  // Stable vendor identifier when available.
  uint32_t vendor_id;
  // Stable device identifier when available.
  uint32_t device_id;
  // Stable revision identifier when available.
  uint32_t revision_id;
  // Optional 128-bit hardware or driver UUID.
  iree_hal_uuid_t uuid;
  // Optional PCI address.
  iree_hal_pci_address_t pci;
  // Optional NUMA node.
  iree_hal_numa_node_t numa;
  // Availability flags for optional fields.
  iree_hal_physical_device_identity_flags_t flags;
} iree_hal_serialized_physical_device_identity_t;

typedef struct iree_hal_serialized_physical_device_spec_t {
  // Stable physical identity fields.
  iree_hal_serialized_physical_device_identity_t identity;
  // Physical device ordinal within the backend.
  uint32_t physical_ordinal;
  // Partition ordinal when the physical device is partitioned.
  uint32_t partition_ordinal;
  // Total partition count for the physical device.
  uint32_t partition_count;
  // Bitmask identifying the physical-device slice covered by this record.
  uint64_t physical_device_affinity;
} iree_hal_serialized_physical_device_spec_t;

typedef struct iree_hal_serialized_device_identity_spec_t {
  // Stable logical device identifier.
  iree_hal_device_spec_serialized_string_t logical_device_id;
  // Human-readable logical device name.
  iree_hal_device_spec_serialized_string_t display_name;
  // HAL driver identifier.
  iree_hal_device_spec_serialized_string_t driver_id;
  // HAL driver version string.
  iree_hal_device_spec_serialized_string_t driver_version;
  // Backend API identifier.
  iree_hal_device_spec_serialized_string_t backend_id;
  // Backend-native logical device path.
  iree_hal_device_spec_serialized_string_t device_path;
  // Human-readable vendor name.
  iree_hal_device_spec_serialized_string_t vendor_name;
  // Stable vendor identifier when available.
  uint32_t vendor_id;
  // Stable device identifier when available.
  uint32_t device_id;
  // Stable revision identifier when available.
  uint32_t revision_id;
  // Logical device ordinal within the driver.
  uint32_t logical_ordinal;
  // Logical device identity flags.
  iree_hal_device_identity_flags_t flags;
} iree_hal_serialized_device_identity_spec_t;

typedef struct iree_hal_serialized_memory_heap_spec_t {
  // Human-readable heap name.
  iree_hal_device_spec_serialized_string_t name;
  // Total heap capacity in bytes.
  uint64_t capacity_bytes;
  // Allocation granularity in bytes.
  uint64_t allocation_granularity;
  // Required allocation alignment in bytes.
  uint64_t allocation_alignment;
  // Maximum single allocation size in bytes.
  uint64_t maximum_allocation_size;
  // Physical device affinity for allocations in this heap.
  uint64_t physical_device_affinity;
  // Stable memory heap flags.
  iree_hal_memory_heap_spec_flags_t flags;
} iree_hal_serialized_memory_heap_spec_t;

typedef struct iree_hal_serialized_memory_type_spec_t {
  // Index of the heap backing this memory type.
  uint32_t heap_index;
  // HAL memory type bits represented by this memory type.
  iree_hal_memory_type_t memory_type;
  // Buffer usage bits accepted for allocations of this memory type.
  iree_hal_buffer_usage_t allowed_buffer_usage;
  // Memory access bits accepted for mappings of this memory type.
  iree_hal_memory_access_t allowed_memory_access;
  // Minimum buffer alignment in bytes.
  uint64_t minimum_alignment;
  // Optimal transfer granularity in bytes.
  uint64_t optimal_transfer_granularity;
  // Stable memory type flags.
  iree_hal_memory_type_spec_flags_t flags;
} iree_hal_serialized_memory_type_spec_t;

typedef struct iree_hal_serialized_device_memory_spec_t {
  // Stable memory capability flags.
  iree_hal_device_memory_spec_flags_t flags;
} iree_hal_serialized_device_memory_spec_t;

typedef struct iree_hal_serialized_device_virtual_memory_spec_t {
  // Stable virtual memory capability flags.
  iree_hal_device_virtual_memory_spec_flags_t flags;
} iree_hal_serialized_device_virtual_memory_spec_t;

typedef struct iree_hal_serialized_queue_family_spec_t {
  // Human-readable queue family name.
  iree_hal_device_spec_serialized_string_t name;
  // Number of queues in the family.
  uint32_t queue_count;
  // Number of priority levels in the family.
  uint32_t priority_count;
  // Valid timestamp bit count.
  uint32_t timestamp_valid_bits;
  // Timestamp frequency in ticks per second.
  uint64_t timestamp_frequency_hz;
  // Physical device affinity for queues in this family.
  uint64_t physical_device_affinity;
  // Queue family role flags.
  iree_hal_queue_family_role_flags_t role_flags;
  // Queue family capability flags.
  iree_hal_queue_family_spec_flags_t flags;
} iree_hal_serialized_queue_family_spec_t;

typedef struct iree_hal_serialized_device_queue_spec_t {
  // Stable queue capability flags.
  iree_hal_device_queue_spec_flags_t flags;
} iree_hal_serialized_device_queue_spec_t;

typedef struct iree_hal_serialized_executable_format_spec_t {
  // Artifact format string.
  iree_hal_device_spec_serialized_string_t format;
  // Caching modes supported for this format.
  iree_hal_executable_caching_mode_t caching_modes;
  // Stable executable format flags.
  iree_hal_executable_format_spec_flags_t flags;
} iree_hal_serialized_executable_format_spec_t;

typedef struct iree_hal_serialized_executable_target_t {
  // Target family such as amdgpu, spirv, ireevm, or local.
  iree_hal_device_spec_serialized_string_t family;
  // Target architecture within the family.
  iree_hal_device_spec_serialized_string_t architecture;
  // Target processor within the architecture.
  iree_hal_device_spec_serialized_string_t processor;
  // Target feature string.
  iree_hal_device_spec_serialized_string_t features;
  // Artifact format accepted by the loader.
  iree_hal_device_spec_serialized_string_t artifact_format;
  // Runtime ABI expected by the artifact.
  iree_hal_device_spec_serialized_string_t runtime_abi;
  // Loader namespace that interprets the artifact.
  iree_hal_device_spec_serialized_string_t loader_namespace;
  // Canonical loader target string used for cache keys and reports.
  iree_hal_device_spec_serialized_string_t loader_target;
  // Metadata schema expected by the loader.
  iree_hal_device_spec_serialized_string_t metadata_schema;
  // Executable target kind.
  iree_hal_executable_target_kind_t kind;
  // Selection priority where larger values are preferred.
  uint32_t priority;
  // Physical device affinity for this target.
  uint64_t physical_device_affinity;
  // Stable executable target flags.
  iree_hal_executable_target_flags_t flags;
} iree_hal_serialized_executable_target_t;

typedef struct iree_hal_serialized_device_executable_spec_t {
  // Stable executable capability flags.
  iree_hal_device_executable_spec_flags_t flags;
} iree_hal_serialized_device_executable_spec_t;

typedef struct iree_hal_serialized_device_spec_facet_t {
  // Stable facet schema identifier.
  iree_hal_device_spec_serialized_string_t schema_id;
  // Facet schema version.
  uint32_t schema_version;
  // Facet payload bytes.
  iree_hal_device_spec_serialized_bytes_t payload;
} iree_hal_serialized_device_spec_facet_t;

//===----------------------------------------------------------------------===//
// iree_hal_device_spec_t
//===----------------------------------------------------------------------===//

struct iree_hal_device_spec_t {
  // Retain/release counter.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for all owned storage.
  iree_allocator_t host_allocator;
  // Stable digest of the canonical serialized byte image.
  uint64_t digest;
  // Owned string table backing public string views.
  char* string_table;
  // Owned string table byte length.
  iree_host_size_t string_table_length;
  // Owned physical device records.
  iree_hal_physical_device_spec_t* physical_devices;
  // Owned memory heap records.
  iree_hal_memory_heap_spec_t* memory_heaps;
  // Owned memory type records.
  iree_hal_memory_type_spec_t* memory_types;
  // Owned external buffer handle records.
  iree_hal_external_buffer_handle_spec_t* external_buffer_handles;
  // Owned virtual memory class records.
  iree_hal_virtual_memory_class_spec_t* virtual_memory_classes;
  // Owned queue family records.
  iree_hal_queue_family_spec_t* queue_families;
  // Owned external timepoint handle records.
  iree_hal_external_timepoint_handle_spec_t* external_timepoint_handles;
  // Owned executable format records.
  iree_hal_executable_format_spec_t* executable_formats;
  // Owned executable target records.
  iree_hal_executable_target_t* executable_targets;
  // Owned driver-local extension facet records.
  iree_hal_device_spec_facet_t* facets;
  // Number of owned driver-local extension facet records.
  iree_host_size_t facet_count;
  // Owned driver-local extension facet payload storage.
  uint8_t* facet_payload_storage;
  // Owned driver-local extension facet payload storage byte length.
  iree_host_size_t facet_payload_storage_length;
  // Logical device identity facet.
  iree_hal_device_identity_spec_t identity;
  // Memory capability facet.
  iree_hal_device_memory_spec_t memory;
  // Virtual memory capability facet.
  iree_hal_device_virtual_memory_spec_t virtual_memory;
  // Queue capability facet.
  iree_hal_device_queue_spec_t queues;
  // Dispatch capability facet.
  iree_hal_device_dispatch_spec_t dispatch;
  // Timing and profiling capability facet.
  iree_hal_device_timing_spec_t timing;
  // Executable capability facet.
  iree_hal_device_executable_spec_t executables;
};

static uint64_t iree_hal_device_spec_digest_update(
    uint64_t state, iree_const_byte_span_t bytes) {
  for (iree_host_size_t i = 0; i < bytes.data_length; ++i) {
    state ^= bytes.data[i];
    state *= IREE_HAL_DEVICE_SPEC_FNV1A64_PRIME;
  }
  return state;
}

static iree_status_t iree_hal_device_spec_add_size(
    iree_host_size_t value, iree_host_size_t* inout_total) {
  if (IREE_UNLIKELY(value > IREE_HOST_SIZE_MAX - *inout_total)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec storage size overflow");
  }
  *inout_total += value;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_accumulate_string(
    iree_string_view_t value, iree_host_size_t* inout_total) {
  if (IREE_UNLIKELY(value.size && !value.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty device spec string has NULL storage");
  }
  return iree_hal_device_spec_add_size(value.size, inout_total);
}

static iree_status_t iree_hal_device_spec_accumulate_bytes(
    iree_const_byte_span_t value, iree_host_size_t* inout_total) {
  if (IREE_UNLIKELY(value.data_length && !value.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty device spec payload has NULL storage");
  }
  return iree_hal_device_spec_add_size(value.data_length, inout_total);
}

static iree_string_view_t iree_hal_device_spec_copy_string(
    iree_string_view_t value, char* storage,
    iree_host_size_t* inout_storage_offset) {
  if (iree_string_view_is_empty(value)) return iree_string_view_empty();
  iree_string_view_t result =
      iree_make_string_view(storage + *inout_storage_offset, value.size);
  memcpy((void*)result.data, value.data, value.size);
  *inout_storage_offset += value.size;
  return result;
}

static iree_const_byte_span_t iree_hal_device_spec_copy_bytes(
    iree_const_byte_span_t value, uint8_t* storage,
    iree_host_size_t* inout_storage_offset) {
  if (iree_const_byte_span_is_empty(value)) {
    return iree_const_byte_span_empty();
  }
  iree_const_byte_span_t result = iree_make_const_byte_span(
      storage + *inout_storage_offset, value.data_length);
  memcpy((void*)result.data, value.data, value.data_length);
  *inout_storage_offset += value.data_length;
  return result;
}

static iree_status_t iree_hal_device_spec_validate_count_pointer(
    iree_host_size_t count, const void* ptr, const char* field_name) {
  if (IREE_UNLIKELY(count && !ptr)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s has count %" PRIhsz
                            " but NULL storage",
                            field_name, count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_validate_params(
    const iree_hal_device_spec_params_t* params) {
  if (!params) return iree_ok_status();
  const iree_hal_device_identity_spec_t* identity = params->identity;
  if (identity) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        identity->physical_device_count, identity->physical_devices,
        "identity.physical_devices"));
  }
  const iree_hal_device_memory_spec_t* memory = params->memory;
  if (memory) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        memory->heap_count, memory->heaps, "memory.heaps"));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        memory->memory_type_count, memory->memory_types,
        "memory.memory_types"));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        memory->external_buffer_handle_count, memory->external_buffer_handles,
        "memory.external_buffer_handles"));
    for (iree_host_size_t i = 0; i < memory->memory_type_count; ++i) {
      if (memory->memory_types[i].heap_index >= memory->heap_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "device spec memory type %" PRIhsz
            " references heap %u outside heap count %" PRIhsz,
            i, memory->memory_types[i].heap_index, memory->heap_count);
      }
    }
  }
  const iree_hal_device_virtual_memory_spec_t* virtual_memory =
      params->virtual_memory;
  if (virtual_memory) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        virtual_memory->class_count, virtual_memory->classes,
        "virtual_memory.classes"));
  }
  const iree_hal_device_queue_spec_t* queues = params->queues;
  if (queues) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        queues->family_count, queues->families, "queues.families"));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        queues->external_timepoint_handle_count,
        queues->external_timepoint_handles,
        "queues.external_timepoint_handles"));
  }
  const iree_hal_device_executable_spec_t* executables = params->executables;
  if (executables) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        executables->format_count, executables->formats,
        "executables.formats"));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
        executables->target_count, executables->targets,
        "executables.targets"));
    for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
      if (executables->targets[i].kind >
          IREE_HAL_EXECUTABLE_TARGET_KIND_COMPOSITE) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "device spec executable target %" PRIhsz
                                " has invalid kind %" PRIu32,
                                i, executables->targets[i].kind);
      }
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_count_pointer(
      params->facet_count, params->facets, "facets"));
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_count_strings_and_payloads(
    const iree_hal_device_spec_params_t* params,
    iree_host_size_t* out_string_table_length,
    iree_host_size_t* out_payload_table_length) {
  iree_host_size_t string_table_length = 0;
  iree_host_size_t payload_table_length = 0;
  if (!params) {
    *out_string_table_length = 0;
    *out_payload_table_length = 0;
    return iree_ok_status();
  }
  const iree_hal_device_identity_spec_t* identity = params->identity;
  if (identity) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->logical_device_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->display_name, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->driver_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->driver_version, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->backend_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->device_path, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        identity->vendor_name, &string_table_length));
    for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          identity->physical_devices[i].identity.display_name,
          &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          identity->physical_devices[i].identity.backend_path,
          &string_table_length));
    }
  }
  const iree_hal_device_memory_spec_t* memory = params->memory;
  if (memory) {
    for (iree_host_size_t i = 0; i < memory->heap_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          memory->heaps[i].name, &string_table_length));
    }
  }
  const iree_hal_device_queue_spec_t* queues = params->queues;
  if (queues) {
    for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          queues->families[i].name, &string_table_length));
    }
  }
  const iree_hal_device_executable_spec_t* executables = params->executables;
  if (executables) {
    for (iree_host_size_t i = 0; i < executables->format_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          executables->formats[i].format, &string_table_length));
    }
    for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
      const iree_hal_executable_target_t* target = &executables->targets[i];
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->family, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->architecture, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->processor, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->features, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->artifact_format, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->runtime_abi, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->loader_namespace, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->loader_target, &string_table_length));
      IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
          target->metadata_schema, &string_table_length));
    }
  }
  for (iree_host_size_t i = 0; i < params->facet_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_string(
        params->facets[i].schema_id, &string_table_length));
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_accumulate_bytes(
        params->facets[i].payload, &payload_table_length));
  }
  *out_string_table_length = string_table_length;
  *out_payload_table_length = payload_table_length;
  return iree_ok_status();
}

static void iree_hal_device_spec_destroy(iree_hal_device_spec_t* spec) {
  iree_allocator_t host_allocator = spec->host_allocator;
  iree_allocator_free(host_allocator, spec->string_table);
  iree_allocator_free(host_allocator, spec->physical_devices);
  iree_allocator_free(host_allocator, spec->memory_heaps);
  iree_allocator_free(host_allocator, spec->memory_types);
  iree_allocator_free(host_allocator, spec->external_buffer_handles);
  iree_allocator_free(host_allocator, spec->virtual_memory_classes);
  iree_allocator_free(host_allocator, spec->queue_families);
  iree_allocator_free(host_allocator, spec->external_timepoint_handles);
  iree_allocator_free(host_allocator, spec->executable_formats);
  iree_allocator_free(host_allocator, spec->executable_targets);
  iree_allocator_free(host_allocator, spec->facets);
  iree_allocator_free(host_allocator, spec->facet_payload_storage);
  iree_allocator_free(host_allocator, spec);
}

static iree_status_t iree_hal_device_spec_clone_array(
    iree_allocator_t host_allocator, iree_host_size_t count,
    iree_host_size_t element_size, const void* source, void** out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;
  if (!count) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, count,
                                                   element_size, out_target));
  memcpy(*out_target, source, count * element_size);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_spec_create(
    const iree_hal_device_spec_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_params(params));

  iree_host_size_t string_table_length = 0;
  iree_host_size_t facet_payload_storage_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_count_strings_and_payloads(
      params, &string_table_length, &facet_payload_storage_length));

  iree_hal_device_spec_t* spec = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*spec), (void**)&spec));
  memset(spec, 0, sizeof(*spec));
  iree_atomic_ref_count_init(&spec->ref_count);
  spec->host_allocator = host_allocator;
  spec->string_table_length = string_table_length;
  spec->facet_payload_storage_length = facet_payload_storage_length;

  iree_status_t status = iree_ok_status();
  if (string_table_length) {
    status = iree_allocator_malloc_uninitialized(
        host_allocator, string_table_length, (void**)&spec->string_table);
  }
  if (iree_status_is_ok(status) && facet_payload_storage_length) {
    status = iree_allocator_malloc_uninitialized(
        host_allocator, facet_payload_storage_length,
        (void**)&spec->facet_payload_storage);
  }

  char* string_storage = spec->string_table;
  iree_host_size_t string_offset = 0;
  uint8_t* payload_storage = spec->facet_payload_storage;
  iree_host_size_t payload_offset = 0;

  if (iree_status_is_ok(status) && params && params->identity) {
    spec->identity = *params->identity;
    spec->identity.logical_device_id = iree_hal_device_spec_copy_string(
        params->identity->logical_device_id, string_storage, &string_offset);
    spec->identity.display_name = iree_hal_device_spec_copy_string(
        params->identity->display_name, string_storage, &string_offset);
    spec->identity.driver_id = iree_hal_device_spec_copy_string(
        params->identity->driver_id, string_storage, &string_offset);
    spec->identity.driver_version = iree_hal_device_spec_copy_string(
        params->identity->driver_version, string_storage, &string_offset);
    spec->identity.backend_id = iree_hal_device_spec_copy_string(
        params->identity->backend_id, string_storage, &string_offset);
    spec->identity.device_path = iree_hal_device_spec_copy_string(
        params->identity->device_path, string_storage, &string_offset);
    spec->identity.vendor_name = iree_hal_device_spec_copy_string(
        params->identity->vendor_name, string_storage, &string_offset);
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->identity->physical_device_count,
        sizeof(*spec->physical_devices), params->identity->physical_devices,
        (void**)&spec->physical_devices);
    spec->identity.physical_devices = spec->physical_devices;
    for (iree_host_size_t i = 0;
         i < spec->identity.physical_device_count && iree_status_is_ok(status);
         ++i) {
      spec->physical_devices[i].identity.display_name =
          iree_hal_device_spec_copy_string(
              params->identity->physical_devices[i].identity.display_name,
              string_storage, &string_offset);
      spec->physical_devices[i].identity.backend_path =
          iree_hal_device_spec_copy_string(
              params->identity->physical_devices[i].identity.backend_path,
              string_storage, &string_offset);
    }
  }
  if (iree_status_is_ok(status) && params && params->memory) {
    spec->memory = *params->memory;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->memory->heap_count, sizeof(*spec->memory_heaps),
        params->memory->heaps, (void**)&spec->memory_heaps);
    spec->memory.heaps = spec->memory_heaps;
    for (iree_host_size_t i = 0;
         i < spec->memory.heap_count && iree_status_is_ok(status); ++i) {
      spec->memory_heaps[i].name = iree_hal_device_spec_copy_string(
          params->memory->heaps[i].name, string_storage, &string_offset);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_clone_array(
          host_allocator, params->memory->memory_type_count,
          sizeof(*spec->memory_types), params->memory->memory_types,
          (void**)&spec->memory_types);
    }
    spec->memory.memory_types = spec->memory_types;
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_clone_array(
          host_allocator, params->memory->external_buffer_handle_count,
          sizeof(*spec->external_buffer_handles),
          params->memory->external_buffer_handles,
          (void**)&spec->external_buffer_handles);
    }
    spec->memory.external_buffer_handles = spec->external_buffer_handles;
  }
  if (iree_status_is_ok(status) && params && params->virtual_memory) {
    spec->virtual_memory = *params->virtual_memory;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->virtual_memory->class_count,
        sizeof(*spec->virtual_memory_classes), params->virtual_memory->classes,
        (void**)&spec->virtual_memory_classes);
    spec->virtual_memory.classes = spec->virtual_memory_classes;
  }
  if (iree_status_is_ok(status) && params && params->queues) {
    spec->queues = *params->queues;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->queues->family_count,
        sizeof(*spec->queue_families), params->queues->families,
        (void**)&spec->queue_families);
    spec->queues.families = spec->queue_families;
    for (iree_host_size_t i = 0;
         i < spec->queues.family_count && iree_status_is_ok(status); ++i) {
      spec->queue_families[i].name = iree_hal_device_spec_copy_string(
          params->queues->families[i].name, string_storage, &string_offset);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_clone_array(
          host_allocator, params->queues->external_timepoint_handle_count,
          sizeof(*spec->external_timepoint_handles),
          params->queues->external_timepoint_handles,
          (void**)&spec->external_timepoint_handles);
    }
    spec->queues.external_timepoint_handles = spec->external_timepoint_handles;
  }
  if (iree_status_is_ok(status) && params && params->dispatch) {
    spec->dispatch = *params->dispatch;
  }
  if (iree_status_is_ok(status) && params && params->timing) {
    spec->timing = *params->timing;
  }
  if (iree_status_is_ok(status) && params && params->executables) {
    spec->executables = *params->executables;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->executables->format_count,
        sizeof(*spec->executable_formats), params->executables->formats,
        (void**)&spec->executable_formats);
    spec->executables.formats = spec->executable_formats;
    for (iree_host_size_t i = 0;
         i < spec->executables.format_count && iree_status_is_ok(status); ++i) {
      spec->executable_formats[i].format = iree_hal_device_spec_copy_string(
          params->executables->formats[i].format, string_storage,
          &string_offset);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_clone_array(
          host_allocator, params->executables->target_count,
          sizeof(*spec->executable_targets), params->executables->targets,
          (void**)&spec->executable_targets);
    }
    spec->executables.targets = spec->executable_targets;
    for (iree_host_size_t i = 0;
         i < spec->executables.target_count && iree_status_is_ok(status); ++i) {
      const iree_hal_executable_target_t* source =
          &params->executables->targets[i];
      iree_hal_executable_target_t* target = &spec->executable_targets[i];
      target->family = iree_hal_device_spec_copy_string(
          source->family, string_storage, &string_offset);
      target->architecture = iree_hal_device_spec_copy_string(
          source->architecture, string_storage, &string_offset);
      target->processor = iree_hal_device_spec_copy_string(
          source->processor, string_storage, &string_offset);
      target->features = iree_hal_device_spec_copy_string(
          source->features, string_storage, &string_offset);
      target->artifact_format = iree_hal_device_spec_copy_string(
          source->artifact_format, string_storage, &string_offset);
      target->runtime_abi = iree_hal_device_spec_copy_string(
          source->runtime_abi, string_storage, &string_offset);
      target->loader_namespace = iree_hal_device_spec_copy_string(
          source->loader_namespace, string_storage, &string_offset);
      target->loader_target = iree_hal_device_spec_copy_string(
          source->loader_target, string_storage, &string_offset);
      target->metadata_schema = iree_hal_device_spec_copy_string(
          source->metadata_schema, string_storage, &string_offset);
    }
  }
  if (iree_status_is_ok(status) && params && params->facet_count) {
    spec->facet_count = params->facet_count;
    status = iree_hal_device_spec_clone_array(
        host_allocator, params->facet_count, sizeof(*spec->facets),
        params->facets, (void**)&spec->facets);
    for (iree_host_size_t i = 0;
         i < params->facet_count && iree_status_is_ok(status); ++i) {
      spec->facets[i].schema_id = iree_hal_device_spec_copy_string(
          params->facets[i].schema_id, string_storage, &string_offset);
      spec->facets[i].payload = iree_hal_device_spec_copy_bytes(
          params->facets[i].payload, payload_storage, &payload_offset);
    }
  }

  iree_byte_span_t serialized_bytes = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_device_spec_serialize(spec, host_allocator, &serialized_bytes);
  }
  if (iree_status_is_ok(status)) {
    spec->digest = iree_hal_device_spec_digest_update(
        IREE_HAL_DEVICE_SPEC_FNV1A64_OFFSET_BASIS,
        iree_const_cast_byte_span(serialized_bytes));
    iree_allocator_free(host_allocator, serialized_bytes.data);
    *out_spec = spec;
  } else {
    iree_allocator_free(host_allocator, serialized_bytes.data);
    iree_hal_device_spec_destroy(spec);
  }

  return status;
}

IREE_API_EXPORT void iree_hal_device_spec_retain(iree_hal_device_spec_t* spec) {
  if (IREE_LIKELY(spec)) {
    iree_atomic_ref_count_inc(&spec->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_device_spec_release(
    iree_hal_device_spec_t* spec) {
  if (IREE_LIKELY(spec) && iree_atomic_ref_count_dec(&spec->ref_count) == 1) {
    iree_hal_device_spec_destroy(spec);
  }
}

IREE_API_EXPORT uint64_t
iree_hal_device_spec_digest(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return spec->digest;
}

static iree_status_t iree_hal_device_spec_layout_bytes(
    iree_host_size_t byte_length, iree_host_size_t alignment,
    iree_host_size_t* inout_offset, uint64_t* out_offset) {
  iree_host_size_t aligned_offset = iree_host_align(*inout_offset, alignment);
  if (IREE_UNLIKELY(aligned_offset < *inout_offset)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec serialized offset overflow");
  }
  if (IREE_UNLIKELY(byte_length > IREE_HOST_SIZE_MAX - aligned_offset)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec serialized range overflow");
  }
  *out_offset = aligned_offset;
  *inout_offset = aligned_offset + byte_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_layout_section(
    iree_host_size_t count, iree_host_size_t element_size,
    iree_host_size_t alignment, iree_host_size_t* inout_offset,
    iree_hal_device_spec_serialized_section_t* out_section) {
  out_section->count = count;
  out_section->element_size = element_size;
  if (!count) {
    out_section->offset = 0;
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(count > IREE_HOST_SIZE_MAX / element_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec serialized section size overflow");
  }
  return iree_hal_device_spec_layout_bytes(count * element_size, alignment,
                                           inout_offset, &out_section->offset);
}

typedef struct iree_hal_device_spec_string_writer_t {
  // Base pointer of the serialized string table.
  uint8_t* base;
  // Current write offset within the serialized string table.
  iree_host_size_t offset;
} iree_hal_device_spec_string_writer_t;

static iree_hal_device_spec_serialized_string_t
iree_hal_device_spec_write_string(
    iree_string_view_t value,
    iree_hal_device_spec_string_writer_t* string_writer) {
  iree_hal_device_spec_serialized_string_t result = {0, 0};
  if (iree_string_view_is_empty(value)) return result;
  result.offset = string_writer->offset;
  result.length = value.size;
  memcpy(string_writer->base + string_writer->offset, value.data, value.size);
  string_writer->offset += value.size;
  return result;
}

typedef struct iree_hal_device_spec_payload_writer_t {
  // Base pointer of the serialized facet payload table.
  uint8_t* base;
  // Current write offset within the serialized facet payload table.
  iree_host_size_t offset;
} iree_hal_device_spec_payload_writer_t;

static iree_hal_device_spec_serialized_bytes_t
iree_hal_device_spec_write_payload(
    iree_const_byte_span_t value,
    iree_hal_device_spec_payload_writer_t* payload_writer) {
  iree_hal_device_spec_serialized_bytes_t result = {0, 0};
  if (iree_const_byte_span_is_empty(value)) return result;
  result.offset = payload_writer->offset;
  result.length = value.data_length;
  memcpy(payload_writer->base + payload_writer->offset, value.data,
         value.data_length);
  payload_writer->offset += value.data_length;
  return result;
}

IREE_API_EXPORT iree_status_t iree_hal_device_spec_serialize(
    const iree_hal_device_spec_t* spec, iree_allocator_t host_allocator,
    iree_byte_span_t* out_bytes) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(out_bytes);
  *out_bytes = iree_byte_span_empty();

  iree_hal_device_spec_serialized_header_t header = {0};
  header.magic = IREE_HAL_DEVICE_SPEC_MAGIC;
  header.version = IREE_HAL_DEVICE_SPEC_VERSION;
  header.header_length = sizeof(header);

  iree_host_size_t offset = sizeof(header);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
      sizeof(iree_hal_serialized_device_identity_spec_t),
      iree_alignof(iree_hal_serialized_device_identity_spec_t), &offset,
      &header.identity_offset));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
      sizeof(iree_hal_serialized_device_memory_spec_t),
      iree_alignof(iree_hal_serialized_device_memory_spec_t), &offset,
      &header.memory_offset));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
      sizeof(iree_hal_serialized_device_virtual_memory_spec_t),
      iree_alignof(iree_hal_serialized_device_virtual_memory_spec_t), &offset,
      &header.virtual_memory_offset));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
      sizeof(iree_hal_serialized_device_queue_spec_t),
      iree_alignof(iree_hal_serialized_device_queue_spec_t), &offset,
      &header.queues_offset));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
      sizeof(iree_hal_device_dispatch_spec_t),
      iree_alignof(iree_hal_device_dispatch_spec_t), &offset,
      &header.dispatch_offset));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
      sizeof(iree_hal_device_timing_spec_t),
      iree_alignof(iree_hal_device_timing_spec_t), &offset,
      &header.timing_offset));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
      sizeof(iree_hal_serialized_device_executable_spec_t),
      iree_alignof(iree_hal_serialized_device_executable_spec_t), &offset,
      &header.executables_offset));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->identity.physical_device_count,
      sizeof(iree_hal_serialized_physical_device_spec_t),
      iree_alignof(iree_hal_serialized_physical_device_spec_t), &offset,
      &header.physical_devices));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->memory.heap_count, sizeof(iree_hal_serialized_memory_heap_spec_t),
      iree_alignof(iree_hal_serialized_memory_heap_spec_t), &offset,
      &header.memory_heaps));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->memory.memory_type_count,
      sizeof(iree_hal_serialized_memory_type_spec_t),
      iree_alignof(iree_hal_serialized_memory_type_spec_t), &offset,
      &header.memory_types));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->memory.external_buffer_handle_count,
      sizeof(iree_hal_external_buffer_handle_spec_t),
      iree_alignof(iree_hal_external_buffer_handle_spec_t), &offset,
      &header.external_buffer_handles));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->virtual_memory.class_count,
      sizeof(iree_hal_virtual_memory_class_spec_t),
      iree_alignof(iree_hal_virtual_memory_class_spec_t), &offset,
      &header.virtual_memory_classes));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->queues.family_count,
      sizeof(iree_hal_serialized_queue_family_spec_t),
      iree_alignof(iree_hal_serialized_queue_family_spec_t), &offset,
      &header.queue_families));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->queues.external_timepoint_handle_count,
      sizeof(iree_hal_external_timepoint_handle_spec_t),
      iree_alignof(iree_hal_external_timepoint_handle_spec_t), &offset,
      &header.external_timepoint_handles));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->executables.format_count,
      sizeof(iree_hal_serialized_executable_format_spec_t),
      iree_alignof(iree_hal_serialized_executable_format_spec_t), &offset,
      &header.executable_formats));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->executables.target_count,
      sizeof(iree_hal_serialized_executable_target_t),
      iree_alignof(iree_hal_serialized_executable_target_t), &offset,
      &header.executable_targets));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_section(
      spec->facet_count, sizeof(iree_hal_serialized_device_spec_facet_t),
      iree_alignof(iree_hal_serialized_device_spec_facet_t), &offset,
      &header.facets));
  header.string_table.length = spec->string_table_length;
  if (header.string_table.length) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
        header.string_table.length, 1, &offset, &header.string_table.offset));
  }
  header.facet_payloads.length = spec->facet_payload_storage_length;
  if (header.facet_payloads.length) {
    IREE_RETURN_IF_ERROR(iree_hal_device_spec_layout_bytes(
        header.facet_payloads.length, 1, &offset,
        &header.facet_payloads.offset));
  }
  header.total_length = offset;

  uint8_t* bytes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, offset, (void**)&bytes));
  memcpy(bytes, &header, sizeof(header));

  iree_hal_device_spec_string_writer_t string_writer = {
      .base = bytes + header.string_table.offset,
      .offset = 0,
  };
  iree_hal_device_spec_payload_writer_t payload_writer = {
      .base = bytes + header.facet_payloads.offset,
      .offset = 0,
  };

  iree_hal_serialized_device_identity_spec_t* identity =
      (iree_hal_serialized_device_identity_spec_t*)(bytes +
                                                    header.identity_offset);
  *identity = (iree_hal_serialized_device_identity_spec_t){
      .logical_device_id = iree_hal_device_spec_write_string(
          spec->identity.logical_device_id, &string_writer),
      .display_name = iree_hal_device_spec_write_string(
          spec->identity.display_name, &string_writer),
      .driver_id = iree_hal_device_spec_write_string(spec->identity.driver_id,
                                                     &string_writer),
      .driver_version = iree_hal_device_spec_write_string(
          spec->identity.driver_version, &string_writer),
      .backend_id = iree_hal_device_spec_write_string(spec->identity.backend_id,
                                                      &string_writer),
      .device_path = iree_hal_device_spec_write_string(
          spec->identity.device_path, &string_writer),
      .vendor_name = iree_hal_device_spec_write_string(
          spec->identity.vendor_name, &string_writer),
      .vendor_id = spec->identity.vendor_id,
      .device_id = spec->identity.device_id,
      .revision_id = spec->identity.revision_id,
      .logical_ordinal = spec->identity.logical_ordinal,
      .flags = spec->identity.flags,
  };

  iree_hal_serialized_physical_device_spec_t* physical_devices =
      (iree_hal_serialized_physical_device_spec_t*)(bytes +
                                                    header.physical_devices
                                                        .offset);
  for (iree_host_size_t i = 0; i < spec->identity.physical_device_count; ++i) {
    const iree_hal_physical_device_spec_t* source =
        &spec->identity.physical_devices[i];
    physical_devices[i] = (iree_hal_serialized_physical_device_spec_t){
        .identity =
            {
                .display_name = iree_hal_device_spec_write_string(
                    source->identity.display_name, &string_writer),
                .backend_path = iree_hal_device_spec_write_string(
                    source->identity.backend_path, &string_writer),
                .vendor_id = source->identity.vendor_id,
                .device_id = source->identity.device_id,
                .revision_id = source->identity.revision_id,
                .uuid = source->identity.uuid,
                .pci = source->identity.pci,
                .numa = source->identity.numa,
                .flags = source->identity.flags,
            },
        .physical_ordinal = source->physical_ordinal,
        .partition_ordinal = source->partition_ordinal,
        .partition_count = source->partition_count,
        .physical_device_affinity = source->physical_device_affinity,
    };
  }

  *(iree_hal_serialized_device_memory_spec_t*)(bytes + header.memory_offset) =
      (iree_hal_serialized_device_memory_spec_t){
          .flags = spec->memory.flags,
      };
  iree_hal_serialized_memory_heap_spec_t* heaps =
      (iree_hal_serialized_memory_heap_spec_t*)(bytes +
                                                header.memory_heaps.offset);
  for (iree_host_size_t i = 0; i < spec->memory.heap_count; ++i) {
    const iree_hal_memory_heap_spec_t* source = &spec->memory.heaps[i];
    heaps[i] = (iree_hal_serialized_memory_heap_spec_t){
        .name = iree_hal_device_spec_write_string(source->name, &string_writer),
        .capacity_bytes = source->capacity_bytes,
        .allocation_granularity = source->allocation_granularity,
        .allocation_alignment = source->allocation_alignment,
        .maximum_allocation_size = source->maximum_allocation_size,
        .physical_device_affinity = source->physical_device_affinity,
        .flags = source->flags,
    };
  }
  for (iree_host_size_t i = 0; i < spec->memory.memory_type_count; ++i) {
    ((iree_hal_serialized_memory_type_spec_t*)(bytes +
                                               header.memory_types.offset))[i] =
        (iree_hal_serialized_memory_type_spec_t){
            .heap_index = spec->memory.memory_types[i].heap_index,
            .memory_type = spec->memory.memory_types[i].memory_type,
            .allowed_buffer_usage =
                spec->memory.memory_types[i].allowed_buffer_usage,
            .allowed_memory_access =
                spec->memory.memory_types[i].allowed_memory_access,
            .minimum_alignment = spec->memory.memory_types[i].minimum_alignment,
            .optimal_transfer_granularity =
                spec->memory.memory_types[i].optimal_transfer_granularity,
            .flags = spec->memory.memory_types[i].flags,
        };
  }
  if (spec->memory.external_buffer_handle_count) {
    memcpy(bytes + header.external_buffer_handles.offset,
           spec->memory.external_buffer_handles,
           spec->memory.external_buffer_handle_count *
               sizeof(*spec->memory.external_buffer_handles));
  }

  *(iree_hal_serialized_device_virtual_memory_spec_t*)(bytes +
                                                       header
                                                           .virtual_memory_offset) =
      (iree_hal_serialized_device_virtual_memory_spec_t){
          .flags = spec->virtual_memory.flags,
      };
  if (spec->virtual_memory.class_count) {
    memcpy(bytes + header.virtual_memory_classes.offset,
           spec->virtual_memory.classes,
           spec->virtual_memory.class_count *
               sizeof(*spec->virtual_memory.classes));
  }

  *(iree_hal_serialized_device_queue_spec_t*)(bytes + header.queues_offset) =
      (iree_hal_serialized_device_queue_spec_t){
          .flags = spec->queues.flags,
      };
  iree_hal_serialized_queue_family_spec_t* families =
      (iree_hal_serialized_queue_family_spec_t*)(bytes +
                                                 header.queue_families.offset);
  for (iree_host_size_t i = 0; i < spec->queues.family_count; ++i) {
    const iree_hal_queue_family_spec_t* source = &spec->queues.families[i];
    families[i] = (iree_hal_serialized_queue_family_spec_t){
        .name = iree_hal_device_spec_write_string(source->name, &string_writer),
        .queue_count = source->queue_count,
        .priority_count = source->priority_count,
        .timestamp_valid_bits = source->timestamp_valid_bits,
        .timestamp_frequency_hz = source->timestamp_frequency_hz,
        .physical_device_affinity = source->physical_device_affinity,
        .role_flags = source->role_flags,
        .flags = source->flags,
    };
  }
  if (spec->queues.external_timepoint_handle_count) {
    memcpy(bytes + header.external_timepoint_handles.offset,
           spec->queues.external_timepoint_handles,
           spec->queues.external_timepoint_handle_count *
               sizeof(*spec->queues.external_timepoint_handles));
  }

  memcpy(bytes + header.dispatch_offset, &spec->dispatch,
         sizeof(spec->dispatch));
  memcpy(bytes + header.timing_offset, &spec->timing, sizeof(spec->timing));

  *(iree_hal_serialized_device_executable_spec_t*)(bytes +
                                                   header.executables_offset) =
      (iree_hal_serialized_device_executable_spec_t){
          .flags = spec->executables.flags,
      };
  iree_hal_serialized_executable_format_spec_t* formats =
      (iree_hal_serialized_executable_format_spec_t*)(bytes +
                                                      header.executable_formats
                                                          .offset);
  for (iree_host_size_t i = 0; i < spec->executables.format_count; ++i) {
    formats[i] = (iree_hal_serialized_executable_format_spec_t){
        .format = iree_hal_device_spec_write_string(
            spec->executables.formats[i].format, &string_writer),
        .caching_modes = spec->executables.formats[i].caching_modes,
        .flags = spec->executables.formats[i].flags,
    };
  }
  iree_hal_serialized_executable_target_t* targets =
      (iree_hal_serialized_executable_target_t*)(bytes +
                                                 header.executable_targets
                                                     .offset);
  for (iree_host_size_t i = 0; i < spec->executables.target_count; ++i) {
    const iree_hal_executable_target_t* source = &spec->executables.targets[i];
    targets[i] = (iree_hal_serialized_executable_target_t){
        .family =
            iree_hal_device_spec_write_string(source->family, &string_writer),
        .architecture = iree_hal_device_spec_write_string(source->architecture,
                                                          &string_writer),
        .processor = iree_hal_device_spec_write_string(source->processor,
                                                       &string_writer),
        .features =
            iree_hal_device_spec_write_string(source->features, &string_writer),
        .artifact_format = iree_hal_device_spec_write_string(
            source->artifact_format, &string_writer),
        .runtime_abi = iree_hal_device_spec_write_string(source->runtime_abi,
                                                         &string_writer),
        .loader_namespace = iree_hal_device_spec_write_string(
            source->loader_namespace, &string_writer),
        .loader_target = iree_hal_device_spec_write_string(
            source->loader_target, &string_writer),
        .metadata_schema = iree_hal_device_spec_write_string(
            source->metadata_schema, &string_writer),
        .kind = source->kind,
        .priority = source->priority,
        .physical_device_affinity = source->physical_device_affinity,
        .flags = source->flags,
    };
  }

  iree_hal_serialized_device_spec_facet_t* facets =
      (iree_hal_serialized_device_spec_facet_t*)(bytes + header.facets.offset);
  for (iree_host_size_t i = 0; i < spec->facet_count; ++i) {
    facets[i] = (iree_hal_serialized_device_spec_facet_t){
        .schema_id = iree_hal_device_spec_write_string(
            spec->facets[i].schema_id, &string_writer),
        .schema_version = spec->facets[i].schema_version,
        .payload = iree_hal_device_spec_write_payload(spec->facets[i].payload,
                                                      &payload_writer),
    };
  }

  *out_bytes = iree_make_byte_span(bytes, offset);
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_validate_serialized_range(
    iree_const_byte_span_t bytes, uint64_t offset, uint64_t length,
    iree_host_size_t alignment, const char* field_name) {
  if (IREE_UNLIKELY(offset > IREE_HOST_SIZE_MAX ||
                    length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s range exceeds host size",
                            field_name);
  }
  if (IREE_UNLIKELY(
          offset && alignment &&
          !iree_host_size_has_alignment((iree_host_size_t)offset, alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s range is misaligned", field_name);
  }
  const iree_host_size_t host_offset = (iree_host_size_t)offset;
  const iree_host_size_t host_length = (iree_host_size_t)length;
  if (IREE_UNLIKELY(host_offset > bytes.data_length ||
                    host_length > bytes.data_length - host_offset)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s range is out of bounds",
                            field_name);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_validate_serialized_section(
    iree_const_byte_span_t bytes,
    iree_hal_device_spec_serialized_section_t section,
    iree_host_size_t expected_element_size, iree_host_size_t alignment,
    const char* section_name) {
  if (IREE_UNLIKELY(section.element_size != expected_element_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s element size %" PRIu64
                            " does not match expected %" PRIhsz,
                            section_name, section.element_size,
                            expected_element_size);
  }
  if (IREE_UNLIKELY(section.count > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s count exceeds host size",
                            section_name);
  }
  if (!section.count) {
    if (IREE_UNLIKELY(section.offset != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "device spec empty %s section must have zero offset", section_name);
    }
    return iree_ok_status();
  }
  if (IREE_UNLIKELY((iree_host_size_t)section.count >
                    IREE_HOST_SIZE_MAX / expected_element_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec %s byte length overflows",
                            section_name);
  }
  return iree_hal_device_spec_validate_serialized_range(
      bytes, section.offset, section.count * (uint64_t)expected_element_size,
      alignment, section_name);
}

static iree_status_t iree_hal_device_spec_read_serialized_string(
    iree_const_byte_span_t bytes,
    iree_hal_device_spec_serialized_header_t header,
    iree_hal_device_spec_serialized_string_t serialized_string,
    iree_string_view_t* out_value) {
  if (!serialized_string.length) {
    *out_value = iree_string_view_empty();
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(serialized_string.offset > IREE_HOST_SIZE_MAX ||
                    serialized_string.length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec string range exceeds host size");
  }
  const iree_host_size_t string_offset =
      (iree_host_size_t)serialized_string.offset;
  const iree_host_size_t string_length =
      (iree_host_size_t)serialized_string.length;
  if (IREE_UNLIKELY(string_offset > header.string_table.length ||
                    string_length >
                        header.string_table.length - string_offset)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec string range is out of bounds");
  }
  *out_value = iree_make_string_view(
      (const char*)bytes.data + header.string_table.offset + string_offset,
      string_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_read_serialized_payload(
    iree_const_byte_span_t bytes,
    iree_hal_device_spec_serialized_header_t header,
    iree_hal_device_spec_serialized_bytes_t serialized_bytes,
    iree_const_byte_span_t* out_value) {
  if (!serialized_bytes.length) {
    *out_value = iree_const_byte_span_empty();
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(serialized_bytes.offset > IREE_HOST_SIZE_MAX ||
                    serialized_bytes.length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec payload range exceeds host size");
  }
  const iree_host_size_t payload_offset =
      (iree_host_size_t)serialized_bytes.offset;
  const iree_host_size_t payload_length =
      (iree_host_size_t)serialized_bytes.length;
  if (IREE_UNLIKELY(payload_offset > header.facet_payloads.length ||
                    payload_length >
                        header.facet_payloads.length - payload_offset)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec payload range is out of bounds");
  }
  *out_value = iree_make_const_byte_span(
      bytes.data + header.facet_payloads.offset + payload_offset,
      payload_length);
  return iree_ok_status();
}

static void iree_hal_device_spec_copy_serialized_element(
    iree_const_byte_span_t bytes,
    iree_hal_device_spec_serialized_section_t section, iree_host_size_t index,
    iree_host_size_t element_size, void* out_element) {
  memcpy(out_element, bytes.data + section.offset + index * element_size,
         element_size);
}

static void iree_hal_device_spec_free_parse_arrays(
    iree_allocator_t host_allocator,
    iree_hal_physical_device_spec_t* physical_devices,
    iree_hal_memory_heap_spec_t* memory_heaps,
    iree_hal_memory_type_spec_t* memory_types,
    iree_hal_external_buffer_handle_spec_t* external_buffer_handles,
    iree_hal_virtual_memory_class_spec_t* virtual_memory_classes,
    iree_hal_queue_family_spec_t* queue_families,
    iree_hal_external_timepoint_handle_spec_t* external_timepoint_handles,
    iree_hal_executable_format_spec_t* executable_formats,
    iree_hal_executable_target_t* executable_targets,
    iree_hal_device_spec_facet_t* facets) {
  iree_allocator_free(host_allocator, physical_devices);
  iree_allocator_free(host_allocator, memory_heaps);
  iree_allocator_free(host_allocator, memory_types);
  iree_allocator_free(host_allocator, external_buffer_handles);
  iree_allocator_free(host_allocator, virtual_memory_classes);
  iree_allocator_free(host_allocator, queue_families);
  iree_allocator_free(host_allocator, external_timepoint_handles);
  iree_allocator_free(host_allocator, executable_formats);
  iree_allocator_free(host_allocator, executable_targets);
  iree_allocator_free(host_allocator, facets);
}

static iree_status_t iree_hal_device_spec_allocate_parse_array(
    iree_allocator_t host_allocator,
    iree_hal_device_spec_serialized_section_t section,
    iree_host_size_t element_size, void** out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;
  if (!section.count) return iree_ok_status();
  return iree_allocator_malloc_array(host_allocator,
                                     (iree_host_size_t)section.count,
                                     element_size, out_target);
}

IREE_API_EXPORT iree_status_t iree_hal_device_spec_parse(
    iree_const_byte_span_t bytes, iree_allocator_t host_allocator,
    iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;
  if (IREE_UNLIKELY(bytes.data_length <
                    sizeof(iree_hal_device_spec_serialized_header_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec image is too small");
  }
  iree_hal_device_spec_serialized_header_t header;
  memcpy(&header, bytes.data, sizeof(header));
  if (IREE_UNLIKELY(header.magic != IREE_HAL_DEVICE_SPEC_MAGIC)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec image has invalid magic");
  }
  if (IREE_UNLIKELY(header.version != IREE_HAL_DEVICE_SPEC_VERSION)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device spec image version %" PRIu32 " is unsupported", header.version);
  }
  if (IREE_UNLIKELY(header.header_length != sizeof(header))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec header length is invalid");
  }
  if (IREE_UNLIKELY(header.total_length != bytes.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device spec total length does not match image");
  }

  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.identity_offset,
      sizeof(iree_hal_serialized_device_identity_spec_t),
      iree_alignof(iree_hal_serialized_device_identity_spec_t), "identity"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.memory_offset,
      sizeof(iree_hal_serialized_device_memory_spec_t),
      iree_alignof(iree_hal_serialized_device_memory_spec_t), "memory"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.virtual_memory_offset,
      sizeof(iree_hal_serialized_device_virtual_memory_spec_t),
      iree_alignof(iree_hal_serialized_device_virtual_memory_spec_t),
      "virtual_memory"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.queues_offset,
      sizeof(iree_hal_serialized_device_queue_spec_t),
      iree_alignof(iree_hal_serialized_device_queue_spec_t), "queues"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.dispatch_offset, sizeof(iree_hal_device_dispatch_spec_t),
      iree_alignof(iree_hal_device_dispatch_spec_t), "dispatch"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.timing_offset, sizeof(iree_hal_device_timing_spec_t),
      iree_alignof(iree_hal_device_timing_spec_t), "timing"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.executables_offset,
      sizeof(iree_hal_serialized_device_executable_spec_t),
      iree_alignof(iree_hal_serialized_device_executable_spec_t),
      "executables"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.physical_devices,
      sizeof(iree_hal_serialized_physical_device_spec_t),
      iree_alignof(iree_hal_serialized_physical_device_spec_t),
      "physical_devices"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.memory_heaps,
      sizeof(iree_hal_serialized_memory_heap_spec_t),
      iree_alignof(iree_hal_serialized_memory_heap_spec_t), "memory_heaps"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.memory_types,
      sizeof(iree_hal_serialized_memory_type_spec_t),
      iree_alignof(iree_hal_serialized_memory_type_spec_t), "memory_types"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.external_buffer_handles,
      sizeof(iree_hal_external_buffer_handle_spec_t),
      iree_alignof(iree_hal_external_buffer_handle_spec_t),
      "external_buffer_handles"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.virtual_memory_classes,
      sizeof(iree_hal_virtual_memory_class_spec_t),
      iree_alignof(iree_hal_virtual_memory_class_spec_t),
      "virtual_memory_classes"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.queue_families,
      sizeof(iree_hal_serialized_queue_family_spec_t),
      iree_alignof(iree_hal_serialized_queue_family_spec_t), "queue_families"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.external_timepoint_handles,
      sizeof(iree_hal_external_timepoint_handle_spec_t),
      iree_alignof(iree_hal_external_timepoint_handle_spec_t),
      "external_timepoint_handles"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.executable_formats,
      sizeof(iree_hal_serialized_executable_format_spec_t),
      iree_alignof(iree_hal_serialized_executable_format_spec_t),
      "executable_formats"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.executable_targets,
      sizeof(iree_hal_serialized_executable_target_t),
      iree_alignof(iree_hal_serialized_executable_target_t),
      "executable_targets"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_section(
      bytes, header.facets, sizeof(iree_hal_serialized_device_spec_facet_t),
      iree_alignof(iree_hal_serialized_device_spec_facet_t), "facets"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.string_table.offset, header.string_table.length, 1,
      "string_table"));
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_validate_serialized_range(
      bytes, header.facet_payloads.offset, header.facet_payloads.length, 1,
      "facet_payloads"));

  iree_status_t status = iree_ok_status();
  iree_hal_physical_device_spec_t* physical_devices = NULL;
  iree_hal_memory_heap_spec_t* memory_heaps = NULL;
  iree_hal_memory_type_spec_t* memory_types = NULL;
  iree_hal_external_buffer_handle_spec_t* external_buffer_handles = NULL;
  iree_hal_virtual_memory_class_spec_t* virtual_memory_classes = NULL;
  iree_hal_queue_family_spec_t* queue_families = NULL;
  iree_hal_external_timepoint_handle_spec_t* external_timepoint_handles = NULL;
  iree_hal_executable_format_spec_t* executable_formats = NULL;
  iree_hal_executable_target_t* executable_targets = NULL;
  iree_hal_device_spec_facet_t* facets = NULL;

  status = iree_hal_device_spec_allocate_parse_array(
      host_allocator, header.physical_devices, sizeof(*physical_devices),
      (void**)&physical_devices);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.memory_heaps, sizeof(*memory_heaps),
        (void**)&memory_heaps);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.memory_types, sizeof(*memory_types),
        (void**)&memory_types);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.external_buffer_handles,
        sizeof(*external_buffer_handles), (void**)&external_buffer_handles);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.virtual_memory_classes,
        sizeof(*virtual_memory_classes), (void**)&virtual_memory_classes);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.queue_families, sizeof(*queue_families),
        (void**)&queue_families);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.external_timepoint_handles,
        sizeof(*external_timepoint_handles),
        (void**)&external_timepoint_handles);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.executable_formats, sizeof(*executable_formats),
        (void**)&executable_formats);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.executable_targets, sizeof(*executable_targets),
        (void**)&executable_targets);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_allocate_parse_array(
        host_allocator, header.facets, sizeof(*facets), (void**)&facets);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_device_spec_free_parse_arrays(
        host_allocator, physical_devices, memory_heaps, memory_types,
        external_buffer_handles, virtual_memory_classes, queue_families,
        external_timepoint_handles, executable_formats, executable_targets,
        facets);
    return status;
  }

  iree_hal_serialized_device_identity_spec_t serialized_identity;
  memcpy(&serialized_identity, bytes.data + header.identity_offset,
         sizeof(serialized_identity));
  iree_hal_device_identity_spec_t identity = {
      .vendor_id = serialized_identity.vendor_id,
      .device_id = serialized_identity.device_id,
      .revision_id = serialized_identity.revision_id,
      .logical_ordinal = serialized_identity.logical_ordinal,
      .physical_device_count = (iree_host_size_t)header.physical_devices.count,
      .physical_devices = physical_devices,
      .flags = serialized_identity.flags,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_identity.logical_device_id,
        &identity.logical_device_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_identity.display_name,
        &identity.display_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_identity.driver_id, &identity.driver_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_identity.driver_version,
        &identity.driver_version);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_identity.backend_id, &identity.backend_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_identity.device_path, &identity.device_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_identity.vendor_name, &identity.vendor_name);
  }

  for (iree_host_size_t i = 0;
       i < (iree_host_size_t)header.physical_devices.count &&
       iree_status_is_ok(status);
       ++i) {
    iree_hal_serialized_physical_device_spec_t serialized_physical_device;
    iree_hal_device_spec_copy_serialized_element(
        bytes, header.physical_devices, i, sizeof(serialized_physical_device),
        &serialized_physical_device);
    physical_devices[i] = (iree_hal_physical_device_spec_t){
        .identity =
            {
                .vendor_id = serialized_physical_device.identity.vendor_id,
                .device_id = serialized_physical_device.identity.device_id,
                .revision_id = serialized_physical_device.identity.revision_id,
                .uuid = serialized_physical_device.identity.uuid,
                .pci = serialized_physical_device.identity.pci,
                .numa = serialized_physical_device.identity.numa,
                .flags = serialized_physical_device.identity.flags,
            },
        .physical_ordinal = serialized_physical_device.physical_ordinal,
        .partition_ordinal = serialized_physical_device.partition_ordinal,
        .partition_count = serialized_physical_device.partition_count,
        .physical_device_affinity =
            serialized_physical_device.physical_device_affinity,
    };
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_physical_device.identity.display_name,
        &physical_devices[i].identity.display_name);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_physical_device.identity.backend_path,
          &physical_devices[i].identity.backend_path);
    }
  }

  iree_hal_serialized_device_memory_spec_t serialized_memory;
  memcpy(&serialized_memory, bytes.data + header.memory_offset,
         sizeof(serialized_memory));
  iree_hal_device_memory_spec_t memory = {
      .heap_count = (iree_host_size_t)header.memory_heaps.count,
      .heaps = memory_heaps,
      .memory_type_count = (iree_host_size_t)header.memory_types.count,
      .memory_types = memory_types,
      .external_buffer_handle_count =
          (iree_host_size_t)header.external_buffer_handles.count,
      .external_buffer_handles = external_buffer_handles,
      .flags = serialized_memory.flags,
  };
  for (iree_host_size_t i = 0;
       i < memory.heap_count && iree_status_is_ok(status); ++i) {
    iree_hal_serialized_memory_heap_spec_t serialized_heap;
    iree_hal_device_spec_copy_serialized_element(bytes, header.memory_heaps, i,
                                                 sizeof(serialized_heap),
                                                 &serialized_heap);
    memory_heaps[i] = (iree_hal_memory_heap_spec_t){
        .capacity_bytes = serialized_heap.capacity_bytes,
        .allocation_granularity = serialized_heap.allocation_granularity,
        .allocation_alignment = serialized_heap.allocation_alignment,
        .maximum_allocation_size = serialized_heap.maximum_allocation_size,
        .physical_device_affinity = serialized_heap.physical_device_affinity,
        .flags = serialized_heap.flags,
    };
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_heap.name, &memory_heaps[i].name);
  }
  for (iree_host_size_t i = 0; i < memory.memory_type_count; ++i) {
    iree_hal_serialized_memory_type_spec_t serialized_memory_type;
    iree_hal_device_spec_copy_serialized_element(bytes, header.memory_types, i,
                                                 sizeof(serialized_memory_type),
                                                 &serialized_memory_type);
    memory_types[i] = (iree_hal_memory_type_spec_t){
        .heap_index = serialized_memory_type.heap_index,
        .memory_type = serialized_memory_type.memory_type,
        .allowed_buffer_usage = serialized_memory_type.allowed_buffer_usage,
        .allowed_memory_access = serialized_memory_type.allowed_memory_access,
        .minimum_alignment = serialized_memory_type.minimum_alignment,
        .optimal_transfer_granularity =
            serialized_memory_type.optimal_transfer_granularity,
        .flags = serialized_memory_type.flags,
    };
  }
  for (iree_host_size_t i = 0; i < memory.external_buffer_handle_count; ++i) {
    iree_hal_device_spec_copy_serialized_element(
        bytes, header.external_buffer_handles, i,
        sizeof(external_buffer_handles[i]), &external_buffer_handles[i]);
  }

  iree_hal_serialized_device_virtual_memory_spec_t serialized_virtual_memory;
  memcpy(&serialized_virtual_memory, bytes.data + header.virtual_memory_offset,
         sizeof(serialized_virtual_memory));
  iree_hal_device_virtual_memory_spec_t virtual_memory = {
      .class_count = (iree_host_size_t)header.virtual_memory_classes.count,
      .classes = virtual_memory_classes,
      .flags = serialized_virtual_memory.flags,
  };
  for (iree_host_size_t i = 0; i < virtual_memory.class_count; ++i) {
    iree_hal_device_spec_copy_serialized_element(
        bytes, header.virtual_memory_classes, i,
        sizeof(virtual_memory_classes[i]), &virtual_memory_classes[i]);
  }

  iree_hal_serialized_device_queue_spec_t serialized_queues;
  memcpy(&serialized_queues, bytes.data + header.queues_offset,
         sizeof(serialized_queues));
  iree_hal_device_queue_spec_t queues = {
      .family_count = (iree_host_size_t)header.queue_families.count,
      .families = queue_families,
      .external_timepoint_handle_count =
          (iree_host_size_t)header.external_timepoint_handles.count,
      .external_timepoint_handles = external_timepoint_handles,
      .flags = serialized_queues.flags,
  };
  for (iree_host_size_t i = 0;
       i < queues.family_count && iree_status_is_ok(status); ++i) {
    iree_hal_serialized_queue_family_spec_t serialized_family;
    iree_hal_device_spec_copy_serialized_element(bytes, header.queue_families,
                                                 i, sizeof(serialized_family),
                                                 &serialized_family);
    queue_families[i] = (iree_hal_queue_family_spec_t){
        .queue_count = serialized_family.queue_count,
        .priority_count = serialized_family.priority_count,
        .timestamp_valid_bits = serialized_family.timestamp_valid_bits,
        .timestamp_frequency_hz = serialized_family.timestamp_frequency_hz,
        .physical_device_affinity = serialized_family.physical_device_affinity,
        .role_flags = serialized_family.role_flags,
        .flags = serialized_family.flags,
    };
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_family.name, &queue_families[i].name);
  }
  for (iree_host_size_t i = 0; i < queues.external_timepoint_handle_count;
       ++i) {
    iree_hal_device_spec_copy_serialized_element(
        bytes, header.external_timepoint_handles, i,
        sizeof(external_timepoint_handles[i]), &external_timepoint_handles[i]);
  }

  iree_hal_device_dispatch_spec_t dispatch;
  memcpy(&dispatch, bytes.data + header.dispatch_offset, sizeof(dispatch));
  iree_hal_device_timing_spec_t timing;
  memcpy(&timing, bytes.data + header.timing_offset, sizeof(timing));

  iree_hal_serialized_device_executable_spec_t serialized_executables;
  memcpy(&serialized_executables, bytes.data + header.executables_offset,
         sizeof(serialized_executables));
  iree_hal_device_executable_spec_t executables = {
      .format_count = (iree_host_size_t)header.executable_formats.count,
      .formats = executable_formats,
      .target_count = (iree_host_size_t)header.executable_targets.count,
      .targets = executable_targets,
      .flags = serialized_executables.flags,
  };
  for (iree_host_size_t i = 0;
       i < executables.format_count && iree_status_is_ok(status); ++i) {
    iree_hal_serialized_executable_format_spec_t serialized_format;
    iree_hal_device_spec_copy_serialized_element(
        bytes, header.executable_formats, i, sizeof(serialized_format),
        &serialized_format);
    executable_formats[i] = (iree_hal_executable_format_spec_t){
        .caching_modes = serialized_format.caching_modes,
        .flags = serialized_format.flags,
    };
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_format.format, &executable_formats[i].format);
  }
  for (iree_host_size_t i = 0;
       i < executables.target_count && iree_status_is_ok(status); ++i) {
    iree_hal_serialized_executable_target_t serialized_target;
    iree_hal_device_spec_copy_serialized_element(
        bytes, header.executable_targets, i, sizeof(serialized_target),
        &serialized_target);
    executable_targets[i] = (iree_hal_executable_target_t){
        .kind = serialized_target.kind,
        .priority = serialized_target.priority,
        .physical_device_affinity = serialized_target.physical_device_affinity,
        .flags = serialized_target.flags,
    };
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_target.family, &executable_targets[i].family);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.architecture,
          &executable_targets[i].architecture);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.processor,
          &executable_targets[i].processor);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.features,
          &executable_targets[i].features);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.artifact_format,
          &executable_targets[i].artifact_format);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.runtime_abi,
          &executable_targets[i].runtime_abi);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.loader_namespace,
          &executable_targets[i].loader_namespace);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.loader_target,
          &executable_targets[i].loader_target);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_string(
          bytes, header, serialized_target.metadata_schema,
          &executable_targets[i].metadata_schema);
    }
  }

  for (iree_host_size_t i = 0;
       i < (iree_host_size_t)header.facets.count && iree_status_is_ok(status);
       ++i) {
    iree_hal_serialized_device_spec_facet_t serialized_facet;
    iree_hal_device_spec_copy_serialized_element(
        bytes, header.facets, i, sizeof(serialized_facet), &serialized_facet);
    facets[i].schema_version = serialized_facet.schema_version;
    status = iree_hal_device_spec_read_serialized_string(
        bytes, header, serialized_facet.schema_id, &facets[i].schema_id);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_read_serialized_payload(
          bytes, header, serialized_facet.payload, &facets[i].payload);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_device_spec_params_t params = {
        .identity = &identity,
        .memory = &memory,
        .virtual_memory = &virtual_memory,
        .queues = &queues,
        .dispatch = &dispatch,
        .timing = &timing,
        .executables = &executables,
        .facet_count = (iree_host_size_t)header.facets.count,
        .facets = facets,
    };
    status = iree_hal_device_spec_create(&params, host_allocator, out_spec);
  }

  iree_hal_device_spec_free_parse_arrays(
      host_allocator, physical_devices, memory_heaps, memory_types,
      external_buffer_handles, virtual_memory_classes, queue_families,
      external_timepoint_handles, executable_formats, executable_targets,
      facets);
  return status;
}

static bool iree_hal_device_spec_mask_overlaps(uint64_t available_mask,
                                               uint64_t requested_mask) {
  return requested_mask == 0 || (available_mask & requested_mask) != 0;
}

static bool iree_hal_device_spec_all_bits_available(uint64_t available_bits,
                                                    uint64_t requested_bits) {
  return requested_bits == 0 ||
         (available_bits & requested_bits) == requested_bits;
}

IREE_API_EXPORT const iree_hal_device_identity_spec_t*
iree_hal_device_spec_identity(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->identity;
}

IREE_API_EXPORT const iree_hal_device_memory_spec_t*
iree_hal_device_spec_memory(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->memory;
}

IREE_API_EXPORT const iree_hal_external_buffer_handle_spec_t*
iree_hal_device_spec_find_external_buffer_handle(
    const iree_hal_device_spec_t* spec,
    const iree_hal_external_buffer_handle_selection_t* selection) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);
  for (iree_host_size_t i = 0; i < spec->memory.external_buffer_handle_count;
       ++i) {
    const iree_hal_external_buffer_handle_spec_t* handle =
        &spec->memory.external_buffer_handles[i];
    if (!iree_hal_device_spec_mask_overlaps(handle->handle_type_mask,
                                            selection->handle_type_mask)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->direction_flags,
                                                 selection->direction_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->allowed_buffer_usage,
                                                 selection->buffer_usage)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->allowed_memory_access,
                                                 selection->memory_access)) {
      continue;
    }
    if (!iree_hal_device_spec_mask_overlaps(
            handle->compatible_memory_type_mask,
            selection->compatible_memory_type_mask)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->flags,
                                                 selection->capability_flags)) {
      continue;
    }
    return handle;
  }
  return NULL;
}

IREE_API_EXPORT const iree_hal_device_virtual_memory_spec_t*
iree_hal_device_spec_virtual_memory(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->virtual_memory;
}

IREE_API_EXPORT const iree_hal_virtual_memory_class_spec_t*
iree_hal_device_spec_find_virtual_memory_class(
    const iree_hal_device_spec_t* spec,
    const iree_hal_virtual_memory_class_selection_t* selection) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);
  for (iree_host_size_t i = 0; i < spec->virtual_memory.class_count; ++i) {
    const iree_hal_virtual_memory_class_spec_t* memory_class =
        &spec->virtual_memory.classes[i];
    if (!iree_hal_device_spec_mask_overlaps(
            memory_class->compatible_memory_type_mask,
            selection->compatible_memory_type_mask)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(
            memory_class->allowed_buffer_usage, selection->buffer_usage)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(
            memory_class->allowed_memory_access, selection->memory_access)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(memory_class->operation_flags,
                                                 selection->operation_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(memory_class->protection_flags,
                                                 selection->protection_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(memory_class->advice_flags,
                                                 selection->advice_flags)) {
      continue;
    }
    return memory_class;
  }
  return NULL;
}

IREE_API_EXPORT const iree_hal_device_queue_spec_t* iree_hal_device_spec_queues(
    const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->queues;
}

IREE_API_EXPORT const iree_hal_external_timepoint_handle_spec_t*
iree_hal_device_spec_find_external_timepoint_handle(
    const iree_hal_device_spec_t* spec,
    const iree_hal_external_timepoint_handle_selection_t* selection) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);
  for (iree_host_size_t i = 0; i < spec->queues.external_timepoint_handle_count;
       ++i) {
    const iree_hal_external_timepoint_handle_spec_t* handle =
        &spec->queues.external_timepoint_handles[i];
    if (selection->handle_type != IREE_HAL_EXTERNAL_TIMEPOINT_TYPE_NONE &&
        handle->handle_type != selection->handle_type) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->direction_flags,
                                                 selection->direction_flags)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->compatibility,
                                                 selection->compatibility)) {
      continue;
    }
    if (!iree_hal_device_spec_all_bits_available(handle->flags,
                                                 selection->capability_flags)) {
      continue;
    }
    return handle;
  }
  return NULL;
}

IREE_API_EXPORT const iree_hal_device_dispatch_spec_t*
iree_hal_device_spec_dispatch(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->dispatch;
}

IREE_API_EXPORT const iree_hal_device_timing_spec_t*
iree_hal_device_spec_timing(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->timing;
}

IREE_API_EXPORT const iree_hal_device_executable_spec_t*
iree_hal_device_spec_executables(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return &spec->executables;
}

IREE_API_EXPORT iree_host_size_t
iree_hal_device_spec_facet_count(const iree_hal_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(spec);
  return spec->facet_count;
}

IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_device_spec_facet_at(const iree_hal_device_spec_t* spec,
                              iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(spec);
  return index < spec->facet_count ? &spec->facets[index] : NULL;
}

IREE_API_EXPORT const iree_hal_device_spec_facet_t*
iree_hal_device_spec_find_facet(const iree_hal_device_spec_t* spec,
                                iree_string_view_t schema_id) {
  IREE_ASSERT_ARGUMENT(spec);
  for (iree_host_size_t i = 0; i < spec->facet_count; ++i) {
    if (iree_string_view_equal(spec->facets[i].schema_id, schema_id)) {
      return &spec->facets[i];
    }
  }
  return NULL;
}

static bool iree_hal_device_spec_selection_string_matches(
    iree_string_view_t filter, iree_string_view_t value) {
  return iree_string_view_is_empty(filter) ||
         iree_string_view_equal(filter, value);
}

static bool iree_hal_device_spec_target_matches_selection(
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_target_selection_t* selection) {
  if (!iree_hal_device_spec_selection_string_matches(selection->family,
                                                     target->family)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(selection->architecture,
                                                     target->architecture)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(selection->processor,
                                                     target->processor)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(selection->features,
                                                     target->features)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(selection->artifact_format,
                                                     target->artifact_format)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(selection->runtime_abi,
                                                     target->runtime_abi)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(
          selection->loader_namespace, target->loader_namespace)) {
    return false;
  }
  if (!iree_hal_device_spec_selection_string_matches(selection->loader_target,
                                                     target->loader_target)) {
    return false;
  }
  if (selection->physical_device_affinity && target->physical_device_affinity &&
      (selection->physical_device_affinity &
       target->physical_device_affinity) == 0) {
    return false;
  }
  switch (selection->policy) {
    case IREE_HAL_EXECUTABLE_TARGET_SELECTION_POLICY_EXACT_DEVICE:
      return target->kind == IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT;
    case IREE_HAL_EXECUTABLE_TARGET_SELECTION_POLICY_MATCH_FIELDS:
      return true;
    case IREE_HAL_EXECUTABLE_TARGET_SELECTION_POLICY_COMPATIBLE_GENERIC:
      return target->kind == IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC;
    default:
      return false;
  }
}

IREE_API_EXPORT iree_hal_executable_target_selection_result_t
iree_hal_device_spec_select_executable_target(
    const iree_hal_device_spec_t* spec,
    const iree_hal_executable_target_selection_t* selection,
    const iree_hal_executable_target_t** out_target) {
  IREE_ASSERT_ARGUMENT(spec);
  IREE_ASSERT_ARGUMENT(selection);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;

  const iree_hal_executable_target_t* selected_target = NULL;
  iree_hal_executable_target_selection_result_t result =
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_NO_MATCH;
  for (iree_host_size_t i = 0; i < spec->executables.target_count; ++i) {
    const iree_hal_executable_target_t* candidate =
        &spec->executables.targets[i];
    if (!iree_hal_device_spec_target_matches_selection(candidate, selection)) {
      continue;
    }
    if (!selected_target || candidate->priority > selected_target->priority) {
      selected_target = candidate;
      result = IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_SELECTED;
    } else if (candidate->priority == selected_target->priority) {
      result = IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_AMBIGUOUS;
    }
  }
  if (result == IREE_HAL_EXECUTABLE_TARGET_SELECTION_RESULT_SELECTED) {
    *out_target = selected_target;
  }
  return result;
}
