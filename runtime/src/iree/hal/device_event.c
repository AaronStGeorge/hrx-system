// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_event.h"

#include <inttypes.h>
#include <stdio.h>

static const char* iree_hal_device_event_type_string(
    iree_hal_device_event_type_t type) {
  switch (type) {
    case IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE:
      return "driver_failure";
    case IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT:
      return "asan_report";
    case IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT:
      return "ubsan_report";
    case IREE_HAL_DEVICE_EVENT_TYPE_PRINTF:
      return "printf";
    case IREE_HAL_DEVICE_EVENT_TYPE_HOST_CALL:
      return "host_call";
    case IREE_HAL_DEVICE_EVENT_TYPE_NONE:
      return "none";
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_event_severity_string(
    iree_hal_device_event_severity_t severity) {
  switch (severity) {
    case IREE_HAL_DEVICE_EVENT_SEVERITY_TRACE:
      return "trace";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_INFO:
      return "info";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_WARNING:
      return "warning";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR:
      return "error";
    case IREE_HAL_DEVICE_EVENT_SEVERITY_FATAL:
      return "fatal";
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_asan_access_kind_string(
    iree_hal_device_asan_access_kind_t access_kind) {
  switch (access_kind) {
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ:
      return "read";
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE:
      return "write";
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_ATOMIC:
      return "atomic";
    case IREE_HAL_DEVICE_ASAN_ACCESS_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static const char* iree_hal_device_ubsan_check_kind_string(
    iree_hal_device_ubsan_check_kind_t check_kind) {
  switch (check_kind) {
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_INTEGER_OVERFLOW:
      return "integer_overflow";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_DIVIDE_BY_ZERO:
      return "divide_by_zero";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ALIGNMENT:
      return "alignment";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_FLOAT_NAN_CONTRACT:
      return "float_nan_contract";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNREACHABLE:
      return "unreachable";
    case IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

static bool iree_hal_device_event_payload_has_prefix(
    iree_const_byte_span_t payload, iree_host_size_t prefix_length) {
  return payload.data && payload.data_length >= prefix_length;
}

static void iree_hal_device_event_sink_stderr_print_source(
    const iree_hal_device_event_source_t* source) {
  const char* driver_id_data =
      source->driver_id.data ? source->driver_id.data : "";
  const char* device_id_data =
      source->device_id.data ? source->device_id.data : "";
  fprintf(stderr,
          " driver=%.*s device=%.*s physical_device=%u queue=%u "
          "executable=0x%016" PRIx64 " export=%u",
          (int)source->driver_id.size, driver_id_data,
          (int)source->device_id.size, device_id_data,
          source->physical_device_ordinal, source->queue_ordinal,
          source->executable_id, source->export_ordinal);
}

static void iree_hal_device_event_sink_stderr_print_site(
    const iree_hal_device_event_site_t* site) {
  if (!site) return;
  fprintf(stderr, " site=0x%016" PRIx64, site->site_id);
  if (!iree_string_view_is_empty(site->source_file)) {
    const char* source_file_data =
        site->source_file.data ? site->source_file.data : "";
    fprintf(stderr, " source=\"%.*s:%u:%u-%u:%u\"", (int)site->source_file.size,
            source_file_data, site->start_line, site->start_column,
            site->end_line, site->end_column);
  }
  if (!iree_string_view_is_empty(site->function_name)) {
    const char* function_name_data =
        site->function_name.data ? site->function_name.data : "";
    fprintf(stderr, " function=\"%.*s\"", (int)site->function_name.size,
            function_name_data);
  }
  if (!iree_string_view_is_empty(site->operation_name)) {
    const char* operation_name_data =
        site->operation_name.data ? site->operation_name.data : "";
    fprintf(stderr, " operation=\"%.*s\"", (int)site->operation_name.size,
            operation_name_data);
  }
  if (!iree_const_byte_span_is_empty(site->producer_payload)) {
    fprintf(stderr, " site_payload_bytes=%" PRIhsz,
            site->producer_payload.data_length);
  }
}

static void iree_hal_device_event_sink_stderr_print_asan(
    const iree_hal_device_asan_report_t* report) {
  fprintf(stderr,
          " access=%s fault_address=0x%016" PRIx64 " access_length=%" PRIu64
          " site_id=0x%016" PRIx64 " shadow_address=0x%016" PRIx64
          " shadow_value=0x%016" PRIx64
          " workgroup=(%u,%u,%u) workitem=(%u,%u,%u)"
          " dispatch=0x%016" PRIx64,
          iree_hal_device_asan_access_kind_string(report->access_kind),
          report->fault_address, report->access_length, report->site_id,
          report->shadow_address, report->shadow_value, report->workgroup_id[0],
          report->workgroup_id[1], report->workgroup_id[2],
          report->workitem_id[0], report->workitem_id[1],
          report->workitem_id[2], report->source_dispatch_ptr);
}

static void iree_hal_device_event_sink_stderr_print_ubsan(
    const iree_hal_device_ubsan_report_t* report) {
  fprintf(stderr,
          " check=%s site_id=0x%016" PRIx64 " operand0=0x%016" PRIx64
          " operand1=0x%016" PRIx64
          " workgroup=(%u,%u,%u)"
          " workitem=(%u,%u,%u) dispatch=0x%016" PRIx64,
          iree_hal_device_ubsan_check_kind_string(report->check_kind),
          report->site_id, report->operand0, report->operand1,
          report->workgroup_id[0], report->workgroup_id[1],
          report->workgroup_id[2], report->workitem_id[0],
          report->workitem_id[1], report->workitem_id[2],
          report->source_dispatch_ptr);
}

static void iree_hal_device_event_sink_stderr_print_printf(
    const iree_hal_device_printf_event_t* event) {
  if (!iree_string_view_is_empty(event->text)) {
    const char* text_data = event->text.data ? event->text.data : "";
    fprintf(stderr, " text=\"%.*s\"", (int)event->text.size, text_data);
  } else {
    fprintf(stderr, " format_id=0x%016" PRIx64 " argument_bytes=%" PRIhsz,
            event->format_id, event->arguments.data_length);
  }
}

static void iree_hal_device_event_sink_stderr_print_driver_failure(
    const iree_hal_device_driver_failure_event_t* event) {
  fprintf(stderr, " status=%s(%d) backend_result=0x%016" PRIx64,
          iree_status_code_string(event->status_code), event->status_code,
          event->backend_result_code);
  if (!iree_string_view_is_empty(event->message)) {
    const char* message_data = event->message.data ? event->message.data : "";
    fprintf(stderr, " message=\"%.*s\"", (int)event->message.size,
            message_data);
  }
}

static void iree_hal_device_event_sink_stderr_callback(
    void* user_data, const iree_hal_device_event_t* event) {
  (void)user_data;
  fprintf(
      stderr, "IREE HAL device event: type=%s severity=%s sequence=%" PRIu64,
      iree_hal_device_event_type_string(event->type),
      iree_hal_device_event_severity_string(event->severity), event->sequence);
  iree_hal_device_event_sink_stderr_print_source(&event->source);
  iree_hal_device_event_sink_stderr_print_site(event->site);

  switch (event->type) {
    case IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_asan_report_t))) {
        iree_hal_device_event_sink_stderr_print_asan(
            (const iree_hal_device_asan_report_t*)event->payload.data);
      }
      break;
    case IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_ubsan_report_t))) {
        iree_hal_device_event_sink_stderr_print_ubsan(
            (const iree_hal_device_ubsan_report_t*)event->payload.data);
      }
      break;
    case IREE_HAL_DEVICE_EVENT_TYPE_PRINTF:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_printf_event_t))) {
        iree_hal_device_event_sink_stderr_print_printf(
            (const iree_hal_device_printf_event_t*)event->payload.data);
      }
      break;
    case IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE:
      if (iree_hal_device_event_payload_has_prefix(
              event->payload, sizeof(iree_hal_device_driver_failure_event_t))) {
        iree_hal_device_event_sink_stderr_print_driver_failure(
            (const iree_hal_device_driver_failure_event_t*)event->payload.data);
      }
      break;
    default:
      if (!iree_const_byte_span_is_empty(event->payload)) {
        fprintf(stderr, " payload_bytes=%" PRIhsz, event->payload.data_length);
      }
      break;
  }

  fprintf(stderr, "\n");
}

IREE_API_EXPORT iree_hal_device_event_sink_t
iree_hal_device_event_sink_stderr(void) {
  iree_hal_device_event_sink_t sink;
  sink.fn = iree_hal_device_event_sink_stderr_callback;
  sink.user_data = NULL;
  return sink;
}
