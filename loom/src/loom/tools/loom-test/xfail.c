// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-test/xfail.h"

#include <string.h>

#include "loom/tooling/cli/loomc_result.h"
#include "loomc/iree.h"

static iree_status_t loom_test_parse_diagnostic_ref(iree_string_view_t value,
                                                    loom_error_ref_t* out_ref) {
  if (!loom_error_ref_parse(value, out_ref)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected canonical diagnostic identity DOMAIN/NNN, got '%.*s'",
        (int)value.size, value.data);
  }
  // Target diagnostics live in optional catalog shards that the test runner
  // does not own. Execution still fails loud for an unknown identity because
  // no emitted diagnostic can match the parsed reference.
  return iree_ok_status();
}

static const char* loom_test_xfail_policy_flag(
    loom_test_xfail_policy_t policy) {
  return policy == LOOM_TEST_XFAIL_POLICY_STRICT ? "--xfail"
                                                 : "--allow-failure";
}

static iree_status_t loom_test_parse_diagnostic_refs(
    loom_test_xfail_t* xfail, iree_allocator_t allocator) {
  xfail->diagnostic_count = 1;
  for (iree_host_size_t i = 0; i < xfail->diagnostic.size; ++i) {
    xfail->diagnostic_count += xfail->diagnostic.data[i] == ',';
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, xfail->diagnostic_count, sizeof(*xfail->diagnostic_refs),
      (void**)&xfail->diagnostic_refs));

  iree_string_view_t remaining = xfail->diagnostic;
  for (iree_host_size_t i = 0; i < xfail->diagnostic_count; ++i) {
    iree_string_view_t diagnostic = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &diagnostic, &remaining);
    if (iree_string_view_is_empty(diagnostic)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%s value for '%.*s' contains an empty diagnostic identity",
          loom_test_xfail_policy_flag(xfail->policy), (int)xfail->record.size,
          xfail->record.data);
    }
    IREE_RETURN_IF_ERROR(
        loom_test_parse_diagnostic_ref(diagnostic, &xfail->diagnostic_refs[i]));
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (xfail->diagnostic_refs[j] == xfail->diagnostic_refs[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "%s value for '%.*s' repeats diagnostic '%.*s'",
                                loom_test_xfail_policy_flag(xfail->policy),
                                (int)xfail->record.size, xfail->record.data,
                                (int)diagnostic.size, diagnostic.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_test_append_xfail_values(
    iree_string_view_list_t values, loom_test_xfail_policy_t policy,
    iree_allocator_t allocator, loom_test_xfail_list_t* list) {
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    iree_string_view_t record = iree_string_view_empty();
    iree_string_view_t diagnostic = iree_string_view_empty();
    if (iree_string_view_split(values.values[i], '=', &record, &diagnostic) <
            0 ||
        record.size < 2 || record.data[0] != '@' ||
        iree_string_view_is_empty(diagnostic)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s value '%.*s' must use "
                              "'@record=DOMAIN/NNN[,DOMAIN/NNN...]'",
                              loom_test_xfail_policy_flag(policy),
                              (int)values.values[i].size,
                              values.values[i].data);
    }
    for (iree_host_size_t j = 0; j < list->count; ++j) {
      if (iree_string_view_equal(list->values[j].record, record)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "expected-failure qualification repeats record '%.*s'",
            (int)record.size, record.data);
      }
    }
    loom_test_xfail_t* xfail = &list->values[list->count++];
    xfail->record = record;
    xfail->diagnostic = diagnostic;
    xfail->policy = policy;
    IREE_RETURN_IF_ERROR(loom_test_parse_diagnostic_refs(xfail, allocator));
  }
  return iree_ok_status();
}

iree_status_t loom_test_xfail_list_initialize(
    iree_string_view_list_t strict_values,
    iree_string_view_list_t allow_failure_values, iree_allocator_t allocator,
    loom_test_xfail_list_t* out_list) {
  *out_list = (loom_test_xfail_list_t){0};
  const iree_host_size_t capacity =
      strict_values.count + allow_failure_values.count;
  if (capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, capacity,
                                                   sizeof(*out_list->values),
                                                   (void**)&out_list->values));
  memset(out_list->values, 0, capacity * sizeof(*out_list->values));
  iree_status_t status = loom_test_append_xfail_values(
      strict_values, LOOM_TEST_XFAIL_POLICY_STRICT, allocator, out_list);
  if (iree_status_is_ok(status)) {
    status = loom_test_append_xfail_values(allow_failure_values,
                                           LOOM_TEST_XFAIL_POLICY_ALLOW_PASS,
                                           allocator, out_list);
  }
  if (!iree_status_is_ok(status)) {
    loom_test_xfail_list_deinitialize(out_list, allocator);
  }
  return status;
}

void loom_test_xfail_list_deinitialize(loom_test_xfail_list_t* list,
                                       iree_allocator_t allocator) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_allocator_free(allocator, list->values[i].diagnostic_refs);
  }
  iree_allocator_free(allocator, list->values);
  *list = (loom_test_xfail_list_t){0};
}

static iree_string_view_t loom_test_xfail_record_name(
    const loom_test_xfail_t* xfail) {
  return iree_string_view_substr(xfail->record, 1, IREE_HOST_SIZE_MAX);
}

loom_test_xfail_t* loom_test_xfail_list_find(loom_test_xfail_list_t* list,
                                             iree_string_view_t record_name) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (iree_string_view_equal(loom_test_xfail_record_name(&list->values[i]),
                               record_name)) {
      return &list->values[i];
    }
  }
  return NULL;
}

iree_status_t loom_test_validate_xfails(
    loom_test_xfail_list_t* xfails,
    const loom_testbench_module_plan_t* module_plan,
    iree_string_view_t selected_record_name) {
  for (iree_host_size_t xfail_index = 0; xfail_index < xfails->count;
       ++xfail_index) {
    const iree_string_view_t record_name =
        loom_test_xfail_record_name(&xfails->values[xfail_index]);
    iree_host_size_t match_count = 0;
    for (iree_host_size_t i = 0; i < module_plan->case_count; ++i) {
      match_count +=
          iree_string_view_equal(module_plan->cases[i].name, record_name);
    }
    for (iree_host_size_t i = 0; i < module_plan->scenario_count; ++i) {
      match_count +=
          iree_string_view_equal(module_plan->scenarios[i].name, record_name);
    }
    if (match_count != 1) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "%s record '%.*s' matched %zu check records; expected exactly one",
          loom_test_xfail_policy_flag(xfails->values[xfail_index].policy),
          (int)xfails->values[xfail_index].record.size,
          xfails->values[xfail_index].record.data, match_count);
    }
    if (!iree_string_view_is_empty(selected_record_name) &&
        !iree_string_view_equal(record_name, selected_record_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%s record '%.*s' is not selected by --case=@%.*s",
          loom_test_xfail_policy_flag(xfails->values[xfail_index].policy),
          (int)xfails->values[xfail_index].record.size,
          xfails->values[xfail_index].record.data,
          (int)selected_record_name.size, selected_record_name.data);
    }
  }
  return iree_ok_status();
}

static bool loom_test_xfail_accepts_diagnostic(const loom_test_xfail_t* xfail,
                                               loom_error_ref_t ref) {
  for (iree_host_size_t i = 0; i < xfail->diagnostic_count; ++i) {
    if (xfail->diagnostic_refs[i] == ref) {
      return true;
    }
  }
  return false;
}

void loom_test_diagnostic_capture_begin(loom_test_diagnostic_capture_t* capture,
                                        loom_test_xfail_t* active_xfail) {
  capture->active_xfail = active_xfail;
  capture->first_error_ref = LOOM_ERROR_REF_NONE;
  capture->matched_expected_diagnostic = false;
}

static iree_status_t loom_test_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_test_diagnostic_capture_t* capture =
      (loom_test_diagnostic_capture_t*)user_data;
  if (diagnostic->severity == LOOM_DIAGNOSTIC_ERROR) {
    const loom_error_ref_t ref = loom_error_def_ref(diagnostic->error);
    if (!loom_error_ref_is_set(capture->first_error_ref)) {
      capture->first_error_ref = ref;
    }
    if (capture->active_xfail != NULL &&
        loom_test_xfail_accepts_diagnostic(capture->active_xfail, ref)) {
      capture->matched_expected_diagnostic = true;
    }
  }
  return loom_diagnostic_stderr_sink(NULL, diagnostic);
}

loom_diagnostic_sink_t loom_test_diagnostic_capture_sink(
    loom_test_diagnostic_capture_t* capture) {
  return (loom_diagnostic_sink_t){
      .fn = loom_test_capture_diagnostic,
      .user_data = capture,
  };
}

iree_status_t loom_test_diagnostic_capture_loomc_result(
    void* user_data, const loomc_result_t* result) {
  loom_test_diagnostic_capture_t* capture = user_data;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic->severity != LOOMC_DIAGNOSTIC_SEVERITY_ERROR) {
      continue;
    }
    loom_error_ref_t ref = LOOM_ERROR_REF_NONE;
    if (!loom_error_ref_parse(iree_string_view_from_loomc(diagnostic->code),
                              &ref)) {
      continue;
    }
    if (!loom_error_ref_is_set(capture->first_error_ref)) {
      capture->first_error_ref = ref;
    }
    if (capture->active_xfail != NULL &&
        loom_test_xfail_accepts_diagnostic(capture->active_xfail, ref)) {
      capture->matched_expected_diagnostic = true;
    }
  }
  bool succeeded = false;
  return loom_tooling_cli_print_loomc_result(stderr, result, &succeeded);
}

void loom_test_capture_expectation_report(
    loom_test_diagnostic_capture_t* capture,
    const loom_testbench_expectation_report_t* report) {
  if (capture->active_xfail == NULL || report == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < report->failure_count; ++i) {
    const loom_error_ref_t ref = report->failures[i].diagnostic_ref;
    if (!loom_error_ref_is_set(capture->first_error_ref)) {
      capture->first_error_ref = ref;
    }
    if (loom_test_xfail_accepts_diagnostic(capture->active_xfail, ref)) {
      capture->matched_expected_diagnostic = true;
    }
  }
}

void loom_test_finish_xfail(loom_test_xfail_t* xfail,
                            const loom_test_diagnostic_capture_t* capture,
                            loom_test_xfail_outcome_t outcome,
                            iree_host_size_t accepted_failed_sample_count,
                            iree_host_size_t accepted_failed_trial_count) {
  xfail->outcome = outcome;
  xfail->observed_diagnostic_ref = capture->first_error_ref;
  xfail->accepted_failed_sample_count = accepted_failed_sample_count;
  xfail->accepted_failed_trial_count = accepted_failed_trial_count;
}

bool loom_test_xfail_try_accept_compile_failure(
    loom_test_xfail_t* xfail, const loom_test_diagnostic_capture_t* capture,
    iree_status_t* inout_status) {
  if (xfail == NULL || iree_status_is_ok(*inout_status) ||
      iree_status_code(*inout_status) != IREE_STATUS_FAILED_PRECONDITION ||
      !capture->matched_expected_diagnostic) {
    return false;
  }
  iree_status_free(*inout_status);
  *inout_status = iree_ok_status();
  loom_test_finish_xfail(xfail, capture,
                         LOOM_TEST_XFAIL_OUTCOME_EXPECTED_FAILURE, 0, 0);
  return true;
}

loom_test_xfail_counts_t loom_test_count_xfails(
    const loom_test_xfail_list_t* xfails) {
  loom_test_xfail_counts_t counts = {0};
  for (iree_host_size_t i = 0; i < xfails->count; ++i) {
    const loom_test_xfail_t* xfail = &xfails->values[i];
    switch (xfail->outcome) {
      case LOOM_TEST_XFAIL_OUTCOME_EXPECTED_FAILURE:
        ++counts.xfail_count;
        counts.accepted_failed_sample_count +=
            xfail->accepted_failed_sample_count;
        counts.accepted_failed_trial_count +=
            xfail->accepted_failed_trial_count;
        break;
      case LOOM_TEST_XFAIL_OUTCOME_UNEXPECTED_PASS:
        ++counts.xpass_count;
        break;
      case LOOM_TEST_XFAIL_OUTCOME_ALLOWED_PASS:
        break;
      case LOOM_TEST_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH:
      case LOOM_TEST_XFAIL_OUTCOME_PENDING:
        ++counts.mismatch_count;
        break;
    }
  }
  return counts;
}

static iree_string_view_t loom_test_xfail_outcome_name(
    loom_test_xfail_outcome_t outcome) {
  switch (outcome) {
    case LOOM_TEST_XFAIL_OUTCOME_EXPECTED_FAILURE:
      return IREE_SV("xfail");
    case LOOM_TEST_XFAIL_OUTCOME_ALLOWED_PASS:
      return IREE_SV("pass");
    case LOOM_TEST_XFAIL_OUTCOME_UNEXPECTED_PASS:
      return IREE_SV("xpass");
    case LOOM_TEST_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH:
      return IREE_SV("diagnostic_mismatch");
    case LOOM_TEST_XFAIL_OUTCOME_PENDING:
    default:
      return IREE_SV("not_executed");
  }
}

static iree_string_view_t loom_test_xfail_policy_name(
    loom_test_xfail_policy_t policy) {
  return policy == LOOM_TEST_XFAIL_POLICY_STRICT ? IREE_SV("strict")
                                                 : IREE_SV("allow_failure");
}

iree_status_t loom_test_write_xfails_json(const loom_test_xfail_list_t* xfails,
                                          loom_json_object_writer_t* report) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(report, IREE_SV("xfails")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(report->stream, &array));
  for (iree_host_size_t i = 0; i < xfails->count; ++i) {
    const loom_test_xfail_t* xfail = &xfails->values[i];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(array.stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("record"), xfail->record));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("diagnostic"), xfail->diagnostic));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("outcome"),
        loom_test_xfail_outcome_name(xfail->outcome)));
    if (xfail->outcome == LOOM_TEST_XFAIL_OUTCOME_DIAGNOSTIC_MISMATCH &&
        loom_error_ref_is_set(xfail->observed_diagnostic_ref)) {
      char observed_diagnostic[32] = {0};
      const int length = snprintf(
          observed_diagnostic, sizeof(observed_diagnostic), "%s/%03u",
          loom_error_domain_name(
              loom_error_ref_domain(xfail->observed_diagnostic_ref)),
          (unsigned)loom_error_ref_code(xfail->observed_diagnostic_ref));
      if (length < 0 ||
          (iree_host_size_t)length >= sizeof(observed_diagnostic)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "diagnostic identity exceeded report storage");
      }
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &object, IREE_SV("observed_diagnostic"),
          iree_make_string_view(observed_diagnostic, length)));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("policy"),
        loom_test_xfail_policy_name(xfail->policy)));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}
