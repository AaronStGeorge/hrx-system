// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/reference.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/modules/hal/types.h"

typedef struct loom_testbench_reference_rank2_view_t {
  // Borrowed HAL buffer view.
  iree_hal_buffer_view_t* view;
  // Element type stored by |view|.
  iree_hal_element_type_t element_type;
  // First row-major dimension.
  iree_host_size_t rows;
  // Second row-major dimension.
  iree_host_size_t columns;
  // Dense byte count for one element.
  iree_host_size_t element_size;
} loom_testbench_reference_rank2_view_t;

typedef struct loom_testbench_reference_rank4_view_t {
  // Borrowed HAL buffer view.
  iree_hal_buffer_view_t* view;
  // Element type stored by |view|.
  iree_hal_element_type_t element_type;
  // First tile-grid dimension.
  iree_host_size_t outer_rows;
  // Second tile-grid dimension.
  iree_host_size_t outer_columns;
  // First per-tile matrix dimension.
  iree_host_size_t inner_rows;
  // Second per-tile matrix dimension.
  iree_host_size_t inner_columns;
  // Dense byte count for one element.
  iree_host_size_t element_size;
} loom_testbench_reference_rank4_view_t;

typedef struct loom_testbench_reference_matmul_result_t {
  // Borrowed f32 row-major result values.
  const float* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_testbench_reference_matmul_result_t;

static iree_status_t loom_testbench_reference_variant_buffer_view(
    const iree_vm_variant_t* variant, iree_hal_buffer_view_t** out_view) {
  *out_view = NULL;
  if (!iree_vm_variant_is_ref(*variant) ||
      !iree_hal_buffer_view_isa(variant->ref)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference oracle expects buffer view inputs");
  }
  *out_view = iree_hal_buffer_view_deref(variant->ref);
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_dim_to_host_size(
    iree_hal_dim_t dim, iree_host_size_t* out_value) {
  if (dim > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer view dimension %" PRIu64
                            " exceeds host size range",
                            (uint64_t)dim);
  }
  *out_value = (iree_host_size_t)dim;
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_rank2_view_initialize(
    iree_hal_buffer_view_t* view,
    loom_testbench_reference_rank2_view_t* out_view) {
  if (iree_hal_buffer_view_shape_rank(view) != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference.matmul expects rank-2 buffer views");
  }
  *out_view = (loom_testbench_reference_rank2_view_t){
      .view = view,
      .element_type = iree_hal_buffer_view_element_type(view),
      .element_size = iree_hal_element_dense_byte_count(
          iree_hal_buffer_view_element_type(view)),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 0), &out_view->rows));
  return loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 1), &out_view->columns);
}

static iree_status_t loom_testbench_reference_rank4_view_initialize(
    iree_hal_buffer_view_t* view,
    loom_testbench_reference_rank4_view_t* out_view) {
  if (iree_hal_buffer_view_shape_rank(view) != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference.tiled_matmul expects rank-4 buffer views");
  }
  *out_view = (loom_testbench_reference_rank4_view_t){
      .view = view,
      .element_type = iree_hal_buffer_view_element_type(view),
      .element_size = iree_hal_element_dense_byte_count(
          iree_hal_buffer_view_element_type(view)),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 0), &out_view->outer_rows));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 1), &out_view->outer_columns));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 2), &out_view->inner_rows));
  return loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 3), &out_view->inner_columns);
}

static double loom_testbench_reference_load_floating_element_at(
    iree_hal_element_type_t element_type, iree_host_size_t element_size,
    const uint8_t* data, iree_host_size_t element_offset) {
  const uint8_t* element_data = data + element_offset * element_size;
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16: {
      uint16_t bits = 0;
      memcpy(&bits, element_data, sizeof(bits));
      return (double)iree_math_f16_to_f32(bits);
    }
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16: {
      uint16_t bits = 0;
      memcpy(&bits, element_data, sizeof(bits));
      return (double)iree_math_bf16_to_f32(bits);
    }
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32: {
      float value = 0.0f;
      memcpy(&value, element_data, sizeof(value));
      return (double)value;
    }
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64: {
      double value = 0.0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    default:
      return 0.0;
  }
}

static double loom_testbench_reference_load_rank2_element(
    const loom_testbench_reference_rank2_view_t* view, const uint8_t* data,
    iree_host_size_t row, iree_host_size_t column) {
  return loom_testbench_reference_load_floating_element_at(
      view->element_type, view->element_size, data,
      row * view->columns + column);
}

static double loom_testbench_reference_load_rank4_element(
    const loom_testbench_reference_rank4_view_t* view, const uint8_t* data,
    iree_host_size_t outer_row, iree_host_size_t outer_column,
    iree_host_size_t inner_row, iree_host_size_t inner_column) {
  const iree_host_size_t element_offset =
      (((outer_row * view->outer_columns + outer_column) * view->inner_rows +
        inner_row) *
           view->inner_columns +
       inner_column);
  return loom_testbench_reference_load_floating_element_at(
      view->element_type, view->element_size, data, element_offset);
}

static bool loom_testbench_reference_element_type_is_floating(
    iree_hal_element_type_t element_type) {
  return element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 ||
         element_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 ||
         element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
         element_type == IREE_HAL_ELEMENT_TYPE_FLOAT_64;
}

static iree_status_t loom_testbench_reference_validate_matmul_views(
    const loom_testbench_reference_rank2_view_t* lhs,
    const loom_testbench_reference_rank2_view_t* rhs,
    const loom_testbench_reference_rank2_view_t* init) {
  if (!loom_testbench_reference_element_type_is_floating(lhs->element_type) ||
      !loom_testbench_reference_element_type_is_floating(rhs->element_type) ||
      !loom_testbench_reference_element_type_is_floating(init->element_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference.matmul expects floating-point buffer elements");
  }
  if (lhs->columns != rhs->rows || lhs->rows != init->rows ||
      rhs->columns != init->columns) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference.matmul shape mismatch: lhs[%" PRIhsz
                            ", %" PRIhsz "], rhs[%" PRIhsz ", %" PRIhsz
                            "], init[%" PRIhsz ", %" PRIhsz "]",
                            lhs->rows, lhs->columns, rhs->rows, rhs->columns,
                            init->rows, init->columns);
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_validate_tiled_matmul_views(
    const loom_testbench_reference_rank4_view_t* lhs,
    const loom_testbench_reference_rank4_view_t* rhs,
    const loom_testbench_reference_rank4_view_t* init) {
  if (!loom_testbench_reference_element_type_is_floating(lhs->element_type) ||
      !loom_testbench_reference_element_type_is_floating(rhs->element_type) ||
      !loom_testbench_reference_element_type_is_floating(init->element_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference.tiled_matmul expects floating-point buffer elements");
  }
  if (lhs->outer_columns != rhs->outer_rows ||
      lhs->outer_rows != init->outer_rows ||
      rhs->outer_columns != init->outer_columns ||
      lhs->inner_columns != rhs->inner_rows ||
      lhs->inner_rows != init->inner_rows ||
      rhs->inner_columns != init->inner_columns) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference.tiled_matmul shape mismatch: lhs[%" PRIhsz ", %" PRIhsz
        ", %" PRIhsz ", %" PRIhsz "], rhs[%" PRIhsz ", %" PRIhsz ", %" PRIhsz
        ", %" PRIhsz "], init[%" PRIhsz ", %" PRIhsz ", %" PRIhsz ", %" PRIhsz
        "]",
        lhs->outer_rows, lhs->outer_columns, lhs->inner_rows,
        lhs->inner_columns, rhs->outer_rows, rhs->outer_columns,
        rhs->inner_rows, rhs->inner_columns, init->outer_rows,
        init->outer_columns, init->inner_rows, init->inner_columns);
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_map_view_read(
    iree_hal_buffer_view_t* view, iree_hal_buffer_mapping_t* out_mapping) {
  return iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(view), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, out_mapping);
}

static iree_status_t loom_testbench_reference_compute_matmul(
    const loom_testbench_reference_rank2_view_t* lhs,
    const loom_testbench_reference_rank2_view_t* rhs,
    const loom_testbench_reference_rank2_view_t* init, float* result) {
  iree_hal_buffer_mapping_t lhs_mapping = {0};
  iree_hal_buffer_mapping_t rhs_mapping = {0};
  iree_hal_buffer_mapping_t init_mapping = {0};
  bool lhs_mapped = false;
  bool rhs_mapped = false;
  bool init_mapped = false;
  iree_status_t status =
      loom_testbench_reference_map_view_read(lhs->view, &lhs_mapping);
  if (iree_status_is_ok(status)) {
    lhs_mapped = true;
    status = loom_testbench_reference_map_view_read(rhs->view, &rhs_mapping);
  }
  if (iree_status_is_ok(status)) {
    rhs_mapped = true;
    status = loom_testbench_reference_map_view_read(init->view, &init_mapping);
  }
  if (iree_status_is_ok(status)) {
    init_mapped = true;
    for (iree_host_size_t row = 0; row < lhs->rows; ++row) {
      for (iree_host_size_t column = 0; column < rhs->columns; ++column) {
        double accumulator = loom_testbench_reference_load_rank2_element(
            init, init_mapping.contents.data, row, column);
        for (iree_host_size_t k = 0; k < lhs->columns; ++k) {
          const double lhs_value = loom_testbench_reference_load_rank2_element(
              lhs, lhs_mapping.contents.data, row, k);
          const double rhs_value = loom_testbench_reference_load_rank2_element(
              rhs, rhs_mapping.contents.data, k, column);
          accumulator += lhs_value * rhs_value;
        }
        result[row * rhs->columns + column] = (float)accumulator;
      }
    }
  }
  if (init_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&init_mapping));
  }
  if (rhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&rhs_mapping));
  }
  if (lhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&lhs_mapping));
  }
  return status;
}

static iree_status_t loom_testbench_reference_compute_tiled_matmul(
    const loom_testbench_reference_rank4_view_t* lhs,
    const loom_testbench_reference_rank4_view_t* rhs,
    const loom_testbench_reference_rank4_view_t* init, float* result) {
  iree_hal_buffer_mapping_t lhs_mapping = {0};
  iree_hal_buffer_mapping_t rhs_mapping = {0};
  iree_hal_buffer_mapping_t init_mapping = {0};
  bool lhs_mapped = false;
  bool rhs_mapped = false;
  bool init_mapped = false;
  iree_status_t status =
      loom_testbench_reference_map_view_read(lhs->view, &lhs_mapping);
  if (iree_status_is_ok(status)) {
    lhs_mapped = true;
    status = loom_testbench_reference_map_view_read(rhs->view, &rhs_mapping);
  }
  if (iree_status_is_ok(status)) {
    rhs_mapped = true;
    status = loom_testbench_reference_map_view_read(init->view, &init_mapping);
  }
  if (iree_status_is_ok(status)) {
    init_mapped = true;
    for (iree_host_size_t outer_row = 0; outer_row < lhs->outer_rows;
         ++outer_row) {
      for (iree_host_size_t outer_column = 0; outer_column < rhs->outer_columns;
           ++outer_column) {
        for (iree_host_size_t inner_row = 0; inner_row < lhs->inner_rows;
             ++inner_row) {
          for (iree_host_size_t inner_column = 0;
               inner_column < rhs->inner_columns; ++inner_column) {
            double accumulator = loom_testbench_reference_load_rank4_element(
                init, init_mapping.contents.data, outer_row, outer_column,
                inner_row, inner_column);
            for (iree_host_size_t outer_k = 0; outer_k < lhs->outer_columns;
                 ++outer_k) {
              for (iree_host_size_t inner_k = 0; inner_k < lhs->inner_columns;
                   ++inner_k) {
                const double lhs_value =
                    loom_testbench_reference_load_rank4_element(
                        lhs, lhs_mapping.contents.data, outer_row, outer_k,
                        inner_row, inner_k);
                const double rhs_value =
                    loom_testbench_reference_load_rank4_element(
                        rhs, rhs_mapping.contents.data, outer_k, outer_column,
                        inner_k, inner_column);
                accumulator += lhs_value * rhs_value;
              }
            }
            const iree_host_size_t result_offset =
                (((outer_row * rhs->outer_columns + outer_column) *
                      lhs->inner_rows +
                  inner_row) *
                     rhs->inner_columns +
                 inner_column);
            result[result_offset] = (float)accumulator;
          }
        }
      }
    }
  }
  if (init_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&init_mapping));
  }
  if (rhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&rhs_mapping));
  }
  if (lhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&lhs_mapping));
  }
  return status;
}

static iree_status_t loom_testbench_reference_fill_matmul_result(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  const loom_testbench_reference_matmul_result_t* result =
      (const loom_testbench_reference_matmul_result_t*)user_data;
  iree_host_size_t byte_count = 0;
  if (!iree_host_size_checked_mul(result->count, sizeof(result->values[0]),
                                  &byte_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reference.matmul result byte count overflow");
  }
  if (mapping->contents.data_length < byte_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reference.matmul result mapping is too small");
  }
  memcpy(mapping->contents.data, result->values, byte_count);
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_allocate_matmul_result(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    iree_host_size_t rows, iree_host_size_t columns, const float* values,
    iree_host_size_t value_count, iree_hal_buffer_view_t** out_buffer_view) {
  const iree_hal_dim_t shape[2] = {
      (iree_hal_dim_t)rows,
      (iree_hal_dim_t)columns,
  };
  loom_testbench_reference_matmul_result_t result = {
      .values = values,
      .count = value_count,
  };
  return iree_hal_buffer_view_generate_buffer(
      options->device, options->device_allocator, IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      options->result_buffer_params,
      loom_testbench_reference_fill_matmul_result, &result, out_buffer_view);
}

static iree_status_t loom_testbench_reference_allocate_tiled_matmul_result(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    iree_host_size_t outer_rows, iree_host_size_t outer_columns,
    iree_host_size_t inner_rows, iree_host_size_t inner_columns,
    const float* values, iree_host_size_t value_count,
    iree_hal_buffer_view_t** out_buffer_view) {
  const iree_hal_dim_t shape[4] = {
      (iree_hal_dim_t)outer_rows,
      (iree_hal_dim_t)outer_columns,
      (iree_hal_dim_t)inner_rows,
      (iree_hal_dim_t)inner_columns,
  };
  loom_testbench_reference_matmul_result_t result = {
      .values = values,
      .count = value_count,
  };
  return iree_hal_buffer_view_generate_buffer(
      options->device, options->device_allocator, IREE_ARRAYSIZE(shape), shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      options->result_buffer_params,
      loom_testbench_reference_fill_matmul_result, &result, out_buffer_view);
}

static iree_status_t loom_testbench_reference_matmul_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const iree_vm_variant_t* inputs,
    iree_host_size_t result_count, iree_vm_variant_t* out_results) {
  (void)invocation;
  const loom_testbench_reference_matmul_oracle_options_t* options =
      (const loom_testbench_reference_matmul_oracle_options_t*)user_data;
  if (input_count != 3 || result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference.matmul expects 3 inputs and 1 result");
  }
  if (options->device_allocator == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "reference.matmul requires a configured HAL allocator");
  }

  iree_hal_buffer_view_t* lhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[0], &lhs_view));
  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[1], &rhs_view));
  iree_hal_buffer_view_t* init_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[2], &init_view));

  loom_testbench_reference_rank2_view_t lhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank2_view_initialize(lhs_view, &lhs));
  loom_testbench_reference_rank2_view_t rhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank2_view_initialize(rhs_view, &rhs));
  loom_testbench_reference_rank2_view_t init = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank2_view_initialize(init_view, &init));
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_validate_matmul_views(&lhs, &rhs, &init));

  iree_host_size_t result_count_checked = 0;
  if (!iree_host_size_checked_mul(lhs.rows, rhs.columns,
                                  &result_count_checked)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reference.matmul result element count overflow");
  }

  iree_allocator_t host_allocator = options->host_allocator;
  if (iree_allocator_is_null(host_allocator)) {
    host_allocator = iree_allocator_system();
  }
  float* result_values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, result_count_checked, sizeof(*result_values),
      (void**)&result_values));
  iree_status_t status =
      loom_testbench_reference_compute_matmul(&lhs, &rhs, &init, result_values);
  iree_hal_buffer_view_t* result_view = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_testbench_reference_allocate_matmul_result(
        options, lhs.rows, rhs.columns, result_values, result_count_checked,
        &result_view);
  }
  iree_allocator_free(host_allocator, result_values);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_vm_ref_t result_ref = iree_hal_buffer_view_move_ref(result_view);
  out_results[0] = iree_vm_make_variant_ref_assign(result_ref);
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_tiled_matmul_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const iree_vm_variant_t* inputs,
    iree_host_size_t result_count, iree_vm_variant_t* out_results) {
  (void)invocation;
  const loom_testbench_reference_matmul_oracle_options_t* options =
      (const loom_testbench_reference_matmul_oracle_options_t*)user_data;
  if (input_count != 3 || result_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference.tiled_matmul expects 3 inputs and 1 result");
  }
  if (options->device_allocator == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "reference.tiled_matmul requires a configured HAL allocator");
  }

  iree_hal_buffer_view_t* lhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[0], &lhs_view));
  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[1], &rhs_view));
  iree_hal_buffer_view_t* init_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[2], &init_view));

  loom_testbench_reference_rank4_view_t lhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank4_view_initialize(lhs_view, &lhs));
  loom_testbench_reference_rank4_view_t rhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank4_view_initialize(rhs_view, &rhs));
  loom_testbench_reference_rank4_view_t init = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank4_view_initialize(init_view, &init));
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_validate_tiled_matmul_views(&lhs, &rhs, &init));

  iree_host_size_t outer_count = 0;
  if (!iree_host_size_checked_mul(lhs.outer_rows, rhs.outer_columns,
                                  &outer_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference.tiled_matmul result outer element count overflow");
  }
  iree_host_size_t inner_count = 0;
  if (!iree_host_size_checked_mul(lhs.inner_rows, rhs.inner_columns,
                                  &inner_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference.tiled_matmul result inner element count overflow");
  }
  iree_host_size_t result_count_checked = 0;
  if (!iree_host_size_checked_mul(outer_count, inner_count,
                                  &result_count_checked)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference.tiled_matmul result element count overflow");
  }

  iree_allocator_t host_allocator = options->host_allocator;
  if (iree_allocator_is_null(host_allocator)) {
    host_allocator = iree_allocator_system();
  }
  float* result_values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, result_count_checked, sizeof(*result_values),
      (void**)&result_values));
  iree_status_t status = loom_testbench_reference_compute_tiled_matmul(
      &lhs, &rhs, &init, result_values);
  iree_hal_buffer_view_t* result_view = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_testbench_reference_allocate_tiled_matmul_result(
        options, lhs.outer_rows, rhs.outer_columns, lhs.inner_rows,
        rhs.inner_columns, result_values, result_count_checked, &result_view);
  }
  iree_allocator_free(host_allocator, result_values);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_vm_ref_t result_ref = iree_hal_buffer_view_move_ref(result_view);
  out_results[0] = iree_vm_make_variant_ref_assign(result_ref);
  return iree_ok_status();
}

void loom_testbench_reference_matmul_oracle_provider_initialize(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    loom_testbench_oracle_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_oracle_provider_t){
      .name = IREE_SV("reference.matmul"),
      .invoke =
          {
              .fn = loom_testbench_reference_matmul_invoke,
              .user_data = (void*)options,
          },
  };
}

void loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    loom_testbench_oracle_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_oracle_provider_t){
      .name = IREE_SV("reference.tiled_matmul"),
      .invoke =
          {
              .fn = loom_testbench_reference_tiled_matmul_invoke,
              .user_data = (void*)options,
          },
  };
}
