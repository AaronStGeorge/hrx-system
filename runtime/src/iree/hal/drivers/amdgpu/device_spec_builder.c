// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device_spec_builder.h"

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_device_spec_builder_add_facet(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_device_spec_t* spec) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(spec);

  const iree_host_size_t payload_size =
      iree_hal_amdgpu_device_spec_payload_size();
  uint8_t* payload_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      builder->host_allocator, payload_size, (void**)&payload_storage));

  iree_status_t status = iree_hal_amdgpu_device_spec_encode(
      spec, iree_make_byte_span(payload_storage, payload_size));
  if (iree_status_is_ok(status)) {
    iree_hal_device_spec_facet_t facet = {
        .schema_id =
            iree_make_cstring_view(IREE_HAL_AMDGPU_DEVICE_SPEC_SCHEMA_ID),
        .schema_version = IREE_HAL_AMDGPU_DEVICE_SPEC_SCHEMA_VERSION,
        .payload = iree_make_const_byte_span(payload_storage, payload_size),
    };
    status = iree_hal_device_spec_builder_add_facet(builder, &facet);
  }

  iree_allocator_free(builder->host_allocator, payload_storage);
  return status;
}
