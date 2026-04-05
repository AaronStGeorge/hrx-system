// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class OwnedFunctionType {
 public:
  explicit OwnedFunctionType(loom_type_t type) : type_(type) {}
  OwnedFunctionType(const OwnedFunctionType&) = delete;
  OwnedFunctionType& operator=(const OwnedFunctionType&) = delete;
  OwnedFunctionType(OwnedFunctionType&& other) noexcept : type_(other.type_) {
    other.type_ = loom_type_none();
  }
  OwnedFunctionType& operator=(OwnedFunctionType&& other) noexcept {
    if (this == &other) return *this;
    iree_allocator_free(iree_allocator_system(),
                        (void*)loom_type_func_data(type_));
    type_ = other.type_;
    other.type_ = loom_type_none();
    return *this;
  }
  ~OwnedFunctionType() {
    iree_allocator_free(iree_allocator_system(),
                        (void*)loom_type_func_data(type_));
  }

  loom_type_t get() const { return type_; }

 private:
  loom_type_t type_;
};

static OwnedFunctionType BuildFunctionType(const loom_type_t* arg_types,
                                           uint16_t arg_count,
                                           const loom_type_t* result_types,
                                           uint16_t result_count) {
  loom_type_t type = {0};
  IREE_CHECK_OK(loom_type_function_build(arg_types, arg_count, result_types,
                                         result_count, iree_allocator_system(),
                                         &type));
  return OwnedFunctionType(type);
}

TEST(TypesTest, FunctionTypeEqualAndHashAreStructural) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t f64 = loom_type_scalar(LOOM_SCALAR_TYPE_F64);

  OwnedFunctionType first = BuildFunctionType(&i32, 1, &f32, 1);
  OwnedFunctionType duplicate = BuildFunctionType(&i32, 1, &f32, 1);
  OwnedFunctionType different = BuildFunctionType(&i32, 1, &f64, 1);

  EXPECT_TRUE(loom_type_equal(first.get(), duplicate.get()));
  EXPECT_EQ(loom_type_hash(first.get()), loom_type_hash(duplicate.get()));
  EXPECT_FALSE(loom_type_equal(first.get(), different.get()));
}

TEST(TypesTest, DialectTypeEqualAndHashAreStructural) {
  loom_type_t params_a[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t params_b[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t params_c[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I64,
                          loom_dim_pack_static(4), 0),
  };

  loom_type_t first = loom_type_dialect((loom_string_id_t)70000u,
                                        IREE_ARRAYSIZE(params_a), params_a);
  loom_type_t duplicate = loom_type_dialect((loom_string_id_t)70000u,
                                            IREE_ARRAYSIZE(params_b), params_b);
  loom_type_t different_name = loom_type_dialect(
      (loom_string_id_t)70001u, IREE_ARRAYSIZE(params_b), params_b);
  loom_type_t different_params = loom_type_dialect(
      (loom_string_id_t)70000u, IREE_ARRAYSIZE(params_c), params_c);

  EXPECT_EQ(loom_type_dialect_name_id(first), 70000u);
  EXPECT_TRUE(loom_type_equal(first, duplicate));
  EXPECT_EQ(loom_type_hash(first), loom_type_hash(duplicate));
  EXPECT_FALSE(loom_type_equal(first, different_name));
  EXPECT_FALSE(loom_type_equal(first, different_params));
}

}  // namespace
}  // namespace loom
