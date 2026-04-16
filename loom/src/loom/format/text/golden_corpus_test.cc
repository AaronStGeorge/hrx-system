// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C parser/printer identity tests over the checked-in .loom text corpus.
//
// The Python parser/printer also round-trips this corpus. Keeping the same
// corpus active in C catches drift in comments, locations, dialect
// registration, type syntax, and format-element handling before cross-language
// tests have to diagnose it indirectly.

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/pool/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/test/corpus/text/golden_text_corpus.h"

namespace loom {
namespace {

using DialectVtables = const loom_op_vtable_t* const*;
using DialectVtablesFn = DialectVtables (*)(iree_host_size_t* out_count);

struct DialectRegistration {
  loom_dialect_id_t dialect_id;
  DialectVtablesFn vtables_fn;
};

static const DialectRegistration kCorpusDialects[] = {
    {LOOM_DIALECT_TEST, loom_test_dialect_vtables},
    {LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables},
    {LOOM_DIALECT_FUNC, loom_func_dialect_vtables},
    {LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables},
    {LOOM_DIALECT_POOL, loom_pool_dialect_vtables},
    {LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables},
    {LOOM_DIALECT_SCF, loom_scf_dialect_vtables},
    {LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables},
    {LOOM_DIALECT_VIEW, loom_view_dialect_vtables},
    {LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables},
    {LOOM_DIALECT_INDEX, loom_index_dialect_vtables},
    {LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables},
    {LOOM_DIALECT_LLVMIR, loom_llvmir_dialect_vtables},
};

class GoldenCorpusTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    for (const DialectRegistration& registration : kCorpusDialects) {
      RegisterDialect(registration);
    }
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(const DialectRegistration& registration) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = registration.vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, registration.dialect_id, vtables, (uint16_t)count));
  }

  loom_module_t* Parse(iree_string_view_t source, iree_string_view_t filename) {
    loom_text_parse_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, NULL},
        .max_errors = 20,
    };
    loom_module_t* module = NULL;
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &options, &module));
    EXPECT_NE(module, nullptr)
        << "parse failed for " << std::string(filename.data, filename.size);
    return module;
  }

  std::string Print(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string printed(iree_string_builder_buffer(&builder),
                        iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return printed;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(GoldenCorpusTest, CorpusIsNotEmpty) {
  EXPECT_GT(loom_test_corpus_text_size(), 0u);
}

TEST_F(GoldenCorpusTest, TextRoundTripsIdentically) {
  const iree_file_toc_t* corpus = loom_test_corpus_text_create();
  for (size_t i = 0; i < loom_test_corpus_text_size(); ++i) {
    const iree_file_toc_t& file = corpus[i];
    SCOPED_TRACE(file.name);
    iree_string_view_t filename = iree_make_cstring_view(file.name);
    iree_string_view_t source =
        iree_make_string_view(file.data, (iree_host_size_t)file.size);

    loom_module_t* module = Parse(source, filename);
    if (!module) continue;
    std::string printed = Print(module);
    loom_module_free(module);

    EXPECT_EQ(std::string(file.data, file.size), printed);

    loom_module_t* reparsed =
        Parse(iree_make_string_view(printed.data(), printed.size()), filename);
    if (!reparsed) continue;
    std::string reprinted = Print(reparsed);
    loom_module_free(reparsed);

    EXPECT_EQ(printed, reprinted);
  }
}

}  // namespace
}  // namespace loom
