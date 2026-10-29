// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/signature.h"

#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

size_t component_count(Types& types, std::span<const cxx::Type* const> sources,
                       cxx::AST* owner) {
  size_t result = 0;
  for (const auto* source : sources) {
    result += types.partition(source, owner).component_count;
  }
  return result;
}

bool signature_requires_binding(Types& types,
                                std::span<const cxx::Type* const> sources,
                                cxx::AST* owner) {
  for (const auto* source : sources) {
    if (types.requires_binding(source, owner)) {
      return true;
    }
  }
  return false;
}

void append_bound(Types& types, std::span<const cxx::Type* const> sources,
                  cxx::AST* owner, std::span<const loom_value_id_t> identities,
                  BoundTypeStorage& type_storage,
                  std::vector<loom_type_t>& output_types) {
  size_t component_offset = 0;
  for (const auto* source : sources) {
    auto count = types.partition(source, owner).component_count;
    types.append_bound(source, owner,
                       identities.subspan(component_offset, count),
                       type_storage, output_types);
    component_offset += count;
  }
}

}  // namespace

BoundSignature bind_signature(Types& types,
                              std::span<const cxx::Type* const> sources,
                              cxx::AST* owner, loom_builder_t* builder,
                              SignatureIdentityRequirement identities) {
  BoundSignature result;
  bool requires_binding = identities == SignatureIdentityRequirement::Required;
  requires_binding |= signature_requires_binding(types, sources, owner);
  const size_t count = component_count(types, sources, owner);
  result.types.reserve(count);
  if (!requires_binding) {
    for (const auto* source : sources) {
      types.append(source, owner, result.types);
    }
    return result;
  }

  result.identities.resize(count);
  check(loom_builder_reserve_values(builder, result.identities.size(),
                                    result.identities.data()));
  append_bound(types, sources, owner, result.identities, result.type_storage,
               result.types);
  return result;
}

BoundSignature bind_signature_to(Types& types,
                                 std::span<const cxx::Type* const> sources,
                                 cxx::AST* owner,
                                 std::span<const loom_value_id_t> identities) {
  BoundSignature result;
  const size_t count = component_count(types, sources, owner);
  IREE_ASSERT_EQ(identities.size(), count);
  result.types.reserve(count);
  result.identities.assign(identities.begin(), identities.end());
  append_bound(types, sources, owner, result.identities, result.type_storage,
               result.types);
  return result;
}

BoundConditionLoopSignatures bind_condition_loop_signatures(
    Types& types, std::span<const cxx::Type* const> sources, cxx::AST* owner,
    loom_builder_t* builder, std::span<const loom_type_t> header_suffix_types,
    bool require_result_types) {
  BoundConditionLoopSignatures result;
  if (!signature_requires_binding(types, sources, owner)) {
    if (require_result_types) {
      const size_t count = component_count(types, sources, owner);
      result.result_types.reserve(count);
      for (const auto* source : sources) {
        types.append(source, owner, result.result_types);
      }
    }
    return result;
  }

  const size_t source_component_count = component_count(types, sources, owner);
  const size_t header_component_count =
      source_component_count + header_suffix_types.size();
  result.identities.resize(header_component_count + source_component_count);
  check(loom_builder_reserve_values(builder, result.identities.size(),
                                    result.identities.data()));
  result.header_types.reserve(header_component_count);
  append_bound(types, sources, owner,
               std::span<const loom_value_id_t>(result.identities)
                   .first(source_component_count),
               result.header_type_storage, result.header_types);
  result.header_types.insert(result.header_types.end(),
                             header_suffix_types.begin(),
                             header_suffix_types.end());
  result.result_types.reserve(source_component_count);
  append_bound(types, sources, owner,
               std::span<const loom_value_id_t>(result.identities)
                   .subspan(header_component_count, source_component_count),
               result.result_type_storage, result.result_types);
  return result;
}

}  // namespace loom::cxx_import
