// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/initialization.h"

#include <cxx/ast.h>
#include <cxx/ast_rewriter.h>
#include <cxx/function_body.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <cstring>
#include <utility>

#include "loom/import/cxx/source/error.h"
#include "loom/ops/scalar/ops.h"

namespace loom::cxx_import {
namespace {

// Restores the enclosing source current-object binding on every normal or
// diagnostic exit from one recursive initialization.
class ReceiverScope {
 public:
  ReceiverScope(Initialization::Evaluation& evaluation,
                const RecordPartition& record)
      : evaluation_(evaluation) {
    evaluation_.enter_value_receiver(record);
  }
  ReceiverScope(Initialization::Evaluation& evaluation,
                const RecordPartition& record, StorageProjection destination,
                const cxx::Type* object_type)
      : evaluation_(evaluation) {
    evaluation_.enter_storage_receiver(record, destination, object_type);
  }
  ReceiverScope(const ReceiverScope&) = delete;
  ReceiverScope& operator=(const ReceiverScope&) = delete;
  ~ReceiverScope() { evaluation_.leave_receiver(); }

 private:
  Initialization::Evaluation& evaluation_;
};

// Walks the frontend-owned declaration-order plan once. The importer supplies
// only the destination policy; constructor synthesis, omitted members, and
// default-member clauses remain source semantic facts.
template <typename Callback>
void for_each_default_initializer(cxx::TranslationUnit& unit,
                                  Diagnostics& diagnostics,
                                  cxx::FunctionSymbol* constructor,
                                  cxx::AST* owner, Callback&& callback) {
  cxx::ASTRewriter::requireFunctionDefinition(&unit, constructor);
  auto* definition = constructor->resolvedDefinition()->declaration();
  auto* body = definition ? definition->functionBody : nullptr;
  if (!body) {
    diagnostics.reject(unit, owner,
                       "default record constructor has no source "
                       "initialization plan");
  }
  for (auto* initializer : cxx::ListView{cxx::ctor_initializers(body)}) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(initializer->symbol);
    if (!field) {
      diagnostics.reject(unit, initializer,
                         "record value constructors cannot initialize bases "
                         "or delegate");
    }
    auto* source = cxx::memInitializerClause(unit.arena(), initializer);
    callback(field, source, initializer);
  }
}

}  // namespace

loom_string_id_t Initialization::zero_name(loom_scalar_type_t scalar_type) {
  IREE_ASSERT(loom_scalar_type_is_valid(scalar_type));
  auto& name_id = zero_name_ids_[scalar_type];
  if (name_id != LOOM_STRING_ID_INVALID) {
    return name_id;
  }
  static constexpr char kPrefix[] = "zero$";
  char storage[32];
  const char* scalar_name = loom_scalar_type_name(scalar_type);
  const iree_host_size_t prefix_length = sizeof(kPrefix) - 1;
  const iree_host_size_t scalar_length = std::strlen(scalar_name);
  IREE_ASSERT(prefix_length + scalar_length <= sizeof(storage));
  std::memcpy(storage, kPrefix, prefix_length);
  std::memcpy(storage + prefix_length, scalar_name, scalar_length);
  check(loom_builder_intern_string(
      &builder_, iree_make_string_view(storage, prefix_length + scalar_length),
      &name_id));
  return name_id;
}

Value Initialization::value(const cxx::Type* type,
                            cxx::ExpressionAST* initializer,
                            cxx::FunctionSymbol* constructor, cxx::AST* owner,
                            std::optional<Value> retained) {
  type = types_.unqualified(type);
  cxx::List<cxx::ExpressionAST*>* elements = nullptr;
  cxx::VariableSymbol* implicit_element = nullptr;
  bool value_initialized = false;
  if (auto* construction =
          cxx::ast_cast<cxx::TypeConstructionAST>(initializer)) {
    elements = construction->expressionList;
    constructor = construction->constructorSymbol;
    value_initialized = construction->lparenLoc && !elements;
  } else if (auto* construction =
                 cxx::ast_cast<cxx::BracedTypeConstructionAST>(initializer)) {
    auto* list = construction->bracedInitList;
    elements = list ? list->expressionList : nullptr;
    implicit_element = list ? list->implicitElement : nullptr;
    constructor = construction->constructorSymbol;
    value_initialized = list && !elements;
  } else if (auto* list = cxx::ast_cast<cxx::BracedInitListAST>(initializer)) {
    elements = list->expressionList;
    implicit_element = list->implicitElement;
  } else if (auto* list =
                 cxx::ast_cast<cxx::ParenInitializerAST>(initializer)) {
    elements = list->expressionList;
    value_initialized = list->lparenLoc && !elements;
  } else if (initializer) {
    return evaluation_.convert(initializer, type, owner);
  }

  if (constructor && !types_.array(type, owner)) {
    types_.admit_construction(constructor, type, owner);
  }
  if (auto* record = types_.record(type, owner);
      record && constructor &&
      constructor == record->source->defaultConstructor()) {
    return default_record(
        *record, constructor,
        value_initialized &&
            unit_.typeTraits().requires_zero_initialization(type, constructor),
        owner, retained);
  }
  return from_elements(type, elements, implicit_element, constructor, owner,
                       retained);
}

Value Initialization::zero(const cxx::Type* type, cxx::AST* owner) {
  type = types_.unqualified(type);
  if (types_.native_vector(type)) {
    return vectors_.construct({}, type, owner);
  }
  if (auto* array = types_.array(type, owner)) {
    std::vector<loom_value_id_t> components;
    components.reserve(array->component_count);
    for (size_t index = 0; index < array->source->size(); ++index) {
      zero(array->source->elementType(), owner).append_to(components);
    }
    return arena_.capture(*array, components);
  }
  if (auto* record = types_.record(type, owner)) {
    std::vector<loom_value_id_t> components;
    components.reserve(record->component_count);
    for (const auto& member : record->members) {
      zero(member.field->type(), owner).append_to(components);
    }
    return arena_.capture(*record, components);
  }
  const auto& partition = types_.partition(type, owner);
  auto output = types_.get(type, owner);
  if (partition.kind != ValueKind::SSA ||
      loom_type_kind(output) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(unit_, owner,
                        "zero initialization requires an admitted scalar, "
                        "array or record value");
  }
  loom_op_t* op;
  check(loom_scalar_constant_build(
      &builder_, types_.is_float(type) ? loom_attr_f64(0.0) : loom_attr_i64(0),
      output, locations_.get(owner), &op));
  loom_value_id_t result = loom_op_results(op)[0];
  check(loom_module_set_value_name(builder_.module, result,
                                   zero_name(loom_type_element_type(output))));
  return result;
}

Value Initialization::finish_value_receiver(const RecordPartition& record,
                                            cxx::AST* owner) {
  std::vector<loom_value_id_t> components;
  components.reserve(record.component_count);
  for (const auto& member : record.members) {
    evaluation_.receiver_member(member.field, owner).append_to(components);
  }
  return arena_.capture(record, components);
}

Value Initialization::default_record(const RecordPartition& record,
                                     cxx::FunctionSymbol* constructor,
                                     bool zero_initialize, cxx::AST* owner,
                                     std::optional<Value> retained) {
  ReceiverScope receiver(evaluation_, record);
  if (retained && !zero_initialize) {
    for (const auto& member : record.members) {
      evaluation_.publish_receiver_member(
          member.field,
          retained->project(*member.partition, member.component_offset));
    }
  } else if (zero_initialize) {
    for (const auto& member : record.members) {
      evaluation_.publish_receiver_member(member.field,
                                          zero(member.field->type(), owner));
    }
  }

  for_each_default_initializer(
      unit_, diagnostics_, constructor, owner,
      [&](cxx::FieldSymbol* field, cxx::ExpressionAST* source,
          cxx::MemInitializerAST* initializer) {
        std::optional<Value> prior;
        if (retained || zero_initialize) {
          prior = evaluation_.receiver_member(field, initializer);
        }
        evaluation_.publish_receiver_member(
            field, value(field->type(), source, initializer->constructor,
                         initializer, prior));
      });
  return finish_value_receiver(record, owner);
}

Value Initialization::from_elements(const cxx::Type* type,
                                    cxx::List<cxx::ExpressionAST*>* elements,
                                    cxx::VariableSymbol* implicit_element,
                                    cxx::FunctionSymbol* constructor,
                                    cxx::AST* owner,
                                    std::optional<Value> retained) {
  type = types_.unqualified(type);
  if (auto* array = types_.array(type, owner)) {
    std::vector<loom_value_id_t> components;
    components.reserve(array->component_count);
    for (size_t index = 0; index < array->source->size(); ++index) {
      std::optional<Value> prior;
      if (retained) {
        prior = retained->project(*array->element,
                                  index * array->element->component_count);
      }
      Value initialized;
      if (elements) {
        initialized = evaluation_.convert(elements->value,
                                          array->source->elementType(), owner);
        elements = elements->next;
      } else if (implicit_element) {
        initialized =
            value(array->source->elementType(), implicit_element->initializer(),
                  implicit_element->constructor(), owner, prior);
      } else if (constructor) {
        initialized = value(array->source->elementType(), nullptr, constructor,
                            owner, prior);
      } else if (prior) {
        initialized = *prior;
      } else {
        initialized = zero(array->source->elementType(), owner);
      }
      initialized.append_to(components);
    }
    return arena_.capture(*array, components);
  }
  if (auto* record = types_.record(type, owner)) {
    // Single same-type arguments are copy-list/direct initialization, not
    // the first aggregate field. The normalized AST retains that distinction.
    if (elements && !elements->next &&
        types_.unqualified(elements->value->type) == types_.unqualified(type)) {
      return evaluation_.expression(elements->value);
    }
    ReceiverScope receiver(evaluation_, *record);
    for (const auto& member : record->members) {
      auto initialized = elements
                             ? evaluation_.convert(elements->value,
                                                   member.field->type(), owner)
                             : zero(member.field->type(), owner);
      evaluation_.publish_receiver_member(member.field, initialized);
      if (elements) {
        elements = elements->next;
      }
    }
    return finish_value_receiver(*record, owner);
  }
  const auto& partition = types_.partition(type, owner);
  if (partition.kind == ValueKind::Address) {
    if (elements && !elements->next) {
      return evaluation_.convert(elements->value, type, owner);
    }
    diagnostics_.reject(unit_, owner,
                        "address values require one integer or address value");
  }
  if (partition.kind == ValueKind::Buffer ||
      partition.kind == ValueKind::OpaqueDialect ||
      partition.kind == ValueKind::Encoding ||
      partition.kind == ValueKind::Vector ||
      partition.kind == ValueKind::View ||
      partition.kind == ValueKind::Tensor) {
    if (elements && !elements->next &&
        types_.unqualified(elements->value->type) == types_.unqualified(type)) {
      return evaluation_.expression(elements->value);
    }
    diagnostics_.reject(
        unit_, owner,
        "Loom source values require an operation result or a copy");
  }
  if (auto* vector = types_.native_vector(type)) {
    std::vector<loom_value_id_t> components;
    for (auto* element : cxx::ListView{elements}) {
      components.push_back(
          evaluation_.convert(element, vector->elementType(), owner).ssa());
    }
    return vectors_.construct(components, type, owner);
  }
  auto output = types_.get(type, owner);
  if (elements && !elements->next) {
    return evaluation_.convert(elements->value, type, owner);
  }
  if (elements || loom_type_kind(output) != LOOM_TYPE_SCALAR) {
    diagnostics_.reject(
        unit_, owner,
        "value initialization requires a scalar, vector or admitted record");
  }
  return zero(type, owner);
}

void Initialization::default_storage_record(StorageProjection destination,
                                            const RecordPartition& record,
                                            const cxx::Type* object_type,
                                            cxx::FunctionSymbol* constructor,
                                            bool zero_initialize, bool retained,
                                            cxx::AST* owner) {
  types_.admit_construction(constructor, object_type, owner);
  ReceiverScope receiver(evaluation_, record, destination, object_type);
  if (zero_initialize) {
    for (const auto& member : record.members) {
      auto member_type = objects_.member_type(object_type, member.field);
      objects_.store(storage_.member(destination, member.field, owner),
                     zero(member_type, owner), member_type, owner);
      evaluation_.complete_storage_receiver_member(member.field);
    }
  } else if (retained) {
    for (const auto& member : record.members) {
      evaluation_.complete_storage_receiver_member(member.field);
    }
  }

  for_each_default_initializer(
      unit_, diagnostics_, constructor, owner,
      [&](cxx::FieldSymbol* field, cxx::ExpressionAST* source,
          cxx::MemInitializerAST* initializer) {
        auto field_type = objects_.member_type(object_type, field);
        object(storage_.member(destination, field, initializer), field_type,
               source, initializer->constructor, initializer,
               zero_initialize || retained);
        evaluation_.complete_storage_receiver_member(field);
      });
}

// A list initializes subobjects in place: a later default clause may observe
// an already initialized field. Explicit nested clauses still resolve their
// lexical enclosing receiver by field identity. Copy initialization captures
// the complete source value before replacing destination storage.
void Initialization::object(StorageProjection destination,
                            const cxx::Type* type,
                            cxx::ExpressionAST* initializer,
                            cxx::FunctionSymbol* constructor, cxx::AST* owner,
                            bool retained) {
  cxx::Initializer source(initializer);
  bool list = !initializer || source.form() == cxx::InitializerForm::kList ||
              source.form() == cxx::InitializerForm::kParen;
  bool value_initialized = false;
  std::vector<cxx::ExpressionAST*> elements;
  cxx::VariableSymbol* implicit_element = nullptr;
  if (list) {
    elements = source.arguments();
    auto* braced = source.bracedInitList();
    implicit_element = braced ? braced->implicitElement : nullptr;
    if (auto* paren = cxx::ast_cast<cxx::ParenInitializerAST>(initializer)) {
      value_initialized = paren->lparenLoc && !paren->expressionList;
    }
  } else if (unit_.typeTraits().is_class(type)) {
    // A same-type prvalue constructs this destination directly. Evaluating it
    // as a detached value first would hide already initialized subobjects from
    // later clauses that name the destination.
    auto* clause = source.singleExpression();
    if (clause && clause->valueCategory == cxx::ValueCategory::kPrValue &&
        types_.unqualified(clause->type) == types_.unqualified(type)) {
      if (auto* nested = cxx::ast_cast<cxx::NestedExpressionAST>(clause)) {
        object(destination, type, nested->expression, constructor, owner,
               retained);
        return;
      }
      if (auto* select = cxx::ast_cast<cxx::ConditionalExpressionAST>(clause)) {
        evaluation_.conditional_object(destination, type, select, retained);
        return;
      }
      if (auto* binary = cxx::ast_cast<cxx::BinaryExpressionAST>(clause);
          binary && !binary->symbol && binary->op == cxx::TokenKind::T_COMMA) {
        evaluation_.effect(binary->leftExpression);
        object(destination, type, binary->rightExpression, constructor, owner,
               retained);
        return;
      }
      cxx::List<cxx::ExpressionAST*>* arguments = nullptr;
      if (auto* construction =
              cxx::ast_cast<cxx::BracedTypeConstructionAST>(clause)) {
        constructor = construction->constructorSymbol;
        arguments = construction->bracedInitList->expressionList;
        implicit_element = construction->bracedInitList->implicitElement;
        list = true;
      } else if (auto* construction =
                     cxx::ast_cast<cxx::TypeConstructionAST>(clause)) {
        constructor = construction->constructorSymbol;
        arguments = construction->expressionList;
        value_initialized =
            construction->lparenLoc && !construction->expressionList;
        list = true;
      }
      for (auto* argument : cxx::ListView{arguments}) {
        elements.push_back(argument);
      }
    }
  }

  auto* object_type = type;
  auto* shape = types_.unqualified(type);
  if (auto* record = types_.record(shape, owner);
      record && constructor &&
      constructor == record->source->defaultConstructor()) {
    default_storage_record(
        destination, *record, object_type, constructor,
        value_initialized &&
            unit_.typeTraits().requires_zero_initialization(shape, constructor),
        retained, owner);
    return;
  }
  if (list) {
    if (auto* array = cxx::type_cast<cxx::BoundedArrayType>(shape)) {
      for (size_t index = 0; index < array->size(); ++index) {
        const bool explicit_clause = index < elements.size();
        auto* clause = index < elements.size() ? elements[index] : nullptr;
        auto* element_constructor = clause ? nullptr : constructor;
        if (!clause && implicit_element) {
          clause = implicit_element->initializer();
          element_constructor = implicit_element->constructor();
        }
        object(storage_.element(destination, array, index, owner),
               unit_.typeTraits().get_element_type(object_type), clause,
               element_constructor, owner, retained && !explicit_clause);
      }
      return;
    }
    if (auto* record = types_.record(shape, owner)) {
      if (elements.size() == 1 &&
          types_.unqualified(elements[0]->type) == shape) {
        object(destination, object_type, elements[0], nullptr, owner);
        return;
      }
      ReceiverScope receiver(evaluation_, *record, destination, object_type);
      for (size_t index = 0; index < record->members.size(); ++index) {
        const auto& member = record->members[index];
        object(storage_.member(destination, member.field, owner),
               objects_.member_type(object_type, member.field),
               index < elements.size() ? elements[index] : nullptr, nullptr,
               owner, false);
        evaluation_.complete_storage_receiver_member(member.field);
      }
      return;
    }
  }
  auto initialized = value(shape, initializer, constructor, owner);
  objects_.store(destination, initialized, object_type, owner);
}

void Initialization::storage(StorageAllocation allocation,
                             const cxx::Type* type,
                             cxx::ExpressionAST* initializer,
                             cxx::FunctionSymbol* constructor,
                             cxx::AST* owner) {
  if (auto* array =
          cxx::type_cast<cxx::BoundedArrayType>(types_.unqualified(type))) {
    cxx::Initializer source(initializer);
    if (source.form() != cxx::InitializerForm::kList &&
        source.form() != cxx::InitializerForm::kParen) {
      diagnostics_.reject(
          unit_, owner, "automatic arrays require element-wise initialization");
    }
    auto elements = source.arguments();
    auto* braced = source.bracedInitList();
    auto* implicit_element = braced ? braced->implicitElement : nullptr;
    auto* element_type = unit_.typeTraits().get_element_type(type);
    for (size_t index = 0; index < array->size(); ++index) {
      auto* clause = index < elements.size() ? elements[index] : nullptr;
      auto* element_constructor = clause ? nullptr : constructor;
      if (!clause && implicit_element) {
        clause = implicit_element->initializer();
        element_constructor = implicit_element->constructor();
      }
      if (unit_.typeTraits().is_class(element_type) ||
          unit_.typeTraits().is_array(element_type) ||
          types_.native_vector(element_type)) {
        object(
            storage_.element(storage_.project(allocation.pointer, type, owner),
                             array, index, owner),
            element_type, clause, element_constructor, owner);
      } else {
        auto element = value(element_type, clause, element_constructor, owner);
        auto position = scalars_.integer(index, LOOM_SCALAR_TYPE_INDEX,
                                         locations_.get(owner));
        storage_.store({allocation.view, position}, element.ssa(), element_type,
                       owner);
      }
    }
  } else if (unit_.typeTraits().is_class(type)) {
    object(storage_.project(allocation.pointer, type, owner), type, initializer,
           constructor, owner);
  } else {
    storage_.store({allocation.view, std::nullopt},
                   evaluation_.expression(initializer).ssa(), type, owner);
  }
}

}  // namespace loom::cxx_import
