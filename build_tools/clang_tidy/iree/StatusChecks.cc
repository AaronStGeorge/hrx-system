// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/StatusChecks.h"

#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang::tidy::iree {
namespace {

bool IsNamedTypedef(QualType Type, StringRef Name) {
  Type = Type.getLocalUnqualifiedType();
  if (Type.isNull()) {
    return false;
  }
  const auto* Typedef = Type->getAs<TypedefType>();
  return Typedef && Typedef->getDecl()->getName() == Name;
}

QualType ReturnTypeFromCalleeExpr(const Expr* Callee) {
  if (!Callee) {
    return QualType();
  }
  QualType CalleeType = Callee->IgnoreParenImpCasts()->getType();
  if (CalleeType.isNull() || CalleeType->isDependentType()) {
    return QualType();
  }
  CalleeType = CalleeType.getLocalUnqualifiedType();
  if (const auto* Pointer = CalleeType->getAs<PointerType>()) {
    CalleeType = Pointer->getPointeeType().getLocalUnqualifiedType();
  }
  if (const auto* Proto = CalleeType->getAs<FunctionProtoType>()) {
    return Proto->getReturnType();
  }
  if (const auto* NoProto = CalleeType->getAs<FunctionNoProtoType>()) {
    return NoProto->getReturnType();
  }
  return QualType();
}

QualType ReturnTypeFromCall(const CallExpr* Call) {
  if (!Call) {
    return QualType();
  }
  if (const FunctionDecl* Callee = Call->getDirectCallee()) {
    return Callee->getReturnType();
  }
  return ReturnTypeFromCalleeExpr(Call->getCallee());
}

StringRef CalleeName(const CallExpr* Call) {
  if (const FunctionDecl* Callee = Call->getDirectCallee()) {
    if (const auto* Identifier = Callee->getIdentifier()) {
      return Identifier->getName();
    }
  }
  return StringRef();
}

bool IsAllowedExplicitStatusConsumer(const CallExpr* Call) {
  return CalleeName(Call) == "iree_status_ignore";
}

}  // namespace

DiscardedStatusCheck::DiscardedStatusCheck(StringRef Name,
                                           ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void DiscardedStatusCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      expr(ignoringParenCasts(
               callExpr(callee(functionDecl())).bind("status_call")),
           hasParent(compoundStmt()))
          .bind("discarded_expr"),
      this);
}

void DiscardedStatusCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Call = Result.Nodes.getNodeAs<CallExpr>("status_call");
  if (!Call) {
    return;
  }
  if (!IsNamedTypedef(ReturnTypeFromCall(Call), "iree_status_t")) {
    return;
  }
  if (IsAllowedExplicitStatusConsumer(Call)) {
    return;
  }

  if (StringRef Name = CalleeName(Call); !Name.empty()) {
    diag(Call->getExprLoc(),
         "iree_status_t result from %0 must be checked, returned, assigned, "
         "or explicitly consumed")
        << Name;
    return;
  }
  diag(Call->getExprLoc(),
       "iree_status_t result must be checked, returned, assigned, or "
       "explicitly consumed");
}

}  // namespace clang::tidy::iree
