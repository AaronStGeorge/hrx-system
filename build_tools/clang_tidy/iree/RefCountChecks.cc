// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/RefCountChecks.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::iree {
namespace {

bool IsExternalMacroBody(SourceLocation Location,
                         const SourceManager& SourceManager) {
  if (!Location.isMacroID()) {
    return false;
  }
  SourceLocation SpellingLocation = SourceManager.getSpellingLoc(Location);
  if (SourceManager.isInSystemHeader(SpellingLocation)) {
    return true;
  }
  llvm::StringRef Filename = SourceManager.getFilename(SpellingLocation);
  return Filename.contains("/external/") || Filename.starts_with("external/");
}

std::optional<std::string> SourceText(CharSourceRange Range,
                                      const SourceManager& SourceManager,
                                      const LangOptions& LangOptions) {
  bool Invalid = false;
  StringRef Text =
      Lexer::getSourceText(Range, SourceManager, LangOptions, &Invalid);
  if (Invalid || Text.empty()) {
    return std::nullopt;
  }
  return Text.str();
}

std::optional<std::string> FunctionBodySourceText(
    const FunctionDecl* Function, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  const Stmt* Body = Function->getBody();
  if (!Body) {
    return std::nullopt;
  }
  SourceRange BodyRange = Body->getSourceRange();
  if (BodyRange.getBegin().isInvalid() || BodyRange.getEnd().isInvalid()) {
    return std::nullopt;
  }
  return SourceText(CharSourceRange::getTokenRange(BodyRange), SourceManager,
                    LangOptions);
}

bool IsIdentifierCharacter(char Character) {
  return std::isalnum(static_cast<unsigned char>(Character)) ||
         Character == '_';
}

bool SourceContainsIdentifier(StringRef Source, StringRef Identifier) {
  enum class State {
    kCode,
    kLineComment,
    kBlockComment,
    kString,
    kCharacter,
  };
  State ParserState = State::kCode;
  for (size_t I = 0; I < Source.size(); ++I) {
    char Character = Source[I];
    char Next = I + 1 < Source.size() ? Source[I + 1] : '\0';
    switch (ParserState) {
      case State::kCode:
        if (Character == '/' && Next == '/') {
          ParserState = State::kLineComment;
          ++I;
        } else if (Character == '/' && Next == '*') {
          ParserState = State::kBlockComment;
          ++I;
        } else if (Character == '"') {
          ParserState = State::kString;
        } else if (Character == '\'') {
          ParserState = State::kCharacter;
        } else if (Source.substr(I).starts_with(Identifier) &&
                   (I == 0 || !IsIdentifierCharacter(Source[I - 1])) &&
                   (I + Identifier.size() == Source.size() ||
                    !IsIdentifierCharacter(Source[I + Identifier.size()]))) {
          return true;
        }
        break;
      case State::kLineComment:
        if (Character == '\n' || Character == '\r') {
          ParserState = State::kCode;
        }
        break;
      case State::kBlockComment:
        if (Character == '*' && Next == '/') {
          ParserState = State::kCode;
          ++I;
        }
        break;
      case State::kString:
      case State::kCharacter:
        if (Character == '\\' && Next != '\0') {
          ++I;
        } else if ((ParserState == State::kString && Character == '"') ||
                   (ParserState == State::kCharacter && Character == '\'')) {
          ParserState = State::kCode;
        }
        break;
    }
  }
  return false;
}

std::string NormalizedCodeSource(StringRef Source) {
  enum class State {
    kCode,
    kLineComment,
    kBlockComment,
    kString,
    kCharacter,
  };
  State ParserState = State::kCode;
  std::string Result;
  for (size_t I = 0; I < Source.size(); ++I) {
    char Character = Source[I];
    char Next = I + 1 < Source.size() ? Source[I + 1] : '\0';
    switch (ParserState) {
      case State::kCode:
        if (Character == '/' && Next == '/') {
          ParserState = State::kLineComment;
          ++I;
        } else if (Character == '/' && Next == '*') {
          ParserState = State::kBlockComment;
          ++I;
        } else if (Character == '"') {
          ParserState = State::kString;
        } else if (Character == '\'') {
          ParserState = State::kCharacter;
        } else if (!std::isspace(static_cast<unsigned char>(Character))) {
          Result.push_back(Character);
        }
        break;
      case State::kLineComment:
        if (Character == '\n' || Character == '\r') {
          ParserState = State::kCode;
        }
        break;
      case State::kBlockComment:
        if (Character == '*' && Next == '/') {
          ParserState = State::kCode;
          ++I;
        }
        break;
      case State::kString:
      case State::kCharacter:
        if (Character == '\\' && Next != '\0') {
          ++I;
        } else if ((ParserState == State::kString && Character == '"') ||
                   (ParserState == State::kCharacter && Character == '\'')) {
          ParserState = State::kCode;
        }
        break;
    }
  }
  return Result;
}

bool IsRefCountLifecycleFunctionName(StringRef FunctionName) {
  return FunctionName.ends_with("_retain") ||
         FunctionName.ends_with("_release");
}

bool IsNamedTypedef(QualType QualType, StringRef Name) {
  QualType = QualType.getUnqualifiedType();
  const Type* TypePtr = QualType.getTypePtrOrNull();
  if (!TypePtr) {
    return false;
  }
  if (const auto* Typedef = TypePtr->getAs<TypedefType>()) {
    return Typedef->getDecl()->getName() == Name;
  }
  if (const auto* Attributed = dyn_cast<AttributedType>(TypePtr)) {
    return IsNamedTypedef(Attributed->getModifiedType(), Name);
  }
  return false;
}

bool IsRefCountField(const FieldDecl* Field) {
  return Field && IsNamedTypedef(Field->getType(), "iree_atomic_ref_count_t");
}

const RecordDecl* EnclosingRecord(const FieldDecl* Field) {
  return dyn_cast_or_null<RecordDecl>(Field->getDeclContext());
}

const FieldDecl* FirstField(const RecordDecl* Record) {
  if (!Record) {
    return nullptr;
  }
  for (const FieldDecl* Field : Record->fields()) {
    return Field;
  }
  return nullptr;
}

bool IsAllowedRefCountFieldName(const FieldDecl* Field,
                                const SourceManager& SourceManager) {
  if (Field->getName() == "ref_count") {
    return true;
  }

  SourceLocation Location = SourceManager.getExpansionLoc(Field->getLocation());
  StringRef Filename = SourceManager.getFilename(Location);
  return Field->getName() == "counter" &&
         Filename.ends_with("runtime/src/iree/vm/ref.h");
}

bool IsRefCountAnchorField(const FieldDecl* Field,
                           const SourceManager& SourceManager) {
  return IsRefCountField(Field) &&
         Field == FirstField(EnclosingRecord(Field)) &&
         IsAllowedRefCountFieldName(Field, SourceManager);
}

bool IsRefCountAnchoredRecord(const RecordDecl* Record,
                              const SourceManager& SourceManager) {
  return IsRefCountAnchorField(FirstField(Record), SourceManager);
}

bool ShouldDiagnoseRefCountField(const FieldDecl* Field,
                                 const SourceManager& SourceManager) {
  if (!IsRefCountField(Field)) {
    return false;
  }
  const RecordDecl* Record = EnclosingRecord(Field);
  if (Field == FirstField(Record)) {
    return !IsAllowedRefCountFieldName(Field, SourceManager);
  }
  return IsRefCountAnchoredRecord(Record, SourceManager);
}

bool BodyContainsRefCountMutation(StringRef BodyText) {
  return SourceContainsIdentifier(BodyText, "iree_atomic_ref_count_inc") ||
         SourceContainsIdentifier(BodyText, "iree_atomic_ref_count_dec");
}

bool HasIgnoredRefCountDecrement(StringRef NormalizedBody) {
  static constexpr StringRef RefCountDec = "iree_atomic_ref_count_dec";
  size_t SearchPosition = 0;
  while (true) {
    size_t DecPosition = NormalizedBody.find(RefCountDec, SearchPosition);
    if (DecPosition == StringRef::npos) {
      return false;
    }
    bool IsStatementStart = DecPosition == 0 ||
                            NormalizedBody[DecPosition - 1] == '{' ||
                            NormalizedBody[DecPosition - 1] == ';';
    static constexpr StringRef VoidCast = "(void)";
    if (!IsStatementStart && DecPosition >= VoidCast.size()) {
      size_t CastPosition = DecPosition - VoidCast.size();
      IsStatementStart =
          NormalizedBody.substr(CastPosition, VoidCast.size()) == VoidCast &&
          (CastPosition == 0 || NormalizedBody[CastPosition - 1] == '{' ||
           NormalizedBody[CastPosition - 1] == ';');
    }
    if (IsStatementStart) {
      return true;
    }
    SearchPosition = DecPosition + RefCountDec.size();
  }
}

bool RefCountDecrementReferencesParameter(StringRef NormalizedBody,
                                          StringRef ParameterName) {
  static constexpr StringRef RefCountDec = "iree_atomic_ref_count_dec";
  size_t SearchPosition = 0;
  while (true) {
    size_t DecPosition = NormalizedBody.find(RefCountDec, SearchPosition);
    if (DecPosition == StringRef::npos) {
      return false;
    }
    size_t StatementEnd = NormalizedBody.find(';', DecPosition);
    if (StatementEnd == StringRef::npos) {
      StatementEnd = NormalizedBody.size();
    }
    if (SourceContainsIdentifier(
            NormalizedBody.substr(DecPosition, StatementEnd - DecPosition),
            ParameterName)) {
      return true;
    }
    SearchPosition = DecPosition + RefCountDec.size();
  }
}

bool HasEarlyNullReturn(StringRef NormalizedBody, StringRef ParameterName) {
  std::string Parameter = ParameterName.str();
  for (std::string Pattern :
       {"if(!" + Parameter + ")return;", "if(!" + Parameter + "){return;}",
        "if(" + Parameter + "==NULL)return;",
        "if(" + Parameter + "==NULL){return;}",
        "if(NULL==" + Parameter + ")return;",
        "if(NULL==" + Parameter + "){return;}",
        "if(IREE_UNLIKELY(!" + Parameter + "))return;",
        "if(IREE_UNLIKELY(!" + Parameter + ")){return;}"}) {
    if (NormalizedBody.contains(StringRef(Pattern))) {
      return true;
    }
  }
  return false;
}

bool HasGuardedRefCountDecrement(StringRef NormalizedBody,
                                 StringRef ParameterName) {
  static constexpr StringRef RefCountDec = "iree_atomic_ref_count_dec";
  size_t SearchPosition = 0;
  while (true) {
    size_t DecPosition = NormalizedBody.find(RefCountDec, SearchPosition);
    if (DecPosition == StringRef::npos) {
      return false;
    }
    StringRef Prefix = NormalizedBody.substr(0, DecPosition);
    size_t IfPosition = Prefix.rfind("if(");
    size_t StatementPosition = NormalizedBody.rfind(';', DecPosition);
    size_t BlockPosition = NormalizedBody.rfind('{', DecPosition);
    size_t Boundary =
        std::max(StatementPosition == StringRef::npos ? 0 : StatementPosition,
                 BlockPosition == StringRef::npos ? 0 : BlockPosition);
    if (IfPosition != StringRef::npos && IfPosition >= Boundary) {
      StringRef ConditionPrefix =
          NormalizedBody.substr(IfPosition + 3, DecPosition - IfPosition - 3);
      if (ConditionPrefix.contains("&&") &&
          SourceContainsIdentifier(ConditionPrefix, ParameterName)) {
        return true;
      }
    }
    SearchPosition = DecPosition + RefCountDec.size();
  }
}

bool HasArgumentAssert(StringRef NormalizedBody, StringRef ParameterName) {
  std::string Parameter = ParameterName.str();
  for (std::string Pattern : {"IREE_ASSERT_ARGUMENT(" + Parameter + ");",
                              "IREE_ASSERT(" + Parameter + ");"}) {
    if (NormalizedBody.contains(StringRef(Pattern))) {
      return true;
    }
  }
  return false;
}

bool IsRefCountReleaseNullSafe(const FunctionDecl* Function,
                               StringRef NormalizedBody) {
  if (Function->getNumParams() != 1) {
    return true;
  }
  const ParmVarDecl* Parameter = Function->getParamDecl(0);
  if (!Parameter->getType()->isAnyPointerType()) {
    return true;
  }
  StringRef ParameterName = Parameter->getName();
  if (ParameterName.empty() ||
      !RefCountDecrementReferencesParameter(NormalizedBody, ParameterName)) {
    return true;
  }
  if (HasArgumentAssert(NormalizedBody, ParameterName)) {
    return false;
  }
  return HasEarlyNullReturn(NormalizedBody, ParameterName) ||
         HasGuardedRefCountDecrement(NormalizedBody, ParameterName);
}

}  // namespace

RefCountLifecycleCheck::RefCountLifecycleCheck(StringRef Name,
                                               ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void RefCountLifecycleCheck::registerMatchers(
    ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      functionDecl(isDefinition(), unless(isExpansionInSystemHeader()))
          .bind("function"),
      this);
  Finder->addMatcher(
      fieldDecl(unless(isExpansionInSystemHeader())).bind("refcount-field"),
      this);
}

void RefCountLifecycleCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  if (const auto* Field = Result.Nodes.getNodeAs<FieldDecl>("refcount-field")) {
    const SourceManager& SourceManager = *Result.SourceManager;
    if (IsExternalMacroBody(Field->getLocation(), SourceManager)) {
      return;
    }
    if (!ShouldDiagnoseRefCountField(Field, SourceManager)) {
      return;
    }
    diag(SourceManager.getExpansionLoc(Field->getLocation()),
         "iree_atomic_ref_count_t field %0 must model object lifetime; use an "
         "explicit atomic integer type for counters")
        << Field->getName();
    return;
  }

  const auto* Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  if (!Function) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(Function->getLocation(), SourceManager)) {
    return;
  }
  std::optional<std::string> BodyText = FunctionBodySourceText(
      Function, SourceManager, Result.Context->getLangOpts());
  if (!BodyText || !BodyContainsRefCountMutation(*BodyText)) {
    return;
  }
  std::string NormalizedBody = NormalizedCodeSource(*BodyText);
  if (HasIgnoredRefCountDecrement(NormalizedBody)) {
    diag(SourceManager.getExpansionLoc(Function->getLocation()),
         "iree_atomic_ref_count_dec return value must be checked");
    return;
  }
  StringRef FunctionName = Function->getName();
  if (!IsRefCountLifecycleFunctionName(FunctionName)) {
    return;
  }
  if (Function->getReturnType()->isVoidType() &&
      FunctionName.ends_with("_release") &&
      !IsRefCountReleaseNullSafe(Function, NormalizedBody)) {
    diag(SourceManager.getExpansionLoc(Function->getLocation()),
         "refcounted release function %0 must be null-safe")
        << FunctionName;
    return;
  }
  if (Function->getReturnType()->isVoidType()) {
    return;
  }
  diag(SourceManager.getExpansionLoc(Function->getLocation()),
       "refcounted retain/release function %0 must return void")
      << FunctionName;
}

}  // namespace clang::tidy::iree
