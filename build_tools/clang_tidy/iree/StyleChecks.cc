// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/StyleChecks.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
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

std::optional<std::string> SourceText(const Expr* Expr,
                                      const SourceManager& SourceManager,
                                      const LangOptions& LangOptions) {
  CharSourceRange Range =
      CharSourceRange::getTokenRange(Expr->getSourceRange());
  bool Invalid = false;
  StringRef Text =
      Lexer::getSourceText(Range, SourceManager, LangOptions, &Invalid);
  if (Invalid || Text.empty()) {
    return std::nullopt;
  }
  std::string Result = Text.str();
  Result.erase(std::remove_if(Result.begin(), Result.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               Result.end());
  return Result;
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

std::optional<CharSourceRange> StatementCharRange(
    const Stmt* Statement, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  SourceLocation Begin = Statement->getBeginLoc();
  SourceLocation End = Statement->getEndLoc();
  if (Begin.isInvalid() || End.isInvalid() || Begin.isMacroID()) {
    return std::nullopt;
  }
  End = SourceManager.getExpansionLoc(End);
  if (SourceManager.getFileID(Begin) != SourceManager.getFileID(End)) {
    return std::nullopt;
  }
  End = Lexer::getLocForEndOfToken(End, 0, SourceManager, LangOptions);
  if (End.isInvalid()) {
    return std::nullopt;
  }
  std::pair<FileID, unsigned> Decomposed = SourceManager.getDecomposedLoc(End);
  bool Invalid = false;
  StringRef Buffer = SourceManager.getBufferData(Decomposed.first, &Invalid);
  if (Invalid) {
    return std::nullopt;
  }
  if (Decomposed.second < Buffer.size() && Buffer[Decomposed.second] == ';') {
    End = End.getLocWithOffset(1);
  }
  return CharSourceRange::getCharRange(Begin, End);
}

std::optional<std::string> StatementSourceText(
    const Stmt* Statement, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  std::optional<CharSourceRange> Range =
      StatementCharRange(Statement, SourceManager, LangOptions);
  if (!Range) {
    return std::nullopt;
  }
  return SourceText(*Range, SourceManager, LangOptions);
}

const Expr* IgnoreExpressionNoise(const Expr* Expr) {
  if (!Expr) {
    return nullptr;
  }
  return Expr->IgnoreParenImpCasts();
}

bool IsNullPointerConstant(const Expr* Expr, ASTContext& Context) {
  Expr = IgnoreExpressionNoise(Expr);
  return Expr && Expr->isNullPointerConstant(Context,
                                             Expr::NPC_ValueDependentIsNotNull);
}

const Expr* NonNullGuardedExpression(const Expr* Condition,
                                     ASTContext& Context) {
  Condition = IgnoreExpressionNoise(Condition);
  if (!Condition) {
    return nullptr;
  }
  if (Condition->getType()->isAnyPointerType()) {
    return Condition;
  }
  const auto* Binary = dyn_cast<BinaryOperator>(Condition);
  if (!Binary || Binary->getOpcode() != BO_NE) {
    return nullptr;
  }
  const Expr* Lhs = IgnoreExpressionNoise(Binary->getLHS());
  const Expr* Rhs = IgnoreExpressionNoise(Binary->getRHS());
  if (IsNullPointerConstant(Lhs, Context) && Rhs &&
      Rhs->getType()->isAnyPointerType()) {
    return Rhs;
  }
  if (IsNullPointerConstant(Rhs, Context) && Lhs &&
      Lhs->getType()->isAnyPointerType()) {
    return Lhs;
  }
  return nullptr;
}

const CallExpr* CallStatement(const Stmt* Statement) {
  const auto* Expr = dyn_cast_or_null<clang::Expr>(Statement);
  if (!Expr) {
    return nullptr;
  }
  return dyn_cast<CallExpr>(IgnoreExpressionNoise(Expr));
}

bool IsPointerLikeSingleArgumentRelease(const CallExpr* Call,
                                        const SourceManager& SourceManager,
                                        const LangOptions& LangOptions) {
  if (!Call || Call->getNumArgs() != 1) {
    return false;
  }
  std::optional<std::string> CalleeText =
      SourceText(Call->getCallee(), SourceManager, LangOptions);
  if (!CalleeText || !StringRef(*CalleeText).ends_with("_release")) {
    return false;
  }
  const FunctionDecl* Callee = Call->getDirectCallee();
  if (!Callee || Callee->getNumParams() != 1 ||
      !Callee->getReturnType()->isVoidType() ||
      !Callee->getName().ends_with("_release")) {
    return false;
  }
  return Callee->getParamDecl(0)->getType()->isAnyPointerType();
}

bool IsRefCountLifecycleFunctionName(StringRef FunctionName) {
  return FunctionName.ends_with("_retain") ||
         FunctionName.ends_with("_release");
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

std::optional<std::string> FieldSourceText(const FieldDecl* Field,
                                           const SourceManager& SourceManager,
                                           const LangOptions& LangOptions) {
  SourceRange FieldRange = Field->getSourceRange();
  if (FieldRange.getBegin().isInvalid() || FieldRange.getEnd().isInvalid()) {
    return std::nullopt;
  }
  return SourceText(CharSourceRange::getTokenRange(FieldRange), SourceManager,
                    LangOptions);
}

bool IsRefCountFieldDeclarationText(StringRef FieldText) {
  return SourceContainsIdentifier(FieldText, "iree_atomic_ref_count_t");
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

bool BodyContainsRefCountMutation(StringRef BodyText) {
  return SourceContainsIdentifier(BodyText, "iree_atomic_ref_count_inc") ||
         SourceContainsIdentifier(BodyText, "iree_atomic_ref_count_dec");
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

bool IsNullAssignmentToGuard(const Stmt* Statement, const Expr* GuardExpression,
                             ASTContext& Context,
                             const SourceManager& SourceManager,
                             const LangOptions& LangOptions) {
  const auto* Expr = dyn_cast_or_null<clang::Expr>(Statement);
  if (!Expr) {
    return false;
  }
  const auto* Binary = dyn_cast<BinaryOperator>(IgnoreExpressionNoise(Expr));
  if (!Binary || !Binary->isAssignmentOp() ||
      !IsNullPointerConstant(Binary->getRHS(), Context)) {
    return false;
  }
  std::optional<std::string> GuardText =
      SourceText(GuardExpression, SourceManager, LangOptions);
  std::optional<std::string> LhsText =
      SourceText(Binary->getLHS(), SourceManager, LangOptions);
  return GuardText && LhsText && *GuardText == *LhsText;
}

const CallExpr* ReleaseCallInSimpleBranch(const Stmt* Branch,
                                          const Expr* GuardExpression,
                                          ASTContext& Context,
                                          const SourceManager& SourceManager,
                                          const LangOptions& LangOptions) {
  if (const auto* Call = CallStatement(Branch)) {
    return IsPointerLikeSingleArgumentRelease(Call, SourceManager, LangOptions)
               ? Call
               : nullptr;
  }
  const auto* Compound = dyn_cast_or_null<CompoundStmt>(Branch);
  if (!Compound) {
    return nullptr;
  }
  const CallExpr* ReleaseCall = nullptr;
  unsigned StatementCount = 0;
  for (const Stmt* Child : Compound->body()) {
    if (isa<NullStmt>(Child)) {
      continue;
    }
    ++StatementCount;
    if (const auto* Call = CallStatement(Child);
        IsPointerLikeSingleArgumentRelease(Call, SourceManager, LangOptions)) {
      if (ReleaseCall) {
        return nullptr;
      }
      ReleaseCall = Call;
      continue;
    }
    if (IsNullAssignmentToGuard(Child, GuardExpression, Context, SourceManager,
                                LangOptions)) {
      continue;
    }
    return nullptr;
  }
  return StatementCount <= 2 ? ReleaseCall : nullptr;
}

std::optional<std::string> GuardedReleaseReplacementText(
    const IfStmt* If, const Stmt* Branch, const SourceManager& SourceManager,
    const LangOptions& LangOptions) {
  if (CallStatement(Branch)) {
    std::optional<std::string> Replacement =
        StatementSourceText(Branch, SourceManager, LangOptions);
    if (!Replacement || Replacement->find('\n') != std::string::npos) {
      return std::nullopt;
    }
    return Replacement;
  }
  const auto* Compound = dyn_cast<CompoundStmt>(Branch);
  if (!Compound) {
    return std::nullopt;
  }
  SourceLocation IfLocation = If->getIfLoc();
  if (IfLocation.isInvalid() || IfLocation.isMacroID()) {
    return std::nullopt;
  }
  unsigned Column = SourceManager.getSpellingColumnNumber(IfLocation);
  std::string Indent(Column > 0 ? Column - 1 : 0, ' ');
  std::string Replacement;
  for (const Stmt* Child : Compound->body()) {
    if (isa<NullStmt>(Child)) {
      continue;
    }
    std::optional<std::string> ChildText =
        StatementSourceText(Child, SourceManager, LangOptions);
    if (!ChildText || ChildText->find('\n') != std::string::npos) {
      return std::nullopt;
    }
    if (!Replacement.empty()) {
      Replacement.append("\n");
      Replacement.append(Indent);
    }
    Replacement.append(*ChildText);
  }
  return Replacement.empty() ? std::nullopt
                             : std::optional<std::string>(Replacement);
}

}  // namespace

DirectGotoCheck::DirectGotoCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void DirectGotoCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(gotoStmt(unless(isExpansionInSystemHeader())).bind("goto"),
                     this);
}

void DirectGotoCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* Goto = Result.Nodes.getNodeAs<GotoStmt>("goto");
  if (!Goto) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(Goto->getGotoLoc(), SourceManager)) {
    return;
  }
  SourceLocation Location = SourceManager.getExpansionLoc(Goto->getGotoLoc());
  diag(Location,
       "direct goto is not allowed; split lifetime management from mutation "
       "logic or use structured control flow");
}

GuardedReleaseCheck::GuardedReleaseCheck(StringRef Name,
                                         ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context) {}

void GuardedReleaseCheck::registerMatchers(ast_matchers::MatchFinder* Finder) {
  using namespace ast_matchers;
  Finder->addMatcher(
      ifStmt(unless(hasElse(stmt())), hasCondition(expr().bind("condition")),
             hasThen(stmt().bind("then")))
          .bind("if"),
      this);
}

void GuardedReleaseCheck::check(
    const ast_matchers::MatchFinder::MatchResult& Result) {
  const auto* If = Result.Nodes.getNodeAs<IfStmt>("if");
  const auto* Condition = Result.Nodes.getNodeAs<Expr>("condition");
  const auto* Then = Result.Nodes.getNodeAs<Stmt>("then");
  if (!If || !Condition || !Then) {
    return;
  }
  const SourceManager& SourceManager = *Result.SourceManager;
  if (IsExternalMacroBody(If->getIfLoc(), SourceManager)) {
    return;
  }
  ASTContext& Context = *Result.Context;
  const Expr* GuardExpression = NonNullGuardedExpression(Condition, Context);
  if (!GuardExpression) {
    return;
  }
  const CallExpr* ReleaseCall =
      ReleaseCallInSimpleBranch(Then, GuardExpression, Context, SourceManager,
                                Result.Context->getLangOpts());
  if (!ReleaseCall) {
    return;
  }
  std::optional<std::string> GuardText =
      SourceText(GuardExpression, SourceManager, Result.Context->getLangOpts());
  std::optional<std::string> ReleaseArgText = SourceText(
      ReleaseCall->getArg(0), SourceManager, Result.Context->getLangOpts());
  if (!GuardText || !ReleaseArgText || *GuardText != *ReleaseArgText) {
    return;
  }
  DiagnosticBuilder Diagnostic =
      diag(SourceManager.getExpansionLoc(If->getIfLoc()),
           "release functions are null-safe; call %0 unconditionally");
  Diagnostic << ReleaseCall->getDirectCallee()->getName();
  std::optional<CharSourceRange> ReplacementRange =
      StatementCharRange(If, SourceManager, Result.Context->getLangOpts());
  std::optional<std::string> ReplacementText = GuardedReleaseReplacementText(
      If, Then, SourceManager, Result.Context->getLangOpts());
  if (ReplacementRange && ReplacementText) {
    Diagnostic << FixItHint::CreateReplacement(*ReplacementRange,
                                               *ReplacementText);
  }
}

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
    std::optional<std::string> FieldText =
        FieldSourceText(Field, SourceManager, Result.Context->getLangOpts());
    if (!FieldText || !IsRefCountFieldDeclarationText(*FieldText) ||
        IsAllowedRefCountFieldName(Field, SourceManager)) {
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
  StringRef FunctionName = Function->getName();
  if (!IsRefCountLifecycleFunctionName(FunctionName)) {
    return;
  }
  std::optional<std::string> BodyText = FunctionBodySourceText(
      Function, SourceManager, Result.Context->getLangOpts());
  if (!BodyText || !BodyContainsRefCountMutation(*BodyText)) {
    return;
  }
  std::string NormalizedBody = NormalizedCodeSource(*BodyText);
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
