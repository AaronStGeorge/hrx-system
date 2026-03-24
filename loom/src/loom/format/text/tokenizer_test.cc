// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/tokenizer.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// RAII wrapper that deinitializes on scope exit.
class ScopedTokenizer {
 public:
  explicit ScopedTokenizer(const char* text) {
    loom_tokenizer_initialize(iree_make_cstring_view(text), IREE_SV("<test>"),
                              &t_);
  }
  ~ScopedTokenizer() { loom_tokenizer_deinitialize(&t_); }
  loom_tokenizer_t* get() { return &t_; }
  loom_token_t next() { return loom_tokenizer_next(&t_); }
  loom_token_t peek() { return loom_tokenizer_peek(&t_); }

 private:
  loom_tokenizer_t t_;
};

//===----------------------------------------------------------------------===//
// Punctuation
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SingleCharPunctuation) {
  ScopedTokenizer t("( ) { } [ ] < > = : ,");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LBRACE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RBRACE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LBRACKET);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RBRACKET);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RANGLE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EQUALS);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COMMA);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, Arrow) {
  ScopedTokenizer t("->");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_ARROW);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("->")));
}

TEST(Tokenizer, ColonBeforeColon) {
  ScopedTokenizer t(": :");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
}

//===----------------------------------------------------------------------===//
// Numbers
//===----------------------------------------------------------------------===//

TEST(Tokenizer, Integer) {
  ScopedTokenizer t("42");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("42")));
}

TEST(Tokenizer, NegativeInteger) {
  ScopedTokenizer t("-1");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("-1")));
}

TEST(Tokenizer, HexInteger) {
  ScopedTokenizer t("0xFF");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("0xFF")));
}

TEST(Tokenizer, Float) {
  ScopedTokenizer t("3.14");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("3.14")));
}

TEST(Tokenizer, FloatExponent) {
  ScopedTokenizer t("1.0e-5");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("1.0e-5")));
}

TEST(Tokenizer, NegativeFloat) {
  ScopedTokenizer t("-0.5");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_FLOAT);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("-0.5")));
}

TEST(Tokenizer, NegativeArrowDisambiguation) {
  ScopedTokenizer t("-> -1");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_ARROW);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
}

//===----------------------------------------------------------------------===//
// Strings
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SimpleString) {
  ScopedTokenizer t("\"hello\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("hello")));
}

TEST(Tokenizer, StringWithEscapes) {
  ScopedTokenizer t("\"has \\\"quotes\\\"\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
}

TEST(Tokenizer, EmptyString) {
  ScopedTokenizer t("\"\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("")));
}

//===----------------------------------------------------------------------===//
// References
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SSAValue) {
  ScopedTokenizer t("%x %0 %arg0 %M %contract0");
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE) << "token " << i;
  }
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, SSAValueText) {
  ScopedTokenizer t("%tile");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("tile")));
}

TEST(Tokenizer, Symbol) {
  ScopedTokenizer t("@main");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SYMBOL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("main")));
}

TEST(Tokenizer, HashAttr) {
  ScopedTokenizer t("#q8_0");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_HASH_ATTR);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("q8_0")));
}

TEST(Tokenizer, ResultOrdinal) {
  ScopedTokenizer t("#0 #1");
  loom_token_t token0 = t.next();
  EXPECT_EQ(token0.kind, LOOM_TOKEN_RESULT_ORDINAL);
  EXPECT_TRUE(iree_string_view_equal(token0.text, IREE_SV("0")));
  loom_token_t token1 = t.next();
  EXPECT_EQ(token1.kind, LOOM_TOKEN_RESULT_ORDINAL);
  EXPECT_TRUE(iree_string_view_equal(token1.text, IREE_SV("1")));
}

TEST(Tokenizer, BlockLabel) {
  ScopedTokenizer t("^bb0");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_BLOCK_LABEL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("bb0")));
}

//===----------------------------------------------------------------------===//
// Identifiers and op names
//===----------------------------------------------------------------------===//

TEST(Tokenizer, BareIdent) {
  ScopedTokenizer t("tile tensor group to step else");
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT) << "token " << i;
  }
}

TEST(Tokenizer, OpName) {
  ScopedTokenizer t("test.addi tile.contract scalar.addf");
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(t.next().kind, LOOM_TOKEN_OP_NAME) << "token " << i;
  }
}

TEST(Tokenizer, OpNameText) {
  ScopedTokenizer t("test.addi");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_OP_NAME);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("test.addi")));
}

TEST(Tokenizer, TokenKindName) {
  EXPECT_TRUE(iree_string_view_equal(loom_token_kind_name(LOOM_TOKEN_INTEGER),
                                     IREE_SV("integer")));
  EXPECT_TRUE(iree_string_view_equal(loom_token_kind_name(LOOM_TOKEN_ARROW),
                                     IREE_SV("'->'")));
  EXPECT_TRUE(iree_string_view_equal(loom_token_kind_name(LOOM_TOKEN_EOF),
                                     IREE_SV("end of file")));
}

//===----------------------------------------------------------------------===//
// Whitespace and comments
//===----------------------------------------------------------------------===//

TEST(Tokenizer, SkipsWhitespace) {
  ScopedTokenizer t("  42  ");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, SkipsComments) {
  ScopedTokenizer t("42 // comment\n99");
  loom_token_t token0 = t.next();
  EXPECT_EQ(token0.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token0.text, IREE_SV("42")));
  loom_token_t token1 = t.next();
  EXPECT_EQ(token1.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token1.text, IREE_SV("99")));
}

TEST(Tokenizer, EmptyInput) {
  ScopedTokenizer t("");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, OnlyWhitespace) {
  ScopedTokenizer t("   \n\t  ");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, OnlyComment) {
  ScopedTokenizer t("// nothing here");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

//===----------------------------------------------------------------------===//
// Location tracking
//===----------------------------------------------------------------------===//

TEST(Tokenizer, LineAndColumn) {
  ScopedTokenizer t("42\n  99");
  loom_token_t token0 = t.next();
  EXPECT_EQ(token0.line, 1u);
  EXPECT_EQ(token0.column, 1u);
  loom_token_t token1 = t.next();
  EXPECT_EQ(token1.line, 2u);
  EXPECT_EQ(token1.column, 3u);
}

//===----------------------------------------------------------------------===//
// Peek and consume helpers
//===----------------------------------------------------------------------===//

TEST(Tokenizer, PeekDoesNotConsume) {
  ScopedTokenizer t("42");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, At) {
  ScopedTokenizer t("42");
  EXPECT_TRUE(loom_tokenizer_at(t.get(), LOOM_TOKEN_INTEGER));
  EXPECT_FALSE(loom_tokenizer_at(t.get(), LOOM_TOKEN_FLOAT));
}

TEST(Tokenizer, TryConsume) {
  ScopedTokenizer t("42 3.14");
  EXPECT_FALSE(loom_tokenizer_try_consume(t.get(), LOOM_TOKEN_FLOAT));
  EXPECT_TRUE(loom_tokenizer_try_consume(t.get(), LOOM_TOKEN_INTEGER));
  EXPECT_TRUE(loom_tokenizer_try_consume(t.get(), LOOM_TOKEN_FLOAT));
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, TryConsumeKeyword) {
  ScopedTokenizer t("to step");
  EXPECT_FALSE(loom_tokenizer_try_consume_keyword(t.get(), IREE_SV("step")));
  EXPECT_TRUE(loom_tokenizer_try_consume_keyword(t.get(), IREE_SV("to")));
  EXPECT_TRUE(loom_tokenizer_try_consume_keyword(t.get(), IREE_SV("step")));
}

TEST(Tokenizer, NextReturnsExpectedKind) {
  ScopedTokenizer t("42");
  loom_token_t token = loom_tokenizer_next(t.get());
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("42")));
}

TEST(Tokenizer, PeekDoesNotMatchWrongKind) {
  ScopedTokenizer t("42");
  loom_token_t token = loom_tokenizer_peek(t.get());
  EXPECT_EQ(token.kind, LOOM_TOKEN_INTEGER);
  EXPECT_NE(token.kind, LOOM_TOKEN_FLOAT);
}

//===----------------------------------------------------------------------===//
// Error propagation
//===----------------------------------------------------------------------===//

TEST(Tokenizer, UnterminatedStringError) {
  ScopedTokenizer t("\"unterminated");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, ErrorSurvivesMultiplePeeks) {
  ScopedTokenizer t("\"unterminated");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, ScanErrorSurfacedOnNext) {
  ScopedTokenizer t("\"unterminated");
  loom_token_t token = loom_tokenizer_next(t.get());
  EXPECT_EQ(token.kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, ErrorAfterValidTokens) {
  ScopedTokenizer t("42 \"unterminated");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, UnexpectedCharacterError) {
  ScopedTokenizer t("~");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, BarePercentError) {
  ScopedTokenizer t("% ");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, BareAtSignError) {
  ScopedTokenizer t("@ ");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, BareHashError) {
  ScopedTokenizer t("# ");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, BareCaretError) {
  ScopedTokenizer t("^ ");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, BarePercentAtEOFError) {
  ScopedTokenizer t("%");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, BareAtSignAtEOFError) {
  ScopedTokenizer t("@");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, PrefixStrippedFromSSAValue) {
  // Token text should be the name without the '%' prefix.
  ScopedTokenizer t("%arg0");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("arg0")));
  // Source location still points at the '%'.
  EXPECT_EQ(token.column, 1u);
}

TEST(Tokenizer, PrefixStrippedFromSymbol) {
  // Token text should be the name without the '@' prefix.
  ScopedTokenizer t("@main");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SYMBOL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("main")));
  EXPECT_EQ(token.column, 1u);
}

TEST(Tokenizer, PrefixStrippedFromHashAttr) {
  // Token text should be the name without the '#' prefix.
  ScopedTokenizer t("#enc");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_HASH_ATTR);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("enc")));
  EXPECT_EQ(token.column, 1u);
}

TEST(Tokenizer, PrefixStrippedFromResultOrdinal) {
  // Token text should be the ordinal without the '#' prefix.
  ScopedTokenizer t("#42");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_RESULT_ORDINAL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("42")));
  EXPECT_EQ(token.column, 1u);
}

TEST(Tokenizer, PrefixStrippedFromBlockLabel) {
  // Token text should be the label without the '^' prefix.
  ScopedTokenizer t("^entry");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_BLOCK_LABEL);
  EXPECT_TRUE(iree_string_view_equal(token.text, IREE_SV("entry")));
  EXPECT_EQ(token.column, 1u);
}

//===----------------------------------------------------------------------===//
// Angle bracket scanning
//===----------------------------------------------------------------------===//

TEST(Tokenizer, AngleInterior) {
  ScopedTokenizer t("<4x4xf32>");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_equal(interior, IREE_SV("4x4xf32")));
}

TEST(Tokenizer, NestedAngleBrackets) {
  ScopedTokenizer t("<vm.ref<hal.buffer>>");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_equal(interior, IREE_SV("vm.ref<hal.buffer>")));
}

TEST(Tokenizer, AngleInteriorWithEscapedQuotes) {
  // Encoding parameter with escaped quotes: #enc<label="a\"b">
  ScopedTokenizer t("<label=\"a\\\"b\">");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  EXPECT_TRUE(iree_string_view_equal(interior, IREE_SV("label=\"a\\\"b\"")));
}

TEST(Tokenizer, AngleInteriorEscapeAtEOF) {
  // Unterminated string ending with escape at EOF.
  ScopedTokenizer t("<\"foo\\");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_scan_angle_interior(t.get(), &interior));
}

TEST(Tokenizer, AngleInteriorStringWithNewline) {
  // String inside angle brackets containing a newline.
  ScopedTokenizer t("<\"line1\nline2\">");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LANGLE);
  iree_string_view_t interior;
  IREE_ASSERT_OK(loom_tokenizer_scan_angle_interior(t.get(), &interior));
  // After scanning, the tokenizer should have tracked the newline.
  // Next token should report correct line.
  loom_token_t eof = t.next();
  EXPECT_EQ(eof.kind, LOOM_TOKEN_EOF);
  // We started on line 1, the newline inside the string moves to line 2.
  EXPECT_EQ(eof.line, 2u);
}

//===----------------------------------------------------------------------===//
// Complex sequences
//===----------------------------------------------------------------------===//

TEST(Tokenizer, BinaryOp) {
  ScopedTokenizer t("%result = test.addi %lhs, %rhs : i32");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EQUALS);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_OP_NAME);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COMMA);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

TEST(Tokenizer, FunctionSignature) {
  ScopedTokenizer t("test.func @main(%a: f32) -> (f32)");
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_OP_NAME);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SYMBOL);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_COLON);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_ARROW);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_LPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_BARE_IDENT);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_RPAREN);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
}

//===----------------------------------------------------------------------===//
// Unicode and UTF-8
//===----------------------------------------------------------------------===//

TEST(Tokenizer, ChineseStringLiteral) {
  // "你好" — two CJK characters, each 3 bytes in UTF-8.
  // U+4F60 = E4 BD A0, U+597D = E5 A5 BD.
  ScopedTokenizer t("\"\xe4\xbd\xa0\xe5\xa5\xbd\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  // Column tracking: " (col 1), 你 (col 2), 好 (col 3), " (col 4).
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 5u);
  EXPECT_EQ(t.next().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, EmojiStringLiteral) {
  // "🔥" — one emoji, 4 bytes in UTF-8 (U+1F525 = F0 9F 94 A5).
  ScopedTokenizer t("\"\xf0\x9f\x94\xa5\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  // Column tracking: " (col 1), 🔥 (col 2), " (col 3).
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 4u);
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, MixedAsciiAndUtf8InString) {
  // "hello 世界 🌍" — ASCII, two CJK (3 bytes each), space, globe emoji (4
  // bytes).
  ScopedTokenizer t("\"hello \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x8c\x8d\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  // Columns: " h e l l o   世 界   🌍 "
  //           1 2 3 4 5 6 7 8  9 10 11 12
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 13u);
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, TwoByteUtf8ColumnTracking) {
  // "aä" — 'a' is 1 byte, 'ä' (U+00E4) is 2 bytes (C3 A4).
  // Columns: " (col 1), a (col 2), ä (col 3), " (col 4).
  ScopedTokenizer t("\"a\xc3\xa4\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  EXPECT_EQ(token.column, 1u);
  EXPECT_EQ(token.end_column, 5u);
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, ChineseInComment) {
  // Comment with Chinese, followed by a token to verify column tracking.
  ScopedTokenizer t(
      "// \xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c\n"
      "%x");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(token.line, 2u);
  EXPECT_EQ(token.column, 1u);
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, CommentColumnTrackingAfterContent) {
  // "42 // comment\n%x" — verify %x is at line 2, column 1.
  ScopedTokenizer t("42 // comment\n%x");
  loom_token_t num = t.next();
  EXPECT_EQ(num.kind, LOOM_TOKEN_INTEGER);
  EXPECT_EQ(num.line, 1u);
  EXPECT_EQ(num.column, 1u);
  loom_token_t val = t.next();
  EXPECT_EQ(val.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_EQ(val.line, 2u);
  EXPECT_EQ(val.column, 1u);
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, InvalidUtf8InString) {
  // 0xFF is never valid as a UTF-8 lead byte.
  ScopedTokenizer t("\"\xff\"");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, InvalidUtf8TruncatedSequence) {
  // Start of a 3-byte sequence (0xE4) followed by only 1 continuation byte,
  // then the closing quote. The closing quote is not a valid continuation byte,
  // so the sequence is invalid.
  ScopedTokenizer t("\"\xe4\xbd\"");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, InvalidUtf8OverlongSequence) {
  // Overlong encoding of '/' (U+002F): 0xC0 0xAF.
  ScopedTokenizer t("\"\xc0\xaf\"");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, InvalidUtf8SurrogateRejected) {
  // UTF-8 encoding of surrogate U+D800: 0xED 0xA0 0x80.
  ScopedTokenizer t("\"\xed\xa0\x80\"");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, InvalidUtf8BareBytes) {
  // Bare continuation byte outside a string — not valid UTF-8.
  ScopedTokenizer t("\x80");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, ValidUnicodeOutsideStringError) {
  // Chinese character outside a string — valid UTF-8 but not a valid token.
  // Should produce an error mentioning the codepoint.
  ScopedTokenizer t("\xe4\xbd\xa0");
  EXPECT_EQ(t.peek().kind, LOOM_TOKEN_EOF);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, Utf8StringFollowedByToken) {
  // Verify tokenizer recovers correctly after a UTF-8 string.
  ScopedTokenizer t("\"\xe4\xbd\xa0\" %x");
  loom_token_t str = t.next();
  EXPECT_EQ(str.kind, LOOM_TOKEN_STRING);
  loom_token_t val = t.next();
  EXPECT_EQ(val.kind, LOOM_TOKEN_SSA_VALUE);
  EXPECT_TRUE(iree_string_view_equal(val.text, IREE_SV("x")));
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

TEST(Tokenizer, Utf8InStringEscapeInteraction) {
  // Escaped quote followed by UTF-8: "\"\xe4\xbd\xa0" inside string.
  ScopedTokenizer t("\"\\\"\xe4\xbd\xa0\"");
  loom_token_t token = t.next();
  EXPECT_EQ(token.kind, LOOM_TOKEN_STRING);
  IREE_EXPECT_OK(loom_tokenizer_consume_status(t.get()));
}

}  // namespace
}  // namespace loom
