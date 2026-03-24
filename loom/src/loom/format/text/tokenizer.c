// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/tokenizer.h"

#include <inttypes.h>
#include <stdarg.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// UTF-8 helpers
//===----------------------------------------------------------------------===//

// Advances past one UTF-8 codepoint, updating position and column.
// Returns the decoded codepoint, or IREE_UNICODE_REPLACEMENT_CHAR for
// invalid sequences (position still advances by 1 to avoid infinite loops).
static inline uint32_t loom_tokenizer_advance_utf8(loom_tokenizer_t* t) {
  uint32_t codepoint = iree_unicode_utf8_decode(t->source, &t->position);
  ++t->column;  // One codepoint = one column, regardless of byte length.
  return codepoint;
}

//===----------------------------------------------------------------------===//
// Character classification
//===----------------------------------------------------------------------===//

// Returns the character at the current position, or '\0' if at EOF.
static inline char loom_tokenizer_char(const loom_tokenizer_t* t) {
  return (t->position < t->source.size) ? t->source.data[t->position] : '\0';
}

// Returns the character at position + offset, or '\0' if out of bounds.
static inline char loom_tokenizer_char_at(const loom_tokenizer_t* t,
                                          iree_host_size_t offset) {
  iree_host_size_t index = t->position + offset;
  return (index < t->source.size) ? t->source.data[index] : '\0';
}

static inline bool loom_is_ident_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         c == '$';
}

static inline bool loom_is_ident_continue(char c) {
  return loom_is_ident_start(c) || (c >= '0' && c <= '9') || c == '.';
}

static inline bool loom_is_ident_continue_no_dot(char c) {
  return loom_is_ident_start(c) || (c >= '0' && c <= '9');
}

static inline bool loom_is_digit(char c) { return c >= '0' && c <= '9'; }

static inline bool loom_is_hex_digit(char c) {
  return loom_is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

void loom_tokenizer_initialize(iree_string_view_t source,
                               iree_string_view_t filename,
                               loom_tokenizer_t* out_tokenizer) {
  out_tokenizer->source = source;
  out_tokenizer->position = 0;
  out_tokenizer->line = 1;
  out_tokenizer->column = 1;
  out_tokenizer->peeked.kind = LOOM_TOKEN_NONE;
  out_tokenizer->filename = filename;
  out_tokenizer->status = iree_ok_status();
}

void loom_tokenizer_deinitialize(loom_tokenizer_t* tokenizer) {
  iree_status_ignore(tokenizer->status);
  tokenizer->status = iree_ok_status();
}

iree_string_view_t loom_token_kind_name(loom_token_kind_t kind) {
  switch (kind) {
    case LOOM_TOKEN_INTEGER:
      return IREE_SV("integer");
    case LOOM_TOKEN_FLOAT:
      return IREE_SV("float");
    case LOOM_TOKEN_STRING:
      return IREE_SV("string");
    case LOOM_TOKEN_SSA_VALUE:
      return IREE_SV("SSA value");
    case LOOM_TOKEN_SYMBOL:
      return IREE_SV("symbol");
    case LOOM_TOKEN_HASH_ATTR:
      return IREE_SV("hash attr");
    case LOOM_TOKEN_RESULT_ORDINAL:
      return IREE_SV("result ordinal");
    case LOOM_TOKEN_BLOCK_LABEL:
      return IREE_SV("block label");
    case LOOM_TOKEN_BARE_IDENT:
      return IREE_SV("identifier");
    case LOOM_TOKEN_OP_NAME:
      return IREE_SV("op name");
    case LOOM_TOKEN_LPAREN:
      return IREE_SV("'('");
    case LOOM_TOKEN_RPAREN:
      return IREE_SV("')'");
    case LOOM_TOKEN_LBRACE:
      return IREE_SV("'{'");
    case LOOM_TOKEN_RBRACE:
      return IREE_SV("'}'");
    case LOOM_TOKEN_LBRACKET:
      return IREE_SV("'['");
    case LOOM_TOKEN_RBRACKET:
      return IREE_SV("']'");
    case LOOM_TOKEN_LANGLE:
      return IREE_SV("'<'");
    case LOOM_TOKEN_RANGLE:
      return IREE_SV("'>'");
    case LOOM_TOKEN_EQUALS:
      return IREE_SV("'='");
    case LOOM_TOKEN_COLON:
      return IREE_SV("':'");
    case LOOM_TOKEN_COMMA:
      return IREE_SV("','");
    case LOOM_TOKEN_ARROW:
      return IREE_SV("'->'");
    case LOOM_TOKEN_PIPE:
      return IREE_SV("'|'");
    case LOOM_TOKEN_EOF:
      return IREE_SV("end of file");
    default:
      return IREE_SV("token");
  }
}

iree_status_t loom_tokenizer_consume_status(loom_tokenizer_t* tokenizer) {
  iree_status_t status = tokenizer->status;
  tokenizer->status = iree_ok_status();
  return status;
}

iree_status_t loom_tokenizer_error(const loom_tokenizer_t* tokenizer,
                                   iree_string_view_t message) {
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "%.*s:%" PRIu32 ":%" PRIu32 ": %.*s",
      (int)tokenizer->filename.size, tokenizer->filename.data, tokenizer->line,
      tokenizer->column, (int)message.size, message.data);
}

// Formatted variant of loom_tokenizer_error for messages with parameters.
IREE_PRINTF_ATTRIBUTE(2, 3)
static iree_status_t loom_tokenizer_errorf(const loom_tokenizer_t* tokenizer,
                                           const char* format, ...) {
  char message[128];
  va_list args;
  va_start(args, format);
  int length = iree_vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  if (length < 0) length = 0;
  if ((size_t)length >= sizeof(message)) length = (int)(sizeof(message) - 1);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "%.*s:%" PRIu32 ":%" PRIu32 ": %.*s",
      (int)tokenizer->filename.size, tokenizer->filename.data, tokenizer->line,
      tokenizer->column, length, message);
}

// Advances past whitespace and comments.
static void loom_tokenizer_skip_whitespace(loom_tokenizer_t* t) {
  while (t->position < t->source.size) {
    char c = t->source.data[t->position];
    if (c == ' ' || c == '\t' || c == '\r') {
      ++t->position;
      ++t->column;
    } else if (c == '\n') {
      ++t->position;
      ++t->line;
      t->column = 1;
    } else if (c == '/' && loom_tokenizer_char_at(t, 1) == '/') {
      // Line comment: skip to end of line with UTF-8-aware column tracking.
      t->position += 2;
      t->column += 2;
      while (t->position < t->source.size &&
             t->source.data[t->position] != '\n') {
        loom_tokenizer_advance_utf8(t);
      }
    } else {
      break;
    }
  }
}

// Creates a token from the range [start, t->position) in the source.
static loom_token_t loom_tokenizer_make_token(const loom_tokenizer_t* t,
                                              loom_token_kind_t kind,
                                              iree_host_size_t start,
                                              uint32_t start_line,
                                              uint32_t start_column) {
  loom_token_t token = {0};
  token.kind = kind;
  token.text =
      iree_make_string_view(t->source.data + start, t->position - start);
  token.line = start_line;
  token.column = start_column;
  token.end_column = t->column;
  return token;
}

// Scans a string literal (opening '"' already matched at position).
// The returned token text is the content without surrounding quotes
// (e.g., "hello" → hello), consistent with how other prefix tokens
// strip their sigils (%, @, #, ^).
static iree_status_t loom_tokenizer_scan_string(loom_tokenizer_t* t,
                                                loom_token_t* out_token) {
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  ++t->position;  // Skip opening '"'.
  ++t->column;
  iree_host_size_t content_start = t->position;

  while (t->position < t->source.size) {
    char c = t->source.data[t->position];
    if (c == '"') {
      // Build token from content (between the quotes), then advance
      // past the closing quote so the next scan starts after it.
      iree_host_size_t content_end = t->position;
      ++t->position;
      ++t->column;
      out_token->kind = LOOM_TOKEN_STRING;
      out_token->text = iree_make_string_view(t->source.data + content_start,
                                              content_end - content_start);
      out_token->line = start_line;
      out_token->column = start_column;
      out_token->end_column = t->column;
      return iree_ok_status();
    }
    if (c == '\\') {
      // Escape: skip the backslash and the next character.
      ++t->position;
      ++t->column;
      if (t->position >= t->source.size) break;
      // Fall through to advance past the escaped character.
    }
    // Multi-byte UTF-8: decode, validate, advance position by sequence length.
    if ((uint8_t)t->source.data[t->position] >= 0x80) {
      uint32_t codepoint = loom_tokenizer_advance_utf8(t);
      if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
        return loom_tokenizer_error(
            t, IREE_SV("invalid UTF-8 byte sequence in string literal"));
      }
      continue;
    }
    if (t->source.data[t->position] == '\n') {
      ++t->line;
      t->column = 1;
    } else {
      ++t->column;
    }
    ++t->position;
  }

  return loom_tokenizer_error(t, IREE_SV("unterminated string literal"));
}

// Scans a number (integer or float). Position is at the first digit or '-'.
static loom_token_t loom_tokenizer_scan_number(loom_tokenizer_t* t) {
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  iree_host_size_t start = t->position;

  // Optional leading '-'.
  if (loom_tokenizer_char(t) == '-') {
    ++t->position;
    ++t->column;
  }

  // Check for hex: 0x...
  if (loom_tokenizer_char(t) == '0' && loom_tokenizer_char_at(t, 1) == 'x') {
    t->position += 2;
    t->column += 2;
    while (loom_is_hex_digit(loom_tokenizer_char(t))) {
      ++t->position;
      ++t->column;
    }
    return loom_tokenizer_make_token(t, LOOM_TOKEN_INTEGER, start, start_line,
                                     start_column);
  }

  // Decimal digits.
  while (loom_is_digit(loom_tokenizer_char(t))) {
    ++t->position;
    ++t->column;
  }

  bool is_float = false;

  // Decimal point.
  if (loom_tokenizer_char(t) == '.' &&
      loom_is_digit(loom_tokenizer_char_at(t, 1))) {
    is_float = true;
    ++t->position;
    ++t->column;
    while (loom_is_digit(loom_tokenizer_char(t))) {
      ++t->position;
      ++t->column;
    }
  }

  // Exponent.
  char e = loom_tokenizer_char(t);
  if (e == 'e' || e == 'E') {
    is_float = true;
    ++t->position;
    ++t->column;
    char sign = loom_tokenizer_char(t);
    if (sign == '+' || sign == '-') {
      ++t->position;
      ++t->column;
    }
    while (loom_is_digit(loom_tokenizer_char(t))) {
      ++t->position;
      ++t->column;
    }
  }

  return loom_tokenizer_make_token(
      t, is_float ? LOOM_TOKEN_FLOAT : LOOM_TOKEN_INTEGER, start, start_line,
      start_column);
}

// Scans an identifier or op name. Position is at the first ident char.
static loom_token_t loom_tokenizer_scan_identifier(loom_tokenizer_t* t) {
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  iree_host_size_t start = t->position;
  bool has_dot = false;

  while (loom_is_ident_continue(loom_tokenizer_char(t))) {
    if (loom_tokenizer_char(t) == '.') {
      has_dot = true;
    }
    ++t->position;
    ++t->column;
  }

  loom_token_kind_t kind = has_dot ? LOOM_TOKEN_OP_NAME : LOOM_TOKEN_BARE_IDENT;
  return loom_tokenizer_make_token(t, kind, start, start_line, start_column);
}

// Scans a '%' prefixed SSA value reference. The returned token text
// is the name without the '%' prefix (e.g., '%x' → text is 'x').
// Rejects bare '%' with no identifier following.
static iree_status_t loom_tokenizer_scan_ssa_value(loom_tokenizer_t* t,
                                                   loom_token_t* out_token) {
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  ++t->position;  // Skip '%'.
  ++t->column;

  // Token text starts after the '%' prefix.
  iree_host_size_t name_start = t->position;

  while (loom_is_ident_continue_no_dot(loom_tokenizer_char(t))) {
    ++t->position;
    ++t->column;
  }

  if (t->position == name_start) {
    return loom_tokenizer_error(t, IREE_SV("expected identifier after '%'"));
  }

  *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_SSA_VALUE, name_start,
                                         start_line, start_column);
  return iree_ok_status();
}

// Scans a '@' prefixed symbol reference. The returned token text
// is the name without the leading '@' prefix (e.g., '@main' → 'main').
// Rejects bare '@' with no identifier.
static iree_status_t loom_tokenizer_scan_symbol(loom_tokenizer_t* t,
                                                loom_token_t* out_token) {
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  ++t->position;  // Skip '@'.
  ++t->column;

  // Token text starts after the '@' prefix.
  iree_host_size_t name_start = t->position;

  while (loom_is_ident_continue_no_dot(loom_tokenizer_char(t))) {
    ++t->position;
    ++t->column;
  }

  if (t->position == name_start) {
    return loom_tokenizer_error(t, IREE_SV("expected identifier after '@'"));
  }

  *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_SYMBOL, name_start,
                                         start_line, start_column);
  return iree_ok_status();
}

// Scans a '#' prefixed token (hash attr or result ordinal). The returned
// token text excludes the '#' prefix (e.g., '#q8_0' → 'q8_0', '#0' → '0').
// Rejects bare '#' with no identifier or digit following.
static iree_status_t loom_tokenizer_scan_hash(loom_tokenizer_t* t,
                                              loom_token_t* out_token) {
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  ++t->position;  // Skip '#'.
  ++t->column;

  // Token text starts after the '#' prefix.
  iree_host_size_t name_start = t->position;

  char next = loom_tokenizer_char(t);
  if (loom_is_digit(next)) {
    // Result ordinal: #0, #1.
    while (loom_is_digit(loom_tokenizer_char(t))) {
      ++t->position;
      ++t->column;
    }
    *out_token = loom_tokenizer_make_token(
        t, LOOM_TOKEN_RESULT_ORDINAL, name_start, start_line, start_column);
    return iree_ok_status();
  }

  // Hash attr: #q8_0, #enc.
  while (loom_is_ident_continue_no_dot(loom_tokenizer_char(t))) {
    ++t->position;
    ++t->column;
  }

  if (t->position == name_start) {
    return loom_tokenizer_error(t, IREE_SV("expected identifier after '#'"));
  }

  *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_HASH_ATTR, name_start,
                                         start_line, start_column);
  return iree_ok_status();
}

// Scans a '^' prefixed block label. The returned token text excludes
// the '^' prefix (e.g., '^bb0' → 'bb0'). Rejects bare '^'.
static iree_status_t loom_tokenizer_scan_block_label(loom_tokenizer_t* t,
                                                     loom_token_t* out_token) {
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  ++t->position;  // Skip '^'.
  ++t->column;

  // Token text starts after the '^' prefix.
  iree_host_size_t name_start = t->position;

  while (loom_is_ident_continue_no_dot(loom_tokenizer_char(t))) {
    ++t->position;
    ++t->column;
  }

  if (t->position == name_start) {
    return loom_tokenizer_error(t, IREE_SV("expected identifier after '^'"));
  }

  *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_BLOCK_LABEL, name_start,
                                         start_line, start_column);
  return iree_ok_status();
}

// Scans the next token from the source. Returns an error for malformed
// input (e.g., unterminated strings).
static iree_status_t loom_tokenizer_scan(loom_tokenizer_t* t,
                                         loom_token_t* out_token) {
  loom_tokenizer_skip_whitespace(t);

  if (t->position >= t->source.size) {
    out_token->kind = LOOM_TOKEN_EOF;
    out_token->text = iree_string_view_empty();
    out_token->line = t->line;
    out_token->column = t->column;
    return iree_ok_status();
  }

  char c = t->source.data[t->position];
  uint32_t start_line = t->line;
  uint32_t start_column = t->column;
  iree_host_size_t start = t->position;

  // Single-character punctuation.
  switch (c) {
    case '(':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_LPAREN, start,
                                             start_line, start_column);
      return iree_ok_status();
    case ')':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_RPAREN, start,
                                             start_line, start_column);
      return iree_ok_status();
    case '{':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_LBRACE, start,
                                             start_line, start_column);
      return iree_ok_status();
    case '}':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_RBRACE, start,
                                             start_line, start_column);
      return iree_ok_status();
    case '[':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_LBRACKET, start,
                                             start_line, start_column);
      return iree_ok_status();
    case ']':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_RBRACKET, start,
                                             start_line, start_column);
      return iree_ok_status();
    case '<':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_LANGLE, start,
                                             start_line, start_column);
      return iree_ok_status();
    case '>':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_RANGLE, start,
                                             start_line, start_column);
      return iree_ok_status();
    case '=':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_EQUALS, start,
                                             start_line, start_column);
      return iree_ok_status();
    case ',':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_COMMA, start,
                                             start_line, start_column);
      return iree_ok_status();
    case '|':
      ++t->position;
      ++t->column;
      *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_PIPE, start,
                                             start_line, start_column);
      return iree_ok_status();
    default:
      break;
  }

  // Multi-character tokens.
  if (c == '-' && loom_tokenizer_char_at(t, 1) == '>') {
    t->position += 2;
    t->column += 2;
    *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_ARROW, start,
                                           start_line, start_column);
    return iree_ok_status();
  }

  if (c == ':') {
    ++t->position;
    ++t->column;
    *out_token = loom_tokenizer_make_token(t, LOOM_TOKEN_COLON, start,
                                           start_line, start_column);
    return iree_ok_status();
  }

  // Negative number: '-' followed by digit.
  if (c == '-' && loom_is_digit(loom_tokenizer_char_at(t, 1))) {
    *out_token = loom_tokenizer_scan_number(t);
    return iree_ok_status();
  }

  // Number.
  if (loom_is_digit(c)) {
    *out_token = loom_tokenizer_scan_number(t);
    return iree_ok_status();
  }

  // String literal.
  if (c == '"') {
    return loom_tokenizer_scan_string(t, out_token);
  }

  // SSA value.
  if (c == '%') {
    return loom_tokenizer_scan_ssa_value(t, out_token);
  }

  // Symbol.
  if (c == '@') {
    return loom_tokenizer_scan_symbol(t, out_token);
  }

  // Hash (attr or result ordinal).
  if (c == '#') {
    return loom_tokenizer_scan_hash(t, out_token);
  }

  // Block label.
  if (c == '^') {
    return loom_tokenizer_scan_block_label(t, out_token);
  }

  // Identifier or op name.
  if (loom_is_ident_start(c)) {
    *out_token = loom_tokenizer_scan_identifier(t);
    return iree_ok_status();
  }

  // Non-ASCII: attempt UTF-8 decode for a diagnostic error message.
  // Valid UTF-8 outside strings/comments is still an error — identifiers are
  // ASCII-only — but we report the codepoint for clarity.
  if ((uint8_t)c >= 0x80) {
    iree_host_size_t saved_position = t->position;
    uint32_t codepoint = iree_unicode_utf8_decode(t->source, &t->position);
    t->position = saved_position;  // Don't consume — we're about to error.
    if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
      return loom_tokenizer_error(t, IREE_SV("invalid UTF-8 byte sequence"));
    }
    return loom_tokenizer_errorf(t, "unexpected Unicode character U+%04" PRIX32,
                                 codepoint);
  }

  return loom_tokenizer_error(t, IREE_SV("unexpected character"));
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

loom_token_t loom_tokenizer_peek(loom_tokenizer_t* tokenizer) {
  if (tokenizer->peeked.kind == LOOM_TOKEN_NONE) {
    // If a previous scan already failed, keep returning EOF.
    if (!iree_status_is_ok(tokenizer->status)) {
      tokenizer->peeked.kind = LOOM_TOKEN_EOF;
      tokenizer->peeked.text = iree_string_view_empty();
      tokenizer->peeked.line = tokenizer->line;
      tokenizer->peeked.column = tokenizer->column;
      return tokenizer->peeked;
    }
    iree_status_t status = loom_tokenizer_scan(tokenizer, &tokenizer->peeked);
    if (!iree_status_is_ok(status)) {
      // Store the error for the caller to retrieve.
      tokenizer->status = status;
      tokenizer->peeked.kind = LOOM_TOKEN_EOF;
      tokenizer->peeked.text = iree_string_view_empty();
      tokenizer->peeked.line = tokenizer->line;
      tokenizer->peeked.column = tokenizer->column;
    }
  }
  return tokenizer->peeked;
}

loom_token_t loom_tokenizer_next(loom_tokenizer_t* tokenizer) {
  loom_token_t token = loom_tokenizer_peek(tokenizer);
  tokenizer->peeked.kind = LOOM_TOKEN_NONE;
  tokenizer->consumed_end_line = token.line;
  tokenizer->consumed_end_column = token.end_column;
  return token;
}

bool loom_tokenizer_at(loom_tokenizer_t* tokenizer, loom_token_kind_t kind) {
  return loom_tokenizer_peek(tokenizer).kind == kind;
}

bool loom_tokenizer_at_keyword(loom_tokenizer_t* tokenizer,
                               iree_string_view_t text) {
  loom_token_t token = loom_tokenizer_peek(tokenizer);
  return token.kind == LOOM_TOKEN_BARE_IDENT &&
         iree_string_view_equal(token.text, text);
}

bool loom_tokenizer_try_consume(loom_tokenizer_t* tokenizer,
                                loom_token_kind_t kind) {
  loom_token_t token = loom_tokenizer_peek(tokenizer);
  if (token.kind == kind) {
    tokenizer->peeked.kind = LOOM_TOKEN_NONE;
    tokenizer->consumed_end_line = token.line;
    tokenizer->consumed_end_column = token.end_column;
    return true;
  }
  return false;
}

bool loom_tokenizer_try_consume_keyword(loom_tokenizer_t* tokenizer,
                                        iree_string_view_t text) {
  loom_token_t token = loom_tokenizer_peek(tokenizer);
  if (token.kind == LOOM_TOKEN_BARE_IDENT &&
      iree_string_view_equal(token.text, text)) {
    tokenizer->peeked.kind = LOOM_TOKEN_NONE;
    tokenizer->consumed_end_line = token.line;
    tokenizer->consumed_end_column = token.end_column;
    return true;
  }
  return false;
}

iree_status_t loom_tokenizer_scan_angle_interior(
    loom_tokenizer_t* tokenizer, iree_string_view_t* out_interior) {
  iree_host_size_t start = tokenizer->position;
  int depth = 1;
  while (tokenizer->position < tokenizer->source.size && depth > 0) {
    char c = tokenizer->source.data[tokenizer->position];
    if (c == '<') {
      ++depth;
      ++tokenizer->column;
    } else if (c == '>') {
      --depth;
      if (depth == 0) {
        *out_interior = iree_make_string_view(tokenizer->source.data + start,
                                              tokenizer->position - start);
        ++tokenizer->position;
        ++tokenizer->column;
        tokenizer->consumed_end_line = tokenizer->line;
        tokenizer->consumed_end_column = tokenizer->column;
        return iree_ok_status();
      }
      ++tokenizer->column;
    } else if (c == '"') {
      // Skip string literals inside angle brackets.
      ++tokenizer->position;
      ++tokenizer->column;
      while (tokenizer->position < tokenizer->source.size) {
        char sc = tokenizer->source.data[tokenizer->position];
        if (sc == '"') {
          ++tokenizer->position;
          ++tokenizer->column;
          break;
        }
        if (sc == '\\') {
          ++tokenizer->position;
          ++tokenizer->column;
          if (tokenizer->position >= tokenizer->source.size) break;
        }
        if ((uint8_t)tokenizer->source.data[tokenizer->position] >= 0x80) {
          uint32_t codepoint = loom_tokenizer_advance_utf8(tokenizer);
          if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
            return loom_tokenizer_error(
                tokenizer,
                IREE_SV("invalid UTF-8 byte sequence in string literal"));
          }
          continue;
        }
        if (tokenizer->source.data[tokenizer->position] == '\n') {
          ++tokenizer->line;
          tokenizer->column = 1;
        } else {
          ++tokenizer->column;
        }
        ++tokenizer->position;
      }
      continue;
    } else if (c == '\n') {
      ++tokenizer->line;
      tokenizer->column = 1;
    } else if ((uint8_t)c >= 0x80) {
      // Multi-byte UTF-8 outside a string in angle interior.
      uint32_t codepoint = loom_tokenizer_advance_utf8(tokenizer);
      if (codepoint == IREE_UNICODE_REPLACEMENT_CHAR) {
        return loom_tokenizer_error(tokenizer,
                                    IREE_SV("invalid UTF-8 byte sequence"));
      }
      continue;  // Position already advanced by decode; skip ++position.
    } else {
      ++tokenizer->column;
    }
    ++tokenizer->position;
  }
  return loom_tokenizer_error(tokenizer, IREE_SV("unterminated angle bracket"));
}
