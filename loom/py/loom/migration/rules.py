# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in Loom source migration rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loom.diagnostics import DiagnosticSeverity
from loom.dialect.memory import MemorySpace
from loom.format.text.tokenizer import Token, Tokenizer, TokenKind
from loom.migration.source import (
    MigrationSourceDiagnostic,
    SourceDocument,
    SourceEdit,
)

BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID = "buffer.assume.memory_space.attr_dict"

_BUFFER_ASSUME_MEMORY_SPACE_OP = "buffer.assume.memory_space"
_MEMORY_SPACE_ATTR = "memory_space"


@dataclass(frozen=True, slots=True)
class MigrationRuleApplication:
    """Edits and diagnostics produced by one source migration rule."""

    edits: tuple[SourceEdit, ...] = ()
    diagnostics: tuple[MigrationSourceDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class MigrationRule:
    """One source migration rule registered with the driver."""

    rule_id: str
    rewrite: Callable[[SourceDocument], MigrationRuleApplication]


@dataclass(frozen=True, slots=True)
class _AssumeMemorySpaceRewrite:
    insert: SourceEdit
    remove: SourceEdit


def _migrate_buffer_assume_memory_space_attr_dict(
    document: SourceDocument,
) -> MigrationRuleApplication:
    if _BUFFER_ASSUME_MEMORY_SPACE_OP not in document.text:
        return MigrationRuleApplication()

    tokens = _tokenize_source(document)
    edits: list[SourceEdit] = []
    diagnostics: list[MigrationSourceDiagnostic] = []

    for index, token in enumerate(tokens):
        if not _is_buffer_assume_memory_space(token):
            continue
        next_token = _token_at(tokens, index + 1)
        if next_token is not None and next_token.kind == TokenKind.LANGLE:
            continue
        rewrite_or_diagnostic = _parse_buffer_assume_memory_space_attr_dict(
            document,
            tokens,
            index,
        )
        if isinstance(rewrite_or_diagnostic, MigrationSourceDiagnostic):
            diagnostics.append(rewrite_or_diagnostic)
        else:
            edits.append(rewrite_or_diagnostic.insert)
            edits.append(rewrite_or_diagnostic.remove)

    return MigrationRuleApplication(
        edits=tuple(sorted(edits, key=lambda edit: (edit.byte_start, edit.byte_end))),
        diagnostics=tuple(diagnostics),
    )


def _tokenize_source(document: SourceDocument) -> tuple[Token, ...]:
    tokenizer = Tokenizer(document.text, str(document.filename or "<input>"))
    tokens: list[Token] = []
    while not tokenizer.at(TokenKind.EOF):
        tokens.append(tokenizer.next())
    return tuple(tokens)


def _is_buffer_assume_memory_space(token: Token) -> bool:
    return (
        token.kind == TokenKind.OP_NAME and token.text == _BUFFER_ASSUME_MEMORY_SPACE_OP
    )


def _parse_buffer_assume_memory_space_attr_dict(
    document: SourceDocument,
    tokens: tuple[Token, ...],
    op_index: int,
) -> _AssumeMemorySpaceRewrite | MigrationSourceDiagnostic:
    op_token = tokens[op_index]
    operand_token = _token_at(tokens, op_index + 1)
    lbrace_token = _token_at(tokens, op_index + 2)
    attr_name_token = _token_at(tokens, op_index + 3)
    equals_token = _token_at(tokens, op_index + 4)
    attr_value_token = _token_at(tokens, op_index + 5)
    rbrace_token = _token_at(tokens, op_index + 6)
    colon_token = _token_at(tokens, op_index + 7)

    if (
        operand_token is None
        or lbrace_token is None
        or attr_name_token is None
        or equals_token is None
        or attr_value_token is None
        or rbrace_token is None
        or colon_token is None
        or operand_token.kind != TokenKind.SSA_VALUE
        or lbrace_token.kind != TokenKind.LBRACE
        or attr_name_token.kind != TokenKind.BARE_IDENT
        or attr_name_token.text != _MEMORY_SPACE_ATTR
        or equals_token.kind != TokenKind.EQUALS
        or attr_value_token.kind != TokenKind.BARE_IDENT
        or rbrace_token.kind != TokenKind.RBRACE
        or colon_token.kind != TokenKind.COLON
    ):
        return _malformed_buffer_assume_memory_space_diagnostic(document, op_token)

    if attr_value_token.text not in MemorySpace.keywords:
        return _malformed_buffer_assume_memory_space_diagnostic(
            document,
            attr_value_token,
            message=f"unknown memory space '{attr_value_token.text}'",
            fixup_hint="Use one of: " + ", ".join(MemorySpace.keywords) + ".",
        )

    removal_start = _simple_intertoken_removal_start(
        document,
        operand_token,
        lbrace_token,
    )
    if removal_start is None:
        return _malformed_buffer_assume_memory_space_diagnostic(
            document,
            lbrace_token,
            message=(
                "legacy buffer.assume.memory_space attr dict must follow the "
                "buffer operand directly"
            ),
            fixup_hint=(
                "Move comments outside the legacy attr dict form before "
                "running loom-migrate."
            ),
        )

    return _AssumeMemorySpaceRewrite(
        insert=SourceEdit(
            byte_start=document.byte_offset_at(op_token.end_location),
            byte_end=document.byte_offset_at(op_token.end_location),
            replacement_text=f"<{attr_value_token.text}>",
            rule_id=BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID,
        ),
        remove=SourceEdit(
            byte_start=removal_start,
            byte_end=document.byte_offset_at(rbrace_token.end_location),
            replacement_text="",
            rule_id=BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID,
        ),
    )


def _token_at(tokens: tuple[Token, ...], index: int) -> Token | None:
    return tokens[index] if 0 <= index < len(tokens) else None


def _simple_intertoken_removal_start(
    document: SourceDocument,
    operand_token: Token,
    lbrace_token: Token,
) -> int | None:
    operand_end = operand_token.end_location.offset
    lbrace_start = lbrace_token.location.offset
    between = document.text[operand_end:lbrace_start]
    if "\n" in between or between.strip() != "":
        return None
    return document.byte_offset_at(operand_token.end_location)


def _malformed_buffer_assume_memory_space_diagnostic(
    document: SourceDocument,
    token: Token,
    *,
    message: str | None = None,
    fixup_hint: str = ("Use buffer.assume.memory_space<global> %buffer : buffer."),
) -> MigrationSourceDiagnostic:
    return MigrationSourceDiagnostic(
        severity=DiagnosticSeverity.ERROR,
        message=message
        or (
            "expected legacy buffer.assume.memory_space attr-dict spelling "
            "or current template-parameter spelling"
        ),
        source_range=document.token_source_range(token),
        rule_id=BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID,
        fixup_hint=fixup_hint,
    )


BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE = MigrationRule(
    rule_id=BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID,
    rewrite=_migrate_buffer_assume_memory_space_attr_dict,
)

DEFAULT_MIGRATION_RULES: tuple[MigrationRule, ...] = (
    BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE,
)


__all__ = [
    "BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE",
    "BUFFER_ASSUME_MEMORY_SPACE_ATTR_DICT_RULE_ID",
    "DEFAULT_MIGRATION_RULES",
    "MigrationRule",
    "MigrationRuleApplication",
]
