# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Translation-unit configuration projection into Loom C++."""

from __future__ import annotations

import json

from loom.format.cxx.errors import CxxRenderError
from loom.format.cxx.kernel_configurations import CxxKernelConfigurationRenderer
from loom.format.cxx.naming import clean_identifier
from loom.ir import (
    EncodingInstance,
    EncodingRole,
    EncodingType,
    Operation,
    Predicate,
    PredicateListAttr,
    ScalarType,
)


class CxxConfigRenderer(CxxKernelConfigurationRenderer):
    """Renders reachable config symbols and binds their reads."""

    def _plan_config_names(self) -> None:
        used_names = self.translation_unit_names
        symbols: list[str] = []
        for operation in self.module.body.ops:
            if operation.name not in {"config.decl", "config.def"}:
                continue
            if id(operation) in self.projected_config_declarations:
                continue
            symbol = operation.attributes.get("symbol")
            if not isinstance(symbol, str):
                raise CxxRenderError(operation, "config symbol is not a string")
            symbols.append(symbol)

        components = {
            symbol: [
                clean_identifier(component, "config") for component in symbol.split(".")
            ]
            for symbol in symbols
        }

        def suffix(symbol: str, width: int) -> str:
            return "_".join(components[symbol][-width:])

        for symbol in symbols:
            base = suffix(symbol, len(components[symbol]))
            for width in range(1, len(components[symbol]) + 1):
                candidate = suffix(symbol, width)
                if candidate in used_names or any(
                    other != symbol
                    and suffix(other, min(width, len(components[other]))) == candidate
                    for other in symbols
                ):
                    continue
                base = candidate
                break
            name = base
            ordinal = 2
            while name in used_names:
                name = f"{base}_{ordinal}"
                ordinal += 1
            used_names.add(name)
            self.config_names[symbol] = name

    def _config_source_type(self, operation: Operation) -> str:
        if len(operation.results) != 1:
            raise CxxRenderError(operation, "config symbol must define one value")
        result = operation.results[0]
        value_type = self.module.values[result].type
        if not isinstance(value_type, (EncodingType, ScalarType)):
            raise CxxRenderError(
                operation,
                f"config type {value_type!r} has no C++ global projection",
            )
        if isinstance(value_type, EncodingType) and value_type.role not in {
            EncodingRole.SCHEMA,
            EncodingRole.TRANSFORM,
        }:
            raise CxxRenderError(
                operation,
                "ranked layout and storage configs require a source rank "
                "that is not present in their Loom value type",
            )
        return self.cpp_type(result)

    def _config_predicate(self, operation: Operation, predicate: Predicate) -> str:
        subject = operation.results[0]
        if (
            not predicate.args
            or predicate.args[0].tag != "value"
            or predicate.args[0].value != subject
            or any(argument.tag != "const" for argument in predicate.args[1:])
        ):
            raise CxxRenderError(
                operation,
                "config predicate is not a subject with constant arguments",
            )
        arguments = ", ".join(
            self._config_predicate_constant(argument.value, subject)
            for argument in predicate.args[1:]
        )
        return f"loom::predicate::{predicate.kind}({arguments})"

    def _config_predicate_constant(self, value: int, subject: int) -> str:
        value_type = self.module.values[subject].type
        if not isinstance(value_type, ScalarType):
            raise CxxRenderError(
                None, "encoding config predicates cannot carry constants"
            )
        type_name = repr(value_type)
        if type_name == "i1":
            if value not in {0, 1}:
                raise CxxRenderError(
                    None, "Boolean config predicate constant is not zero or one"
                )
            return "true" if value else "false"
        if type_name.startswith("f") or type_name == "bf16":
            return self._literal(float(value), subject)
        if type_name in {"i8", "i16"}:
            return f"static_cast<{self.cpp_type(subject)}>({value})"
        if type_name == "i64" and subject in self.signed_values:
            if value == -(1 << 63):
                return "(-9223372036854775807ll - 1ll)"
            return f"{value}ll"
        return self._predicate_constant(value, subject)

    def _config_attributes(self, operation: Operation) -> str:
        symbol = operation.attributes.get("symbol")
        if not isinstance(symbol, str):
            raise CxxRenderError(operation, "config symbol is not a string")
        attributes = [f"loom::config({json.dumps(symbol)})"]
        predicates = operation.attributes.get("predicates")
        if predicates is not None:
            if not isinstance(predicates, PredicateListAttr) or not predicates:
                raise CxxRenderError(
                    operation, "config predicates are not a nonempty predicate list"
                )
            expression = " && ".join(
                self._config_predicate(operation, predicate) for predicate in predicates
            )
            attributes.append(f"loom::where({expression})")
        return "[[" + ", ".join(attributes) + "]]"

    def _render_configs(self) -> bool:
        rendered = False
        for operation in self.module.body.ops:
            if operation.name not in {"config.decl", "config.def"}:
                continue
            if id(operation) in self.projected_config_declarations:
                continue
            symbol = operation.attributes.get("symbol")
            if not isinstance(symbol, str) or symbol not in self.config_names:
                raise CxxRenderError(operation, "config has no planned C++ name")
            self._validate_projected_attributes(
                operation,
                ("symbol", "predicates")
                if operation.name == "config.decl"
                else ("symbol", "value", "predicates"),
            )
            type_name = self._config_source_type(operation)
            name = self.config_names[symbol]
            self.writer.line(self._config_attributes(operation))
            if operation.name == "config.decl":
                self.writer.line(f"extern const {type_name} {name};")
            else:
                value = operation.attributes.get("value")
                if isinstance(value, EncodingInstance):
                    raise CxxRenderError(
                        operation,
                        "encoding config definitions have no C++ constant "
                        "initializer projection",
                    )
                self.writer.line(
                    f"const {type_name} {name} = "
                    f"{self._literal(value, operation.results[0])};"
                )
            self.rendered_config_operations.add(id(operation))
            rendered = True
        return rendered

    def _render_config_get(self, operation: Operation) -> tuple[str, ...]:
        symbol = operation.attributes.get("config")
        if not isinstance(symbol, str):
            raise CxxRenderError(operation, "config.get symbol is not a string")
        declaration = self.config_symbols.get(symbol)
        name = self.config_names.get(symbol)
        if declaration is None or name is None:
            raise CxxRenderError(
                operation,
                f"config.get symbol {symbol!r} has no projected global declaration",
            )
        if len(operation.results) != 1 or len(declaration.results) != 1:
            raise CxxRenderError(operation, "config.get must produce one value")
        result = operation.results[0]
        if (
            self.module.values[result].type
            != self.module.values[declaration.results[0]].type
        ):
            raise CxxRenderError(
                operation, "config.get type differs from its global declaration"
            )
        self.expressions[result] = name
        self.expression_types[result] = self.cpp_type(declaration.results[0])
        return ("config",)
