# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Descriptor-backed contract emission forms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import Op
from loom.target.contracts.descriptors import (
    _immediate_has_default,
    _require_descriptor,
    _require_descriptor_operand,
    _require_immediate,
    _require_input_descriptor_role,
    _require_output_descriptor_role,
    _validate_immediate_literal,
    _validate_required_descriptor_operands,
)
from loom.target.contracts.immediates import (
    AttrProject,
    SourceMemoryProject,
    SourceMemoryProjectKind,
    ValueProject,
)
from loom.target.contracts.kinds import SourceValueKind
from loom.target.contracts.patterns import TypePattern
from loom.target.contracts.source import ValueRef
from loom.target.contracts.source_memory import SourceMemoryConstraint
from loom.target.low_descriptors import Descriptor, DescriptorSet


@unique
class DescriptorEmitForm(Enum):
    """Descriptor emission form used by the target-low lowering interpreter."""

    AUTO = "auto"
    OP = "op"
    CONST = "const"
    FIRST_LANE = "first_lane"
    PER_LANE = "per_lane"
    ACCUMULATE_LANES = "accumulate_lanes"


@unique
class DescriptorAccumulatorSeed(Enum):
    """Initial accumulator source for lane-accumulating descriptor emits."""

    OPERAND = "operand"
    FIRST_LANE = "first_lane"


@unique
class DescriptorAccumulatorTree(Enum):
    """Lane-combining tree shape for lane-accumulating descriptor emits."""

    CHAIN = "chain"
    BALANCED = "balanced"


type ResultTypeBinding = ValueRef | TypePattern


@dataclass(frozen=True, slots=True)
class EmitDescriptorOp:
    """Emits one descriptor-backed low op from source fields."""

    descriptor: Descriptor
    operands: Mapping[str, ValueRef] | None = None
    results: Mapping[str, ValueRef] | None = None
    result_types: Mapping[str, ResultTypeBinding] | None = None
    immediates: (
        Mapping[str, AttrProject | ValueProject | SourceMemoryProject | int]
        | Sequence[AttrProject]
    ) = ()
    form: DescriptorEmitForm = DescriptorEmitForm.AUTO
    swap_first_two_operands: bool = False
    copy_operands: Sequence[str] = ()
    accumulator: str | None = None
    accumulator_seed: DescriptorAccumulatorSeed = DescriptorAccumulatorSeed.OPERAND
    accumulator_tree: DescriptorAccumulatorTree = DescriptorAccumulatorTree.CHAIN
    skip_first_lane: bool = False
    source_memory: SourceMemoryConstraint | None = None

    def __post_init__(self) -> None:
        operand_bindings = self.operands if self.operands is not None else {}
        result_bindings = self.results if self.results is not None else {}
        result_type_bindings = (
            self.result_types if self.result_types is not None else None
        )
        object.__setattr__(self, "operands", dict(operand_bindings))
        object.__setattr__(self, "results", dict(result_bindings))
        object.__setattr__(
            self,
            "result_types",
            None if result_type_bindings is None else dict(result_type_bindings),
        )
        object.__setattr__(self, "copy_operands", tuple(self.copy_operands))
        for operand in self.copy_operands:
            if not operand:
                raise ValueError("copied descriptor operand name must be non-empty")
        if self.accumulator is not None and not self.accumulator:
            raise ValueError("descriptor accumulator field must be non-empty")

    def validate(
        self,
        source_op: Op,
        descriptor_set: DescriptorSet,
        defined_temporaries: set[str],
    ) -> tuple[str, ...]:
        _require_descriptor(descriptor_set, self.descriptor)
        operand_bindings = dict(self.operands) if self.operands is not None else {}
        result_bindings = dict(self.results) if self.results is not None else {}
        for descriptor_field, value_ref in operand_bindings.items():
            operand = _require_descriptor_operand(
                self.descriptor, descriptor_field, "descriptor operand binding"
            )
            _require_input_descriptor_role(self.descriptor, operand)
            value_ref.validate(
                source_op,
                f"descriptor operand '{descriptor_field}'",
                defined_temporaries=defined_temporaries,
            )
        produced_temporaries = []
        for descriptor_field, value_ref in result_bindings.items():
            operand = _require_descriptor_operand(
                self.descriptor, descriptor_field, "descriptor result binding"
            )
            _require_output_descriptor_role(self.descriptor, operand)
            if value_ref.kind not in (
                SourceValueKind.RESULT,
                SourceValueKind.TEMPORARY,
            ):
                raise ValueError(
                    f"{source_op.name}: descriptor result '{descriptor_field}' "
                    "must bind a source result or temporary"
                )
            if value_ref.kind == SourceValueKind.TEMPORARY:
                if (
                    value_ref.field in defined_temporaries
                    or value_ref.field in produced_temporaries
                ):
                    raise ValueError(
                        f"{source_op.name}: descriptor result '{descriptor_field}' "
                        f"redefines temporary '{value_ref.field}'"
                    )
                produced_temporaries.append(value_ref.field)
            else:
                value_ref.validate(source_op, f"descriptor result '{descriptor_field}'")
        result_type_bindings = (
            dict(self.result_types) if self.result_types is not None else {}
        )
        for descriptor_field, binding in result_type_bindings.items():
            _require_descriptor_operand(
                self.descriptor, descriptor_field, "descriptor result type binding"
            )
            if isinstance(binding, ValueRef):
                binding.validate(
                    source_op,
                    f"descriptor result type '{descriptor_field}'",
                    defined_temporaries=defined_temporaries,
                )
        _validate_required_descriptor_operands(
            source_op,
            self.descriptor,
            operand_bindings.keys(),
            result_bindings.keys(),
        )
        if self.swap_first_two_operands and len(operand_bindings) < 2:
            raise ValueError(
                f"{source_op.name}: descriptor operand swap needs at least two operands"
            )
        for descriptor_field in self.copy_operands:
            if descriptor_field not in operand_bindings:
                raise ValueError(
                    f"{source_op.name}: copied descriptor operand "
                    f"'{descriptor_field}' is not an operand binding"
                )
        if self.form == DescriptorEmitForm.ACCUMULATE_LANES:
            if self.accumulator is None:
                raise ValueError(
                    f"{source_op.name}: accumulate-lanes emit needs an accumulator"
                )
            if self.accumulator not in operand_bindings:
                raise ValueError(
                    f"{source_op.name}: accumulator '{self.accumulator}' "
                    "is not a descriptor operand binding"
                )
        elif self.accumulator is not None:
            raise ValueError(
                f"{source_op.name}: accumulator is only valid for "
                "accumulate-lanes emits"
            )
        if (
            self.accumulator_seed != DescriptorAccumulatorSeed.OPERAND
            and self.form != DescriptorEmitForm.ACCUMULATE_LANES
        ):
            raise ValueError(
                f"{source_op.name}: accumulator seed is only valid for "
                "accumulate-lanes emits"
            )
        if (
            self.accumulator_tree != DescriptorAccumulatorTree.CHAIN
            and self.form != DescriptorEmitForm.ACCUMULATE_LANES
        ):
            raise ValueError(
                f"{source_op.name}: accumulator tree is only valid for "
                "accumulate-lanes emits"
            )
        if self.skip_first_lane and self.form != DescriptorEmitForm.ACCUMULATE_LANES:
            raise ValueError(
                f"{source_op.name}: skip-first-lane is only valid for "
                "accumulate-lanes emits"
            )
        if (
            self.skip_first_lane
            and self.accumulator_seed == DescriptorAccumulatorSeed.FIRST_LANE
        ):
            raise ValueError(
                f"{source_op.name}: skip-first-lane cannot be combined with "
                "first-lane accumulator seeding"
            )
        if self.source_memory is not None and self.form not in (
            DescriptorEmitForm.AUTO,
            DescriptorEmitForm.OP,
        ):
            raise ValueError(f"{source_op.name}: source memory requires an op emit")
        if self.source_memory is not None:
            self.source_memory.validate(source_op)
        self._validate_immediates(source_op)
        return tuple(produced_temporaries)

    def _validate_immediates(self, source_op: Op) -> None:
        if isinstance(self.immediates, Mapping):
            bound_names = set[str]()
            for immediate_name, binding in self.immediates.items():
                immediate = _require_immediate(
                    self.descriptor,
                    immediate_name,
                    "descriptor immediate binding",
                )
                if isinstance(binding, AttrProject | ValueProject):
                    binding.validate(source_op, self.descriptor, immediate_name)
                elif isinstance(binding, SourceMemoryProject):
                    if self.source_memory is None:
                        raise ValueError(
                            f"{source_op.name}: source-memory immediate "
                            f"'{immediate_name}' needs a source-memory emit"
                        )
                    binding.validate(source_op, self.descriptor, immediate_name)
                    if (
                        binding.kind == SourceMemoryProjectKind.DYNAMIC_BYTE_STRIDE
                        and binding.dynamic_term_index
                        >= self.source_memory.dynamic_term_count
                    ):
                        raise ValueError(
                            f"{source_op.name}: source-memory immediate "
                            f"'{immediate_name}' references dynamic term "
                            f"{binding.dynamic_term_index}, but the source-memory "
                            "constraint only selects "
                            f"{self.source_memory.dynamic_term_count}"
                        )
                else:
                    _validate_immediate_literal(
                        source_op, self.descriptor, immediate, binding
                    )
                bound_names.add(immediate_name)
        else:
            bound_names = set[str]()
            for projection in self.immediates:
                projection.validate(source_op, self.descriptor, None)
                bound_names.update(projection.target_names)
        for immediate in self.descriptor.immediates:
            if _immediate_has_default(immediate):
                continue
            if immediate.field_name not in bound_names:
                raise ValueError(
                    f"{source_op.name}: descriptor '{self.descriptor.key}' "
                    f"immediate '{immediate.field_name}' is not bound"
                )
