# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native turn-domain sine with source-defined cardinal zero signs."""

from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import F32_VGPR_MATERIALIZER
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    TypePattern,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import DescriptorSet


def sin_turns_rules(
    descriptors: DescriptorSet, source_op: Op, type_pattern: TypePattern
) -> tuple[DescriptorRule, ...]:
    sine = descriptor_by_key(descriptors, "amdgpu.v_sin_f32")
    sign = descriptor_by_key(descriptors, "amdgpu.v_and_b32.lit")
    compare = descriptor_by_key(descriptors, "amdgpu.v_cmp_oeq_f32.src0_inline")
    select = descriptor_by_key(descriptors, "amdgpu.v_cndmask_b32")
    typed_guards = tuple(
        Guard.value_type(field, type_pattern) for field in ("input", "result")
    )
    vector = type_pattern.kind == "vector"
    form = DescriptorEmitForm.PER_LANE if vector else DescriptorEmitForm.OP
    source = ValueRef.operand("input", materializer=F32_VGPR_MATERIALIZER.name)
    return (
        DescriptorRule(
            source_op=source_op,
            descriptor=sine,
            guards=(
                *typed_guards,
                Guard.instance_flags_has_all("fastmath", "nsz"),
                Guard.descriptor_available(sine),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=sine,
                    operands={"input": ValueRef.operand("input")},
                    results={"dst": ValueRef.result("result")},
                    form=form,
                ),
            ),
        ),
        DescriptorRule(
            source_op=source_op,
            descriptor=sine,
            guards=(
                *typed_guards,
                Guard.instance_flags_has_none("fastmath", "nsz"),
                *(
                    Guard.descriptor_available(item)
                    for item in (sine, sign, compare, select)
                ),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=sine,
                    operands={"input": source},
                    results={"dst": ValueRef.temporary("sine")},
                    result_types={"dst": DescriptorResultType()},
                    form=form,
                ),
                # Native sine returns +0 at negative nonzero half-integers.
                # AFN does not permit discarding the source-defined zero sign.
                EmitDescriptorOp(
                    descriptor=sign,
                    operands={"rhs": source},
                    results={"dst": ValueRef.temporary("signed_zero")},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"imm32": 0x80000000},
                    form=form,
                ),
                EmitDescriptorOp(
                    descriptor=compare,
                    operands={"rhs": ValueRef.temporary("sine")},
                    results={"mask": ValueRef.temporary("is_zero")},
                    result_types={"mask": DescriptorResultType()},
                    immediates={"lhs": 0},
                    form=form,
                ),
                EmitDescriptorOp(
                    descriptor=select,
                    operands={
                        "false_value": ValueRef.temporary("sine"),
                        "true_value": ValueRef.temporary("signed_zero"),
                        "mask": ValueRef.temporary("is_zero"),
                    },
                    results={"dst": ValueRef.result("result")},
                    form=form,
                ),
            ),
        ),
    )
