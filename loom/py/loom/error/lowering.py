# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LOWERING domain — pass legality and unsupported mappings."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_LOWERING_015: Low pure call targets an impure callee.
ERR_LOWERING_015 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=15,
    severity=Severity.ERROR,
    summary="Low pure call targets an impure callee.",
    message=(
        "low pure call callee '@{callee_name}' through '{boundary_name}' "
        "does not have a pure contract"
    ),
    params=(
        ErrorParam("callee_name", ParamKind.STRING),
        ErrorParam("boundary_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Remove the pure marker, mark the direct callee pure after proving its "
        "body has no observable effects, or route the call through a boundary "
        "whose contract is explicitly pure"
    ),
)

# ERR_LOWERING_019: Low operation reads an undefined register part.
ERR_LOWERING_019 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=19,
    severity=Severity.ERROR,
    summary="Low operation reads an undefined register part.",
    message=(
        "low function '@{function_name}' op '{op_name}' operand "
        "'{field_name}' requires register part mask {required_mask}, but "
        "the value defines only mask {defined_mask}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("field_name", ParamKind.STRING),
        ErrorParam("required_mask", ParamKind.U64),
        ErrorParam("defined_mask", ParamKind.U64),
    ),
    fix_hint=(
        "Compose the missing register part with an explicitly tied partial "
        "write before using the value as a full-register operand"
    ),
)

# ERR_LOWERING_020: Static vector scalarization lane count is not representable.
ERR_LOWERING_020 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=20,
    severity=Severity.ERROR,
    summary="Static vector scalarization lane count is not representable.",
    message=(
        "{op_name} cannot be lowered by {pass_name} because static vector type "
        "{vector_type} has more lanes than scalarization can represent"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("vector_type", ParamKind.TYPE),
    ),
    fix_hint=(
        "Refine the vector shape before scalarization or lower it with a "
        "target primitive that preserves the vector aggregate"
    ),
)

# ERR_LOWERING_022: Kernel async group has no wait in the current stream.
ERR_LOWERING_022 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=22,
    severity=Severity.ERROR,
    summary="Kernel async group has no wait in the current stream.",
    message=(
        "{phase_name} requires {op_name} to be waited in the current "
        "straight-line async stream"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the async group in the same straight-line stream or lower it "
        "with a pipeline-aware async strategy"
    ),
)

# ERR_LOWERING_023: Kernel async group is carried outside the stream.
ERR_LOWERING_023 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=23,
    severity=Severity.ERROR,
    summary="Kernel async group is carried outside the stream.",
    message=(
        "{phase_name} cannot lower {op_name} whose group value has a non-wait use"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Keep async groups in a straight-line group/wait stream or run a "
        "pipeline-aware legality path before lowering"
    ),
)

# ERR_LOWERING_024: Kernel async movement cannot be described.
ERR_LOWERING_024 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=24,
    severity=Severity.ERROR,
    summary="Kernel async movement cannot be described.",
    message=(
        "{phase_name} cannot describe the movement endpoints for {op_name}; "
        "movement rejection bits are {rejection_bits}"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("rejection_bits", ParamKind.U64),
    ),
    fix_hint=(
        "Refine async transfer operands so their source and destination view "
        "regions can be described by movement analysis"
    ),
)

# ERR_LOWERING_025: Kernel async token producer is not an async view movement.
ERR_LOWERING_025 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=25,
    severity=Severity.ERROR,
    summary="Kernel async token producer is not an async view movement.",
    message=(
        "{phase_name} requires the token producer for {op_name} to be an async "
        "view movement"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Group only tokens produced by kernel async transfer operations whose "
        "destination is a view"
    ),
)

# ERR_LOWERING_026: Kernel async destination overlaps a pending destination.
ERR_LOWERING_026 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=26,
    severity=Severity.ERROR,
    summary="Kernel async destination overlaps a pending destination.",
    message=(
        "{phase_name} found {op_name} whose destination may overlap an earlier "
        "uncompleted async destination"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the earlier async group before issuing an overlapping async "
        "destination or prove the destination views are disjoint"
    ),
)

# ERR_LOWERING_027: Synchronous write overlaps a pending async destination.
ERR_LOWERING_027 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=27,
    severity=Severity.ERROR,
    summary="Synchronous write overlaps a pending async destination.",
    message=(
        "{phase_name} found {op_name} writing a view that may overlap a "
        "pending async destination before wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the pending async group before the synchronous write or prove the "
        "written view is disjoint"
    ),
)

# ERR_LOWERING_028: Synchronous read overlaps a pending async destination.
ERR_LOWERING_028 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=28,
    severity=Severity.ERROR,
    summary="Synchronous read overlaps a pending async destination.",
    message=(
        "{phase_name} found {op_name} reading a view that may observe a "
        "pending async destination before wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the pending async group before the synchronous read or prove the "
        "read view is disjoint"
    ),
)

# ERR_LOWERING_029: Kernel async wait references an uncommitted group.
ERR_LOWERING_029 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=29,
    severity=Severity.ERROR,
    summary="Kernel async wait references an uncommitted group.",
    message=(
        "{phase_name} requires {op_name} to wait a group committed in the "
        "current straight-line async stream"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Move the wait after the matching kernel.async.group in the same block "
        "or use a pipeline-aware async lowering path"
    ),
)

# ERR_LOWERING_030: Kernel async wait references an already completed group.
ERR_LOWERING_030 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=30,
    severity=Severity.ERROR,
    summary="Kernel async wait references an already completed group.",
    message=(
        "{phase_name} found {op_name} waiting an async group already completed "
        "by an earlier wait"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Remove the duplicate wait or wait a younger group that is still outstanding"
    ),
)

# ERR_LOWERING_031: Kernel async wait count does not match stream depth.
ERR_LOWERING_031 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=31,
    severity=Severity.ERROR,
    summary="Kernel async wait count does not match stream depth.",
    message=(
        "{phase_name} found {op_name} with newer_groups {actual_newer_groups}, "
        "but {expected_newer_groups} younger async groups remain outstanding"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
        ErrorParam("actual_newer_groups", ParamKind.I64),
        ErrorParam("expected_newer_groups", ParamKind.U64),
    ),
    fix_hint=(
        "Set newer_groups to the number of younger uncompleted groups that "
        "remain after this wait"
    ),
)

# ERR_LOWERING_032: Kernel async group leaves a block before wait.
ERR_LOWERING_032 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=32,
    severity=Severity.ERROR,
    summary="Kernel async group leaves a block before wait.",
    message=("{phase_name} requires {op_name} to be waited before leaving its block"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Wait the async group along every control-flow path before leaving the "
        "block or move the async stream into a pipeline-aware region"
    ),
)

# ERR_LOWERING_033: Kernel async group token has no producer op.
ERR_LOWERING_033 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=33,
    severity=Severity.ERROR,
    summary="Kernel async group token has no producer op.",
    message=(
        "{phase_name} requires every token operand of {op_name} to be produced "
        "by a local async transfer op"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("phase_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Commit only tokens produced in the current function by kernel async "
        "transfer operations"
    ),
)

# ERR_LOWERING_034: Vector transform permutation is not statically proven.
ERR_LOWERING_034 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=34,
    severity=Severity.ERROR,
    summary="Vector transform permutation is not statically proven.",
    message=("{pass_name} requires {op_name} permutation lanes to be statically exact"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the permutation lane facts before vector scalarization or "
        "specialize the transform through a target primitive"
    ),
)

# ERR_LOWERING_035: Vector transform permutation repeats a source lane.
ERR_LOWERING_035 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=35,
    severity=Severity.ERROR,
    summary="Vector transform permutation repeats a source lane.",
    message=(
        "{pass_name} requires {op_name} permutation to name each source lane "
        "once per last-axis slice"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a bijective permutation for each last-axis slice before vector "
        "scalarization"
    ),
)

# ERR_LOWERING_036: Vector transform permutation source lane is out of bounds.
ERR_LOWERING_036 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=36,
    severity=Severity.ERROR,
    summary="Vector transform permutation source lane is out of bounds.",
    message=(
        "{pass_name} requires {op_name} permutation lanes to reference "
        "in-bounds source lanes"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint="Clamp or specialize permutation lanes to the source last-axis extent",
)

# ERR_LOWERING_037: SCF to CFG requires a positive counted-loop step.
ERR_LOWERING_037 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=37,
    severity=Severity.ERROR,
    summary="SCF to CFG requires a positive counted-loop step.",
    message="{pass_name} requires {op_name} step to be fact-proven positive",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the loop step facts or normalize the loop before converting "
        "structured control flow to CFG"
    ),
)

# ERR_LOWERING_038: SCF to CFG cannot preserve tied result ownership.
ERR_LOWERING_038 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=38,
    severity=Severity.ERROR,
    summary="SCF to CFG cannot preserve tied result ownership.",
    message="{pass_name} cannot preserve tied result ownership on {op_name}",
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Lower ownership transfers before CFG conversion or keep the "
        "structured op until CFG block arguments can model the transfer"
    ),
)

# ERR_LOWERING_039: Vector transform descriptor is not locally decodable.
ERR_LOWERING_039 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=39,
    severity=Severity.ERROR,
    summary="Vector transform descriptor is not locally decodable.",
    message=(
        "{pass_name} requires {op_name} transform to be a local "
        "#numeric_transform encoding.define"
    ),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Specialize the transform descriptor into the function before vector "
        "scalarization or lower the transform through a target primitive"
    ),
)

# ERR_LOWERING_040: Vector transform last-axis extent is dynamic.
ERR_LOWERING_040 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=40,
    severity=Severity.ERROR,
    summary="Vector transform last-axis extent is dynamic.",
    message=("{pass_name} requires {op_name} last-axis transform extent to be static"),
    params=(
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("pass_name", ParamKind.STRING),
    ),
    fix_hint=(
        "Refine the transform last-axis extent before vector scalarization or "
        "lower the transform through a target primitive"
    ),
)

# ERR_LOWERING_043: Boundary fact refinement did not converge.
ERR_LOWERING_043 = ErrorDef(
    domain=ErrorDomain.LOWERING,
    code=43,
    severity=Severity.ERROR,
    summary="Boundary fact refinement did not converge.",
    message=(
        "{pass_name} did not converge boundary facts in {max_iterations} iteration(s)"
    ),
    params=(
        ErrorParam("pass_name", ParamKind.STRING),
        ErrorParam("max_iterations", ParamKind.U32),
    ),
    fix_hint=(
        "Specialize recursive/SCC summaries or raise the pass iteration limit "
        "only after proving additional iterations are bounded"
    ),
)

ALL_LOWERING_ERRORS: tuple[ErrorDef, ...] = (
    ERR_LOWERING_015,
    ERR_LOWERING_019,
    ERR_LOWERING_020,
    ERR_LOWERING_022,
    ERR_LOWERING_023,
    ERR_LOWERING_024,
    ERR_LOWERING_025,
    ERR_LOWERING_026,
    ERR_LOWERING_027,
    ERR_LOWERING_028,
    ERR_LOWERING_029,
    ERR_LOWERING_030,
    ERR_LOWERING_031,
    ERR_LOWERING_032,
    ERR_LOWERING_033,
    ERR_LOWERING_034,
    ERR_LOWERING_035,
    ERR_LOWERING_036,
    ERR_LOWERING_037,
    ERR_LOWERING_038,
    ERR_LOWERING_039,
    ERR_LOWERING_040,
    ERR_LOWERING_043,
)
