# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BACKEND domain — target codegen feedback and emission."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_BACKEND_001: Target legality rejected a codegen subject.
ERR_BACKEND_001 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=1,
    severity=Severity.ERROR,
    summary="Target legality rejected codegen input.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "rejected {subject_kind} '{subject_name}' in '@{function_name}': {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("subject_kind", ParamKind.STRING),
        ErrorParam("subject_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Specialize, decompose, or retarget {subject_kind} '{subject_name}' "
        "before backend codegen for target '{target_key}'"
    ),
)

# ERR_BACKEND_002: Target contract selection or rejection was recorded.
ERR_BACKEND_002 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=2,
    severity=Severity.REMARK,
    summary="Target contract decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} contract '{contract_key}' for '@{function_name}': {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("contract_key", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
)

# ERR_BACKEND_003: Register pressure peak was observed.
ERR_BACKEND_003 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=3,
    severity=Severity.REMARK,
    summary="Register pressure peak observed.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' peak {value_class} pressure is {peak} "
        "unit(s) against budget {budget} at block '{block_name}' "
        "near '{operation_name}'; contributors: {contributors}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("peak", ParamKind.U32),
        ErrorParam("block_name", ParamKind.STRING),
        ErrorParam("operation_name", ParamKind.STRING),
        ErrorParam("contributors", ParamKind.STRING_LIST),
    ),
)

# ERR_BACKEND_004: Register pressure exceeded a requested budget.
ERR_BACKEND_004 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=4,
    severity=Severity.WARNING,
    summary="Register pressure budget exceeded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' exceeded {value_class} pressure budget {budget} "
        "with peak {peak} in '{region_name}': {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("peak", ParamKind.U32),
        ErrorParam("region_name", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Reduce live values, unroll less, split accumulation, or select a "
        "target contract with lower {value_class} pressure"
    ),
)

# ERR_BACKEND_005: Register allocation failed.
ERR_BACKEND_005 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=5,
    severity=Severity.ERROR,
    summary="Register allocation failed.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "failed to allocate {value_class} registers for '@{function_name}' "
        "with budget {budget} and peak {peak}: {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("peak", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Lower pressure before allocation or allow spill codegen for "
        "{value_class} values"
    ),
)

# ERR_BACKEND_006: Register coalescing decision was recorded.
ERR_BACKEND_006 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=6,
    severity=Severity.REMARK,
    summary="Register coalescing decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} coalescing '{value_name}' with '{partner_value_name}' "
        "in '@{function_name}': {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("partner_value_name", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
)

# ERR_BACKEND_007: Live range split was inserted.
ERR_BACKEND_007 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=7,
    severity=Severity.REMARK,
    summary="Live range split inserted.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "split {value_class} value '{value_name}' in '@{function_name}' into "
        "{split_count} range(s): {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("split_count", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
)

# ERR_BACKEND_008: Spill traffic was predicted before allocation.
ERR_BACKEND_008 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=8,
    severity=Severity.WARNING,
    summary="Spill traffic predicted.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "predicts spilling {value_class} value '{value_name}' in "
        "'@{function_name}' for {spill_bytes} byte(s), {store_count} "
        "store(s), and {reload_count} reload(s): {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("spill_bytes", ParamKind.U32),
        ErrorParam("store_count", ParamKind.U32),
        ErrorParam("reload_count", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Use the pressure contributors for '{value_name}' to shorten the "
        "live range or choose a lower-pressure configuration"
    ),
)

# ERR_BACKEND_009: Spill/reload IR was inserted.
ERR_BACKEND_009 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=9,
    severity=Severity.WARNING,
    summary="Spill and reload operations inserted.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "inserted spill slot '{slot_name}' for {value_class} value "
        "'{value_name}' in '@{function_name}' using {spill_bytes} byte(s), "
        "{store_count} store(s), and {reload_count} reload(s)"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("slot_name", ParamKind.STRING),
        ErrorParam("spill_bytes", ParamKind.U32),
        ErrorParam("store_count", ParamKind.U32),
        ErrorParam("reload_count", ParamKind.U32),
    ),
)

# ERR_BACKEND_010: Occupancy or resource estimate was recorded.
ERR_BACKEND_010 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=10,
    severity=Severity.REMARK,
    summary="Occupancy/resource estimate recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' uses {used} of {budget} {resource_name} unit(s) "
        "with estimated occupancy {occupancy_percent}% limited by "
        "'{limiting_resource}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("resource_name", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("used", ParamKind.U32),
        ErrorParam("occupancy_percent", ParamKind.U32),
        ErrorParam("limiting_resource", ParamKind.STRING),
    ),
)

# ERR_BACKEND_011: Scheduling hazard handling was recorded.
ERR_BACKEND_011 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=11,
    severity=Severity.REMARK,
    summary="Scheduling hazard decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "handled {hazard_kind} hazard for descriptor '{descriptor_key}' in "
        "'@{function_name}' by '{inserted_action}' with cost {cycle_cost} "
        "cycle(s): {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_key", ParamKind.STRING),
        ErrorParam("hazard_kind", ParamKind.STRING),
        ErrorParam("inserted_action", ParamKind.STRING),
        ErrorParam("cycle_cost", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
)

# ERR_BACKEND_012: Artifact emission failed.
ERR_BACKEND_012 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=12,
    severity=Severity.ERROR,
    summary="Backend artifact emission failed.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "emitter '{emitter_name}' failed to emit {artifact_kind} for "
        "'@{function_name}': {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("emitter_name", ParamKind.STRING),
        ErrorParam("artifact_kind", ParamKind.STRING),
        ErrorParam("reason", ParamKind.STRING),
    ),
    fix_hint=(
        "Check target package support, descriptor encoding hooks, relocation "
        "records, and artifact metadata for emitter '{emitter_name}'"
    ),
)

# ERR_BACKEND_013: Schedule resource bottleneck estimate was recorded.
ERR_BACKEND_013 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=13,
    severity=Severity.REMARK,
    summary="Schedule resource bottleneck estimate recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' resource '{resource_name}' is a schedule "
        "bottleneck: {estimated_min_cycles} estimated cycle(s) from "
        "{total_unit_cycles} unit-cycle(s), {use_count} use(s), capacity "
        "{capacity_per_cycle}/cycle, peak {peak_units_per_cycle}/cycle, "
        "contention group {contention_group}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("resource_name", ParamKind.STRING),
        ErrorParam("capacity_per_cycle", ParamKind.U32),
        ErrorParam("contention_group", ParamKind.U32),
        ErrorParam("use_count", ParamKind.U32),
        ErrorParam("total_unit_cycles", ParamKind.U64),
        ErrorParam("estimated_min_cycles", ParamKind.U64),
        ErrorParam("peak_units_per_cycle", ParamKind.U32),
    ),
)

# ERR_BACKEND_014: Schedule hazard gap requires downstream materialization.
ERR_BACKEND_014 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=14,
    severity=Severity.REMARK,
    summary="Schedule hazard gap requires materialization.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' descriptor '{descriptor_key}' has unresolved "
        "{hazard_kind} hazard on {reference_kind} '{reference_name}': "
        "requires distance {required_distance}, schedule distance "
        "{actual_distance}, and delay/wait {required_delay} cycle(s) between "
        "packet {producer_packet} and packet {consumer_packet}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_key", ParamKind.STRING),
        ErrorParam("hazard_kind", ParamKind.STRING),
        ErrorParam("reference_kind", ParamKind.STRING),
        ErrorParam("reference_name", ParamKind.STRING),
        ErrorParam("required_distance", ParamKind.U32),
        ErrorParam("actual_distance", ParamKind.U32),
        ErrorParam("required_delay", ParamKind.U32),
        ErrorParam("producer_packet", ParamKind.U32),
        ErrorParam("consumer_packet", ParamKind.U32),
    ),
)

# ERR_BACKEND_015: Schedule candidate selection was recorded.
ERR_BACKEND_015 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=15,
    severity=Severity.REMARK,
    summary="Schedule candidate selection recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' pressure scheduler chose '{chosen_packet}' over "
        "'{rejected_packet}' at block '{block_name}' ordinal "
        "{scheduled_ordinal}: {candidate_count} ready candidate(s), chosen "
        "projected/killed/produced {chosen_projected_live_units}/"
        "{chosen_killed_live_units}/{chosen_produced_live_units}, rejected "
        "projected/killed/produced {rejected_projected_live_units}/"
        "{rejected_killed_live_units}/{rejected_produced_live_units}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("block_name", ParamKind.STRING),
        ErrorParam("scheduled_ordinal", ParamKind.U32),
        ErrorParam("candidate_count", ParamKind.U32),
        ErrorParam("chosen_packet", ParamKind.STRING),
        ErrorParam("rejected_packet", ParamKind.STRING),
        ErrorParam("chosen_projected_live_units", ParamKind.U64),
        ErrorParam("chosen_killed_live_units", ParamKind.U64),
        ErrorParam("chosen_produced_live_units", ParamKind.U64),
        ErrorParam("rejected_projected_live_units", ParamKind.U64),
        ErrorParam("rejected_killed_live_units", ParamKind.U64),
        ErrorParam("rejected_produced_live_units", ParamKind.U64),
    ),
)

# ERR_BACKEND_016: Schedule model quality was recorded.
ERR_BACKEND_016 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=16,
    severity=Severity.REMARK,
    summary="Schedule model quality recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' schedule class '{schedule_class}' uses "
        "{model_quality} model quality with {latency_kind} latency "
        "{latency_cycles} cycle(s), {issue_use_count} resource row(s), "
        "{hazard_count} hazard row(s), and {use_count} scheduled use(s)"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("schedule_class", ParamKind.STRING),
        ErrorParam("model_quality", ParamKind.STRING),
        ErrorParam("latency_kind", ParamKind.STRING),
        ErrorParam("latency_cycles", ParamKind.U32),
        ErrorParam("issue_use_count", ParamKind.U32),
        ErrorParam("hazard_count", ParamKind.U32),
        ErrorParam("use_count", ParamKind.U32),
    ),
)

# ERR_BACKEND_017: Target memory access selection was recorded.
ERR_BACKEND_017 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=17,
    severity=Severity.REMARK,
    summary="Target memory access decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} {operation_kind} memory packet '{packet_key}' for "
        "'@{function_name}' in {memory_space}: element {element_bytes} "
        "byte(s), vector lanes {vector_lanes}, dynamic stride "
        "{dynamic_stride_bytes} byte(s), vector lane stride "
        "{vector_lane_stride_bytes} byte(s), bank stride {bank_stride_words} "
        "word(s), conflict degree {bank_conflict_degree}: {reason}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("operation_kind", ParamKind.STRING),
        ErrorParam("packet_key", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("element_bytes", ParamKind.U32),
        ErrorParam("vector_lanes", ParamKind.U32),
        ErrorParam("dynamic_stride_bytes", ParamKind.U32),
        ErrorParam("vector_lane_stride_bytes", ParamKind.U32),
        ErrorParam("bank_stride_words", ParamKind.U32),
        ErrorParam("bank_conflict_degree", ParamKind.U32),
        ErrorParam("reason", ParamKind.STRING),
    ),
)

ALL_BACKEND_ERRORS: tuple[ErrorDef, ...] = (
    ERR_BACKEND_001,
    ERR_BACKEND_002,
    ERR_BACKEND_003,
    ERR_BACKEND_004,
    ERR_BACKEND_005,
    ERR_BACKEND_006,
    ERR_BACKEND_007,
    ERR_BACKEND_008,
    ERR_BACKEND_009,
    ERR_BACKEND_010,
    ERR_BACKEND_011,
    ERR_BACKEND_012,
    ERR_BACKEND_013,
    ERR_BACKEND_014,
    ERR_BACKEND_015,
    ERR_BACKEND_016,
    ERR_BACKEND_017,
)
