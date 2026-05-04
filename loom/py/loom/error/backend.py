# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BACKEND domain — target codegen feedback and emission."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_BACKEND_001: Deprecated catch-all target legality rejection.
#
# Do not add new uses. Target-owned legality failures need domain-specific
# diagnostics with stable error identities and typed parameters.
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

# ERR_BACKEND_005: Register allocation failed.
ERR_BACKEND_005 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=5,
    severity=Severity.ERROR,
    summary="Register allocation failed.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "failed to allocate {value_class} registers for '@{function_name}' "
        "with budget {budget}, peak {peak}, and failure '{failure_kind}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("peak", ParamKind.U32),
        ErrorParam("failure_kind", ParamKind.STRING),
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
        "in '@{function_name}' with constraint '{coalescing_constraint}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("partner_value_name", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("coalescing_constraint", ParamKind.STRING),
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
        "store(s), {reload_count} reload(s), and cause '{spill_cause}'"
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
        ErrorParam("spill_cause", ParamKind.STRING),
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
        "inserted spill storage '{storage_name}' for {value_class} value "
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
        ErrorParam("storage_name", ParamKind.STRING),
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
        "'@{function_name}' scheduler chose '{chosen_packet}' over "
        "'{rejected_packet}' at block '{block_name}' ordinal "
        "{scheduled_ordinal}: {candidate_count} ready candidate(s), chosen "
        "dep-latency/latency/projected/killed/produced "
        "{chosen_dependency_latency_cycles}/{chosen_latency_cycles}/"
        "{chosen_projected_live_units}/"
        "{chosen_killed_live_units}/{chosen_produced_live_units}, chosen "
        "cliff class/units/penalty/next "
        "{chosen_pressure_cliff_reg_class_id}/{chosen_pressure_cliff_units}/"
        "{chosen_pressure_cliff_penalty}/"
        "{chosen_units_until_pressure_cliff}, rejected "
        "dep-latency/latency/projected/killed/produced "
        "{rejected_dependency_latency_cycles}/{rejected_latency_cycles}/"
        "{rejected_projected_live_units}/"
        "{rejected_killed_live_units}/{rejected_produced_live_units}, rejected "
        "cliff class/units/penalty/next "
        "{rejected_pressure_cliff_reg_class_id}/"
        "{rejected_pressure_cliff_units}/{rejected_pressure_cliff_penalty}/"
        "{rejected_units_until_pressure_cliff}"
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
        ErrorParam("chosen_dependency_latency_cycles", ParamKind.U32),
        ErrorParam("chosen_latency_cycles", ParamKind.U32),
        ErrorParam("chosen_projected_live_units", ParamKind.U64),
        ErrorParam("chosen_killed_live_units", ParamKind.U64),
        ErrorParam("chosen_produced_live_units", ParamKind.U64),
        ErrorParam("chosen_pressure_cliff_reg_class_id", ParamKind.U32),
        ErrorParam("chosen_pressure_cliff_units", ParamKind.U32),
        ErrorParam("chosen_pressure_cliff_penalty", ParamKind.U32),
        ErrorParam("chosen_units_until_pressure_cliff", ParamKind.U32),
        ErrorParam("rejected_dependency_latency_cycles", ParamKind.U32),
        ErrorParam("rejected_latency_cycles", ParamKind.U32),
        ErrorParam("rejected_projected_live_units", ParamKind.U64),
        ErrorParam("rejected_killed_live_units", ParamKind.U64),
        ErrorParam("rejected_produced_live_units", ParamKind.U64),
        ErrorParam("rejected_pressure_cliff_reg_class_id", ParamKind.U32),
        ErrorParam("rejected_pressure_cliff_units", ParamKind.U32),
        ErrorParam("rejected_pressure_cliff_penalty", ParamKind.U32),
        ErrorParam("rejected_units_until_pressure_cliff", ParamKind.U32),
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
        "word(s), conflict degree {bank_conflict_degree}, bank conflict "
        "'{bank_conflict_kind}'"
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
        ErrorParam("bank_conflict_kind", ParamKind.STRING),
    ),
)

# ERR_BACKEND_018: Target-low packet selection was recorded.
ERR_BACKEND_018 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=18,
    severity=Severity.REMARK,
    summary="Target-low packet decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} {packet_category} low packet '{packet_key}' for "
        "'@{function_name}': operands {operand_count}, results "
        "{result_count}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("packet_category", ParamKind.STRING),
        ErrorParam("packet_key", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("operand_count", ParamKind.U32),
        ErrorParam("result_count", ParamKind.U32),
    ),
)

ALL_BACKEND_ERRORS: tuple[ErrorDef, ...] = (
    ERR_BACKEND_001,
    ERR_BACKEND_003,
    ERR_BACKEND_005,
    ERR_BACKEND_006,
    ERR_BACKEND_008,
    ERR_BACKEND_009,
    ERR_BACKEND_010,
    ERR_BACKEND_013,
    ERR_BACKEND_014,
    ERR_BACKEND_015,
    ERR_BACKEND_016,
    ERR_BACKEND_017,
    ERR_BACKEND_018,
)
