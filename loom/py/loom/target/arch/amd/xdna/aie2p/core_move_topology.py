# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete native allocation-move topology for the AIE2P core."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.amd.xdna.aie.schedule import itinerary_payload
from loom.target.arch.amd.xdna.aie2p.core_machine_data import (
    CORE_MACHINE_TABLE,
    DIMENSION_FIELDS,
)
from loom.target.arch.amd.xdna.aie2p.core_schedule_data import CORE_SCHEDULE_TABLE

_SCALAR_MOVE_ITINERARY_PREFIX = "II_MOV_alu_mv_mv_mv_scl_"
_SCALAR_MOVE_GENERIC_ITINERARY = "II_MOV_alu_mv_mv_mv_scl"
AGGREGATE_MOVE_COMPONENT_FORMS = {
    "scalar": "MOV_alu_mv_mv_mv_scl",
    "accumulator_1024": "VMOV_alu_mv_mv_cm",
    "fifo_512": "VMOV_alu_mv_mv_x",
}


@dataclass(frozen=True, slots=True)
class ScalarMoveTopology:
    """Factored exact schedules for the native scalar-move register domain."""

    destination_categories: tuple[tuple[str, ...], ...]
    source_categories: tuple[tuple[str, ...], ...]
    itinerary_names: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class AggregateMoveRecipe:
    """Fixed native decomposition shared by compatible aggregate classes."""

    name: str
    machine_classes: tuple[str, ...]
    component_paths: tuple[tuple[str, ...], ...]
    component_kind: str
    non_value_paths: tuple[tuple[str, ...], ...] = ()


def _build_scalar_move_topology() -> ScalarMoveTopology:
    classes = {row.name: row for row in CORE_MACHINE_TABLE.register_classes}
    adapters = {row.name: row for row in CORE_MACHINE_TABLE.register_adapters}
    itineraries = {row.name: row for row in CORE_SCHEDULE_TABLE.itineraries}
    destination_registers = tuple(
        sorted(
            register
            for register, _ in adapters["OP_mMvSclDst"].effective_register_encodings
        )
    )
    source_registers = tuple(
        sorted(
            register
            for register, _ in adapters["OP_mMvSclSrc"].effective_register_encodings
            if register != "CORE_ID"
        )
    )

    specialized = {}
    destination_class_names = set()
    source_class_names = set()
    for destination_class in CORE_MACHINE_TABLE.register_classes:
        for source_class in CORE_MACHINE_TABLE.register_classes:
            name = (
                f"{_SCALAR_MOVE_ITINERARY_PREFIX}"
                f"{destination_class.name}_{source_class.name}"
            )
            itinerary = itineraries.get(name)
            if itinerary is None:
                continue
            if (
                itinerary.operand_cycles != (1, 1)
                or itinerary.bypasses
                or itinerary.memory is not None
                or itinerary.micro_ops != 1
            ):
                raise ValueError(f"{name}: scalar move timing contract changed")
            specialized[(destination_class.name, source_class.name)] = itinerary
            destination_class_names.add(destination_class.name)
            source_class_names.add(source_class.name)
    if len(specialized) != len(destination_class_names) * len(source_class_names):
        raise ValueError("AIE2P specialized scalar move itinerary matrix is sparse")

    destination_classes_by_register = {
        register: tuple(
            row
            for name in sorted(destination_class_names)
            for row in (classes[name],)
            if register in row.candidates
        )
        for register in destination_registers
    }
    source_classes_by_register = {
        register: tuple(
            row
            for name in sorted(source_class_names)
            for row in (classes[name],)
            if register in row.candidates
        )
        for register in source_registers
    }

    def narrowest(rows, register: str):
        if not rows:
            raise ValueError(f"{register}: scalar move register has no exact itinerary")
        candidate_count = min(len(row.candidates) for row in rows)
        return tuple(row for row in rows if len(row.candidates) == candidate_count)

    destination_choices = {
        register: narrowest(rows, register)
        for register, rows in destination_classes_by_register.items()
    }
    source_choices = {
        register: narrowest(rows, register)
        for register, rows in source_classes_by_register.items()
    }

    # Equal-specificity source classes must describe the same physical bank.
    # Proving every tied product lets generation select one stable name without
    # turning source-class vocabulary into runtime policy.
    selected = {}
    for destination in destination_registers:
        for source in source_registers:
            rows = tuple(
                specialized[(destination_class.name, source_class.name)]
                for destination_class in destination_choices[destination]
                for source_class in source_choices[source]
            )
            payloads = {itinerary_payload(row) for row in rows}
            if len(payloads) != 1:
                raise ValueError(
                    f"{source} -> {destination}: equally specific scalar "
                    "itineraries disagree"
                )
            selected[(destination, source)] = rows[0]

    destination_patterns = {}
    for destination in destination_registers:
        pattern = tuple(
            itinerary_payload(selected[(destination, source)])
            for source in source_registers
        )
        destination_patterns.setdefault(pattern, []).append(destination)
    source_patterns = {}
    for source in source_registers:
        pattern = tuple(
            itinerary_payload(selected[(destination, source)])
            for destination in destination_registers
        )
        source_patterns.setdefault(pattern, []).append(source)

    destination_categories = tuple(
        tuple(registers) for registers in destination_patterns.values()
    )
    source_categories = tuple(
        tuple(registers) for registers in source_patterns.values()
    )
    itinerary_names = []
    for destination_category in destination_categories:
        row = []
        for source_category in source_categories:
            payloads = {
                itinerary_payload(selected[(destination, source)])
                for destination in destination_category
                for source in source_category
            }
            if len(payloads) != 1:
                raise ValueError("factored scalar move category is not exact")
            row.append(selected[(destination_category[0], source_category[0])].name)
        # CORE_ID has no specialized oracle itinerary. The owned generic row is
        # deliberately conservative and is already used by read.core.id.
        row.append(_SCALAR_MOVE_GENERIC_ITINERARY)
        itinerary_names.append(tuple(row))

    return ScalarMoveTopology(
        destination_categories=destination_categories,
        source_categories=(*source_categories, ("CORE_ID",)),
        itinerary_names=tuple(itinerary_names),
    )


SCALAR_MOVE_TOPOLOGY = _build_scalar_move_topology()


def _validate_aggregate_move_recipes(
    recipes: tuple[AggregateMoveRecipe, ...],
) -> None:
    classes = {row.name: row for row in CORE_MACHINE_TABLE.register_classes}
    registers = {row.name: row for row in CORE_MACHINE_TABLE.physical_registers}
    adapters = {row.name: row for row in CORE_MACHINE_TABLE.register_adapters}
    forms = {row.name: row for row in CORE_MACHINE_TABLE.forms}
    classified_classes = set()
    for recipe in recipes:
        form = forms[AGGREGATE_MOVE_COMPONENT_FORMS[recipe.component_kind]]
        if len(form.outputs) != 1 or len(form.inputs) != 1:
            raise ValueError(f"{form.name}: component move must have two operands")
        destination_registers = {
            register
            for register, _ in adapters[
                form.outputs[0].type_name
            ].effective_register_encodings
        }
        source_registers = {
            register
            for register, _ in adapters[
                form.inputs[0].type_name
            ].effective_register_encodings
        }
        for class_name in recipe.machine_classes:
            if class_name in classified_classes:
                raise ValueError(f"{class_name}: duplicate aggregate move recipe")
            classified_classes.add(class_name)
            register_class = classes[class_name]
            for candidate in register_class.candidates:
                aggregate = registers[candidate]
                selected = []
                for path in (*recipe.component_paths, *recipe.non_value_paths):
                    component = aggregate
                    for step in path:
                        try:
                            index = component.subregister_indices.index(step)
                        except ValueError as exc:
                            raise ValueError(
                                f"{candidate}: move path has no {step} component"
                            ) from exc
                        component = registers[component.subregisters[index]]
                    selected.append(component)
                components = selected[: len(recipe.component_paths)]
                component_units = [
                    unit for component in selected for unit in component.atomic_units
                ]
                if len(component_units) != len(set(component_units)) or set(
                    component_units
                ) != set(aggregate.atomic_units):
                    raise ValueError(
                        f"{candidate}: move components do not partition native storage"
                    )
                for component in components:
                    if (
                        component.name not in destination_registers
                        or component.name not in source_registers
                    ):
                        raise ValueError(
                            f"{candidate}: {component.name} is not encodable by "
                            f"{form.name}"
                        )


AGGREGATE_MOVE_RECIPES = (
    AggregateMoveRecipe(
        name="scalar_pair",
        machine_classes=("eL", "eLPredicate", "mL8m"),
        component_paths=(("sub_l_even",), ("sub_l_odd",)),
        component_kind="scalar",
    ),
    AggregateMoveRecipe(
        name="dimension_2d",
        machine_classes=("eD",),
        component_paths=tuple(path for _, path, _ in DIMENSION_FIELDS["eD"]),
        component_kind="scalar",
    ),
    AggregateMoveRecipe(
        name="dimension_3d",
        machine_classes=("eDS",),
        component_paths=tuple(path for _, path, _ in DIMENSION_FIELDS["eDS"]),
        component_kind="scalar",
        # Native 3D addressing owns an upper wrap register that carries no
        # value state; the canonical field model deliberately omits it.
        non_value_paths=(("sub_hi_dim", "sub_mod"),),
    ),
    AggregateMoveRecipe(
        name="accumulator_2048",
        machine_classes=("eDM",),
        component_paths=(("sub_1024_acc_lo",), ("sub_1024_acc_hi",)),
        component_kind="accumulator_1024",
    ),
    AggregateMoveRecipe(
        name="fifo_1024",
        machine_classes=("eLdFifoReg", "mStFifo"),
        component_paths=(("sub_lo_fifo",), ("sub_hi_fifo",)),
        component_kind="fifo_512",
    ),
)

_validate_aggregate_move_recipes(AGGREGATE_MOVE_RECIPES)
