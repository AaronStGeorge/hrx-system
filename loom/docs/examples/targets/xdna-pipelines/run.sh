#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=loom/docs/examples/example.shlib
source "${script_dir}/../../example.shlib"

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  printf 'usage: %s source.cxx output-directory [device-profile]\n' "$0" >&2
  exit 64
fi

source_file="$1"
output_dir="$2"
profile="${3:-amd.xdna.strix_halo.17f0_11}"
loom_format="${LOOM_FORMAT:-loom-format}"
loom_import_cxx="${LOOM_IMPORT_CXX:-loom-import-cxx}"
loom_compile="${LOOM_COMPILE:-loom-compile}"
loom_example_require_tool "${loom_format}"
loom_example_require_tool "${loom_import_cxx}"
loom_example_require_tool "${loom_compile}"
mkdir -p -- "${output_dir}"

loom_example_section "Import the authored C++ into Loom High IR"
loom_example_run_tool loom-import-cxx "${loom_import_cxx}" \
  --root=ffn_gate_up_quadratic_bf16 "${source_file}" \
  --output="${output_dir}/imported.loom"
loom_example_run_tool loom-format "${loom_format}" --check \
  "${output_dir}/imported.loom"

loom_example_section "Compile the pipeline and retain its compilation boundaries"
loom_example_run_tool loom-compile "${loom_compile}" \
  "${output_dir}/imported.loom" \
  --root=@ffn_gate_up_quadratic_bf16 \
  --target="amd.xdna.aie2p:${profile}" --format=xdna \
  --config=ffn_gate_up.input_size=512 \
  --dump-ir-after=outline-pipeline-strands \
  --dump-ir-after=aie2p-lower-pipeline \
  --dump-ir-output="${output_dir}/trajectory/" \
  --compile-report=summary \
  --compile-report-output="${output_dir}/projection.report.json" \
  --output="${output_dir}/projection.xdna"

# Keep complete modules so every displayed boundary can be compiled on its own.
shopt -s nullglob
for boundary in outline-pipeline-strands aie2p-lower-pipeline; do
  cuts=("${output_dir}/trajectory/ir/"*"-after-${boundary}.loom")
  if [[ "${#cuts[@]}" -ne 1 ]]; then
    printf 'expected one %s boundary, received %s\n' \
      "${boundary}" "${#cuts[@]}" >&2
    exit 1
  fi
  # Trace comments contain build-local paths, outside the program's semantics.
  sed '/^[[:space:]]*\/\//d' "${cuts[0]}" >"${output_dir}/${boundary}.input.loom"
  loom_example_run_tool loom-format "${loom_format}" "${output_dir}/${boundary}.input.loom" \
    --output="${output_dir}/${boundary}.loom"
  loom_example_run_tool loom-format "${loom_format}" \
    "${output_dir}/${boundary}.loom" --to=bc \
    --output="${output_dir}/${boundary}.loombc"

  loom_example_section "Resume from ${boundary} text and bytecode"
  for encoding in loom loombc; do
    loom_example_run_tool loom-compile "${loom_compile}" \
      "${output_dir}/${boundary}.${encoding}" \
      --root=@ffn_gate_up_quadratic_bf16 --format=xdna \
      --output="${output_dir}/${boundary}.${encoding}.xdna"
  done
  cmp -- "${output_dir}/${boundary}.loom.xdna" \
    "${output_dir}/${boundary}.loombc.xdna"
done
