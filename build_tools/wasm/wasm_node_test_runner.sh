#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bundle.mjs> [args...]" >&2
  exit 1
fi

node_bin="${IREE_WASM_NODE:-}"
if [[ -z "${node_bin}" ]]; then
  node_bin="$(command -v node || true)"
fi
if [[ -z "${node_bin}" ]]; then
  node_bin="$(command -v nodejs || true)"
fi
if [[ -z "${node_bin}" ]]; then
  echo "Unable to find node. Set IREE_WASM_NODE to a Node.js executable." >&2
  exit 1
fi

node_realpath="$(readlink -f "${node_bin}" || true)"
if [[ "${node_realpath}" == "/usr/bin/snap" &&
      -x "/snap/node/current/bin/node" ]]; then
  node_bin="/snap/node/current/bin/node"
fi

exec "${node_bin}" "$@"
