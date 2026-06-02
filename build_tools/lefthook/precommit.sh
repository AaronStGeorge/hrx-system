#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if [[ -d "${REPO_ROOT}/.venv/bin" ]]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

PYTHON="${PYTHON:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

"${PYTHON}" build_tools/lefthook/presubmit.py --fix --staged --hygiene
"${PYTHON}" build_tools/lefthook/presubmit.py --check --staged --hygiene
