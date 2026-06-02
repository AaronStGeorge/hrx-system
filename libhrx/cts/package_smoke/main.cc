// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <hrx_runtime_cxx.h>

int main() {
  hrx::runtime::device_ptr device;
  return !device && hrx::runtime::format_status(hrx_ok_status()) == "OK" ? 0
                                                                         : 1;
}
