// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_SOURCES_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_SOURCES_H_

#include "iree/base/api.h"
#include "loom/serve/runtime/program/package.h"

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed program inputs, independent of modality and registration policy.
// The owner retains every view through asynchronous model preparation. Authored
// locators remain separate from the local base used by an I/O consumer.
typedef struct loom_serve_sources_t {
  // Local file base; empty requires absolute locators for filesystem inputs.
  // Package-relative programs/metadata do not use this base.
  iree_string_view_t base_directory;
  // Cold VM program declaring stages and model metadata.
  iree_string_view_t prepare;
  // Warm VM program orchestrating requests and stage submissions.
  iree_string_view_t control;
  // Number of shared VM, native command, kernel, and configuration providers.
  iree_host_size_t count;
  // Authored provider paths; symbols establish dependencies, not list order.
  const iree_string_view_t* values;
  // Optional borrowed immutable package for relative source/metadata locators.
  // Absolute locators still name server-local files. No missing-member
  // fallback.
  loom_serve_package_t* package;
} loom_serve_sources_t;

// Captures every locator and retains the immutable package in one allocation.
// The result is independent of the input views. This preserves the namespace,
// not filesystem contents; sources_open still performs the eventual read.
// Failure publishes NULL. Release with sources_free and the same allocator.
iree_status_t loom_serve_sources_clone(const loom_serve_sources_t* sources,
                                       iree_allocator_t host_allocator,
                                       loom_serve_sources_t** out_sources);

// Releases a cloned namespace and its package reference. NULL is accepted.
void loom_serve_sources_free(loom_serve_sources_t* sources,
                             iree_allocator_t host_allocator);

// Loads one source or metadata asset, retaining its bytes independently of the
// environment. File reads belong on preparation workers, not a scheduler turn.
iree_status_t loom_serve_sources_open(const loom_serve_sources_t* sources,
                                      iree_string_view_t locator,
                                      iree_allocator_t host_allocator,
                                      loomc_source_t** out_source);

// Reports whether a locator is absolute in the local filesystem namespace.
bool loom_serve_path_is_absolute(iree_string_view_t path);

// Binds an authored locator to a local I/O base without changing its owner.
// Absolute locators ignore base; relative locators require a nonempty base.
// Windows drive-relative and root-relative paths reject instead of inheriting
// process drive state. Does not open the file or resolve symlinks. The caller
// frees the result.
iree_status_t loom_serve_path_resolve(iree_string_view_t base,
                                      iree_string_view_t path,
                                      iree_allocator_t host_allocator,
                                      char** out_path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_SOURCES_H_
