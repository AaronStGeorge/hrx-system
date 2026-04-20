// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "third_party/hsa-runtime-headers/include/hsa/hsa.h"

#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/debugging.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/presets.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

struct HsaApi {
  // Loaded HSA runtime shared library.
  iree_dynamic_library_t* library = nullptr;
  // `hsa_init` entry point.
  decltype(&hsa_init) hsa_init = nullptr;
  // `hsa_shut_down` entry point.
  decltype(&hsa_shut_down) hsa_shut_down = nullptr;
  // `hsa_status_string` entry point.
  decltype(&hsa_status_string) hsa_status_string = nullptr;
  // `hsa_iterate_agents` entry point.
  decltype(&hsa_iterate_agents) hsa_iterate_agents = nullptr;
  // `hsa_agent_get_info` entry point.
  decltype(&hsa_agent_get_info) hsa_agent_get_info = nullptr;
  // `hsa_agent_iterate_isas` entry point.
  decltype(&hsa_agent_iterate_isas) hsa_agent_iterate_isas = nullptr;
  // `hsa_isa_get_info_alt` entry point.
  decltype(&hsa_isa_get_info_alt) hsa_isa_get_info_alt = nullptr;
  // `hsa_code_object_reader_create_from_memory` entry point.
  decltype(&hsa_code_object_reader_create_from_memory)
      hsa_code_object_reader_create_from_memory = nullptr;
  // `hsa_code_object_reader_destroy` entry point.
  decltype(&hsa_code_object_reader_destroy) hsa_code_object_reader_destroy =
      nullptr;
  // `hsa_executable_create_alt` entry point.
  decltype(&hsa_executable_create_alt) hsa_executable_create_alt = nullptr;
  // `hsa_executable_destroy` entry point.
  decltype(&hsa_executable_destroy) hsa_executable_destroy = nullptr;
  // `hsa_executable_load_agent_code_object` entry point.
  decltype(&hsa_executable_load_agent_code_object)
      hsa_executable_load_agent_code_object = nullptr;
  // `hsa_executable_freeze` entry point.
  decltype(&hsa_executable_freeze) hsa_executable_freeze = nullptr;
  // `hsa_executable_get_symbol_by_name` entry point.
  decltype(&hsa_executable_get_symbol_by_name)
      hsa_executable_get_symbol_by_name = nullptr;
  // `hsa_executable_symbol_get_info` entry point.
  decltype(&hsa_executable_symbol_get_info) hsa_executable_symbol_get_info =
      nullptr;
};

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transient HSACO writer storage.
  iree_arena_allocator_t arena_ = {0};
};

StreamPtr CreateStream() {
  iree_io_stream_t* stream = nullptr;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      1024, iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

std::string StreamBytes(iree_io_stream_t* stream) {
  const iree_io_stream_pos_t length = iree_io_stream_length(stream);
  IREE_ASSERT_GE(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(
      iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
  return bytes;
}

std::string StatusToStringAndFree(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = nullptr;
  iree_host_size_t buffer_length = 0;
  std::string result = iree_status_code_string(iree_status_code(status));
  if (iree_status_to_string(status, &allocator, &buffer, &buffer_length)) {
    result.assign(buffer, buffer_length);
    iree_allocator_free(allocator, buffer);
  }
  iree_status_free(status);
  return result;
}

std::string HsaStatusString(const HsaApi& api, hsa_status_t status) {
  const char* status_string = nullptr;
  if (api.hsa_status_string != nullptr &&
      api.hsa_status_string(status, &status_string) == HSA_STATUS_SUCCESS &&
      status_string != nullptr) {
    return status_string;
  }
  return std::to_string((int)status);
}

template <typename Fn, typename... Args>
hsa_status_t CallHsa(Fn fn, Args... args) {
  IREE_LEAK_CHECK_DISABLE_PUSH();
  hsa_status_t status = fn(args...);
  IREE_LEAK_CHECK_DISABLE_POP();
  return status;
}

template <typename Fn>
iree_status_t LoadHsaSymbol(HsaApi* api, const char* symbol_name, Fn* out_fn) {
  IREE_ASSERT_ARGUMENT(api);
  IREE_ASSERT_ARGUMENT(out_fn);
  void* symbol = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_dynamic_library_lookup_symbol(api->library, symbol_name, &symbol));
  *out_fn = reinterpret_cast<Fn>(symbol);
  return iree_ok_status();
}

void AppendHsaLibraryCandidate(std::vector<std::string>* candidates,
                               std::string candidate) {
  IREE_ASSERT_ARGUMENT(candidates);
  if (candidate.empty()) {
    return;
  }
  for (const std::string& existing_candidate : *candidates) {
    if (existing_candidate == candidate) {
      return;
    }
  }
  candidates->push_back(std::move(candidate));
}

void AppendHsaLibraryCandidates(std::vector<std::string>* candidates) {
  IREE_ASSERT_ARGUMENT(candidates);
  static const char* kLibraryNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
      "hsa-runtime64.dll",
#else
      "libhsa-runtime64.so.1",
      "libhsa-runtime64.so",
#endif  // IREE_PLATFORM_WINDOWS
  };
  const char* env_path = std::getenv("IREE_HAL_AMDGPU_LIBHSA_PATH");
  if (env_path != nullptr && env_path[0] != '\0') {
    const std::string path_fragment(env_path);
    AppendHsaLibraryCandidate(candidates, path_fragment);
    for (const char* library_name : kLibraryNames) {
      std::string candidate_path = path_fragment;
      if (candidate_path.back() != '/') {
        candidate_path.push_back('/');
      }
      candidate_path.append(library_name);
      AppendHsaLibraryCandidate(candidates, std::move(candidate_path));
    }
  }
  for (const char* library_name : kLibraryNames) {
    AppendHsaLibraryCandidate(candidates, library_name);
  }
}

iree_status_t LoadHsaLibrary(HsaApi* api) {
  IREE_ASSERT_ARGUMENT(api);
  std::vector<std::string> candidates;
  AppendHsaLibraryCandidates(&candidates);

  std::string failures;
  for (const std::string& candidate : candidates) {
    iree_status_t status = iree_dynamic_library_load_from_file(
        candidate.c_str(), IREE_DYNAMIC_LIBRARY_FLAG_NONE,
        iree_allocator_system(), &api->library);
    if (iree_status_is_ok(status)) {
      return iree_ok_status();
    }
    failures.append("\n  Tried: ");
    failures.append(candidate);
    failures.append("\n    ");
    failures.append(StatusToStringAndFree(status));
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "HSA runtime unavailable; specify "
                          "IREE_HAL_AMDGPU_LIBHSA_PATH if the runtime is "
                          "not on the loader path:%s",
                          failures.c_str());
}

void UnloadHsaApi(HsaApi* api) {
  IREE_ASSERT_ARGUMENT(api);
  if (api->library != nullptr) {
    iree_dynamic_library_release(api->library);
  }
  *api = {};
}

iree_status_t LoadHsaApi(HsaApi* out_api) {
  IREE_ASSERT_ARGUMENT(out_api);
  *out_api = {};
  HsaApi api = {};
  iree_status_t status = LoadHsaLibrary(&api);
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_init", &api.hsa_init);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_shut_down", &api.hsa_shut_down);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_status_string", &api.hsa_status_string);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_iterate_agents", &api.hsa_iterate_agents);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_agent_get_info", &api.hsa_agent_get_info);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_agent_iterate_isas",
                           &api.hsa_agent_iterate_isas);
  }
  if (iree_status_is_ok(status)) {
    status =
        LoadHsaSymbol(&api, "hsa_isa_get_info_alt", &api.hsa_isa_get_info_alt);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_code_object_reader_create_from_memory",
                           &api.hsa_code_object_reader_create_from_memory);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_code_object_reader_destroy",
                           &api.hsa_code_object_reader_destroy);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_create_alt",
                           &api.hsa_executable_create_alt);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_destroy",
                           &api.hsa_executable_destroy);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_load_agent_code_object",
                           &api.hsa_executable_load_agent_code_object);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_freeze",
                           &api.hsa_executable_freeze);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_get_symbol_by_name",
                           &api.hsa_executable_get_symbol_by_name);
  }
  if (iree_status_is_ok(status)) {
    status = LoadHsaSymbol(&api, "hsa_executable_symbol_get_info",
                           &api.hsa_executable_symbol_get_info);
  }
  if (iree_status_is_ok(status)) {
    *out_api = api;
  } else {
    UnloadHsaApi(&api);
  }
  return status;
}

class HsaRuntime {
 public:
  HsaRuntime() = default;
  HsaRuntime(const HsaRuntime&) = delete;
  HsaRuntime& operator=(const HsaRuntime&) = delete;

  ~HsaRuntime() {
    if (initialized_) {
      EXPECT_EQ(CallHsa(api_.hsa_shut_down), HSA_STATUS_SUCCESS);
    }
    UnloadHsaApi(&api_);
  }

  const HsaApi& api() const { return api_; }

  HsaApi* mutable_api() { return &api_; }

  void MarkInitialized() { initialized_ = true; }

 private:
  // Dynamically loaded HSA API surface.
  HsaApi api_ = {};
  // True once `hsa_init` has successfully incremented the runtime refcount.
  bool initialized_ = false;
};

class HsaCodeObjectReader {
 public:
  explicit HsaCodeObjectReader(const HsaApi* api) : api_(api) {}
  HsaCodeObjectReader(const HsaCodeObjectReader&) = delete;
  HsaCodeObjectReader& operator=(const HsaCodeObjectReader&) = delete;

  ~HsaCodeObjectReader() { Reset(); }

  hsa_code_object_reader_t get() const { return reader_; }

  hsa_code_object_reader_t* out() {
    Reset();
    return &reader_;
  }

  void Reset() {
    if (reader_.handle != 0) {
      EXPECT_EQ(CallHsa(api_->hsa_code_object_reader_destroy, reader_),
                HSA_STATUS_SUCCESS);
      reader_ = {};
    }
  }

 private:
  // HSA entry points used to release the reader.
  const HsaApi* api_ = nullptr;
  // Owned HSA code object reader handle.
  hsa_code_object_reader_t reader_ = {};
};

class HsaExecutable {
 public:
  explicit HsaExecutable(const HsaApi* api) : api_(api) {}
  HsaExecutable(const HsaExecutable&) = delete;
  HsaExecutable& operator=(const HsaExecutable&) = delete;

  ~HsaExecutable() { Reset(); }

  hsa_executable_t get() const { return executable_; }

  hsa_executable_t* out() {
    Reset();
    return &executable_;
  }

  void Reset() {
    if (executable_.handle != 0) {
      EXPECT_EQ(CallHsa(api_->hsa_executable_destroy, executable_),
                HSA_STATUS_SUCCESS);
      executable_ = {};
    }
  }

 private:
  // HSA entry points used to release the executable.
  const HsaApi* api_ = nullptr;
  // Owned HSA executable handle.
  hsa_executable_t executable_ = {};
};

struct GpuAgentSearch {
  // HSA API used while iterating agents.
  const HsaApi* api = nullptr;
  // First discovered GPU agent.
  hsa_agent_t agent = {};
  // HSA-reported agent name used for diagnostics.
  std::string agent_name;
  // True once |agent| has been populated.
  bool found = false;
};

hsa_status_t FindFirstGpuAgent(hsa_agent_t agent, void* user_data) {
  GpuAgentSearch* search = reinterpret_cast<GpuAgentSearch*>(user_data);
  hsa_device_type_t device_type = HSA_DEVICE_TYPE_CPU;
  hsa_status_t status = CallHsa(search->api->hsa_agent_get_info, agent,
                                HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (device_type == HSA_DEVICE_TYPE_GPU) {
    char agent_name[64] = {0};
    status = CallHsa(search->api->hsa_agent_get_info, agent,
                     HSA_AGENT_INFO_NAME, agent_name);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
    search->agent = agent;
    search->agent_name = agent_name;
    search->found = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

iree_status_t QueryHsaIsaName(const HsaApi& api, hsa_isa_t isa,
                              std::string* out_name) {
  IREE_ASSERT_ARGUMENT(out_name);
  *out_name = {};
  uint32_t name_length = 0;
  hsa_status_t status = CallHsa(api.hsa_isa_get_info_alt, isa,
                                HSA_ISA_INFO_NAME_LENGTH, &name_length);
  if (status != HSA_STATUS_SUCCESS) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "hsa_isa_get_info_alt(HSA_ISA_INFO_NAME_LENGTH) failed: %s",
        HsaStatusString(api, status).c_str());
  }
  if (name_length == 0 || name_length > 1024) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "HSA ISA name has invalid length %" PRIu32,
                            name_length);
  }

  std::string buffer((size_t)name_length + 1u, '\0');
  status =
      CallHsa(api.hsa_isa_get_info_alt, isa, HSA_ISA_INFO_NAME, buffer.data());
  if (status != HSA_STATUS_SUCCESS) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "hsa_isa_get_info_alt(HSA_ISA_INFO_NAME) failed: %s",
        HsaStatusString(api, status).c_str());
  }

  size_t actual_length = 0;
  while (actual_length < buffer.size() && buffer[actual_length] != '\0') {
    ++actual_length;
  }
  if (actual_length == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HSA ISA name is empty");
  }
  *out_name = buffer.substr(0, actual_length);
  return iree_ok_status();
}

struct AgentIsaSearch {
  // HSA API used while iterating agent ISAs.
  const HsaApi* api = nullptr;
  // First HSA ISA name reported for the selected GPU agent.
  std::string isa_name;
  // IREE status text captured if ISA name query fails inside the callback.
  std::string failure;
  // True once |isa_name| has been populated.
  bool found = false;
};

hsa_status_t FindFirstAgentIsa(hsa_isa_t isa, void* user_data) {
  AgentIsaSearch* search = reinterpret_cast<AgentIsaSearch*>(user_data);
  iree_status_t status = QueryHsaIsaName(*search->api, isa, &search->isa_name);
  if (!iree_status_is_ok(status)) {
    search->failure = StatusToStringAndFree(status);
    return HSA_STATUS_ERROR;
  }
  search->found = true;
  return HSA_STATUS_INFO_BREAK;
}

bool TryFindFirstGpuAgent(const HsaApi& api, hsa_agent_t* out_agent,
                          std::string* out_agent_name,
                          std::string* out_skip_reason) {
  IREE_ASSERT_ARGUMENT(out_agent);
  IREE_ASSERT_ARGUMENT(out_agent_name);
  IREE_ASSERT_ARGUMENT(out_skip_reason);
  GpuAgentSearch search = {.api = &api};
  hsa_status_t status =
      CallHsa(api.hsa_iterate_agents, FindFirstGpuAgent, &search);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    ADD_FAILURE() << "hsa_iterate_agents failed: "
                  << HsaStatusString(api, status);
    return false;
  }
  if (!search.found) {
    *out_skip_reason = "no HSA GPU agent found";
    return false;
  }
  *out_agent = search.agent;
  *out_agent_name = search.agent_name;
  return true;
}

bool TryParseAmdhsaTargetCpu(const std::string& target_id,
                             std::string* out_target_cpu,
                             std::string* out_feature_suffix,
                             std::string* out_error) {
  IREE_ASSERT_ARGUMENT(out_target_cpu);
  IREE_ASSERT_ARGUMENT(out_feature_suffix);
  IREE_ASSERT_ARGUMENT(out_error);
  *out_target_cpu = {};
  *out_feature_suffix = {};
  *out_error = {};
  loom_amdgpu_amdhsa_target_id_t parsed_target_id = {};
  iree_status_t status = loom_amdgpu_target_info_parse_amdhsa_target_id(
      iree_make_string_view(target_id.data(), target_id.size()),
      &parsed_target_id);
  if (!iree_status_is_ok(status)) {
    *out_error = StatusToStringAndFree(status);
    return false;
  }
  out_target_cpu->assign(parsed_target_id.processor->target_cpu.data,
                         parsed_target_id.processor->target_cpu.size);
  out_feature_suffix->assign(parsed_target_id.feature_suffix.data,
                             parsed_target_id.feature_suffix.size);
  return true;
}

struct AmdgpuHsaTarget {
  // HSA GPU agent that owns the queried target ISA.
  hsa_agent_t agent = {};
  // HSA-reported GPU agent name used for diagnostics.
  std::string agent_name;
  // Full HSA target id reported by the runtime for |agent|.
  std::string isa_name;
  // Target CPU parsed out of |isa_name|.
  std::string target_cpu;
  // Target-feature suffix parsed out of |isa_name|.
  std::string feature_suffix;
};

bool TryDiscoverCurrentAmdgpuTarget(const HsaApi& api,
                                    AmdgpuHsaTarget* out_target,
                                    std::string* out_skip_reason) {
  IREE_ASSERT_ARGUMENT(out_target);
  IREE_ASSERT_ARGUMENT(out_skip_reason);
  *out_target = {};
  hsa_agent_t agent = {};
  std::string agent_name;
  if (!TryFindFirstGpuAgent(api, &agent, &agent_name, out_skip_reason)) {
    return false;
  }

  AgentIsaSearch isa_search = {.api = &api};
  hsa_status_t status = CallHsa(api.hsa_agent_iterate_isas, agent,
                                FindFirstAgentIsa, &isa_search);
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    if (!isa_search.failure.empty()) {
      ADD_FAILURE() << isa_search.failure;
    } else {
      ADD_FAILURE() << "hsa_agent_iterate_isas failed: "
                    << HsaStatusString(api, status);
    }
    return false;
  }
  if (!isa_search.found) {
    *out_skip_reason =
        "HSA GPU agent '" + agent_name + "' did not report any ISAs";
    return false;
  }

  std::string target_cpu;
  std::string feature_suffix;
  std::string parse_error;
  if (!TryParseAmdhsaTargetCpu(isa_search.isa_name, &target_cpu,
                               &feature_suffix, &parse_error)) {
    *out_skip_reason = parse_error;
    return false;
  }

  *out_target = {
      .agent = agent,
      .agent_name = std::move(agent_name),
      .isa_name = std::move(isa_search.isa_name),
      .target_cpu = std::move(target_cpu),
      .feature_suffix = std::move(feature_suffix),
  };
  return true;
}

const loom_op_t* FindFirstLowFunction(loom_module_t* module) {
  loom_block_t* block = loom_module_block(module);
  const loom_op_t* op = nullptr;
  loom_block_for_each_op(block, op) {
    if (loom_low_func_def_isa(op)) {
      return op;
    }
  }
  return nullptr;
}

class LowNoopKernelCompiler {
 public:
  LowNoopKernelCompiler() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_CHECK_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                      &context_));
    loom_amdgpu_low_descriptor_registry_initialize(&target_registry_);
  }

  LowNoopKernelCompiler(const LowNoopKernelCompiler&) = delete;
  LowNoopKernelCompiler& operator=(const LowNoopKernelCompiler&) = delete;

  ~LowNoopKernelCompiler() {
    ResetModule();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t EncodeReturnOnlyKernel(iree_string_view_t preset_key,
                                       iree_const_byte_span_t* out_text,
                                       iree_arena_allocator_t* arena) {
    IREE_ASSERT_ARGUMENT(out_text);
    IREE_ASSERT_ARGUMENT(arena);
    *out_text = iree_const_byte_span_empty();
    std::string source = "target.preset @gfx_target {key = \"";
    source.append(preset_key.data, preset_key.size);
    source += "\", source = @loom_kernel}\n";
    source += "low.func.def target(@gfx_target) @loom_kernel() {\n";
    source += "  low.return\n";
    source += "}\n";
    IREE_RETURN_IF_ERROR(ParseSource(source));

    const loom_target_preset_registry_t preset_registry =
        loom_target_low_descriptor_registry_presets(&target_registry_);
    iree_host_size_t expanded_preset_count = 0;
    IREE_RETURN_IF_ERROR(loom_target_expand_presets(module_, &preset_registry,
                                                    &expanded_preset_count));
    if (expanded_preset_count != 1) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU HSA low no-op kernel expanded %" PRIhsz
                              " target presets instead of one",
                              expanded_preset_count);
    }

    loom_low_verify_options_t verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t verify_result = {};
    IREE_RETURN_IF_ERROR(
        loom_low_verify_module(module_, &verify_options, &verify_result));
    if (verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU HSA low no-op kernel failed low "
                              "verification");
    }

    const loom_op_t* low_function = FindFirstLowFunction(module_);
    if (low_function == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU HSA low no-op kernel has no low func");
    }

    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
    };
    loom_low_packetization_t packetization = {};
    IREE_RETURN_IF_ERROR(loom_low_packetize_function(
        module_, low_function, &packetization_options, arena, &packetization));
    return loom_amdgpu_encode_instruction_stream(
        &packetization.schedule, &packetization.allocation, out_text, arena);
  }

 private:
  void ResetModule() {
    if (module_ != nullptr) {
      loom_module_free(module_);
      module_ = nullptr;
    }
  }

  iree_status_t ParseSource(const std::string& source) {
    ResetModule();
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    return loom_text_parse(iree_make_string_view(source.data(), source.size()),
                           IREE_SV("amdgpu_hsaco_hsa_noop.loom"), &context_,
                           &block_pool_, &parse_options, &module_);
  }

  // Block pool backing parser and context allocations.
  iree_arena_block_pool_t block_pool_ = {0};
  // Loom context containing dialect/type registration for parsing and verify.
  loom_context_t context_ = {};
  // Parsed module owned by this compiler instance.
  loom_module_t* module_ = nullptr;
  // AMDGPU-only descriptor/preset registry used by low verification.
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

struct NoopKernelProgram {
  // Exported kernel entry name.
  std::string name;
  // Required workgroup size in the X dimension.
  uint16_t workgroup_size_x = 1;
  // Required workgroup size in the Y dimension.
  uint16_t workgroup_size_y = 1;
  // Required workgroup size in the Z dimension.
  uint16_t workgroup_size_z = 1;
};

loom_amdgpu_metadata_kernel_t MetadataForNoopKernelProgram(
    const NoopKernelProgram& program, iree_string_view_t descriptor_symbol,
    uint32_t wavefront_size) {
  return {
      .name = iree_make_string_view(program.name.data(), program.name.size()),
      .descriptor_symbol = descriptor_symbol,
      .kernarg_segment_size = 0,
      .kernarg_segment_alignment = 8,
      .wavefront_size = wavefront_size,
      .group_segment_fixed_size = 0,
      .private_segment_fixed_size = 0,
      .sgpr_count = 0,
      .vgpr_count = 0,
      .max_flat_workgroup_size =
          static_cast<uint32_t>(program.workgroup_size_x) *
          static_cast<uint32_t>(program.workgroup_size_y) *
          static_cast<uint32_t>(program.workgroup_size_z),
      .required_workgroup_size =
          {
              .x = program.workgroup_size_x,
              .y = program.workgroup_size_y,
              .z = program.workgroup_size_z,
          },
      .has_required_workgroup_size = true,
      .arguments = nullptr,
      .argument_count = 0,
  };
}

iree_status_t CompileNoopKernelProgramForAmdgpu(
    const NoopKernelProgram& program, const AmdgpuHsaTarget& target,
    std::string* out_hsaco) {
  IREE_ASSERT_ARGUMENT(out_hsaco);
  *out_hsaco = {};
  if (!target.feature_suffix.empty()) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU HSACO target-feature suffixes are not supported yet: %s",
        target.feature_suffix.c_str());
  }
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_processor(
      iree_make_string_view(target.target_cpu.data(), target.target_cpu.size()),
      &processor));

  std::string descriptor_symbol = program.name + ".kd";
  TestArena arena;
  LowNoopKernelCompiler compiler;
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(compiler.EncodeReturnOnlyKernel(
      processor->low_preset_key, &text, arena.arena()));
  const loom_amdgpu_hsaco_kernel_t kernel = {
      .metadata = MetadataForNoopKernelProgram(
          program,
          iree_make_string_view(descriptor_symbol.data(),
                                descriptor_symbol.size()),
          processor->default_wavefront_size),
      .text = text,
  };
  const loom_amdgpu_hsaco_file_t file = {
      .target =
          iree_make_string_view(target.isa_name.data(), target.isa_name.size()),
      .target_cpu = iree_make_string_view(target.target_cpu.data(),
                                          target.target_cpu.size()),
      .kernels = &kernel,
      .kernel_count = 1,
  };

  StreamPtr stream = CreateStream();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_write_file(&file, stream.get(), arena.arena()));
  *out_hsaco = StreamBytes(stream.get());
  return iree_ok_status();
}

TEST(AmdgpuHsacoHsaTest, LoadsLowReturnNoopKernelAndResolvesDescriptorSymbol) {
  HsaRuntime runtime;
  iree_status_t load_status = LoadHsaApi(runtime.mutable_api());
  if (!iree_status_is_ok(load_status)) {
    GTEST_SKIP() << "HSA runtime unavailable: "
                 << StatusToStringAndFree(load_status);
  }
  const HsaApi& api = runtime.api();

  hsa_status_t status = CallHsa(api.hsa_init);
  if (status != HSA_STATUS_SUCCESS) {
    GTEST_SKIP() << "hsa_init failed: " << HsaStatusString(api, status);
  }
  runtime.MarkInitialized();

  AmdgpuHsaTarget target = {};
  std::string skip_reason;
  if (!TryDiscoverCurrentAmdgpuTarget(api, &target, &skip_reason)) {
    if (!skip_reason.empty()) {
      GTEST_SKIP() << skip_reason;
    }
    return;
  }

  const NoopKernelProgram program = {
      .name = "loom_kernel",
      .workgroup_size_x = 64,
      .workgroup_size_y = 1,
      .workgroup_size_z = 1,
  };
  std::string hsaco;
  iree_status_t compile_status =
      CompileNoopKernelProgramForAmdgpu(program, target, &hsaco);
  if (iree_status_is_unimplemented(compile_status)) {
    GTEST_SKIP() << "Loom AMDGPU HSACO compiler does not support current ISA '"
                 << target.isa_name << "' from HSA agent '" << target.agent_name
                 << "': " << StatusToStringAndFree(compile_status);
  }
  IREE_ASSERT_OK(compile_status);

  HsaCodeObjectReader reader(&api);
  ASSERT_EQ(CallHsa(api.hsa_code_object_reader_create_from_memory, hsaco.data(),
                    hsaco.size(), reader.out()),
            HSA_STATUS_SUCCESS);

  HsaExecutable executable(&api);
  status = CallHsa(api.hsa_executable_create_alt, HSA_PROFILE_FULL,
                   HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                   executable.out());
  ASSERT_EQ(status, HSA_STATUS_SUCCESS) << HsaStatusString(api, status);

  status = CallHsa(api.hsa_executable_load_agent_code_object, executable.get(),
                   target.agent, reader.get(), nullptr, nullptr);
  reader.Reset();
  ASSERT_EQ(status, HSA_STATUS_SUCCESS) << HsaStatusString(api, status);

  status = CallHsa(api.hsa_executable_freeze, executable.get(), nullptr);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS) << HsaStatusString(api, status);

  hsa_executable_symbol_t symbol = {};
  status = CallHsa(api.hsa_executable_get_symbol_by_name, executable.get(),
                   "loom_kernel.kd", &target.agent, &symbol);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS) << HsaStatusString(api, status);

  uint64_t kernel_object = 0;
  status = CallHsa(api.hsa_executable_symbol_get_info, symbol,
                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS) << HsaStatusString(api, status);
  EXPECT_NE(kernel_object, 0u);

  uint32_t kernarg_segment_size = UINT32_MAX;
  status = CallHsa(api.hsa_executable_symbol_get_info, symbol,
                   HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                   &kernarg_segment_size);
  EXPECT_EQ(status, HSA_STATUS_SUCCESS) << HsaStatusString(api, status);
  EXPECT_EQ(kernarg_segment_size, 0u);
}

}  // namespace
}  // namespace loom
