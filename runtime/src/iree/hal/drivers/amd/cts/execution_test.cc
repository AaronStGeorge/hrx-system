// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>
#include <tuple>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amd/cts/affine_images.h"
#include "iree/hal/drivers/amd/cts/cpu_executable.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"
#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::cts {
namespace {

using ::iree::hal::cts::Ref;

enum Engine { kCpu, kNpu, kGpu, kEngineCount };

enum class SubmissionMode { kDefaults, kExplicit, kRecordedIndirect };

struct Boundary {
  // Combined queue-local execution and visibility actions.
  iree_hal_barrier_t barrier = {
      .source_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask =
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
  };
  // Per-edge range recipes captured alongside their combined global effects.
  std::array<iree_hal_buffer_barrier_t, kEngineCount> ranges = {};
};

struct Invocation {
  // Acquires from the actual producers of the invocation's inputs.
  Boundary before;
  // Releases to subsequent native and CPU consumers.
  Boundary after;
  // Optional reusable recording with only indirect buffer references.
  Ref<iree_hal_command_buffer_t> commands;
};

struct Participant {
  // Independently created device retained by this fixture and the sealed group.
  Ref<iree_hal_device_t> device;
  // Borrowed queue from the device's dispatch-capable family.
  iree_hal_queue_t* queue = nullptr;
  // Native implementation's semaphore, signaled only by its own queue.
  Ref<iree_hal_semaphore_t> timeline;
  // Loaded arithmetic executable owned until all submissions complete.
  Ref<iree_hal_executable_t> executable;
  // Reflected function in the executable.
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
};

class ExecutionTest
    : public ::testing::TestWithParam<std::tuple<bool, SubmissionMode>> {
 protected:
  static constexpr size_t kElementCount = 16;
  static constexpr size_t kBytes = kElementCount * sizeof(uint32_t);
  using Payload = std::array<uint32_t, 5 * kElementCount>;

  static void SetUpTestSuite() {
    // KFD's primary VM belongs to the process. Keep one canonical native
    // instance while independently constructing and releasing each device
    // group.
    IREE_ASSERT_OK(
        iree_hal_driver_registry_allocate(iree_allocator_system(), &registry_));
    IREE_ASSERT_OK(iree_hal_amd_driver_module_register(registry_));
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("rdna"), &driver_));
  }

  static void TearDownTestSuite() {
    iree_hal_driver_release(driver_);
    driver_ = nullptr;
    iree_hal_driver_registry_free(registry_);
    registry_ = nullptr;
  }

  void SetUp() override {
    IREE_ASSERT_OK(context_.Initialize(iree_allocator_system(),
                                       iree_hal_device_event_sink_stderr()));
    iree_hal_driver_t* xdna_driver = nullptr;
    IREE_ASSERT_OK(iree_hal_driver_registry_acquire_driver(
        registry_, IREE_SV("xdna"), &xdna_driver));
    EXPECT_EQ(driver_, xdna_driver);
    iree_hal_driver_release(xdna_driver);
    ASSERT_NO_FATAL_FAILURE(CreateNativeDevice(kNpu, IREE_SV("xdna")));
    if (IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(CreateNativeDevice(kGpu, IREE_SV("rdna")));
    if (IsSkipped()) {
      return;
    }

    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(4, &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &cpu_executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(iree_allocator_system(),
                                                  &cpu_allocator_));
    iree_hal_task_device_params_t task_params;
    iree_hal_task_device_params_initialize(&task_params);
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("cpu"), &task_params, 1, &cpu_executor_, 0, nullptr,
        cpu_allocator_, context_.params(), iree_allocator_system(),
        participants_[kCpu].device.out()));

    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder,
                                             context_.frontier_tracker());
    status = iree_ok_status();
    for (size_t i = 0; i < kEngineCount && iree_status_is_ok(status); ++i) {
      status = iree_hal_device_group_builder_add_device(
          &builder, participants_[i].device);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), group_.out());
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);
    for (auto& participant : participants_) {
      participant.queue = iree_hal_device_queue(participant.device, 0, 0);
      ASSERT_NE(participant.queue, nullptr);
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          participant.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_NONE, participant.timeline.out()));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        participants_[kCpu].device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, start_.out()));
    ASSERT_NO_FATAL_FAILURE(CreatePool());
    ASSERT_NO_FATAL_FAILURE(LoadExecutables());
  }

  void TearDown() override {
    start_.reset();
    pool_.reset();
    group_.reset();
    for (auto& participant : participants_) {
      participant.executable.reset();
      participant.timeline.reset();
      participant.device.reset();
    }
    iree_hal_allocator_release(cpu_allocator_);
    iree_task_executor_release(cpu_executor_);
    context_.Deinitialize();
  }

  void CreateNativeDevice(Engine engine, iree_string_view_t scheme) {
    iree_host_size_t count = 0;
    iree_hal_device_info_t* infos = nullptr;
    IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
        driver_, scheme, iree_allocator_system(), &count, &infos));
    iree_allocator_free(iree_allocator_system(), infos);
    if (!count) {
      GTEST_SKIP() << "No native " << std::string(scheme.data, scheme.size)
                   << " endpoint";
    }
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_uri(
        driver_, scheme, context_.params(), iree_allocator_system(),
        participants_[engine].device.out()));
  }

  void CreatePool() {
    const iree_hal_buffer_interface_t interfaces[] = {
        IREE_HAL_BUFFER_INTERFACE_HOST, IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
        IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
    std::array<iree_hal_pool_family_access_t, kEngineCount> families = {};
    for (size_t i = 0; i < families.size(); ++i) {
      families[i].family = iree_hal_queue_family(participants_[i].queue);
      families[i].usage =
          IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
      families[i].interfaces = UINT64_C(1) << interfaces[i];
    }
    iree_hal_pool_scope_t scope = {
        .family_count = families.size(),
        .families = families.data(),
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        group_, scope, &options, iree_allocator_system(), pool_.out()));
  }

  void Load(Engine engine, iree_const_byte_span_t bytes,
            iree_string_view_t name) {
    auto& participant = participants_[engine];
    const auto* targets = iree_hal_device_spec_executables(
        iree_hal_device_spec(participant.device));
    ASSERT_EQ(targets->target_count, 1u);
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data = bytes;
    IREE_ASSERT_OK(iree_hal_executable_load(
        iree_hal_queue_family(participant.queue), &targets->targets[0], &params,
        participant.executable.out()));
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        participant.executable, name, &participant.function));
  }

  void LoadExecutables() {
    auto& cpu = participants_[kCpu];
    cpu.executable.reset(
        CreateCpuAddExecutable(iree_hal_queue_family(cpu.queue)));
    cpu.function = iree_hal_executable_function_from_index(0);
    const auto* npu_targets = iree_hal_device_spec_executables(
        iree_hal_device_spec(participants_[kNpu].device));
    ASSERT_EQ(npu_targets->target_count, 1u);
    const bool halo =
        iree_string_view_equal(npu_targets->targets[0].target_key,
                               IREE_SV("amd.xdna.strix_halo.17f0_11"));
    const auto* npu_image = halo ? iree_hal_amd_xdna_test_mul_i32_create()
                                 : iree_hal_amd_xdna_test_mul_i32_npu4_create();
    ASSERT_NO_FATAL_FAILURE(Load(
        kNpu, iree_make_const_byte_span(npu_image[0].data, npu_image[0].size),
        IREE_SV("mul_i32")));
    const auto* gpu_targets = iree_hal_device_spec_executables(
        iree_hal_device_spec(participants_[kGpu].device));
    ASSERT_EQ(gpu_targets->target_count, 1u);
    iree_hal_amd_gpu_target_identity_t identity;
    IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        gpu_targets->targets[0].target_key, &identity));
    ASSERT_TRUE(identity.version.major == 11 || identity.version.major == 12);
    const char* selector =
        identity.version.major == 11 ? "gfx11-generic" : "gfx12-generic";
    const auto* gpu_images = iree_hal_amd_cts_affine_images_create();
    for (size_t i = 0; i < iree_hal_amd_cts_affine_images_size(); ++i) {
      if (strstr(gpu_images[i].name, selector)) {
        ASSERT_NO_FATAL_FAILURE(Load(
            kGpu,
            iree_make_const_byte_span(gpu_images[i].data, gpu_images[i].size),
            IREE_SV("affine_i32")));
        break;
      }
    }
    ASSERT_TRUE(participants_[kGpu].executable);
  }

  static iree_hal_dispatch_config_t Config(Engine engine) {
    auto config = iree_hal_make_static_dispatch_config(
        engine == kCpu ? kElementCount : 1, 1, 1);
    config.workgroup_size[0] = engine == kGpu ? 32 : 1;
    config.workgroup_size[1] = config.workgroup_size[2] = 1;
    return config;
  }

  void AddTransition(Engine producer, Engine consumer,
                     iree_hal_memory_transition_action_t action,
                     iree_hal_buffer_ref_t reference, Boundary* boundary) {
    auto table = iree_hal_pool_transition_table(pool_);
    iree_hal_memory_scope_t scopes[2];
    const Engine engines[] = {producer, consumer};
    for (size_t i = 0; i < IREE_ARRAYSIZE(engines); ++i) {
      const iree_hal_memory_site_t site = {
          .kind = IREE_HAL_MEMORY_SITE_QUEUE,
          .family = iree_hal_queue_family(participants_[engines[i]].queue),
      };
      IREE_ASSERT_OK(
          iree_hal_device_group_resolve_memory_scope(group_, site, &scopes[i]));
    }
    iree_hal_memory_transition_pair_t pair;
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, scopes[0], scopes[1], action, &pair));
    const auto transition = iree_hal_memory_transition_query(table, pair);
    const auto effects = action == IREE_HAL_MEMORY_TRANSITION_RELEASE
                             ? transition.release
                             : transition.acquire;
    ASSERT_TRUE(iree_hal_memory_effects_is_supported(effects));
    boundary->barrier.effects =
        iree_hal_memory_effects_combine(boundary->barrier.effects, effects);
    const auto* recipe = iree_hal_memory_transition_recipe(table, pair, action);
    if (recipe) {
      auto& count = boundary->barrier.buffer_barrier_count;
      ASSERT_LT(count, boundary->ranges.size());
      boundary->ranges[count++] = {.buffer_ref = reference, .recipe = recipe};
      boundary->barrier.buffer_barriers = boundary->ranges.data();
    }
  }

  void Record(Engine engine, iree_hal_buffer_ref_list_t bindings,
              uint32_t addend, Invocation* invocation) {
    auto& participant = participants_[engine];
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(participant.queue),
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*binding_capacity=*/4, invocation->commands.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(invocation->commands));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(
        invocation->commands, &invocation->before.barrier));
    const auto constants =
        engine == kNpu ? iree_const_byte_span_empty()
                       : iree_make_const_byte_span(&addend, sizeof(addend));
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        invocation->commands, participant.executable, participant.function,
        Config(engine), constants, bindings, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(invocation->commands,
                                                   &invocation->after.barrier));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(invocation->commands));
  }

  void Dispatch(Engine engine, iree_hal_semaphore_t* wait, uint64_t wait_value,
                uint64_t signal_value, iree_hal_buffer_ref_list_t bindings,
                uint32_t addend, const Invocation& invocation,
                iree_hal_buffer_binding_table_t binding_table) {
    auto& participant = participants_[engine];
    auto* signal = participant.timeline.get();
    if (invocation.commands) {
      IREE_ASSERT_OK(iree_hal_queue_execute(
          participant.queue, {1, &wait, &wait_value},
          {1, &signal, &signal_value}, invocation.commands, binding_table,
          IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
      return;
    }
    const iree_hal_barrier_list_t before = {1, &invocation.before.barrier};
    const iree_hal_barrier_list_t after = {1, &invocation.after.barrier};
    const iree_hal_queue_barriers_t barriers = {&before, &after};
    const auto constants =
        engine == kNpu ? iree_const_byte_span_empty()
                       : iree_make_const_byte_span(&addend, sizeof(addend));
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        participant.queue, {1, &wait, &wait_value}, {1, &signal, &signal_value},
        participant.executable, participant.function, Config(engine), constants,
        bindings,
        std::get<1>(GetParam()) == SubmissionMode::kDefaults ? nullptr
                                                             : &barriers,
        IREE_HAL_DISPATCH_FLAG_NONE));
  }

  // CPU worker resources outlive the Task device using them.
  iree_task_executor_t* cpu_executor_ = nullptr;
  // Ordinary Task allocation source, independent of the shared physical pool.
  iree_hal_allocator_t* cpu_allocator_ = nullptr;
  // Proactors and frontier identities shared by all three device instances.
  ::iree::hal::cts::DeviceCreateContext context_;
  // Process-level registry holding the native VM across independent cases.
  static inline iree_hal_driver_registry_t* registry_ = nullptr;
  // Canonical AMD driver acquired through both public engine schemes.
  static inline iree_hal_driver_t* driver_ = nullptr;
  // Actual Task, XDNA, and PM4 implementations; no native submission bypass.
  std::array<Participant, kEngineCount> participants_;
  // Sealed topology carrying the complete participants and visibility scopes.
  Ref<iree_hal_device_group_t> group_;
  // One allocation source qualified for all three execution interfaces.
  Ref<iree_hal_pool_t> pool_;
  // Host-controlled gate keeping every consumer pending until capture finishes.
  Ref<iree_hal_semaphore_t> start_;
};

TEST_P(ExecutionTest, ConsumerFirstKernelsShareOneBacking) {
  Ref<iree_hal_buffer_t> root;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(sizeof(Payload), 64),
      iree_infinite_timeout(), root.out()));
  std::array<Ref<iree_hal_buffer_t>, 3> views;
  std::array<iree_hal_buffer_ref_t, 3> bindings;
  for (size_t i = 0; i < views.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_buffer_subspan(root, (i + 1) * kBytes, kBytes,
                                           iree_allocator_system(),
                                           views[i].out()));
    bindings[i] = iree_hal_make_buffer_ref(views[i], 0, kBytes);
  }
  const bool npu_first = std::get<0>(GetParam());
  const auto mode = std::get<1>(GetParam());
  const bool recorded = mode == SubmissionMode::kRecordedIndirect;
  const Engine first = npu_first ? kNpu : kGpu;
  const Engine second = npu_first ? kGpu : kNpu;
  const auto* gpu_binding = &bindings[npu_first ? 2 : 0];
  const iree_hal_buffer_binding_t table_bindings[] = {
      {views[0], 0, kBytes},
      {views[1], 0, kBytes},
      {views[2], 0, kBytes},
      {root, 0, sizeof(Payload)},
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      IREE_ARRAYSIZE(table_bindings), table_bindings};
  Invocation producer, native_first, native_second, consumer;
  Boundary upload_release, download_acquire;
  const auto root_ref = iree_hal_make_buffer_ref(root, 0, sizeof(Payload));
  const auto recorded_root_ref =
      recorded ? iree_hal_make_indirect_buffer_ref(3, 0, sizeof(Payload))
               : root_ref;
  if (mode != SubmissionMode::kDefaults) {
    // Upload also produces the unchanged RHS that the NPU reads in either
    // engine order, independently of the producer's later LHS update.
    for (Engine engine : {kNpu, kGpu}) {
      ASSERT_NO_FATAL_FAILURE(AddTransition(kCpu, engine,
                                            IREE_HAL_MEMORY_TRANSITION_RELEASE,
                                            root_ref, &upload_release));
    }
    ASSERT_NO_FATAL_FAILURE(AddTransition(kCpu, first,
                                          IREE_HAL_MEMORY_TRANSITION_RELEASE,
                                          recorded_root_ref, &producer.after));
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(kCpu, first, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                      recorded_root_ref, &native_first.before));
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(first, second, IREE_HAL_MEMORY_TRANSITION_RELEASE,
                      recorded_root_ref, &native_first.after));
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(first, second, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                      recorded_root_ref, &native_second.before));
    if (first == kGpu) {
      ASSERT_NO_FATAL_FAILURE(
          AddTransition(kCpu, kNpu, IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                        recorded_root_ref, &native_second.before));
      // Final CPU verification reads the GPU-modified LHS as well as the NPU
      // result; the NPU's output release cannot publish somebody else's writes.
      ASSERT_NO_FATAL_FAILURE(
          AddTransition(kGpu, kCpu, IREE_HAL_MEMORY_TRANSITION_RELEASE,
                        recorded_root_ref, &native_first.after));
      ASSERT_NO_FATAL_FAILURE(AddTransition(kGpu, kCpu,
                                            IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                                            root_ref, &download_acquire));
    }
    ASSERT_NO_FATAL_FAILURE(
        AddTransition(second, kCpu, IREE_HAL_MEMORY_TRANSITION_RELEASE,
                      recorded_root_ref, &native_second.after));
    ASSERT_NO_FATAL_FAILURE(AddTransition(second, kCpu,
                                          IREE_HAL_MEMORY_TRANSITION_ACQUIRE,
                                          recorded_root_ref, &consumer.before));
  }
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_barrier_list_t upload_after = {1, &upload_release.barrier};
  const iree_hal_barrier_list_t download_before = {1,
                                                   &download_acquire.barrier};
  const iree_hal_queue_barriers_t upload_barriers = {&empty, &upload_after};
  const iree_hal_queue_barriers_t download_barriers = {&download_before,
                                                       &empty};
  if (recorded) {
    std::array<iree_hal_buffer_ref_t, 3> references;
    for (size_t i = 0; i < references.size(); ++i) {
      references[i] = iree_hal_make_indirect_buffer_ref(i, 0, kBytes);
    }
    const auto gpu_reference = references[npu_first ? 2 : 0];
    ASSERT_NO_FATAL_FAILURE(Record(kCpu, {1, &references[0]}, 7, &producer));
    ASSERT_NO_FATAL_FAILURE(Record(
        first,
        first == kNpu
            ? iree_hal_buffer_ref_list_t{references.size(), references.data()}
            : iree_hal_buffer_ref_list_t{1, &gpu_reference},
        29, &native_first));
    ASSERT_NO_FATAL_FAILURE(Record(
        second,
        second == kNpu
            ? iree_hal_buffer_ref_list_t{references.size(), references.data()}
            : iree_hal_buffer_ref_list_t{1, &gpu_reference},
        29, &native_second));
    ASSERT_NO_FATAL_FAILURE(Record(kCpu, {1, &references[2]}, 100, &consumer));
  }
  auto* cpu_timeline = participants_[kCpu].timeline.get();
  auto* start = start_.get();
  for (uint32_t replay = 0; replay < 3; ++replay) {
    SCOPED_TRACE(replay);
    uint64_t gate = replay + 1;
    uint64_t initialized = replay * 4 + 1, produced = initialized + 1,
             consumed = initialized + 2, downloaded = initialized + 3;
    Payload initial;
    initial.fill(0xA5A5A5A5u);
    for (size_t i = 0; i < kElementCount; ++i) {
      initial[kElementCount + i] =
          static_cast<uint32_t>(i * 97 + 13 + replay * 211);
      initial[2 * kElementCount + i] =
          static_cast<uint32_t>(0xFFFFFF00u + i * 11 + replay);
    }
    Payload result = {};
    iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
    };
    download.download.source_buffer = root;
    download.download.target = result.data();
    download.download.length = sizeof(result);
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        participants_[kCpu].queue, {1, &cpu_timeline, &consumed},
        {1, &cpu_timeline, &downloaded}, 1, &download,
        mode == SubmissionMode::kDefaults ? nullptr : &download_barriers));
    ASSERT_NO_FATAL_FAILURE(Dispatch(kCpu, participants_[second].timeline, gate,
                                     consumed, {1, &bindings[2]}, 100, consumer,
                                     binding_table));
    // Submit every consumer before its producer. Each native engine signals its
    // own semaphore, so both native-to-foreign admission directions are
    // required.
    ASSERT_NO_FATAL_FAILURE(Dispatch(
        second, participants_[first].timeline, gate, gate,
        second == kNpu
            ? iree_hal_buffer_ref_list_t{bindings.size(), bindings.data()}
            : iree_hal_buffer_ref_list_t{1, gpu_binding},
        29, native_second, binding_table));
    ASSERT_NO_FATAL_FAILURE(Dispatch(
        first, cpu_timeline, produced, gate,
        first == kNpu
            ? iree_hal_buffer_ref_list_t{bindings.size(), bindings.data()}
            : iree_hal_buffer_ref_list_t{1, gpu_binding},
        29, native_first, binding_table));
    ASSERT_NO_FATAL_FAILURE(Dispatch(kCpu, cpu_timeline, initialized, produced,
                                     {1, &bindings[0]}, 7, producer,
                                     binding_table));
    iree_hal_transfer_operation_t upload = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
    };
    upload.update.source_buffer = initial.data();
    upload.update.target_buffer = root;
    upload.update.length = sizeof(initial);
    IREE_ASSERT_OK(iree_hal_queue_transfer(
        participants_[kCpu].queue, {1, &start, &gate},
        {1, &cpu_timeline, &initialized}, 1, &upload,
        mode == SubmissionMode::kDefaults ? nullptr : &upload_barriers));
    IREE_ASSERT_OK(iree_hal_semaphore_signal(start, gate, nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(cpu_timeline, downloaded,
                                           iree_infinite_timeout(), 0));
    Payload expected = initial;
    for (size_t i = 0; i < kElementCount; ++i) {
      uint32_t lhs = initial[kElementCount + i] + 7;
      if (first == kGpu) {
        lhs = lhs * 3 + 29;
      }
      expected[kElementCount + i] = lhs;
      uint32_t value = lhs * initial[2 * kElementCount + i];
      if (second == kGpu) {
        value = value * 3 + 29;
      }
      expected[3 * kElementCount + i] = value + 100;
    }
    EXPECT_EQ(result, expected);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SharedBacking, ExecutionTest,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(SubmissionMode::kDefaults,
                                         SubmissionMode::kExplicit,
                                         SubmissionMode::kRecordedIndirect)),
    [](const ::testing::TestParamInfo<ExecutionTest::ParamType>& info) {
      std::string name =
          std::get<0>(info.param) ? "CpuNpuGpuCpu" : "CpuGpuNpuCpu";
      switch (std::get<1>(info.param)) {
        case SubmissionMode::kDefaults:
          return name + "Defaults";
        case SubmissionMode::kExplicit:
          return name + "Explicit";
        case SubmissionMode::kRecordedIndirect:
          return name + "RecordedIndirect";
      }
      return name;
    });

}  // namespace
}  // namespace iree::hal::amd::cts
