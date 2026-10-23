// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/model.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_profile_file_record_t MakeChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk;
  memset(&chunk, 0, sizeof(chunk));
  chunk.header.record_type = IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK;
  chunk.header.session_id = 1;
  chunk.content_type = IREE_SV("application/vnd.iree.test");
  chunk.payload = iree_make_const_byte_span(payload.data(), payload.size());
  return chunk;
}

static iree_hal_profile_file_record_t MakeCommandBuffersChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeDevicesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeQueuesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeExecutablesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeExecutableFunctionsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeCommandOperationsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeMetricSourcesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeMetricDescriptorsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS;
  return chunk;
}

template <typename T>
static void AppendPlainRecord(std::vector<uint8_t>* payload, const T& record) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record));
  memcpy(payload->data() + offset, &record, sizeof(record));
}

template <typename T>
static void AppendRecordWithPayload(std::vector<uint8_t>* payload,
                                    const T& record,
                                    iree_string_view_t trailing_payload) {
  AppendPlainRecord(payload, record);
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + trailing_payload.size);
  memcpy(payload->data() + offset, trailing_payload.data,
         trailing_payload.size);
}

static void AppendCommandBuffer(std::vector<uint8_t>* payload,
                                uint64_t command_buffer_id) {
  iree_hal_profile_command_buffer_record_t record =
      iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = command_buffer_id;
  record.physical_device_ordinal = 0;
  AppendPlainRecord(payload, record);
}

static void AppendQueue(std::vector<uint8_t>* payload,
                        uint32_t physical_device_ordinal,
                        uint32_t queue_ordinal, uint64_t stream_id) {
  iree_hal_profile_queue_record_t record =
      iree_hal_profile_queue_record_default();
  record.physical_device_ordinal = physical_device_ordinal;
  record.queue_ordinal = queue_ordinal;
  record.stream_id = stream_id;
  AppendPlainRecord(payload, record);
}

static void AppendCommandOperation(
    std::vector<uint8_t>* payload,
    const iree_hal_profile_command_operation_record_t& record) {
  AppendPlainRecord(payload, record);
}

static void AppendExecutable(std::vector<uint8_t>* payload,
                             uint64_t executable_id, uint32_t function_count) {
  iree_hal_profile_executable_record_t record =
      iree_hal_profile_executable_record_default();
  record.executable_id = executable_id;
  record.function_count = function_count;
  AppendPlainRecord(payload, record);
}

static void AppendExecutableFunction(std::vector<uint8_t>* payload,
                                     uint64_t executable_id,
                                     uint32_t function_ordinal,
                                     iree_string_view_t name) {
  iree_hal_profile_executable_function_record_t record =
      iree_hal_profile_executable_function_record_default();
  record.executable_id = executable_id;
  record.function_ordinal = function_ordinal;
  record.name_length = name.size;
  record.record_length = sizeof(record) + name.size;
  AppendRecordWithPayload(payload, record, name);
}

static void AppendMetricSource(std::vector<uint8_t>* payload,
                               uint64_t source_id, iree_string_view_t name) {
  iree_hal_profile_device_metric_source_record_t record =
      iree_hal_profile_device_metric_source_record_default();
  record.source_id = source_id;
  record.physical_device_ordinal = 0;
  record.device_class = IREE_HAL_PROFILE_DEVICE_CLASS_GPU;
  record.metric_count = 1;
  record.name_length = name.size;
  record.record_length = sizeof(record) + record.name_length;
  AppendRecordWithPayload(payload, record, name);
}

static void AppendMetricDescriptor(std::vector<uint8_t>* payload,
                                   uint64_t source_id, uint64_t metric_id,
                                   iree_string_view_t name,
                                   iree_string_view_t description) {
  iree_hal_profile_device_metric_descriptor_record_t record =
      iree_hal_profile_device_metric_descriptor_record_default();
  record.source_id = source_id;
  record.metric_id = metric_id;
  record.unit = IREE_HAL_PROFILE_METRIC_UNIT_HERTZ;
  record.value_kind = IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64;
  record.semantic = IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT;
  record.plot_hint = IREE_HAL_PROFILE_METRIC_PLOT_HINT_FREQUENCY;
  record.name_length = name.size;
  record.description_length = description.size;
  record.record_length = sizeof(record) + name.size + description.size;
  AppendPlainRecord(payload, record);
  const iree_host_size_t name_offset = payload->size();
  payload->resize(name_offset + name.size + description.size);
  memcpy(payload->data() + name_offset, name.data, name.size);
  memcpy(payload->data() + name_offset + name.size, description.data,
         description.size);
}

static iree_hal_profile_clock_correlation_record_t MakeClockSample(
    uint64_t sample_id, uint64_t device_tick, uint64_t host_cpu_timestamp_ns) {
  iree_hal_profile_clock_correlation_record_t sample =
      iree_hal_profile_clock_correlation_record_default();
  sample.flags = IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
                 IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP;
  sample.sample_id = sample_id;
  sample.device_tick = device_tick;
  sample.host_cpu_timestamp_ns = host_cpu_timestamp_ns;
  return sample;
}

TEST(ProfileClockFitTest, MapsTicksWithIntegerRounding) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.first_clock_sample =
      MakeClockSample(10, (1ull << 60) + 3, 900000000000000000ull);
  device.last_clock_sample =
      MakeClockSample(11, (1ull << 60) + 10, 900000000000000010ull);

  iree_profile_model_clock_fit_t fit;
  ASSERT_TRUE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      &fit));
  EXPECT_EQ(10u, fit.first_sample_id);
  EXPECT_EQ(11u, fit.last_sample_id);
  EXPECT_EQ(7u, fit.device_tick_span);
  EXPECT_EQ(10u, fit.time_span_ns);

  int64_t time_ns = 0;
  EXPECT_TRUE(
      iree_profile_model_clock_fit_map_tick(&fit, (1ull << 60) + 6, &time_ns));
  EXPECT_EQ(900000000000000004ll, time_ns);
  EXPECT_TRUE(
      iree_profile_model_clock_fit_map_tick(&fit, (1ull << 60), &time_ns));
  EXPECT_EQ(899999999999999996ll, time_ns);

  int64_t duration_ns = 0;
  EXPECT_TRUE(
      iree_profile_model_clock_fit_scale_ticks_to_ns(&fit, 7, &duration_ns));
  EXPECT_EQ(10, duration_ns);
  EXPECT_TRUE(
      iree_profile_model_clock_fit_scale_ticks_to_ns(&fit, 4, &duration_ns));
  EXPECT_EQ(6, duration_ns);
}

TEST(ProfileClockFitTest, FitsIreeHostTimeFromBracketMidpoints) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.first_clock_sample = MakeClockSample(1, 1000, 5000);
  device.first_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  device.first_clock_sample.host_time_begin_ns = 100;
  device.first_clock_sample.host_time_end_ns = 103;
  device.last_clock_sample = MakeClockSample(2, 1010, 6000);
  device.last_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  device.last_clock_sample.host_time_begin_ns = 170;
  device.last_clock_sample.host_time_end_ns = 173;

  iree_profile_model_clock_fit_t fit;
  ASSERT_TRUE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_IREE_HOST_TIME_NS, &fit));
  EXPECT_EQ(101, fit.first_time_ns);
  EXPECT_EQ(171, fit.last_time_ns);

  int64_t time_ns = 0;
  EXPECT_TRUE(iree_profile_model_clock_fit_map_tick(&fit, 1005, &time_ns));
  EXPECT_EQ(136, time_ns);
}

TEST(ProfileClockFitTest, RejectsInvalidDeviceTickAlignment) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.invalid_clock_alignment_sample_count = 1;
  device.first_clock_sample = MakeClockSample(1, 1000, 5000);
  device.last_clock_sample = MakeClockSample(2, 1010, 6000);
  device.last_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED;

  iree_profile_model_clock_fit_t fit;
  EXPECT_FALSE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      &fit));
}

TEST(ProfileDurationScaleTest, UsesMetadataWithoutHostClockFit) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.metadata_flags = IREE_HAL_PROFILE_DEVICE_FLAG_TIMESTAMP_FREQUENCY;
  device.timestamp_frequency_hz = 100000000;
  device.clock_sample_count = 2;
  device.invalid_clock_alignment_sample_count = 2;
  device.first_clock_sample = MakeClockSample(1, 1000, 5000);
  device.last_clock_sample = MakeClockSample(2, 2000, 15000);
  device.first_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED;
  device.last_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK_UNALIGNED;

  iree_profile_model_clock_fit_t clock_fit;
  EXPECT_FALSE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      &clock_fit));

  iree_profile_model_duration_scale_t duration_scale;
  ASSERT_TRUE(iree_profile_model_device_try_resolve_duration_scale(
      &device, &duration_scale));
  EXPECT_EQ(IREE_PROFILE_MODEL_DURATION_SCALE_SOURCE_DEVICE_METADATA,
            duration_scale.source);
  EXPECT_EQ(100000000u, duration_scale.device_tick_span);
  EXPECT_EQ(1000000000u, duration_scale.time_span_ns);
  int64_t duration_ns = 0;
  EXPECT_TRUE(iree_profile_model_duration_scale_ticks_to_ns(&duration_scale,
                                                            123, &duration_ns));
  EXPECT_EQ(1230, duration_ns);
}

TEST(ProfileDurationScaleTest, UsesClockCorrelationForOlderProfiles) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.first_clock_sample = MakeClockSample(1, 1000, 5000);
  device.last_clock_sample = MakeClockSample(2, 2000, 15000);

  iree_profile_model_duration_scale_t duration_scale;
  ASSERT_TRUE(iree_profile_model_device_try_resolve_duration_scale(
      &device, &duration_scale));
  EXPECT_EQ(IREE_PROFILE_MODEL_DURATION_SCALE_SOURCE_CLOCK_CORRELATION,
            duration_scale.source);
  int64_t duration_ns = 0;
  EXPECT_TRUE(iree_profile_model_duration_scale_ticks_to_ns(&duration_scale,
                                                            123, &duration_ns));
  EXPECT_EQ(1230, duration_ns);
}

TEST(ProfileModelTest, AcceptsOriginalDeviceRecordPrefix) {
  iree_hal_profile_device_record_t device_record =
      iree_hal_profile_device_record_default();
  device_record.record_length = IREE_HAL_PROFILE_DEVICE_RECORD_MIN_LENGTH;
  device_record.physical_device_ordinal = 3;
  device_record.queue_count = 1;
  std::vector<uint8_t> payload(IREE_HAL_PROFILE_DEVICE_RECORD_MIN_LENGTH);
  memcpy(payload.data(), &device_record, payload.size());
  iree_hal_profile_file_record_t device_chunk = MakeDevicesChunk(payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &device_chunk));
  const iree_profile_model_device_t* device =
      iree_profile_model_find_device(&model, 1, 3);
  ASSERT_NE(device, nullptr);
  EXPECT_FALSE(
      iree_any_bit_set(device->metadata_flags,
                       IREE_HAL_PROFILE_DEVICE_FLAG_TIMESTAMP_FREQUENCY));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, AcceptsLinearCommandOperationsWithoutBlockStructure) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_operation_chunk));
  ASSERT_EQ(1u, model.command_operation_count);
  EXPECT_FALSE(iree_hal_profile_command_operation_has_block_structure(
      &model.command_operations[0].record));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, LinksExecutableFunctionsByOwner) {
  std::vector<uint8_t> executable_payload;
  AppendExecutable(&executable_payload, 5, 2);
  AppendExecutable(&executable_payload, 7, 1);
  iree_hal_profile_file_record_t executable_chunk =
      MakeExecutablesChunk(executable_payload);

  std::vector<uint8_t> function_payload;
  AppendExecutableFunction(&function_payload, 5, 0, IREE_SV("main"));
  AppendExecutableFunction(&function_payload, 7, 0, IREE_SV("side"));
  AppendExecutableFunction(&function_payload, 5, 1, IREE_SV("tail"));
  iree_hal_profile_file_record_t function_chunk =
      MakeExecutableFunctionsChunk(function_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &executable_chunk));
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &function_chunk));

  const iree_profile_model_executable_t* executable =
      iree_profile_model_find_executable(&model, 1, 5);
  ASSERT_NE(executable, nullptr);
  ASSERT_EQ(2u, executable->function_row_count);
  ASSERT_NE(IREE_HOST_SIZE_MAX, executable->first_function_index);
  const iree_profile_model_function_t* first_function =
      &model.functions[executable->first_function_index];
  EXPECT_EQ(5u, first_function->executable_id);
  EXPECT_EQ(0u, first_function->function_ordinal);
  ASSERT_NE(IREE_HOST_SIZE_MAX, first_function->next_function_index);
  const iree_profile_model_function_t* second_function =
      &model.functions[first_function->next_function_index];
  EXPECT_EQ(5u, second_function->executable_id);
  EXPECT_EQ(1u, second_function->function_ordinal);
  EXPECT_EQ(IREE_HOST_SIZE_MAX, second_function->next_function_index);

  const iree_profile_model_executable_t* other_executable =
      iree_profile_model_find_executable(&model, 1, 7);
  ASSERT_NE(other_executable, nullptr);
  EXPECT_EQ(1u, other_executable->function_row_count);
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, SeparatesSessionLocalMetadataIdentities) {
  constexpr iree_hal_profile_session_id_t kFirstSession = 11;
  constexpr iree_hal_profile_session_id_t kSecondSession = 22;

  std::vector<uint8_t> device_payload;
  iree_hal_profile_device_record_t device_record =
      iree_hal_profile_device_record_default();
  device_record.physical_device_ordinal = 0;
  AppendPlainRecord(&device_payload, device_record);
  iree_hal_profile_file_record_t first_device_chunk =
      MakeDevicesChunk(device_payload);
  first_device_chunk.header.session_id = kFirstSession;
  iree_hal_profile_file_record_t second_device_chunk =
      MakeDevicesChunk(device_payload);
  second_device_chunk.header.session_id = kSecondSession;

  std::vector<uint8_t> queue_payload;
  AppendQueue(&queue_payload, 0, 0, 1);
  iree_hal_profile_file_record_t first_queue_chunk =
      MakeQueuesChunk(queue_payload);
  first_queue_chunk.header.session_id = kFirstSession;
  iree_hal_profile_file_record_t second_queue_chunk =
      MakeQueuesChunk(queue_payload);
  second_queue_chunk.header.session_id = kSecondSession;

  std::vector<uint8_t> executable_payload;
  AppendExecutable(&executable_payload, 1, 1);
  iree_hal_profile_file_record_t first_executable_chunk =
      MakeExecutablesChunk(executable_payload);
  first_executable_chunk.header.session_id = kFirstSession;
  iree_hal_profile_file_record_t second_executable_chunk =
      MakeExecutablesChunk(executable_payload);
  second_executable_chunk.header.session_id = kSecondSession;

  std::vector<uint8_t> first_function_payload;
  AppendExecutableFunction(&first_function_payload, 1, 0,
                           IREE_SV("first_function"));
  iree_hal_profile_file_record_t first_function_chunk =
      MakeExecutableFunctionsChunk(first_function_payload);
  first_function_chunk.header.session_id = kFirstSession;
  std::vector<uint8_t> second_function_payload;
  AppendExecutableFunction(&second_function_payload, 1, 0,
                           IREE_SV("second_function"));
  iree_hal_profile_file_record_t second_function_chunk =
      MakeExecutableFunctionsChunk(second_function_payload);
  second_function_chunk.header.session_id = kSecondSession;

  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t first_command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);
  first_command_buffer_chunk.header.session_id = kFirstSession;
  iree_hal_profile_file_record_t second_command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);
  second_command_buffer_chunk.header.session_id = kSecondSession;

  iree_hal_profile_command_operation_record_t operation_record =
      iree_hal_profile_command_operation_record_default();
  operation_record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL;
  operation_record.command_buffer_id = 1;
  operation_record.command_index = 0;
  std::vector<uint8_t> operation_payload;
  AppendCommandOperation(&operation_payload, operation_record);
  iree_hal_profile_file_record_t first_operation_chunk =
      MakeCommandOperationsChunk(operation_payload);
  first_operation_chunk.header.session_id = kFirstSession;
  iree_hal_profile_file_record_t second_operation_chunk =
      MakeCommandOperationsChunk(operation_payload);
  second_operation_chunk.header.session_id = kSecondSession;

  std::vector<uint8_t> first_source_payload;
  AppendMetricSource(&first_source_payload, 1, IREE_SV("first.metrics"));
  iree_hal_profile_file_record_t first_source_chunk =
      MakeMetricSourcesChunk(first_source_payload);
  first_source_chunk.header.session_id = kFirstSession;
  std::vector<uint8_t> second_source_payload;
  AppendMetricSource(&second_source_payload, 1, IREE_SV("second.metrics"));
  iree_hal_profile_file_record_t second_source_chunk =
      MakeMetricSourcesChunk(second_source_payload);
  second_source_chunk.header.session_id = kSecondSession;

  std::vector<uint8_t> first_descriptor_payload;
  AppendMetricDescriptor(&first_descriptor_payload, 1, 1,
                         IREE_SV("first.metric"), IREE_SV("First metric."));
  iree_hal_profile_file_record_t first_descriptor_chunk =
      MakeMetricDescriptorsChunk(first_descriptor_payload);
  first_descriptor_chunk.header.session_id = kFirstSession;
  std::vector<uint8_t> second_descriptor_payload;
  AppendMetricDescriptor(&second_descriptor_payload, 1, 1,
                         IREE_SV("second.metric"), IREE_SV("Second metric."));
  iree_hal_profile_file_record_t second_descriptor_chunk =
      MakeMetricDescriptorsChunk(second_descriptor_payload);
  second_descriptor_chunk.header.session_id = kSecondSession;

  const iree_hal_profile_file_record_t* chunks[] = {
      &first_device_chunk,         &second_device_chunk,
      &first_queue_chunk,          &second_queue_chunk,
      &first_executable_chunk,     &second_executable_chunk,
      &first_function_chunk,       &second_function_chunk,
      &first_command_buffer_chunk, &second_command_buffer_chunk,
      &first_operation_chunk,      &second_operation_chunk,
      &first_source_chunk,         &second_source_chunk,
      &first_descriptor_chunk,     &second_descriptor_chunk,
  };
  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  for (const iree_hal_profile_file_record_t* chunk : chunks) {
    IREE_ASSERT_OK(iree_profile_model_process_metadata_record(&model, chunk));
  }

  EXPECT_EQ(2u, model.device_count);
  EXPECT_EQ(2u, model.queue_count);
  EXPECT_EQ(2u, model.executable_count);
  EXPECT_EQ(2u, model.function_count);
  EXPECT_EQ(2u, model.command_buffer_count);
  EXPECT_EQ(2u, model.command_operation_count);
  EXPECT_EQ(2u, model.metric_source_count);
  EXPECT_EQ(2u, model.metric_descriptor_count);
  EXPECT_NE(nullptr, iree_profile_model_find_device(&model, kFirstSession, 0));
  EXPECT_NE(nullptr,
            iree_profile_model_find_queue(&model, kSecondSession, 0, 0, 1));

  const iree_profile_model_function_t* first_function =
      iree_profile_model_find_function(&model, kFirstSession, 1, 0);
  const iree_profile_model_function_t* second_function =
      iree_profile_model_find_function(&model, kSecondSession, 1, 0);
  ASSERT_NE(nullptr, first_function);
  ASSERT_NE(nullptr, second_function);
  EXPECT_TRUE(
      iree_string_view_equal(first_function->name, IREE_SV("first_function")));
  EXPECT_TRUE(iree_string_view_equal(second_function->name,
                                     IREE_SV("second_function")));

  const iree_profile_model_command_buffer_t* first_command_buffer =
      iree_profile_model_find_command_buffer(&model, kFirstSession, 1);
  const iree_profile_model_command_buffer_t* second_command_buffer =
      iree_profile_model_find_command_buffer(&model, kSecondSession, 1);
  ASSERT_NE(nullptr, first_command_buffer);
  ASSERT_NE(nullptr, second_command_buffer);
  EXPECT_EQ(
      kFirstSession,
      model.command_operations[first_command_buffer->first_operation_index]
          .session_id);
  EXPECT_EQ(
      kSecondSession,
      model.command_operations[second_command_buffer->first_operation_index]
          .session_id);

  const iree_profile_model_metric_source_t* first_source =
      iree_profile_model_find_metric_source(&model, kFirstSession, 1);
  const iree_profile_model_metric_source_t* second_source =
      iree_profile_model_find_metric_source(&model, kSecondSession, 1);
  ASSERT_NE(nullptr, first_source);
  ASSERT_NE(nullptr, second_source);
  EXPECT_TRUE(
      iree_string_view_equal(first_source->name, IREE_SV("first.metrics")));
  EXPECT_TRUE(
      iree_string_view_equal(second_source->name, IREE_SV("second.metrics")));
  EXPECT_NE(nullptr, iree_profile_model_find_metric_descriptor(
                         &model, kFirstSession, 1, 1));
  EXPECT_NE(nullptr, iree_profile_model_find_metric_descriptor(
                         &model, kSecondSession, 1, 1));

  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsExecutableFunctionWithoutOwner) {
  std::vector<uint8_t> function_payload;
  AppendExecutableFunction(&function_payload, 5, 0, IREE_SV("main"));
  iree_hal_profile_file_record_t function_chunk =
      MakeExecutableFunctionsChunk(function_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_profile_model_process_metadata_record(&model, &function_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsCommandOperationBlockFieldsWithoutFlag) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;
  operation.block_ordinal = 2;
  operation.block_command_ordinal = 3;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_model_process_metadata_record(
                            &model, &command_operation_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsBlockStructureWithoutBlockCoordinates) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;
  operation.flags = IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_BLOCK_STRUCTURE;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_model_process_metadata_record(
                            &model, &command_operation_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, IndexesDeviceMetricDescriptors) {
  std::vector<uint8_t> source_payload;
  AppendMetricSource(&source_payload, 1, IREE_SV("test.metrics"));
  iree_hal_profile_file_record_t source_chunk =
      MakeMetricSourcesChunk(source_payload);

  std::vector<uint8_t> descriptor_payload;
  AppendMetricDescriptor(&descriptor_payload, 1, 2, IREE_SV("test.clock"),
                         IREE_SV("Test clock."));
  iree_hal_profile_file_record_t descriptor_chunk =
      MakeMetricDescriptorsChunk(descriptor_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &source_chunk));
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &descriptor_chunk));

  const iree_profile_model_metric_source_t* source =
      iree_profile_model_find_metric_source(&model, 1, 1);
  ASSERT_NE(source, nullptr);
  EXPECT_TRUE(iree_string_view_equal(source->name, IREE_SV("test.metrics")));

  const iree_profile_model_metric_descriptor_t* descriptor = NULL;
  IREE_ASSERT_OK(iree_profile_model_resolve_metric_descriptor(&model, 1, 1, 2,
                                                              &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor->name, IREE_SV("test.clock")));
  EXPECT_TRUE(
      iree_string_view_equal(descriptor->description, IREE_SV("Test clock.")));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsMissingDeviceMetricDescriptorReference) {
  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);

  const iree_profile_model_metric_descriptor_t* descriptor = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_model_resolve_metric_descriptor(
                            &model, 1, 1, 2, &descriptor));
  EXPECT_EQ(nullptr, descriptor);
  iree_profile_model_deinitialize(&model);
}

}  // namespace
