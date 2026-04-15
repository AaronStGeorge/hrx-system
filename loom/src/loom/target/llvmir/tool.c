// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/tool.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/path.h"

#if defined(IREE_PLATFORM_WINDOWS)

#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#else  // POSIX

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_ANDROID)
#define LOOM_LLVMIR_TOOL_POSIX 1
#endif

#if LOOM_LLVMIR_TOOL_POSIX
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif  // LOOM_LLVMIR_TOOL_POSIX

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Tool and argv utilities
//===----------------------------------------------------------------------===//

void loom_llvmir_toolchain_initialize_from_environment(
    loom_llvmir_toolchain_t* out_toolchain) {
  IREE_ASSERT_ARGUMENT(out_toolchain);
  const char* root_path = getenv("LOOM_LLVMIR_TOOLCHAIN_ROOT");
  if (root_path == NULL || root_path[0] == '\0') {
    root_path = getenv("LOOM_LLVMIR_TOOLCHAIN_BIN");
  }
  *out_toolchain = (loom_llvmir_toolchain_t){
      .root_path = root_path ? iree_make_cstring_view(root_path)
                             : iree_string_view_empty(),
  };
}

iree_string_view_t loom_llvmir_tool_name(loom_llvmir_tool_kind_t tool_kind) {
  switch (tool_kind) {
    case LOOM_LLVMIR_TOOL_LLVM_AS:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("llvm-as.exe");
#else
      return IREE_SV("llvm-as");
#endif
    case LOOM_LLVMIR_TOOL_LLVM_DIS:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("llvm-dis.exe");
#else
      return IREE_SV("llvm-dis");
#endif
    case LOOM_LLVMIR_TOOL_OPT:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("opt.exe");
#else
      return IREE_SV("opt");
#endif
    case LOOM_LLVMIR_TOOL_LLC:
#if defined(IREE_PLATFORM_WINDOWS)
      return IREE_SV("llc.exe");
#else
      return IREE_SV("llc");
#endif
  }
  return iree_string_view_empty();
}

static bool loom_llvmir_string_view_contains_nul(iree_string_view_t value) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] == '\0') return true;
  }
  return false;
}

static iree_status_t loom_llvmir_dup_cstring(iree_string_view_t value,
                                             iree_allocator_t allocator,
                                             char** out_string) {
  *out_string = NULL;
  if (value.data == NULL && value.size != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "string data is required");
  }
  if (loom_llvmir_string_view_contains_nul(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tool arguments cannot contain NUL bytes");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "string size overflow");
  }
  char* string = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&string));
  memcpy(string, value.data, value.size);
  string[value.size] = '\0';
  *out_string = string;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_tool_executable_path(
    const loom_llvmir_toolchain_t* toolchain, loom_llvmir_tool_kind_t tool_kind,
    iree_allocator_t allocator, char** out_executable_path,
    bool* out_search_path) {
  *out_executable_path = NULL;
  *out_search_path = false;

  iree_string_view_t tool_name = loom_llvmir_tool_name(tool_kind);
  if (iree_string_view_is_empty(tool_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown LLVM tool kind %d", (int)tool_kind);
  }

  if (toolchain == NULL || iree_string_view_is_empty(toolchain->root_path)) {
    *out_search_path = true;
    return loom_llvmir_dup_cstring(tool_name, allocator, out_executable_path);
  }

  return iree_file_path_join(toolchain->root_path, tool_name, allocator,
                             out_executable_path);
}

static void loom_llvmir_tool_argv_deinitialize(char** argv,
                                               iree_allocator_t allocator) {
  if (argv == NULL) return;
  for (iree_host_size_t i = 0; argv[i] != NULL; ++i) {
    iree_allocator_free(allocator, argv[i]);
  }
  iree_allocator_free(allocator, argv);
}

static iree_status_t loom_llvmir_tool_argv_allocate(
    char* executable_path, const iree_string_view_t* arguments,
    iree_host_size_t argument_count, iree_allocator_t allocator,
    char*** out_argv) {
  *out_argv = NULL;
  iree_host_size_t argv_count = 0;
  iree_host_size_t total_count = 0;
  if (!iree_host_size_checked_add(argument_count, 1, &argv_count) ||
      !iree_host_size_checked_add(argv_count, 1, &total_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "argv size overflow");
  }
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_mul(total_count, sizeof(char*), &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "argv allocation size overflow");
  }

  char** argv = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&argv));
  memset(argv, 0, total_size);
  argv[0] = executable_path;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < argument_count && iree_status_is_ok(status);
       ++i) {
    status = loom_llvmir_dup_cstring(arguments[i], allocator, &argv[i + 1]);
  }
  if (iree_status_is_ok(status)) {
    *out_argv = argv;
  } else {
    argv[0] = NULL;
    loom_llvmir_tool_argv_deinitialize(argv, allocator);
  }
  return status;
}

static iree_status_t loom_llvmir_tool_append_text(
    iree_string_builder_t* builder, const char* data, iree_host_size_t length) {
  return iree_string_builder_append_string(builder,
                                           iree_make_string_view(data, length));
}

void loom_llvmir_tool_output_deinitialize(loom_llvmir_tool_output_t* output,
                                          iree_allocator_t allocator) {
  if (output == NULL) return;
  iree_allocator_free(allocator, output->data);
  *output = (loom_llvmir_tool_output_t){0};
}

void loom_llvmir_tool_result_deinitialize(loom_llvmir_tool_result_t* result,
                                          iree_allocator_t allocator) {
  if (result == NULL) return;
  loom_llvmir_tool_output_deinitialize(&result->stdout_text, allocator);
  loom_llvmir_tool_output_deinitialize(&result->stderr_text, allocator);
  result->exit_code = 0;
}

bool loom_llvmir_tool_result_succeeded(
    const loom_llvmir_tool_result_t* result) {
  return result != NULL && result->exit_code == 0;
}

static iree_status_t loom_llvmir_tool_output_take(
    iree_string_builder_t* builder, loom_llvmir_tool_output_t* out_output) {
  out_output->length = iree_string_builder_size(builder);
  out_output->data = iree_string_builder_take_storage(builder);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_tool_checked_status(
    loom_llvmir_tool_kind_t tool_kind, const loom_llvmir_tool_result_t* result,
    iree_string_view_t action) {
  iree_string_view_t tool_name = loom_llvmir_tool_name(tool_kind);
  int tool_name_length =
      (int)iree_min(tool_name.size, (iree_host_size_t)INT_MAX);
  int stdout_length =
      (int)iree_min(result->stdout_text.length, (iree_host_size_t)INT_MAX);
  int stderr_length =
      (int)iree_min(result->stderr_text.length, (iree_host_size_t)INT_MAX);
  int action_length = (int)iree_min(action.size, (iree_host_size_t)INT_MAX);
  const char* stdout_data =
      result->stdout_text.data ? result->stdout_text.data : "";
  const char* stderr_data =
      result->stderr_text.data ? result->stderr_text.data : "";
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "LLVM tool %.*s failed while %.*s with exit code %d\nstdout:\n%.*s\n"
      "stderr:\n%.*s",
      tool_name_length, tool_name.data, action_length, action.data,
      result->exit_code, stdout_length, stdout_data, stderr_length,
      stderr_data);
}

//===----------------------------------------------------------------------===//
// Platform capture files and process execution
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

typedef struct loom_llvmir_capture_file_t {
  // Capturing file handle passed to the child process.
  HANDLE handle;
  // Temporary filesystem path backing |handle|.
  char path[MAX_PATH];
} loom_llvmir_capture_file_t;

static iree_status_t loom_llvmir_win32_status(DWORD error,
                                              const char* message) {
  return iree_make_status(iree_status_code_from_win32_error(error),
                          "%s (GetLastError=%lu)", message,
                          (unsigned long)error);
}

static iree_status_t loom_llvmir_capture_file_open(
    const char* prefix, loom_llvmir_capture_file_t* out_file) {
  memset(out_file, 0, sizeof(*out_file));
  char temp_directory[MAX_PATH] = {0};
  DWORD temp_directory_length =
      GetTempPathA(IREE_ARRAYSIZE(temp_directory), temp_directory);
  if (temp_directory_length == 0 ||
      temp_directory_length >= IREE_ARRAYSIZE(temp_directory)) {
    return loom_llvmir_win32_status(GetLastError(),
                                    "failed to resolve temporary directory");
  }

  if (GetTempFileNameA(temp_directory, prefix, 0, out_file->path) == 0) {
    return loom_llvmir_win32_status(GetLastError(),
                                    "failed to allocate temporary file path");
  }

  SECURITY_ATTRIBUTES security_attributes;
  memset(&security_attributes, 0, sizeof(security_attributes));
  security_attributes.nLength = sizeof(security_attributes);
  security_attributes.bInheritHandle = TRUE;

  HANDLE handle = CreateFileA(
      out_file->path, GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      &security_attributes, CREATE_ALWAYS,
      FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED, NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    DeleteFileA(out_file->path);
    memset(out_file, 0, sizeof(*out_file));
    return loom_llvmir_win32_status(error, "failed to open capture file");
  }
  out_file->handle = handle;
  return iree_ok_status();
}

static void loom_llvmir_capture_file_deinitialize(
    loom_llvmir_capture_file_t* file) {
  if (file->handle != NULL && file->handle != INVALID_HANDLE_VALUE) {
    CloseHandle(file->handle);
  }
  if (file->path[0] != '\0') {
    DeleteFileA(file->path);
  }
  memset(file, 0, sizeof(*file));
}

static iree_status_t loom_llvmir_capture_file_read(
    loom_llvmir_capture_file_t* file, iree_allocator_t allocator,
    loom_llvmir_tool_output_t* out_output) {
  *out_output = (loom_llvmir_tool_output_t){0};
  if (SetFilePointer(file->handle, 0, NULL, FILE_BEGIN) ==
      INVALID_SET_FILE_POINTER) {
    DWORD error = GetLastError();
    if (error != NO_ERROR) {
      return loom_llvmir_win32_status(error, "failed to rewind capture file");
    }
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_ok_status();
  uint8_t buffer[4096];
  bool reading = true;
  while (reading && iree_status_is_ok(status)) {
    DWORD bytes_read = 0;
    if (!ReadFile(file->handle, buffer, sizeof(buffer), &bytes_read, NULL)) {
      status =
          loom_llvmir_win32_status(GetLastError(), "failed to read capture");
      break;
    }
    if (bytes_read == 0) {
      reading = false;
    } else {
      status = loom_llvmir_tool_append_text(&builder, (const char*)buffer,
                                            (iree_host_size_t)bytes_read);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_tool_output_take(&builder, out_output);
  }
  if (!iree_status_is_ok(status)) {
    loom_llvmir_tool_output_deinitialize(out_output, allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static bool loom_llvmir_win32_arg_needs_quotes(const char* argument) {
  if (argument[0] == '\0') return true;
  for (const char* p = argument; *p != '\0'; ++p) {
    if (*p == ' ' || *p == '\t' || *p == '"') return true;
  }
  return false;
}

static iree_status_t loom_llvmir_win32_append_char(
    iree_string_builder_t* builder, char value) {
  return iree_string_builder_append_string(builder,
                                           iree_make_string_view(&value, 1));
}

static iree_status_t loom_llvmir_win32_append_repeated_char(
    iree_string_builder_t* builder, char value, iree_host_size_t count) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_llvmir_win32_append_char(builder, value);
  }
  return status;
}

static iree_status_t loom_llvmir_win32_append_quoted_arg(
    iree_string_builder_t* builder, const char* argument) {
  if (!loom_llvmir_win32_arg_needs_quotes(argument)) {
    return iree_string_builder_append_cstring(builder, argument);
  }

  iree_status_t status = loom_llvmir_win32_append_char(builder, '"');
  iree_host_size_t backslash_count = 0;
  for (const char* p = argument; *p != '\0' && iree_status_is_ok(status); ++p) {
    if (*p == '\\') {
      ++backslash_count;
    } else if (*p == '"') {
      status = loom_llvmir_win32_append_repeated_char(builder, '\\',
                                                      backslash_count * 2 + 1);
      if (iree_status_is_ok(status)) {
        status = loom_llvmir_win32_append_char(builder, '"');
      }
      backslash_count = 0;
    } else {
      status = loom_llvmir_win32_append_repeated_char(builder, '\\',
                                                      backslash_count);
      if (iree_status_is_ok(status)) {
        status = loom_llvmir_win32_append_char(builder, *p);
      }
      backslash_count = 0;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_win32_append_repeated_char(builder, '\\',
                                                    backslash_count * 2);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_win32_append_char(builder, '"');
  }
  return status;
}

static iree_status_t loom_llvmir_win32_make_command_line(
    char** argv, iree_allocator_t allocator, char** out_command_line) {
  *out_command_line = NULL;
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; argv[i] != NULL && iree_status_is_ok(status);
       ++i) {
    if (i != 0) {
      status = loom_llvmir_win32_append_char(&builder, ' ');
    }
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_win32_append_quoted_arg(&builder, argv[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_command_line = iree_string_builder_take_storage(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_llvmir_tool_process_run(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_llvmir_tool_result_t* out_result) {
  (void)search_path;
  *out_result = (loom_llvmir_tool_result_t){0};

  loom_llvmir_capture_file_t stdout_file;
  loom_llvmir_capture_file_t stderr_file;
  IREE_RETURN_IF_ERROR(loom_llvmir_capture_file_open("out", &stdout_file));
  iree_status_t status = loom_llvmir_capture_file_open("err", &stderr_file);

  char* command_line = NULL;
  PROCESS_INFORMATION process_info;
  memset(&process_info, 0, sizeof(process_info));
  if (iree_status_is_ok(status)) {
    status =
        loom_llvmir_win32_make_command_line(argv, allocator, &command_line);
  }
  if (iree_status_is_ok(status)) {
    STARTUPINFOA startup_info;
    memset(&startup_info, 0, sizeof(startup_info));
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;
    startup_info.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup_info.hStdOutput = stdout_file.handle;
    startup_info.hStdError = stderr_file.handle;

    if (!CreateProcessA(NULL, command_line, NULL, NULL, TRUE, 0, NULL, NULL,
                        &startup_info, &process_info)) {
      status =
          loom_llvmir_win32_status(GetLastError(), "failed to spawn LLVM tool");
    }
  }
  if (iree_status_is_ok(status)) {
    WaitForSingleObject(process_info.hProcess, INFINITE);
    DWORD exit_code = 1;
    if (!GetExitCodeProcess(process_info.hProcess, &exit_code)) {
      status = loom_llvmir_win32_status(GetLastError(),
                                        "failed to query LLVM tool exit code");
    } else {
      out_result->exit_code = (int)exit_code;
    }
  }
  if (process_info.hThread != NULL) CloseHandle(process_info.hThread);
  if (process_info.hProcess != NULL) CloseHandle(process_info.hProcess);
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_capture_file_read(&stdout_file, allocator,
                                           &out_result->stdout_text);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_capture_file_read(&stderr_file, allocator,
                                           &out_result->stderr_text);
  }
  if (!iree_status_is_ok(status)) {
    loom_llvmir_tool_result_deinitialize(out_result, allocator);
  }
  iree_allocator_free(allocator, command_line);
  loom_llvmir_capture_file_deinitialize(&stderr_file);
  loom_llvmir_capture_file_deinitialize(&stdout_file);
  return status;
}

#elif LOOM_LLVMIR_TOOL_POSIX

typedef struct loom_llvmir_capture_file_t {
  // Open file descriptor passed to the child process.
  int fd;
} loom_llvmir_capture_file_t;

static const char* loom_llvmir_temp_directory(void) {
  const char* temp_directory = getenv("TEST_TMPDIR");
  if (temp_directory != NULL && temp_directory[0] != '\0') {
    return temp_directory;
  }
  temp_directory = getenv("TMPDIR");
  if (temp_directory != NULL && temp_directory[0] != '\0') {
    return temp_directory;
  }
  return "/tmp";
}

static iree_status_t loom_llvmir_posix_status(int error_number,
                                              const char* message) {
  return iree_make_status(iree_status_code_from_errno(error_number),
                          "%s (%d: %s)", message, error_number,
                          strerror(error_number));
}

static iree_status_t loom_llvmir_capture_file_open(
    const char* stem, loom_llvmir_capture_file_t* out_file) {
  out_file->fd = -1;
  char template_path[4096] = {0};
  int length = iree_snprintf(template_path, sizeof(template_path),
                             "%s/loom_llvmir_%s_XXXXXX",
                             loom_llvmir_temp_directory(), stem);
  if (length < 0 || (iree_host_size_t)length >= sizeof(template_path)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "temporary capture path is too long");
  }
  int fd = mkstemp(template_path);
  if (fd < 0) {
    return loom_llvmir_posix_status(errno, "failed to open capture file");
  }
  unlink(template_path);
  out_file->fd = fd;
  return iree_ok_status();
}

static void loom_llvmir_capture_file_deinitialize(
    loom_llvmir_capture_file_t* file) {
  if (file->fd >= 0) {
    close(file->fd);
    file->fd = -1;
  }
}

static iree_status_t loom_llvmir_capture_file_read(
    loom_llvmir_capture_file_t* file, iree_allocator_t allocator,
    loom_llvmir_tool_output_t* out_output) {
  *out_output = (loom_llvmir_tool_output_t){0};
  if (lseek(file->fd, 0, SEEK_SET) < 0) {
    return loom_llvmir_posix_status(errno, "failed to rewind capture file");
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_ok_status();
  char buffer[4096];
  bool reading = true;
  while (reading && iree_status_is_ok(status)) {
    ssize_t bytes_read = read(file->fd, buffer, sizeof(buffer));
    if (bytes_read > 0) {
      status = loom_llvmir_tool_append_text(&builder, buffer,
                                            (iree_host_size_t)bytes_read);
    } else if (bytes_read == 0) {
      reading = false;
    } else if (errno != EINTR) {
      status = loom_llvmir_posix_status(errno, "failed to read capture file");
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_tool_output_take(&builder, out_output);
  }
  if (!iree_status_is_ok(status)) {
    loom_llvmir_tool_output_deinitialize(out_output, allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_llvmir_tool_process_wait(pid_t pid,
                                                   int* out_exit_code) {
  *out_exit_code = 1;
  int status = 0;
  pid_t wait_result = -1;
  do {
    wait_result = waitpid(pid, &status, 0);
  } while (wait_result < 0 && errno == EINTR);
  if (wait_result < 0) {
    return loom_llvmir_posix_status(errno, "failed to wait for LLVM tool");
  }
  if (WIFEXITED(status)) {
    *out_exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    *out_exit_code = 128 + WTERMSIG(status);
  } else {
    *out_exit_code = 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_tool_process_run(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_llvmir_tool_result_t* out_result) {
  *out_result = (loom_llvmir_tool_result_t){0};

  loom_llvmir_capture_file_t stdout_file;
  loom_llvmir_capture_file_t stderr_file;
  IREE_RETURN_IF_ERROR(loom_llvmir_capture_file_open("stdout", &stdout_file));
  iree_status_t status = loom_llvmir_capture_file_open("stderr", &stderr_file);

  posix_spawn_file_actions_t file_actions;
  bool file_actions_initialized = false;
  if (iree_status_is_ok(status)) {
    int spawn_result = posix_spawn_file_actions_init(&file_actions);
    if (spawn_result == 0) {
      file_actions_initialized = true;
    } else {
      status = loom_llvmir_posix_status(spawn_result,
                                        "failed to initialize spawn actions");
    }
  }
  if (iree_status_is_ok(status)) {
    int spawn_result = posix_spawn_file_actions_adddup2(
        &file_actions, stdout_file.fd, STDOUT_FILENO);
    if (spawn_result != 0) {
      status =
          loom_llvmir_posix_status(spawn_result, "failed to redirect stdout");
    }
  }
  if (iree_status_is_ok(status)) {
    int spawn_result = posix_spawn_file_actions_adddup2(
        &file_actions, stderr_file.fd, STDERR_FILENO);
    if (spawn_result != 0) {
      status =
          loom_llvmir_posix_status(spawn_result, "failed to redirect stderr");
    }
  }
  pid_t pid = 0;
  if (iree_status_is_ok(status)) {
    int spawn_result =
        search_path
            ? posix_spawnp(&pid, argv[0], &file_actions, NULL, argv, environ)
            : posix_spawn(&pid, argv[0], &file_actions, NULL, argv, environ);
    if (spawn_result != 0) {
      status =
          loom_llvmir_posix_status(spawn_result, "failed to spawn LLVM tool");
    }
  }
  if (file_actions_initialized) {
    posix_spawn_file_actions_destroy(&file_actions);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_tool_process_wait(pid, &out_result->exit_code);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_capture_file_read(&stdout_file, allocator,
                                           &out_result->stdout_text);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_capture_file_read(&stderr_file, allocator,
                                           &out_result->stderr_text);
  }
  if (!iree_status_is_ok(status)) {
    loom_llvmir_tool_result_deinitialize(out_result, allocator);
  }
  loom_llvmir_capture_file_deinitialize(&stderr_file);
  loom_llvmir_capture_file_deinitialize(&stdout_file);
  return status;
}

#else

static iree_status_t loom_llvmir_tool_process_run(
    char** argv, bool search_path, iree_allocator_t allocator,
    loom_llvmir_tool_result_t* out_result) {
  (void)argv;
  (void)search_path;
  (void)allocator;
  *out_result = (loom_llvmir_tool_result_t){0};
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "LLVM tool execution is unsupported on this platform");
}

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Public execution API
//===----------------------------------------------------------------------===//

iree_status_t loom_llvmir_tool_run(const loom_llvmir_toolchain_t* toolchain,
                                   loom_llvmir_tool_kind_t tool_kind,
                                   const iree_string_view_t* arguments,
                                   iree_host_size_t argument_count,
                                   iree_allocator_t allocator,
                                   loom_llvmir_tool_result_t* out_result) {
  if (out_result == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM tool result output is required");
  }
  *out_result = (loom_llvmir_tool_result_t){0};
  if (arguments == NULL && argument_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM tool arguments are required");
  }

  char* executable_path = NULL;
  bool search_path = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_tool_executable_path(
      toolchain, tool_kind, allocator, &executable_path, &search_path));

  char** argv = NULL;
  iree_status_t status = loom_llvmir_tool_argv_allocate(
      executable_path, arguments, argument_count, allocator, &argv);
  if (iree_status_is_ok(status)) {
    status =
        loom_llvmir_tool_process_run(argv, search_path, allocator, out_result);
  } else {
    iree_allocator_free(allocator, executable_path);
  }
  loom_llvmir_tool_argv_deinitialize(argv, allocator);
  return status;
}

static iree_status_t loom_llvmir_tool_run_checked(
    const loom_llvmir_toolchain_t* toolchain, loom_llvmir_tool_kind_t tool_kind,
    const iree_string_view_t* arguments, iree_host_size_t argument_count,
    iree_allocator_t allocator, iree_string_view_t action) {
  loom_llvmir_tool_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_llvmir_tool_run(
      toolchain, tool_kind, arguments, argument_count, allocator, &result));

  iree_status_t status = iree_ok_status();
  if (!loom_llvmir_tool_result_succeeded(&result)) {
    status = loom_llvmir_tool_checked_status(tool_kind, &result, action);
  }
  loom_llvmir_tool_result_deinitialize(&result, allocator);
  return status;
}

iree_status_t loom_llvmir_tool_query_version(
    const loom_llvmir_toolchain_t* toolchain, loom_llvmir_tool_kind_t tool_kind,
    iree_allocator_t allocator, loom_llvmir_tool_output_t* out_version_text) {
  if (out_version_text == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM version text output is required");
  }
  *out_version_text = (loom_llvmir_tool_output_t){0};
  iree_string_view_t arguments[] = {IREE_SV("--version")};
  loom_llvmir_tool_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_llvmir_tool_run(toolchain, tool_kind, arguments,
                                            IREE_ARRAYSIZE(arguments),
                                            allocator, &result));

  iree_status_t status = iree_ok_status();
  if (loom_llvmir_tool_result_succeeded(&result)) {
    *out_version_text = result.stdout_text;
    result.stdout_text = (loom_llvmir_tool_output_t){0};
  } else {
    status = loom_llvmir_tool_checked_status(
        tool_kind, &result, IREE_SV("querying LLVM tool version"));
  }
  loom_llvmir_tool_result_deinitialize(&result, allocator);
  return status;
}

iree_status_t loom_llvmir_tool_assemble_text_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, iree_allocator_t allocator) {
  iree_string_view_t arguments[] = {
      input_path,
      IREE_SV("-o"),
      output_path,
  };
  return loom_llvmir_tool_run_checked(
      toolchain, LOOM_LLVMIR_TOOL_LLVM_AS, arguments, IREE_ARRAYSIZE(arguments),
      allocator, IREE_SV("assembling LLVM IR text"));
}

iree_status_t loom_llvmir_tool_disassemble_bitcode_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_allocator_t allocator, loom_llvmir_tool_output_t* out_text) {
  if (out_text == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM disassembly output is required");
  }
  *out_text = (loom_llvmir_tool_output_t){0};
  iree_string_view_t arguments[] = {
      input_path,
      IREE_SV("-o"),
      IREE_SV("-"),
  };
  loom_llvmir_tool_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_tool_run(toolchain, LOOM_LLVMIR_TOOL_LLVM_DIS, arguments,
                           IREE_ARRAYSIZE(arguments), allocator, &result));

  iree_status_t status = iree_ok_status();
  if (loom_llvmir_tool_result_succeeded(&result)) {
    *out_text = result.stdout_text;
    result.stdout_text = (loom_llvmir_tool_output_t){0};
  } else {
    status =
        loom_llvmir_tool_checked_status(LOOM_LLVMIR_TOOL_LLVM_DIS, &result,
                                        IREE_SV("disassembling LLVM bitcode"));
  }
  loom_llvmir_tool_result_deinitialize(&result, allocator);
  return status;
}

iree_status_t loom_llvmir_tool_verify_bitcode_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_allocator_t allocator) {
  iree_string_view_t arguments[] = {
      IREE_SV("-passes=verify"),
      input_path,
      IREE_SV("-disable-output"),
  };
  return loom_llvmir_tool_run_checked(
      toolchain, LOOM_LLVMIR_TOOL_OPT, arguments, IREE_ARRAYSIZE(arguments),
      allocator, IREE_SV("verifying LLVM bitcode"));
}

iree_status_t loom_llvmir_tool_compile_object_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator) {
  if (extra_arguments == NULL && extra_argument_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "extra LLVM tool arguments are required");
  }

  iree_host_size_t argument_count = 0;
  if (!iree_host_size_checked_add(extra_argument_count, 4, &argument_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "llc argument count overflow");
  }
  iree_host_size_t argument_size = 0;
  if (!iree_host_size_checked_mul(argument_count, sizeof(iree_string_view_t),
                                  &argument_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "llc argument allocation overflow");
  }

  iree_string_view_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, argument_size, (void**)&arguments));
  arguments[0] = input_path;
  arguments[1] = IREE_SV("-filetype=obj");
  arguments[2] = IREE_SV("-o");
  arguments[3] = output_path;
  for (iree_host_size_t i = 0; i < extra_argument_count; ++i) {
    arguments[i + 4] = extra_arguments[i];
  }

  iree_status_t status = loom_llvmir_tool_run_checked(
      toolchain, LOOM_LLVMIR_TOOL_LLC, arguments, argument_count, allocator,
      IREE_SV("compiling LLVM bitcode to an object file"));
  iree_allocator_free(allocator, arguments);
  return status;
}
