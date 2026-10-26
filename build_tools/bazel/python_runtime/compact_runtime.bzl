# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compacting a hermetic Python runtime into tree artifacts."""

_DEVELOPMENT_DIRECTORIES = {
    "include": None,
    "libs": None,
}

_DEVELOPMENT_EXTENSIONS = {
    ".exp": None,
    ".lib": None,
    ".pdb": None,
}

def _runtime_relative_path(file):
    short_path = file.short_path
    if not short_path.startswith("../"):
        fail("Python runtime input must come from an external repository: {}".format(file))
    repository_name, separator, relative_path = short_path[3:].partition("/")
    if not separator or not repository_name or not relative_path:
        fail("Python runtime input has no repository-relative path: {}".format(file))
    return relative_path

def _is_runtime_file(relative_path):
    path_parts = relative_path.split("/")
    if path_parts[0] in _DEVELOPMENT_DIRECTORIES:
        return False
    basename = path_parts[-1].lower()
    for extension in _DEVELOPMENT_EXTENSIONS:
        if basename.endswith(extension):
            return False
    return True

def _compact_python_runtime_impl(ctx):
    interpreter_relative_path = _runtime_relative_path(ctx.file.interpreter)
    if not interpreter_relative_path.lower().endswith(".exe"):
        fail("Compact Windows Python interpreter must be an .exe: {}".format(interpreter_relative_path))
    included_sources = []
    source_by_relative_path = {}
    directory_names = {}
    root_file_names = {}

    for source in ctx.files.srcs:
        relative_path = _runtime_relative_path(source)
        if relative_path in source_by_relative_path:
            fail("Python runtime contains duplicate path '{}'".format(relative_path))
        source_by_relative_path[relative_path] = source
        if not _is_runtime_file(relative_path):
            continue
        included_sources.append(source)
        path_parts = relative_path.split("/")
        if len(path_parts) == 1:
            root_file_names[relative_path] = None
        else:
            directory_names[path_parts[0]] = None

    if interpreter_relative_path not in root_file_names:
        fail("Python interpreter '{}' is absent from runtime files".format(interpreter_relative_path))

    output_root = ctx.label.name
    output_directories = [
        ctx.actions.declare_directory("{}/{}".format(output_root, directory_name))
        for directory_name in sorted(directory_names)
    ]
    output_files = {
        relative_path: ctx.actions.declare_file("{}/{}".format(output_root, relative_path))
        for relative_path in sorted(root_file_names)
    }
    output_interpreter_actual = output_files[interpreter_relative_path]
    output_interpreter = ctx.actions.declare_file(
        "{}/{}".format(output_root, interpreter_relative_path[:-4] + ".cmd"),
    )
    archive_outputs = output_directories + output_files.values()
    runtime_outputs = archive_outputs + [output_interpreter]
    venv_bin_files = [
        output
        for relative_path, output in output_files.items()
        if relative_path.lower().endswith(".dll")
    ] + [output_interpreter_actual]

    archive = ctx.actions.declare_file(ctx.label.name + ".zip")
    archive_inputs = ctx.actions.args()
    for source in included_sources:
        archive_inputs.add("{}={}".format(_runtime_relative_path(source), source.path))
    archive_inputs.use_param_file("@%s", use_always = True)
    archive_inputs.set_param_file_format("multiline")
    ctx.actions.run(
        arguments = ["c", archive.path, archive_inputs],
        executable = ctx.executable._zipper,
        inputs = included_sources,
        mnemonic = "PyCompactRuntimeArchive",
        outputs = [archive],
        progress_message = "Archiving compact Python runtime %{label}",
    )

    extract_arguments = ctx.actions.args()
    extract_arguments.add("x")
    extract_arguments.add(archive)
    extract_arguments.add("-d")
    extract_arguments.add(output_interpreter_actual.dirname)
    ctx.actions.run(
        arguments = [extract_arguments],
        executable = ctx.executable._zipper,
        inputs = [archive],
        mnemonic = "PyCompactRuntimeExpand",
        outputs = archive_outputs,
        progress_message = "Expanding compact Python runtime %{label}",
    )

    # CPython must not create __pycache__ entries inside Bazel's declared tree
    # outputs. Preserve the launcher as sys.executable so child interpreters
    # also receive -B, including children that otherwise ignore PYTHON* state.
    ctx.actions.write(
        content = """@echo off
set "PYTHONEXECUTABLE=%~f0"
"%~dp0{}" -B %*
exit /b %ERRORLEVEL%
""".format(interpreter_relative_path),
        is_executable = True,
        output = output_interpreter,
    )

    return [
        DefaultInfo(
            executable = output_interpreter,
            files = depset(runtime_outputs),
            runfiles = ctx.runfiles(files = runtime_outputs),
        ),
        OutputGroupInfo(venv_bin_files = depset(venv_bin_files)),
    ]

compact_python_runtime = rule(
    implementation = _compact_python_runtime_impl,
    attrs = {
        "interpreter": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "srcs": attr.label_list(
            allow_files = True,
            allow_empty = False,
        ),
        "_zipper": attr.label(
            cfg = "exec",
            default = "@bazel_tools//tools/zip:zipper",
            executable = True,
        ),
    },
    executable = True,
)
