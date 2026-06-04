# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line interface for dev.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from build_tools.devtools import doctor, help_text, hooks, presubmit, setup
from build_tools.devtools.command_plan import CommandPlan, CommandStep
from build_tools.devtools.environment import (
    REPO_ROOT,
    existing_or_system_environment,
    tool_environment_from_args,
)

LANES = ("bazel", "cmake")
DEFAULT_BAZEL_TARGETS = ("//runtime/...", "//libhrx/...")
PASSTHROUGH_COMMANDS = {
    "bazel": frozenset(("configure", "build", "test", "run", "try", "fuzz")),
    "cmake": frozenset(("configure", "build", "test", "run", "try")),
}
HELP_FLAGS = frozenset(("-h", "--help"))
DEV_OPTIONS_WITH_VALUES = frozenset(
    (
        "--alias-dir",
        "--alias_dir",
        "--base",
        "--cmake-build-dir",
        "--cmake_build_dir",
        "--profile",
        "--tool-root",
        "--tool_root",
    )
)


def expand_option_aliases(option_strings: tuple[str, ...]) -> tuple[str, ...]:
    expanded_option_strings = []
    seen_option_strings = set()
    for option_string in option_strings:
        for candidate_option_string in option_aliases(option_string):
            if candidate_option_string in seen_option_strings:
                continue
            expanded_option_strings.append(candidate_option_string)
            seen_option_strings.add(candidate_option_string)
    return tuple(expanded_option_strings)


def option_aliases(option_string: str) -> tuple[str, ...]:
    if not option_string.startswith("--"):
        return (option_string,)

    aliases = [option_string]
    option_name = option_string[2:]
    if "-" in option_name:
        aliases.append("--" + option_name.replace("-", "_"))
    if "_" in option_name:
        aliases.append("--" + option_name.replace("_", "-"))
    return tuple(aliases)


def add_argument(
    parser: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup,
    *option_strings: str,
    **kwargs,
) -> argparse.Action:
    return parser.add_argument(*expand_option_aliases(option_strings), **kwargs)


def cmake_build_dir(args: argparse.Namespace | None = None) -> Path:
    configured_build_dir = getattr(args, "cmake_build_dir", None)
    if configured_build_dir is not None:
        build_dir = Path(configured_build_dir)
        if build_dir.is_absolute():
            return build_dir
        return REPO_ROOT / build_dir
    return REPO_ROOT.parent / "builds" / REPO_ROOT.name


def forwarded_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def cmake_build_args(backend_args: list[str]) -> list[str]:
    target_names = []
    raw_args = []
    for index, arg in enumerate(backend_args):
        if arg.startswith("-"):
            raw_args = backend_args[index:]
            break
        target_names.append(arg)
    else:
        raw_args = []

    cmake_args = []
    for target_name in target_names:
        cmake_args.extend(["--target", target_name])
    cmake_args.extend(raw_args)
    return cmake_args


def option_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def option_consumes_next_arg(arg: str) -> bool:
    return "=" not in arg and option_name(arg) in DEV_OPTIONS_WITH_VALUES


def find_passthrough_tail(argv: list[str]) -> int | None:
    index = 0
    while index < len(argv):
        arg = argv[index]
        if option_consumes_next_arg(arg):
            index += 2
            continue
        if arg not in LANES:
            index += 1
            continue

        lane = arg
        command_index = index + 1
        while command_index < len(argv):
            command = argv[command_index]
            if option_consumes_next_arg(command):
                command_index += 2
                continue
            if command in PASSTHROUGH_COMMANDS[lane]:
                return command_index + 1
            if command.startswith("-"):
                command_index += 1
                continue
            return None
        return None
    return None


def normalize_passthrough_args(argv: list[str]) -> list[str]:
    tail_index = find_passthrough_tail(argv)
    if tail_index is None:
        return argv

    tail = argv[tail_index:]
    if not tail or tail[0] == "--":
        return argv
    if len(tail) == 1 and tail[0] in HELP_FLAGS:
        return argv
    if not any(arg.startswith("-") for arg in tail):
        return argv
    return [*argv[:tail_index], "--", *tail]


def add_common_options(parser: argparse.ArgumentParser) -> None:
    add_argument(
        parser,
        "--agent-md",
        "--agents-md",
        action="store_true",
        default=argparse.SUPPRESS,
        dest="agent_md",
        help="Print AGENTS.md-ready command guidance.",
    )
    add_argument(
        parser,
        "-n",
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print the command plan without executing it.",
    )
    add_argument(
        parser,
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print extra diagnostics.",
    )
    add_argument(
        parser,
        "--cmake-build-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "CMake build directory for cmake commands. Relative paths are "
            "resolved from the repository root."
        ),
    )


def add_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    *,
    command_help: help_text.CommandHelp,
    help: str,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        name,
        description=command_help.description,
        epilog=command_help.epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=help,
    )


def add_tool_environment_options(parser: argparse.ArgumentParser) -> None:
    mode_group = parser.add_mutually_exclusive_group()
    add_argument(
        mode_group,
        "--venv",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use the repo-local .venv tool environment.",
    )
    add_argument(
        mode_group,
        "--system",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use tools from the system PATH and install nothing.",
    )
    add_argument(
        mode_group,
        "--tool-root",
        type=Path,
        default=argparse.SUPPRESS,
        help="Use or create an external tool environment root.",
    )


def add_profile_option(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    command: str,
) -> None:
    add_argument(
        parser,
        "--profile",
        choices=presubmit.PROFILES,
        default=default,
        help=f"{command} profile. Defaults to {default}.",
    )


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    argv = normalize_passthrough_args(argv)
    parser = argparse.ArgumentParser(
        prog="dev.py",
        description="Repository developer command router.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_options(parser)
    add_tool_environment_options(parser)
    subparsers = parser.add_subparsers(dest="command")

    add_root_commands(subparsers)
    for lane in LANES:
        add_lane_commands(subparsers, lane)

    args = parser.parse_args(argv)
    if not hasattr(args, "agent_md"):
        args.agent_md = False
    if not hasattr(args, "dry_run"):
        args.dry_run = False
    if not hasattr(args, "verbose"):
        args.verbose = False
    selected_tool_modes = [
        bool(getattr(args, "venv", False)),
        bool(getattr(args, "system", False)),
        getattr(args, "tool_root", None) is not None,
    ]
    if sum(selected_tool_modes) > 1:
        parser.error("--venv, --system, and --tool-root are mutually exclusive")
    if getattr(args, "paths", None) and (
        getattr(args, "base", None) is not None
        or getattr(args, "staged", False)
        or getattr(args, "commit", False)
    ):
        parser.error("explicit precommit paths cannot be combined with input mode")
    if args.agent_md:
        lane = getattr(args, "lane", None)
        print_agent_md((lane,) if lane else LANES)
        raise SystemExit(0)
    if not args.command:
        parser.print_help()
        raise SystemExit(2)
    if args.command in LANES and not hasattr(args, "handler"):
        parser.error(f"{args.command} requires a subcommand unless --agents_md is used")
    return args


def add_root_commands(subparsers: argparse._SubParsersAction) -> None:
    setup_parser = add_subparser(
        subparsers,
        "setup",
        command_help=help_text.root_command_help("setup"),
        help="Set up common developer tools.",
    )
    add_common_options(setup_parser)
    add_tool_environment_options(setup_parser)
    add_argument(
        setup_parser, "--alias-dir", type=Path, help="Directory for generated aliases."
    )
    setup_parser.set_defaults(handler=handle_root_setup)

    doctor_parser = add_subparser(
        subparsers,
        "doctor",
        command_help=help_text.root_command_help("doctor"),
        help="Check common developer tools.",
    )
    add_common_options(doctor_parser)
    add_tool_environment_options(doctor_parser)
    doctor_parser.set_defaults(handler=handle_root_doctor)

    tools_parser = subparsers.add_parser(
        "tools", help="Inspect pinned standalone tools."
    )
    add_common_options(tools_parser)
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command", required=True)
    list_parser = tools_subparsers.add_parser(
        "list", help="List known standalone tools."
    )
    add_common_options(list_parser)
    list_parser.set_defaults(handler=handle_tools_list)
    check_parser = tools_subparsers.add_parser(
        "check", help="Check installed standalone tools."
    )
    add_common_options(check_parser)
    add_tool_environment_options(check_parser)
    check_parser.set_defaults(handler=handle_tools_check)


def add_lane_commands(subparsers: argparse._SubParsersAction, lane: str) -> None:
    lane_parser = subparsers.add_parser(
        lane,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=f"{lane} developer commands.",
    )
    add_common_options(lane_parser)
    add_tool_environment_options(lane_parser)
    lane_parser.set_defaults(lane=lane)
    lane_subparsers = lane_parser.add_subparsers(dest=f"{lane}_command")

    setup_parser = add_subparser(
        lane_subparsers,
        "setup",
        command_help=help_text.lane_command_help(lane, "setup"),
        help=f"Set up {lane} tools.",
    )
    add_common_options(setup_parser)
    add_tool_environment_options(setup_parser)
    add_argument(
        setup_parser, "--alias-dir", type=Path, help="Directory for generated aliases."
    )
    setup_parser.set_defaults(handler=handle_lane_setup, lane=lane)

    hook_parser = add_subparser(
        lane_subparsers,
        "hook",
        command_help=help_text.lane_command_help(lane, "hook"),
        help=f"Install {lane} Git hooks.",
    )
    add_common_options(hook_parser)
    add_tool_environment_options(hook_parser)
    add_argument(
        hook_parser, "--verify", action="store_true", help="Run the hook after install."
    )
    add_profile_option(
        hook_parser,
        default=presubmit.precommit_default_profile(lane),
        command="Git hook precommit",
    )
    hook_parser.set_defaults(handler=handle_hook, lane=lane)

    configure_help = help_text.lane_command_help(lane, "configure")
    configure_parser = add_subparser(
        lane_subparsers,
        "configure",
        command_help=configure_help,
        help=f"Configure {lane}.",
    )
    configure_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help=configure_help.arguments,
    )
    configure_parser.set_defaults(handler=handle_configure, lane=lane)

    build_help = help_text.lane_command_help(lane, "build")
    build_parser = add_subparser(
        lane_subparsers,
        "build",
        command_help=build_help,
        help=f"Build with {lane}.",
    )
    build_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help=build_help.arguments,
    )
    build_parser.set_defaults(handler=handle_build, lane=lane)

    test_help = help_text.lane_command_help(lane, "test")
    test_parser = add_subparser(
        lane_subparsers,
        "test",
        command_help=test_help,
        help=f"Test with {lane}.",
    )
    test_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help=test_help.arguments,
    )
    test_parser.set_defaults(handler=handle_test, lane=lane)

    precommit_parser = add_subparser(
        lane_subparsers,
        "precommit",
        command_help=help_text.lane_command_help(lane, "precommit"),
        help=f"Run {lane} checks for local changes.",
    )
    add_common_options(precommit_parser)
    add_tool_environment_options(precommit_parser)
    precommit_input_group = precommit_parser.add_mutually_exclusive_group()
    add_argument(
        precommit_input_group,
        "--base",
        metavar="GIT_REF",
        help="Check branch changes since GIT_REF plus local changes.",
    )
    add_argument(
        precommit_input_group,
        "--commit",
        action="store_true",
        help="Check the Git hook commit scope.",
    )
    add_argument(
        precommit_input_group,
        "--staged",
        action="store_true",
        help="Check only files staged for commit.",
    )
    add_profile_option(
        precommit_parser,
        default=presubmit.precommit_default_profile(lane),
        command="Precommit",
    )
    precommit_parser.add_argument(
        "paths",
        nargs="*",
        help="Explicit repo-relative paths. Used by the generated Git hook.",
    )
    precommit_parser.set_defaults(handler=handle_precommit, lane=lane)

    presubmit_parser = add_subparser(
        lane_subparsers,
        "presubmit",
        command_help=help_text.lane_command_help(lane, "presubmit"),
        help=f"Run {lane} presubmit.",
    )
    add_common_options(presubmit_parser)
    add_tool_environment_options(presubmit_parser)
    add_profile_option(
        presubmit_parser,
        default=presubmit.PRESUBMIT_DEFAULT_PROFILE,
        command="Presubmit",
    )
    add_argument(
        presubmit_parser,
        "--no-project-tests",
        dest="project_tests",
        action="store_false",
        default=True,
        help="Skip runtime/libhrx/loom project tests while still running root devtools tests.",
    )
    presubmit_parser.set_defaults(handler=handle_presubmit, lane=lane)

    fix_parser = add_subparser(
        lane_subparsers,
        "fix",
        command_help=help_text.lane_command_help(lane, "fix"),
        help="Apply staged mechanical fixups.",
    )
    add_common_options(fix_parser)
    add_tool_environment_options(fix_parser)
    fix_parser.set_defaults(handler=handle_fix, lane=lane)

    doctor_parser = add_subparser(
        lane_subparsers,
        "doctor",
        command_help=help_text.lane_command_help(lane, "doctor"),
        help=f"Check {lane} tools.",
    )
    add_common_options(doctor_parser)
    add_tool_environment_options(doctor_parser)
    doctor_parser.set_defaults(handler=handle_lane_doctor, lane=lane)

    for command_name in ("run", "try", "fuzz"):
        if lane == "cmake" and command_name == "fuzz":
            continue
        command_help = help_text.lane_command_help(lane, command_name)
        command_parser = add_subparser(
            lane_subparsers,
            command_name,
            command_help=command_help,
            help=f"{command_name} with {lane}.",
        )
        command_parser.add_argument(
            "args", nargs=argparse.REMAINDER, help=command_help.arguments
        )
        command_parser.set_defaults(
            handler=handle_unimplemented_backend_command,
            lane=lane,
            backend_command=command_name,
        )


def handle_root_setup(args: argparse.Namespace) -> CommandPlan:
    tool_env = tool_environment_from_args(args)
    return setup.common_setup_plan(tool_env)


def handle_lane_setup(args: argparse.Namespace) -> CommandPlan:
    tool_env = tool_environment_from_args(args)
    return setup.setup_plan(args.lane, tool_env, args.alias_dir)


def handle_root_doctor(args: argparse.Namespace) -> CommandPlan:
    return doctor.doctor_plan(None, existing_or_system_environment(args))


def handle_lane_doctor(args: argparse.Namespace) -> CommandPlan:
    return doctor.doctor_plan(args.lane, existing_or_system_environment(args))


def handle_tools_list(args: argparse.Namespace) -> CommandPlan:
    return CommandPlan(
        [
            CommandStep(
                [
                    sys.executable,
                    str(REPO_ROOT / "build_tools/devtools/install.py"),
                    "--list",
                ],
                cwd=REPO_ROOT,
                label="list standalone developer tools",
            )
        ]
    )


def handle_tools_check(args: argparse.Namespace) -> CommandPlan:
    tool_env = existing_or_system_environment(args)
    command = [
        tool_env.python,
        str(REPO_ROOT / "build_tools/devtools/install.py"),
        "--check",
    ]
    if tool_env.bin_dir is not None:
        command += ["--bin-dir", str(tool_env.bin_dir)]
    return CommandPlan(
        [
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label="check standalone developer tools",
            )
        ]
    )


def handle_hook(args: argparse.Namespace) -> CommandPlan:
    return hooks.hook_plan(
        args.lane, existing_or_system_environment(args), args.verify, args.profile
    )


def handle_configure(args: argparse.Namespace) -> CommandPlan:
    tool_env = existing_or_system_environment(args)
    backend_args = forwarded_args(args.args)
    if args.lane == "bazel":
        command = [
            tool_env.python,
            str(REPO_ROOT / "build_tools/bazel/configure.py"),
            *backend_args,
        ]
    else:
        command = [
            tool_env.tool("cmake"),
            "-S",
            str(REPO_ROOT),
            "-B",
            str(cmake_build_dir(args)),
            *backend_args,
        ]
    return CommandPlan(
        [
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label=f"configure {args.lane}",
            )
        ]
    )


def handle_build(args: argparse.Namespace) -> CommandPlan:
    tool_env = existing_or_system_environment(args)
    backend_args = forwarded_args(args.args)
    if args.lane == "bazel":
        targets = backend_args or list(DEFAULT_BAZEL_TARGETS)
        command = [tool_env.tool("bazel"), "build", *targets]
    else:
        command = [
            tool_env.tool("cmake"),
            "--build",
            str(cmake_build_dir(args)),
            *cmake_build_args(backend_args),
        ]
    return CommandPlan(
        [
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label=f"{args.lane} build",
            )
        ]
    )


def handle_test(args: argparse.Namespace) -> CommandPlan:
    tool_env = existing_or_system_environment(args)
    backend_args = forwarded_args(args.args)
    if args.lane == "bazel":
        targets = backend_args or list(DEFAULT_BAZEL_TARGETS)
        command = [tool_env.tool("bazel"), "test", "--config=presubmit", *targets]
    else:
        command = [
            tool_env.tool("ctest"),
            "--test-dir",
            str(cmake_build_dir(args)),
            "--output-on-failure",
            *backend_args,
        ]
    return CommandPlan(
        [
            CommandStep(
                command,
                cwd=REPO_ROOT,
                env=tool_env.path_env(),
                label=f"{args.lane} test",
            )
        ]
    )


def handle_presubmit(args: argparse.Namespace) -> CommandPlan:
    return presubmit.presubmit_plan(
        args.lane,
        existing_or_system_environment(args),
        args.profile,
        verbose=args.verbose,
        project_tests=args.project_tests,
    )


def handle_precommit(args: argparse.Namespace) -> CommandPlan:
    return presubmit.precommit_plan(
        args.lane,
        existing_or_system_environment(args),
        args.profile,
        base=args.base,
        commit=args.commit,
        staged=args.staged,
        paths=args.paths,
        verbose=args.verbose,
    )


def handle_fix(args: argparse.Namespace) -> CommandPlan:
    return presubmit.fix_plan(
        existing_or_system_environment(args), verbose=args.verbose
    )


def handle_unimplemented_backend_command(args: argparse.Namespace) -> CommandPlan:
    return CommandPlan(
        [
            CommandStep(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        f"print('dev.py {args.lane} {args.backend_command} is not wired yet', file=sys.stderr); "
                        "sys.exit(1)"
                    ),
                ],
                cwd=REPO_ROOT,
                label=f"{args.lane} {args.backend_command}",
            )
        ]
    )


def print_agent_md(lanes: tuple[str, ...]) -> None:
    sections = [agent_md_header(lanes), ""]
    for lane in lanes:
        if lane == "bazel":
            sections.append(agent_md_bazel_section())
        elif lane == "cmake":
            sections.append(agent_md_cmake_section())
        else:
            raise ValueError(f"unknown lane: {lane}")
        sections.append("")
    print("\n".join(sections).rstrip() + "\n")


def agent_md_header(lanes: tuple[str, ...]) -> str:
    if lanes == ("bazel",):
        command_shape = "`python dev.py bazel <command>`."
        wrapper_example = "`python dev.py --dry-run bazel build`"
    elif lanes == ("cmake",):
        command_shape = "`python dev.py cmake <command>`."
        wrapper_example = "`python dev.py --dry-run cmake build hrx`"
    else:
        command_shape = (
            "`python dev.py bazel <command>` for Bazel and "
            "`python dev.py cmake <command>` for CMake."
        )
        wrapper_example = "`python dev.py --dry-run bazel build`"
    return f"""## dev.py

Run from repo root: {command_shape} Use `--help` for command details.
Put wrapper options before the lane command: {wrapper_example}.
Long flags accept hyphen or underscore spellings: `--dry-run`, `--dry_run`.
Published build configuration options live in BUILDING.md."""


def agent_md_bazel_section() -> str:
    return """### Bazel

Use Bazel for source-tree builds, tests, and review checks. `build` and `test`
default to `//runtime/...` and `//libhrx/...` when no targets are given.
Use absolute labels such as `//runtime/src/iree/base/...`, not `:target`.

```bash
python dev.py bazel configure
python dev.py bazel configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
python dev.py bazel configure --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm
python dev.py bazel build [targets...]
python dev.py bazel test [targets...]
python dev.py bazel precommit [paths...]
python dev.py bazel precommit --profile paranoid
python dev.py bazel precommit --base origin/main
python dev.py bazel presubmit
```

`precommit` checks staged, unstaged, and untracked local changes.
`precommit --base REF` checks branch changes from the merge base with `REF`
through `HEAD`, plus local changes. `presubmit` is the full CI-shaped check.

Outputs: `bazel-bin/`, `bazel-testlogs/`, and `bazel-out/`."""


def agent_md_cmake_section() -> str:
    build_dir = "../builds/<checkout-name>"
    return f"""### CMake

Use CMake for package and install-test workflows. `configure` writes
`{build_dir}/`. `build TARGET` maps to `cmake --build ... --target TARGET`.

```bash
python dev.py cmake configure
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
python dev.py cmake configure -DIREE_HAL_DRIVER_AMDGPU=OFF -DLIBHRX_BUILD=OFF
python dev.py --cmake-build-dir build/cmake-asan cmake configure -DIREE_ENABLE_ASAN=ON
python dev.py cmake build hrx
python dev.py cmake test -R hrx
python dev.py cmake precommit
python dev.py cmake precommit --profile paranoid
python dev.py cmake presubmit
```

`precommit` checks local changes. `precommit --profile paranoid` adds affected
project CMake/CTest checks. `presubmit` is the full-tree CI-shaped check."""


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    plan = args.handler(args)
    return plan.run(dry_run=args.dry_run, verbose=args.verbose)
