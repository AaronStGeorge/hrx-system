# Execution Tests

Execution tests run command-line tools from YAML manifests and check selected
textual evidence from stdout, stderr, and generated files. The runner executes
tools directly with `subprocess.run`; it does not use a shell.

```yaml
version: 1

cases:
  - name: help mentions important flags
    run:
      tool: fixture
      args:
        - --helpish
    stdout:
      contains:
        - "Usage:"
        - "--output="
    stderr:
      empty: true
```

Each manifest contains one or more cases. A case can use a single `run` block or
an explicit `steps` list:

```yaml
cases:
  - name: write then run
    steps:
      - write:
          path: "{tmp}/input.txt"
          text: "hello\n"
      - run:
          tool: fixture
          args: ["--input={tmp}/input.txt"]
```

Manifest strings support `{srcdir}`, `{tmp}`, `{case}`, `{manifest}`, and
`{tool:name}` substitutions.

Run steps default to `exit: 0`. Stdout and stderr are ignored unless checks are
declared:

```yaml
stdout:
  contains:
    - "first literal"
    - regex: "second .+ regex"
  not_contains:
    - "forbidden literal"
```

`contains` lists are ordered by default. Use `unordered` when order is not part
of the contract:

```yaml
stdout:
  contains:
    unordered:
      - "alpha"
      - "beta"
```

Expected failures and file checks are explicit:

```yaml
exit:
  nonzero: true
stderr:
  contains:
    - "expected diagnostic"
files:
  - path: "{tmp}/module.vmfb"
    non_empty: true
```
