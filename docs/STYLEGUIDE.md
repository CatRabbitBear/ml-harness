# HOUSERULES.md
## Coding & Architecture Rules for ml-harness

These rules apply to all code unless explicitly overridden.

---

## 1. Core is sacred

The `core` package:
- contains orchestration logic, contracts, and lifecycle management
- must remain lightweight and dependency-minimal
- must not import concrete ML implementations

If something feels “plugin-specific”, it probably does not belong in core.

---

## 2. Public vs internal APIs

Public APIs:
- `core.api.run_pipeline`
- dataclasses / types in `core.contracts`

These MUST:
- have docstrings
- remain backward-compatible once introduced
- avoid leaking internal implementation details

Internal modules:
- may change freely
- should not be relied upon by plugins or apps

---

## 3. Contracts first, implementation second

Before adding logic:
- define or update a contract (RunSpec, Plugin, Registry, etc.)
- ensure the contract makes sense for *multiple future models* (HMM, RL, etc.)

Avoid baking assumptions about:
- training style
- metrics shape
- model size
- execution time

---

## 4. Imports and circularity

Rules:
- Inside `core/contracts/*`, import sibling modules directly
  (`from core.contracts.run_spec import RunSpec`)
- Do NOT import `core.contracts` (the package) inside contract submodules
- Use `TYPE_CHECKING` for type-only imports when appropriate

If you see “partially initialized module” errors, stop and fix imports
before proceeding.

---

## 5. Configuration and environment

- Configuration is passed explicitly (objects, dicts, context)
- No hidden reliance on environment variables inside core
- Apps are responsible for reading env / config and injecting dependencies

---

## 6. Tests are specifications

Tests should:
- read like documentation
- assert behaviour, not implementation
- prefer clarity over cleverness

Bad:
```python
assert False
```
Good:
```
with pytest.raises(SomeError):
    ...
```
7. Incremental design rule

When unsure:

    implement the smallest reasonable version

    leave explicit TODO comments

    do not over-generalise early

We are optimising for evolution, not premature completeness.
8. When to stop and ask

Stop and ask before:

    adding a new top-level dependency to core

    introducing a new lifecycle stage

    changing public contracts

    adding implicit plugin discovery

    adding global singletons

End of HOUSERULES.md
