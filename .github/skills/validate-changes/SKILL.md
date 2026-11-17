---
name: validate-changes
description: Inspect changed files and validate Python or Streamlit changes with targeted and, when appropriate, complete test execution.
user-invocable: true
---

# Validate Changes

Use this skill to validate the current repository changes without modifying application code or repository history.

## Procedure

1. Inspect the Git diff:
   - Run `git diff --stat` and `git diff --`.
   - Also inspect staged changes with `git diff --cached --stat` and `git diff --cached --` when present.
2. Identify the changed files and classify them as application code, tests, configuration, documentation, or generated artifacts.
3. Check imports and syntax for every changed Python module:
   - Use `python -m py_compile <changed-files>` for Python syntax.
   - Inspect imports and run the narrowest available import check when dependencies and project structure allow it.
   - For Streamlit modules, validate the Python module syntax and imports without starting a long-running server unless needed.
4. Identify relevant tests by inspecting the repository’s test layout, package configuration, and changed code. Prefer existing focused tests over inventing new commands.
5. Run targeted tests covering the changed behavior. Use the repository’s configured test runner and report the exact command and exit result.
6. Run the complete test suite when the change is cross-cutting, changes shared utilities or configuration, affects multiple modules, or when no narrower test selection provides adequate coverage. If no test suite or test runner exists, state that explicitly.
7. Report actual results only:
   - Never claim a test, syntax check, import check, or build passed unless it was executed in this validation run.
   - Include failures, skips, missing dependencies, and commands that could not be run.
8. Identify unresolved risks, such as untested paths, missing fixtures, dependency/environment limitations, Streamlit runtime behavior not exercised, or changes without automated coverage.

## Safety and scope

- Do not commit, amend, reset, push, or otherwise change Git history.
- Do not automatically fix application code while validating changes.
- Do not expose customer data, credentials, secrets, or raw uploaded rows in output.
- Keep validation reusable across Python modules and Streamlit applications; follow project-specific instructions and existing tooling when available.

## Report format

Summarize:

- Changed files and the relevant validation performed.
- Each command actually executed and its result.
- Targeted test results and complete-suite results, or why they were not run.
- Unresolved risks and any required follow-up.
