# Thinking Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

(1) Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

(2) Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

(3) Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

(4) Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# Coding Guidelines

Simplicity:

- no code > simple code > clever code
- more dependencies is better than more code
- minimize indentation: guard first programming, early returns, happy path first
- maximize code locality: a little duplication is okay. inline small functions (<5 LoC). add top-level helpers only when reused
- use locals by default. inline one-offs. keep shared constants and long prompts global
- prefer pure functions: pass inputs, return results, and isolate side effects
- use `itertools` and `functools` to fold deeply nested loops
- no shared utility modules. each standalone script duplicates its own boilerplate
- use `tyro` for command line parsing. no subcommands or argument groups
- comments are used sparsely and give non-obvious context that an LLM couldn't infer

Avoid LLM smells:

- no trailing commas. prefer single-line imports, dicts and lists over multi-line with trailing commas. take advantage of infinite line width
- no docstrings. use tight technical comments starting with the pound `#` symbol when needed
- avoid unnecessary `print` statements. replace repetitive whitespaces with tabs `\t`
- call `tyro.cli(main)` in `if __name__ == "__main__":` for command line scripts
- import statements must always be at the top of the file
- avoid `from __future__ import annotations` where possible
- never use semicolons. ever. in any file (code, prose, docs). replace with periods or split into separate clauses
- never use emdashes

Robustness:

- never suppress missing library stubs. add them in a separate pr, merge it, then rebase all open prs onto the typed base
- use `assert` statements frequently as invariants. prefer crashing over failing silently. prefer `assert` over runtime exceptions
- never explicitly raise `AssertionError`. always use an `assert` statement, including `assert False, message` for unreachable branches
- maximize test coverage, without adding noise (except for standalone scripts)
- prefer typed enums over strings for states
- use match statements over multiple `isinstance` branches

Performance:

- use `@cache` where it makes sense
- have mechanistic sympathy, design with data oriented programming best-practices

exojit style:

- `JITRuntime` in `exojit.py` uses a class as a namespace for `@staticmethod` functions.
- Prefix private helpers with `_`, as in `_tensor_converter`.
- Call static methods as `ClassName.method(...)`. `backend_metal.py` calls `JITRuntime.compile(proc, compiler=compiler)`.
- Split files into sections with a title line between two banner comments. This section header appears in `backend_metal.py`.

```python
# ===----------------------------------------------------------------------=== #
# metal runtime
# ===----------------------------------------------------------------------=== #
```
