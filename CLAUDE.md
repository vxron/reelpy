# CLAUDE.md

## CI: keep .gitlab-ci.yml in sync with tests/

`.gitlab-ci.yml` runs one CI job per file in `tests/test_*.py` (in the `test`
stage, named `test:<name>` without the `test_` prefix/`.py` suffix). This is
deliberate — it lets slow files (e.g. large fixture-based suites) run in
parallel with everything else instead of blocking the whole pipeline serially.

**Whenever you add, rename, or remove a `tests/test_*.py` file, update
`.gitlab-ci.yml` to match** — add/rename/remove the corresponding `test:<name>`
job. Every `test_*.py` file must have a corresponding job; none should be
silently skipped. Each job just runs `pytest tests/test_<name>.py` — no `-v`,
no `--local` (conftest.py already skips `@pytest.mark.local` tests by default).
