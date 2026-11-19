# Repository Conventions

## Workspace

- All disposable scripts, logs, debug outputs, and intermediate artifacts must go under `.scratch/<task-slug>/`.
- Do not create ad-hoc temporary directories elsewhere in the repository.
- Promote reusable experiments to `experiments/`, regression tests to `test/`, application documentation to `docs/`, and development findings to `context/`.
- `requirements.txt` 必ず UTF-8 (BOM なし) で書き出すこと。PowerShell のリダイレクト (`>`) や `Add-Content` は UTF-16 を生成するため、このリポジトリでは過去に pip 以外のツールが読めなくなった。
- Clean up temporary files created by the current task when they are no longer needed. Never delete unrelated or pre-existing files.

## Git Workflow

- Before editing, inspect the current branch, `git status`, and existing changes.
- Use a dedicated `feat/`, `fix/`, or `chore/` branch for each independent task. Preserve unrelated work; use a separate worktree when isolation is needed.
- Commit autonomously after each coherent, verified change. Do not accumulate unrelated changes in one commit.
- Stage only intended files and inspect the staged diff before committing. Never indiscriminately stage unrelated changes.
- Run relevant tests before committing. Report failures and remaining uncommitted changes.
- Never discard user changes, force-push, rewrite shared history, or perform destructive Git operations without explicit authorization.
- Do not push, merge into `main`, or delete branches without authorization.

## Beads Issue Tracking

- Use `bd` (Beads) for durable task tracking, not separate markdown TODO lists. Run `bd prime` when workflow context is missing or stale; its guidance does not override repository, user, or orchestrator instructions.
- Use `bd ready` to find available work, `bd show <id>` to inspect it, `bd update <id> --claim` to claim it atomically, and `bd close <id>` when finished.
- Before handoff, update issue status and create issues for remaining follow-up work.
- Use `bd remember` for persistent agent memory instead of ad hoc memory files; keep application documentation in `docs/` and development knowledge in `context/` as described below.
- Issues live in the local Dolt database; `.beads/issues.jsonl` is a passive export, not the source of truth. Do not use `bd import` during normal operation.
- Remote issue sync uses `bd dolt push` / `bd dolt pull`, separate from code branches. Obtain explicit authorization before remote sync; follow the Git Workflow above for code commits and pushes.

## Completion

- Report the branch, commits made, verification results, and any remaining worktree changes.
- Prefer leaving the repository clean and the task's temporary workspace free of unnecessary files.

## Documentation and Development Context

- `docs/` documents the **current application**: its behavior, APIs, architecture, configuration, and usage. Keep it consistent with the implementation.
- `context/` stores **development knowledge**: external references, research and discussion summaries, implementation plans, design decisions, and terminology.
- Before implementing a task, consult relevant materials in `context/`. Use `context/README.md` as an index; avoid reading unrelated documents.
- Distinguish verified facts, hypotheses, proposals, and accepted decisions. Preserve source references and summarize lengthy discussions when appropriate.
- Update affected plans, decisions, and documentation when implementation or understanding changes. Do not treat outdated plans as authoritative.
