# Version Control Guidelines

## Branch Model
- `main`: stable, releasable branch.
- `develop` (optional): staging branch for merging multiple features before release.
- Feature work: `feature/<short-description>` (e.g., `feature/data-cleanup`).
- Bug fixes: `fix/<issue>` or `bugfix/<issue>` (e.g., `fix/roc-plot`).
- Hotfixes to production: `hotfix/<issue>` branched from `main`, merged back to both `main` and `develop` (if used).

## Commit Prefixing
- Prefix commit subjects with a type: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `build:`.
- Keep subjects under ~72 characters, present tense, and scoped when helpful (e.g., `feat(preprocessing): add age binning`).
- Use body lines for rationale and context when not obvious.

## Review Expectations
- Open a pull request for any change touching code, data processing, or documentation that affects users.
- Include: purpose, summary of changes, testing performed (`python -m unittest` or targeted tests), and screenshots for visuals.
- At least one reviewer approval before merging; self-merge only for trivial docs or config tweaks after a quick sanity check.
- Resolve comments or capture follow-ups in issues before merge.

## Release Tagging
- Use lightweight tags on `main` after validation: `vMAJOR.MINOR.PATCH` (e.g., `v0.3.0`).
- Bump `PATCH` for fixes, `MINOR` for backward-compatible features, `MAJOR` for breaking changes.
- Tag only after tests pass and artifacts (metrics, plots) are generated for that revision.
