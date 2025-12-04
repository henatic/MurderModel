# Final Delivery Checklist (Phase 6 prep)

Use this list before submission. Check items off as you complete them.

## Paper & Presentation
- [ ] ACM-format paper draft complete (≤15 pages) — outline in `reports/final-paper-outline.md`.
- [ ] Presentation slides prepared and rehearsed — outline in `docs/presentation-outline.md`.
- [ ] Include latest metrics (random/temporal/geo), learning curves, error analysis, and (if done) interpretability plots.

## Code & Artifacts
- [ ] All tests pass (`python -m unittest discover -s tests -p "test_*.py" -v`).
- [ ] Latest models, metrics JSONs, plots in `data/output/` are present and referenced in docs.
- [ ] README updated with usage, split strategies, resample options, troubleshooting.
- [ ] Optional: SHAP/LIME outputs added or deferral documented (`docs/phase5-interpretability.md`).
- [ ] Fairness/threshold notes documented (or explicitly deferred in `docs/phase5-summary.md`).

## Packaging
- [ ] Clean repo (remove transient files, notebooks outputs if any).
- [ ] Ensure `requirements.txt` is current; note env `.venv`.
- [ ] Provide run commands for reproducing key results (train, compare, plots).
- [ ] Archive/zip if required by submission guidelines.

## Documentation
- [ ] Phase summaries updated (through Phase 5).
- [ ] Project roadmap status aligned with actual completion.
- [ ] Version control guidelines followed (branch names, commit prefixes).
- [ ] Include links to key docs: roadmap, phase summaries, comparison, interpretability status, final checklist.
