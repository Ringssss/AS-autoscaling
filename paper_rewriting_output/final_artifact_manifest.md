# Final Artifact Manifest

## Manuscript

- `full_paper.md`: complete English paper in Markdown, 7,161 words.
- `final_paper/main.tex`: USENIX entry point and abstract.
- `final_paper/body.tex`: generated and reviewed two-column body.
- `final_paper/references.bib`: cited literature.
- `final_paper/paper.pdf`: compiled 13-page USENIX paper.

## Figures

- `fig_workload_characterization.pdf`
- `fig_architecture.pdf` and editable `fig_architecture.drawio`
- `fig_context_latency.pdf`
- `fig_gap_overlap.pdf`
- `fig_hotspot.pdf`
- `fig_elasticity.pdf`
- `fig_interference.pdf`
- `fig_control_plane.pdf`
- `fig_replay_policy.pdf`

All plot labels use embedded Arial TrueType fonts. PNG copies are included for quick inspection.

## Evidence and Audit

- `evidence_bank.md` and `claim_register.md`: claim provenance and wording limits.
- `paper_self_review.md`: five-dimension adversarial review and claim-evidence audit.
- `latex_report.md`: reproducible compile result.
- `reference_materials/source_index.md`: paper-reading index.
- `writing_rationale_matrix.md`: unit-level writing decisions and final checks.

## Verification

- AgentShift tests: 39 passed.
- SGLang AgentShift prefix-cache tests: 14 passed.
- Fault campaign: 8/8 passed.
- LaTeX compilation: successful.
- No AgentShift benchmark or SGLang server process remains active.
