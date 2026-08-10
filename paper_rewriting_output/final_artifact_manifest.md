# Final Artifact Manifest

## Manuscript

- `full_paper.md`: complete English paper in Markdown, approximately 8,200 words.
- `final_paper/main.tex`: USENIX entry point and abstract.
- `final_paper/body.tex`: reviewed two-column body with Figure 1 integrated and
  16 remaining figure placeholders.
- `final_paper/references.bib`: cited literature.
- `final_paper/paper.pdf`: stale prior draft; current source has not been rebuilt because no TeX engine is available.

## Figures

- `FIGURE_PLAN.md`: frozen 17-figure sequence, type, evidence status, drawing protocol, and claim.
- `figure_asset_map.md`: maps each slot to evidence and LaTeX labels.
- `figures/fig_motivation.pdf`: vector Figure 1 generated from the raw
  FlowPrefill trace and repeated SGLang experiments.
- `figures/fig_motivation.png`: 600 dpi Figure 1 preview.
- `figures/fig_motivation_data.json`: exact derived values, metric definitions,
  and source paths for Figure 1.
- `final_paper/body.tex`: contains Figure 1 and 16 numbered placeholder floats.
- Existing PDFs in `figures/` are prior data checks, not the final 17-figure set.

Every final experimental plot must use the Times-compatible style and the
palette/layout rules in `FIGURE_PLAN.md`.

## Evidence and Audit

- `evidence_bank.md` and `claim_register.md`: claim provenance and wording limits.
- `paper_self_review.md`: five-dimension adversarial review and claim-evidence audit.
- `latex_report.md`: reproducible compile result.
- `reference_materials/source_index.md`: paper-reading index.
- `writing_rationale_matrix.md`: unit-level writing decisions and final checks.

## Verification

- Previously recorded AgentShift tests: 39 passed.
- Previously recorded SGLang AgentShift prefix-cache tests: 14 passed.
- Recorded fault campaign: 8/8 passed.
- LaTeX static guard: 0 errors, 0 warnings.
- Current PDF compilation: pending a TeX engine.
