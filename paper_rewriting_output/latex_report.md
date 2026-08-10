# LaTeX Build Report

- Source: `final_paper/main.tex` and `final_paper/body.tex`
- Template: `final_paper/usenix-2020-09.sty`
- Current source status: static guard passes with 0 errors and 0 warnings.
- TeX engine in the current environment: not available.
- Current PDF status: **not rebuilt after the 17-figure/130K revision**.

`final_paper/paper.pdf` and `final_paper/main.pdf` are stale outputs from the
previous nine-figure draft. Do not use them to review the current manuscript.
Rebuild the paper with Tectonic, XeLaTeX, or another USENIX-compatible engine
before circulating a PDF.

Recommended command when Tectonic is available:

```bash
cd /home/zhujianian/agentshift/paper_rewriting_output/final_paper
tectonic main.tex --keep-logs --keep-intermediates
cp main.pdf paper.pdf
```

The current draft contains exactly 17 compilable placeholder figure floats.
Detailed drawing instructions and Arial requirements are in `FIGURE_PLAN.md`.
