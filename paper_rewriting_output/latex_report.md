# LaTeX Build Report

- Source: `final_paper/main.tex`
- Template: `final_paper/usenix-2020-09.sty`
- Engine: Tectonic 0.16.9, musl build
- Command: `/tmp/tectonic-musl/tectonic main.tex --keep-logs --keep-intermediates`
- Status: success
- Output: `final_paper/paper.pdf`
- Pages: 13, letter size, two columns
- BibTeX: completed successfully
- LaTeX guard: 0 errors, 0 warnings before compilation

The vendored USENIX 2020 style needed two XeTeX compatibility guards: `breakurl` is skipped because its PostScript hook is unavailable, and explicit pdfTeX-only microtype spacing/kerning options are disabled. Page geometry, fonts, columns, and citation style remain those of the USENIX template.

Compilation reports only underfull box warnings. It reports no undefined citations, missing figures, overfull boxes, or fatal errors. Experimental figure PDFs embed the real Arial family as `ArialMT`; the generator also rejects any font whose family name is not exactly `Arial`.
