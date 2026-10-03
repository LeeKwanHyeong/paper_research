# TitanTPP — anonymous LNCS working draft

Evidence and editorial cutoff: 2 October 2026. This is a working manuscript,
not a submission-ready or independently evaluated final paper.

## Contents

- `main.tex`: anonymous manuscript with four embedded vector figures,
  nine tables, Appendices A--C, and 32 embedded references.
- `llncs.cls`: unmodified official Springer LNCS class, version 2.25.
- `splncs04.bst`: unmodified official Springer bibliography style, provided
  for future BibTeX-based editing.
- `references.bib`: reviewed bibliography source retained for maintenance.
  The present manuscript uses the embedded bibliography in `main.tex`.
- `LNCS_README.txt`: original template distribution instructions.

The main comparison contains 84 completed development-validation runs:
four datasets, seven models, and three seeds. The appendix separately records
11 completed conditions from the ongoing extension as of 07:14 KST. Its
checkpoint binary CPU audit was still pending at that observation cutoff.
No running best-so-far result is presented as a completed result.

Appendix B describes the four train populations with common definitions.
Taxi, Intermittent, and Instacart profiles are reused; RAF's missing train
profile is computed against its frozen population identity. Appendix C uses
all 84 completed selected validation records for quantity/history strata.
Its comparator is fixed by each dataset's overall external-model RMSE.
The analysis, all-model per-seed tables, source hashes, and standalone PDF/PNG
figures are in `reports/titantpp_dataset_appendix_20261002_v1/` at repository
root. No training, model inference, or held-out performance inspection was
performed for this addition.

## Compilation and verification status

Open `main.tex` in the Codex LaTeX editor. Figures and references are embedded;
the official class is the only local TeX input. The standard packages used are
`fontenc`, `amsmath`, `newtxtext`, `newtxmath`, `booktabs`, `array`, `tikz`,
and `hyperref`.

In an existing TeX installation with these packages, the usual build is:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

BibTeX is not required for this version. These commands were not run during
draft creation, and no TeX distribution or plugin was installed.

**Native compilation passed after the Appendix B/C edit.** The first attempt
identified an undefined `\Description` command in the compiler's bundled
LNCS 2.21 class. A `\providecommand` fallback preserves compatibility with
older class versions; the next attempt succeeded. The packaged official
LNCS 2.25 class remains unmodified and has not been separately compiled by
the native tool, which does not read additional local project files.

The prior busy responses are retained as historical verification records.
Current source structure, citation keys, cross-references, and numerical
tables pass separate checks. The two new standalone figure PDFs have been
rendered and visually reviewed. The full manuscript has not been exported
or visually reviewed as a PDF, and its page count is not yet verified.

## Venue and remaining editorial work

The [PAKDD 2027 research-track call](https://pakdd2027.org/pages/calls/research)
specifies Springer LNCS formatting and double-blind review. Its page limit
and submission system were not yet announced when this draft was prepared.
The class and margins have not been modified. Recheck the call before final
pagination and submission.

Next steps are to review the full rendered document, establish the
independent evaluation scope, and incorporate the extension results when
their complete groups and audits are available. Pending training does not
prevent editing this manuscript now. Quantitative claims currently concern
development validation, not an untouched final test population.

Template source: [Springer proceedings author instructions](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines).
