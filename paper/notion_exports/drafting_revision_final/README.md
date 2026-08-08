# Notion mirror: drafting and iterative improvement

Source parent page: [7. Drafting and iterative improvement of the paper](https://app.notion.com/p/2e9bbe405613808cbc77faf7127512d6)

Snapshot date: 2026-08-08 KST

This directory keeps a local Markdown mirror of the Notion pages that sit under the paper drafting task. The mirror is not meant to replace the working files in `paper/drafts/`, `paper/tables/`, `paper/figures/`, or `paper/results/`. It is a navigation and provenance layer: each page-level note points to the local files that currently carry the executable or manuscript-ready content.

## Mirrored pages

| Notion page | Local mirror | Local working files |
|---|---|---|
| TitanTPP 논문 Structure 및 표·그림 계획 (August 14 초안) | [01_structure_table_figure_plan.md](01_structure_table_figure_plan.md) | `paper/drafts/manuscript_section_plan_v0_1.md`, `paper/figures/F1_F3_figure_register.md`, `paper/tables/T1_dataset_statistics.md`, `paper/tables/T2_model_training_contract.md` |
| Reference List | [02_reference_list.md](02_reference_list.md) | `paper/references/related_work_reference_register.md` |
| TitanTPP 논문 Draft | [03_titantpp_paper_draft.md](03_titantpp_paper_draft.md) | `paper/drafts/introduction_v0_1.md`, `paper/drafts/problem_formulation_v0_1.md`, `paper/drafts/methodology_v0_1.md`, `paper/drafts/related_work_outline_v0_1.md` |
| TitanTPP 4-page paper structure revision | [04_four_page_structure_revision.md](04_four_page_structure_revision.md) | `paper/drafts/manuscript_section_plan_v0_1.md`, `paper/results/e300_matched_20260808/result_briefing.md` |
| TitanTPP e300 matched baseline result briefing and draft applicability | [05_e300_result_briefing.md](05_e300_result_briefing.md) | `paper/results/e300_matched_20260808/result_briefing.md`, `paper/results/e300_matched_20260808/tables/preliminary_summary.md` |

## Current status

- 완료: Notion parent page and five child pages were identified.
- 완료: Local paper directory already had most draft/reference/table/result artifacts, but not a page-by-page Notion mirror.
- 완료: This directory now provides that page-by-page mirror and links it to the existing project files.

## Update rule

When a Notion page is substantially edited, update the matching mirror file here and the relevant working file under `paper/`. If a page contains only project-management text, update only this mirror. If a page changes manuscript content, update both the mirror and the manuscript file.
