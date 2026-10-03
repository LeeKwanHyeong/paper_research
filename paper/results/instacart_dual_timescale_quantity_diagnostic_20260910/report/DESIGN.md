---
name: Instacart Dual-timescale Quantity Diagnostic
description: Portable technical report design contract for the frozen-model Instacart quantity diagnosis.
colors:
  bg-primary: { figma: "bg/primary", css: "--ds-surface" }
  bg-secondary: { figma: "bg/secondary", css: "--ds-surface-secondary" }
  bg-tertiary: { figma: "bg/tertiary", css: "--ds-surface-tertiary" }
  border-light: { figma: "border/light", css: "--ds-border-subtle" }
  border-default: { figma: "border/default", css: "--ds-border" }
  text-primary: { figma: "text/primary", css: "--ds-text-primary" }
  text-secondary: { figma: "text/secondary", css: "--ds-text-secondary" }
  text-tertiary: { figma: "text/tertiary", css: "--ds-text-tertiary" }
  icon-accent: { figma: "icon/accent", css: "--ds-blue-bright" }
  chart-primary: { figma: "blue/500", use: "candidate and focal contrasts" }
  chart-support-positive: { figma: "green/700", use: "favorable signed movement when a second tone is needed" }
  chart-support-secondary: { figma: "purple/500", use: "secondary comparison family when category identity requires it" }
typography:
  body: { figma: "text/sm/normal", fontFamily: "System Sans Variable" }
  label: { figma: "text/xs/semibold", fontFamily: "System Sans Variable" }
  supporting: { figma: "text/xs/normal", fontFamily: "System Sans Variable" }
  control: { figma: "text/sm/medium", fontFamily: "System Sans Variable" }
  section: { figma: "heading/md/medium", fontFamily: "System Sans Variable" }
  title: { figma: "heading/2xl", fontFamily: "System Sans Variable" }
spacing:
  compact: { figma: "space-12", value: "24px" }
  section: { figma: "space-24", value: "48px" }
rounded:
  card: { figma: "corner-radius/cr-24", value: "24px" }
elevation:
  shell: { figma: "elevation/01", value: "shared shell elevation" }
surfaces:
  report: { shell_width: "1140px", content_width: "800px", outer_padding_x: "48px", mobile_padding_x: "24px" }
components:
  top-bar: { height: "48px", typography: "text/sm/medium" }
  metric-card: { min_width: "280px", rounded: "corner-radius/cr-24" }
  report-block: { width: "800px default", rounded: "corner-radius/cr-24" }
  chart-block: { width: "full", title: "text/sm/medium" }
  table-list: { width: "800px", header: "text/xs/semibold" }
  popover-menu: { row_height: "32px", surface: "bg/primary" }
implementation:
  renderer: "Data Analytics portable artifact reader"
  generator: "build_report.py"
  artifact: "artifact.json"
  delivery: "npm run report:deliver"
---

# Instacart frozen-model quantity diagnostic

## Overview

이 보고서는 TitanTPP(B), 현재 dual-timescale 후보, raw-RMSE selector에 맞춘 RMTPP와 THP를 동일 Instacart 표본에서 비교하는 기술 진단입니다. 결과를 먼저 제시하고, 그 다음에 수량·이력 구간별 제곱오차 기여와 bias/centered-error 분해, train의 series-disjoint 재현 여부, provenance와 한계를 배치합니다. 보고서는 canonical `artifact.json`에서 생성하며 별도 HTML 구현이나 외부 런타임을 두지 않습니다.

독자는 candidate가 RMTPP 대비 어디서 이기거나 지는지, 그 차이가 B 및 THP와 비교할 때도 어떤 의미인지, 그리고 다음 Backbone 실험을 정당화할 만큼 train 두 fold에서 같은 방향이 유지되는지를 판단해야 합니다. validation과 train의 정확한 같은-표본 수를 본문에 노출합니다.

## Colors

기본 배경과 텍스트는 `bg/primary`, `bg/secondary`, `bg/tertiary`, `border/light`, `border/default`, `text/primary`, `text/secondary`, `text/tertiary`, `icon/accent` 토큰을 사용합니다. 차트의 중심 계열은 `blue/500`을 사용합니다. 유리한 signed movement가 두 번째 시각 채널을 필요로 할 때만 `green/700`을 사용하고, 독립 모델군 구분이 꼭 필요할 때만 `purple/500`을 보조로 사용합니다.

색만으로 부호나 모델 역할을 전달하지 않습니다. 0 기준선, signed 값, 축 레이블, 정렬 순서와 직접 표기를 함께 둡니다. quantity/history 기여 차트는 한 비교의 signed 값이므로 단일 root와 중립 0선이 우선입니다. bias 분해는 구성 요소 이름을 축에 직접 표시합니다.

## Typography

전체 보고서는 `System Sans Variable`을 사용합니다. 본문은 `text/sm/normal`, 표와 차트의 보조 표기는 `text/xs/normal`, 열 머리글은 `text/xs/semibold`, 조작 요소는 `text/sm/medium`, 절 제목은 `heading/md/medium`, 보고서 제목은 `heading/2xl`을 사용합니다. 모든 숫자는 renderer의 tabular-number 표현을 따릅니다.

## Layout

보고서는 `1140px` shell 안에 기본 `800px` 읽기 열을 두고, 바깥 가로 여백은 `48px`입니다. 좁은 화면에서는 `24px` 여백과 세로 스택을 사용합니다. top bar 높이는 `48px`입니다. 긴 축 레이블과 정확한 비교표는 full-width 블록으로 둡니다.

문서 순서는 기술 요약, 핵심 시각 증거, 범위·정의, frozen 추론 및 분해 방법, 불확실성과 한계, 다음 판단 순서입니다. 각 차트 바로 앞에는 읽는 방법과 해석 한계를 설명하는 독립 markdown 블록을 둡니다.

계획한 차트 맵은 다음과 같습니다.

| 보고서 구간 | 분석 질문 | 형식 | 핵심 필드 | 지원하는 주장 |
|---|---|---|---|---|
| 수량 구간 기여 | RMTPP 대비 candidate의 MSE 차이를 어느 수량 구간이 만드는가 | signed bar | quantity stratum, weighted MSE contribution delta, count | 총 RMSE 차이의 수량 구간 위치 |
| 이력 길이 기여 | RMTPP 대비 candidate 차이가 짧거나 긴 이력에 집중되는가 | signed bar | history stratum, weighted MSE contribution delta, count | 이력 길이별 병목 위치 |
| bias 분해 | raw MSE 차이가 평균 level shift와 centered error 중 어디서 생기는가 | grouped bar | split/fold, bias squared, centered MSE | 수준 보정과 구조적 분산 오차 구분 |

네 모델의 validation MAE, RMSE, body MAE(`y≤25`), tail MAE(`y>35`), bias와 표본 수는 정확한 값 조회가 목적이므로 표로 제공합니다. Validation에서 고정한 H1–H5가 train 두 fold에서 재현·부분 재현·미재현됐는지도 작은 감사 표로 제시합니다. 차트는 0선을 포함하고 signed 값을 그대로 표시하며, aggregate와 strata의 합계가 같은 metric 정의를 사용했는지 QA에서 확인합니다.

## Elevation & Depth

`elevation/01`은 artifact shell과 실제 source popover에만 사용합니다. 분석 섹션, 차트, 표는 `border/light`와 간격으로 구분하며 중첩 shadow를 사용하지 않습니다. 상태 경고는 결과의 한계가 판정을 바꿀 때만 사용합니다.

## Shapes

주요 block과 metric card는 `corner-radius/cr-24`를 사용합니다. 작은 source control은 공유 renderer의 control radius를 따릅니다. 차트 막대, 0선, 표 셀에 별도 장식형 모양이나 gradient를 추가하지 않습니다.

## Components

`top-bar`는 보고서 제목과 생성 시각만 담습니다. `metric-card`는 headline 판단에 꼭 필요한 수치가 있을 때만 사용하며 한 카드에 하나의 headline metric을 둡니다. `report-block`은 각 주요 제목과 설명을 분리합니다. `chart-block`은 세 핵심 차트를 각각 독립 full-width row에 배치합니다. `table-list`는 네 모델 validation 지표와 H1–H5 재현 판정을 보여 줍니다. `popover-menu`는 source provenance를 확인하는 shared reader 기능만 사용합니다.

모든 source-backed chart와 table은 canonical source ID를 가집니다. 수치가 들어간 markdown은 한 source로 완전히 뒷받침될 때만 block-level source ID를 사용합니다. 분석 계약, frozen inference audit, validation 분석, train 분석을 서로 다른 source로 구분합니다.

## Do's and Don'ts

Do:

- 실제 `summary.json`, `strata.csv`, `folds.csv`, inference audit와 frozen 계약에 있는 값만 사용합니다.
- validation과 train 표본 수, 동일 표본 조건, metric 단위와 primary contrast를 독자가 보기 전에 정의합니다.
- 후보가 이긴 구간과 진 구간, 반례와 fold 간 불일치를 함께 보여 줍니다.
- train fold는 series-disjoint 설명적 복제로만 표현합니다.
- `report.html`은 `artifact.json`에서 packaged builder로 한 번 생성하고 delivery receipt를 보존합니다.

Don't:

- 현재 dual-timescale 후보를 과거 BOUNDED-QK 후보와 섞지 않습니다.
- 전체 train으로 적합된 frozen model의 두 fold 결과를 out-of-fold 일반화로 표현하지 않습니다.
- seed 42 한 번의 관측을 통계적 반복성이나 인과 효과로 표현하지 않습니다.
- validation 진단으로 held-out test 성능이나 최종 모델 채택을 단정하지 않습니다.
- report 전용 CSS, 원격 script, CDN, 별도 chart runtime을 만들지 않습니다.

독자는 manifest의 제목·본문·block 순서와 호환되는 차트 형식, 표 열, 안전한 레이블을 customize할 수 있습니다. 계산값, 표본 identity, metric 정의, source SHA와 frozen 계약은 customization 대상이 아닙니다.
