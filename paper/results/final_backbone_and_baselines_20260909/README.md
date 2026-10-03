# 마지막 Backbone 후보와 공정한 baseline 비교

최종 갱신: 2026-09-10 07:06 KST. 기준선4개와9행 비교표를 완료했다. 신규 Backbone은 Taxi 시간 기준 실패로 평가를 종료하고 TitanTPP(B)를 유지한다.

**후속 승인 기록:** 2026-09-10 07:37 KST부터 변경하지 않은 후보의 Intermittent 수량 중심 추가 평가 한 건을 별도로 시작했다. 이 폴더의 원래 실패 판정은 유지한다. [추가 평가의 현재 상태와 계약](../dual_timescale_intermittent_quantity_extension_20260910/README.md)을 참조한다.

[최종 후보 평가 보고서](final_campaign_report.md)

[완료된 seed42 validation 비교표](seed42_validation_comparison.md) · [기계 판독 결과](seed42_validation_comparison.json)

## 현재 승인과 종료 조건

사용자는 마지막 공통 Backbone 후보 하나를 5090에서 검증하고, 성능 기준 미달이면 TitanTPP(B)를 유지하도록 승인했다. B는 기존 Hard-LMM + validation raw-RMSE selector다. 5080은 빠진 baseline 네 full fit만 수행한다. 기존 Instacart 결과와 이전 후보 실패 기록은 유지한다.

## 구현한 후보

- 전용 경로: `titantpp_dual_timescale_memory`; branch `codex/hard-lmm-dual-timescale`, source `ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c`.
- 두 encoder 층 사이에서 관측 상태의 전이를 기록하는 동적 memory를 추가한다. 최근 8개 전이와 입력 창 내 더 오래된 전이를 분리하고, 현재 상태 query로 두 경로를 검색한다.
- 각 read에 절대 support `-expm1(-8*mass)`를 곱한 뒤 confidence로 결합한다. 이력이 짧으면 confidence가 정규화 때문에 상쇄되는 문제를 방지한다.
- static Hard-LMM prior는 유지하며, memory 상태는 표본별 관측 prefix에서 매 forward 재구성된다. global은 입력 창 내 오래된 이력을 뜻한다. 고객 간 memory 공유나 무제한 streaming memory가 아니다.
- 같은 seed B와 초기 출력·공유 gradient·RNG가 일치하고, 활성화된 memory에는 Q/K/V/write/fusion gradient가 전달된다. 추가 파라미터 5,319개(+5.92%).
- 새로운 실험 경로를 구현했다는 사실만으로 학술적 독창성이나 성능 우위를 주장하지 않는다.

## 완료된 검증과 실행 상태

- 로컬: 31 passed, CUDA 조건 14개 skip. 별도 기존 모델 회귀 40개 통과.
- 5090: 45 tests passed. CUDA 포함 모든 테스트 실행, source/input 2,314개 SHA 검증.
- CUDA 비용: 같은 batch128에서 step B 대비 1.722배(길이64), 1.847배(길이256). peak allocated VRAM 1.206배/1.178배. 동결 상한(step2배/VRAM3배/parameter1.25배) 통과.
- Taxi e1: 12.35초, 전체 train38,393/validation8,268 대상 감사 및 실제 selected/last 모델과 AdamW 복원 통과.
- Intermittent e1: 107.20초, 전체 train393,824/validation86,285 대상 감사 및 실제 복원 통과.
- Instacart e1: 225.28초, 전체 train1,991,192/validation503,733 대상 감사 및 실제 복원 통과. 세 e1 모두 통과했으며 성능 채택 판정에는 사용하지 않는다.
- 5090 Instacart seed42 full fit: 2026-09-09 23:58:29 KST 시작, 진행 중.
- 5080: 전용 commit `16ebcc8bd535197175f7a1965dee0251992b98b7`, local 50 tests와 RMTPP/THP CUDA smoke 통과. Intermittent RMTPP full fit 실행 중.

## 남은 실행 순서

1. 5090: 진행 중인 Instacart seed42 full fit을 완료한다. 통과하면 Taxi → Intermittent full fit. 최대300/minimum40/patience40를 유지한다.
2. 후보 성능 gate: 각 데이터셋에서 B보다 raw RMSE 엄격히 개선, 전체 MAE 악화1% 이하, body 및 >p99 MAE 악화각2% 이하, legacy clamped time loss 증가0.01 이하. 하나라도 실패하면 후속 후보 실행을 중단하고 B를 유지한다. 결과 후 구조/loss/selector/threshold를 바꾸지 않는다.
3. 5080: Intermittent RMTPP → Intermittent THP → Taxi RMTPP → Taxi THP. baseline과 B의 우열로 중간 중단하지 않고 네 비교를 완성한다. 기술·계약 실패 때만 중단한다.
4. 두 큐의 고정 auditor, 처리 건수, 선택 checkpoint, finite·복원·held-out 부재를 확인하고 validation 표를 통합한다.

추가 seed와 held-out test는 실행하지 않는다. 후보 seed42 세 데이터셋 통과는 후속 재현 검증이 필요한 상태이며 최종 채택 증명이 아니다. legacy time loss를 정상화 Time NLL로 부르지 않는다.

## 시간 추정과 점검

- 5080 네 fit: 초기 실측/과거 기록에서 약2~3시간이 잠정 예상이며, early stop 시점을 예측할 수 없으므로 300epoch 전부 실행되는 보수적 상한은 약9시간이다. 실제 속도와 선택 epoch에 따라 갱신한다.
- 5090: Instacart e1 225.28초를 그대로 적용한 시나리오는 40epoch 약2.5시간, 100epoch 약6.3시간, 300epoch 약18.8시간이다. full fit의 steady-state 시간과 early stop을 아직 측정하지 않았으므로 확정 ETA가 아니다. 긴 이력 Intermittent를 300epoch 소진하면 e1 기준 약8.9시간이며 Taxi 약1시간이다. 세 데이터셋 모두 통과하면서 300epoch를 소진하는 시나리오는 총 약29시간이고 운영 여유가 추가될 수 있다.
- 서버 큐가 다음 job을 자동으로 이어가며 Codex heartbeat가 1시간마다 두 서버를 점검한다. 의미 있는 단계 완료·실패·조치 필요 시 알리고, 두 큐가 종료되면 heartbeat를 중단한다. 기존 paused VNC heartbeat를 최신 두 큐 범위로 갱신해 ACTIVE로 설정했다(automation id: vnc-hard-lmm-5090-screening).

소스·결과 경로는 각 서버 하위 폴더 deployment.json에 기록했다. 동결 계약은 contracts/에 복사했다. 원격 checkpoint와 대형 cache는 이 결과 폴더나 Git에 포함하지 않는다. 기존 main checkout의 사용자 변경 및 develop/master는 수정하지 않았다.

## 2026-09-10 00:58 KST 시간별 점검

5090은 Instacart seed42 epoch16 완료, epoch17 학습 중이다. 현재 raw-RMSE 선택 epoch10의 RMSE는 5.9451657721, MAE 4.0099296698, legacy clamped time loss 3.2158333523이다. B의 RMSE 5.8722169305에는 아직 못 미치며, full fit과 모든 guardrail 감사 전 성능 판정하지 않는다. 관측된 16개 epoch는 모두 train1,991,192건, finite 계산 조건을 만족한다. CUDA PID1746618 및 controller1743507이 살아 있고 GPU46%/1,286MiB를 사용 중이었다.

안전한 history.json의 완료 시각과 job 시작 시각으로 산출한 평균은 약217.5초/epoch(시작 비용 포함)다. best epoch10 이후 개선이 없으면 epoch50에 약02:59 KST 종료하며, 개선되면 patience가 다시 시작된다. 동일 속도의 300epoch 시나리오는 약18:06 KST이다. 이는 고정 완료 약속이 아니다. checkpoint 역직렬화 없이 JSON metadata를 사용했다. 원시 snapshot과 계산은 5090/monitor_20260910_0057.json에 보관했다.

### 5080: 첫 baseline full fit 완료

Intermittent RMTPP는 75epoch로 종료했고, validation raw-RMSE의 earliest best는 epoch35다. 고정 auditor는 passed이며 source/runtime, train393,824/validation86,285 대상, selector와 checkpoint/resume 상태, validation-only 범위를 확인했다. RMSE1.8336068667, MAE0.6695222346으로 B의 RMSE1.4995550721, MAE0.6043404184보다 높다. 이는 Intermittent seed42에서 B가 이 RMTPP 비교군보다 수량 오차가 낮다는 증적이며 모든 모델·seed·held-out 우위 주장이 아니다.

현재 Intermittent THP가 자동으로 이어 실행 중이며 00:59 KST snapshot 기준 epoch44, best epoch37이다. best raw RMSE1.6611799723, 같은 epoch MAE0.6151529871로 아직 B를 넘지 못했지만 완료 전 판정하지 않는다. 실제 속도 약54초/epoch이고 이후 개선이 없으면 epoch77, 약01:29 KST에 종료한다. 이후 Taxi RMTPP → Taxi THP가 같은 5080 큐에서 이어진다. 상태의 running_job_id가 현재 job을 나타내며, next_job_id는 최초 snapshot 값이 남아 있으므로 현재 작업 판별에 사용하지 않는다. 기술 오류는 없었다.

17개 소형 JSON/CSV 증적과 SHA는 5080/monitor_20260910_0059.json 및 5080/snapshot_20260910_0059/에 보관했다. 서버 source/runtime/학습에는 변경하지 않았다. 두 campaign이 계속 실행 중이므로 1시간 heartbeat를 ACTIVE로 유지한다.

## 2026-09-10 01:59 KST 시간별 점검

5090은 Instacart epoch33까지 완료했고, 현재 선택 epoch는21이다. 같은 선택 epoch의 raw RMSE5.8872047643, MAE3.9900239039, legacy clamped time loss3.2174577798이다. B의 RMSE5.8722169305보다 아직0.0149878339 높아 strict raw-RMSE gate를 통과한 상태가 아니다. full fit/guardrail 감사 전 최종 판정하지 않는다. 33개 epoch 모두 finite 및 train1,991,192건을 유지했고 CUDA PID1746618/controller/tmux는 정상이다. GPU51%, 사용1,286MiB였다.

평균216.1초/epoch이며 best epoch21 이후 더 개선되지 않을 경우 epoch61에서 약03:38 KST에 종료한다. 이후 best가 갱신되면 종료 시점도 연장된다. 같은 속도의 300epoch 시나리오는 약17:59 KST이다. 변경·재시작 없이 고정 계약대로 진행한다. 원시 JSON과 계산은 5090/monitor_20260910_0158.json에 보관했다.

5080은 같은 시각 Intermittent THP epoch111, best epoch104까지 진행했다. 선택 epoch104의 RMSE1.6497231014, MAE0.5983332299이다. B보다 RMSE는 높고 MAE는 낮지만 아직 full fit 완료 전이다. 새 best가 없으면 epoch144에서 약02:29 KST 종료하며 이후 Taxi 두 job이 자동 진행한다. Intermittent RMTPP만 완료된 상태이고 그 감사는 계속 passed다. 새로운 완료 receipt·실패·조치 필요는 없으며 두 큐와 GPU는 정상이다. 5080 snapshot은 5080/monitor_20260910_0159.json에 보관했다. 학습과 자동 점검을 유지한다.

## 2026-09-10 03:00 KST 시간별 점검

5090 Instacart seed42는 epoch50 완료, best epoch40이다. best raw RMSE5.8777777952, MAE3.9932746832, legacy clamped time loss3.2132832503이다. B의 RMSE5.8722169305보다0.0055608647 높으므로 strict raw-RMSE gate를 아직 넘지 못했다. 완료 전 최종 gate 판정하지 않는다. 50개 epoch의 finite 및 train1,991,192건 조건이 유지되고 controller/tmux/CUDA가 정상이다. GPU44%, 사용1,286MiB였다.

평균215.0초/epoch, 이후 개선이 없다면 epoch80에서 약04:45 KST 종료한다. 300epoch 시나리오는 약17:53 KST이다. 새 best가 갱신되어 이전 조건부 ETA가 연장되었으며 고정 patience40을 그대로 유지한다. 5090/monitor_20260910_0300.json에 snapshot을 보관했다. 변경·재시작·추가 평가 없이 기존 큐를 유지한다.

### 5080 기준선 큐 완료와 9개 결과 통합

03:00 KST 확인 시 네 job 전부 완료·audit passed다. GPU는 idle이며 큐 tmux는 정상 종료했다. Intermittent RMTPP75e/best35, THP144e/best104; Taxi RMTPP86e/best46, THP233e/best193이다. 추가 seed·held-out·오류는 없다. 원격 학습을 변경하거나 재시작하지 않았다. 51개 소형 JSON/CSV 증적은 5080/completed/와 원본 bundle에 보관했다.

기존 Instacart RMTPP80e/best40와 THP90e/best50의 exact summary/history/launch/audit를 재사용했다. 세 B summary/history/launch는 사전에 동결된 a_vs_b_metrics.json의 SHA와 대조했다. 9개 행에 대해 같은 validation 대상 identity와 quantity hash, train/validation 처리 건수, loss·optimizer·selector, finite history, earliest raw-RMSE checkpoint와 patience40, 수량 구간별 지표 재구성을 확인했다. 결과는 seed42_validation_comparison.md/.json/.csv이며 build_seed42_comparison.py로 재생성 가능하다.

B의 원래 launch는 B/C를 함께 지정해 adaptive strength1을 기록하지만 B의 개별 variant는 count_only_log_regression이고 interface에 adaptive 계약이 없다. production training의 _quantity_variant_model_kwargs는 이 variant에서 빈 kwargs를 반환하여 unweighted log-MSE를 유지한다. Instacart 공용 launcher의 status=running도 동결 증적과 일치하며, 중단한 C를 포함한 공용 상태다. B 자체는 success,112epochs/best72,summary/history/checkpoint-state SHA와 완전한 종료 이력을 별도로 확인했다. 이 상태를 C의 완료로 바꾸거나 추가 학습하지 않았다.

TitanTPP(B)는 Intermittent와 Taxi RMSE에서 두 baseline보다 낮고, Instacart RMSE에서는 두 baseline보다 높다. MAE 최솟값은 Intermittent THP, Taxi·Instacart RMTPP다. 단일 seed validation 결과로 모든 데이터셋·모든 TPP 또는 held-out 우위를 주장하지 않는다.

남은 순서는 5090 Instacart screening 완료·고정 gate 판정 → 통과할 때만 Taxi·Intermittent → 최종 증적 정리다. 후보가 실패하면 기존 B를 유지하며 추가 Backbone 탐색을 하지 않는다. 5080 학습은 재개하지 않는다. 5090이 아직 진행 중이므로 1시간 heartbeat는 ACTIVE로 유지한다.

## 2026-09-10 04:00 KST 시간별 점검

5090에 대한 한 차례 SSH 접속이 192.168.0.71:22 timeout(exit255)으로 실패했다. 따라서 현재 epoch/GPU/완료 여부는 미확인이다. 학습 실패 또는 중단으로 판정하지 않는다. 마지막 정상 증적은03:00 KST의 Instacart epoch50/best40이며 당시 조건부 ETA04:45는 현재 상태를 반영한 갱신치가 아니다. 재시작·중복 실행·source/runtime 변경을 하지 않았다. 실패 사실과 마지막 정상 snapshot 경로를 5090/monitor_20260910_0400.json에 기록했다. 다음 정기 점검에서 상태를 다시 확인한다.

5080은 단일 읽기 점검에서4/4 complete를 유지하며 error없음, GPU0%/compute없음, tmux정상종료를 확인했다. 완료된 결과를 재실행하지 않았다. 5090의 현재 상태를 확인하지 못했으므로 두 큐의 종료를 확정하거나 heartbeat를 중단하지 않는다. 1시간 점검을 유지한다.

## 2026-09-10 05:04 KST 시간별 점검

5090 LAN SSH가 두 번째 정기 점검에서도 timeout이다. 현재 학습 상태는 계속 미확인이며 마지막 정상 증적은03:00 KST다. 이전 조건부 종료 시각이 지났다는 사실만으로 완료·실패·중단을 추정하지 않는다. 같은 서버의 기존 대체 접속 주소가 기록되어 있는지 읽기 전용으로 확인한다. source·runtime·SSH 설정·학습을 변경하지 않았다.

5080은4/4 complete와 오류없음, GPU0%/compute없음을 재확인했다. 새 학습을 시작하지 않는다. 5090 종료 여부를 확인할 때까지 정기 점검을 유지한다.

동일5090 서버의 대체 SSH/Tailscale 주소는 로컬 SSH 설정과 프로젝트 실행 규약에 없었다. 확인된 유일한 주소는192.168.0.71이며 문서의100.118.62.81은 다른5080 서버다. 주소를 추측하거나 다른 호스트를 탐색하지 않았다. 5090 연결 복구 또는 기존 대체 주소 확인을 기다리며 정기 점검을 유지한다.

## 2026-09-10 06:04 KST 시간별 점검

5090 LAN SSH는 세 번째 정기 점검에서도 timeout이다. 현재 학습 상태는 미확인, 마지막 정상 snapshot은03:00 KST다. 학습 종료·실패를 추정하지 않으며 새 ETA를 계산하지 않는다. 05:04 점검에서 연결 확인 요청과 기존 대체 주소 부재를 이미 안내했다. 새 조치나 상태 변화가 없으므로 반복 알림은 생략한다.

5080은 다시4/4 complete, 오류없음, GPU0%/compute없음을 확인했다. 양 서버에 재실행·runtime·source 변경을 하지 않았으며 1시간 점검을 유지한다. 원시 상태와 접속 실패 기록은 각 서버의 monitor_20260910_0604.json에 보관했다.

## 2026-09-10 06:38 KST 사용자 요청 점검: 연결 복구·Instacart 통과

5090 SSH에 다시 접속했다. Instacart 신규 후보는112epochs/best72로 정상 완료했고 raw RMSE·전체 MAE·body MAE·>p99 MAE·legacy clamped time loss의 동결 gate를 모두 통과했다. selected 모델 strict 복원과 finite forward, last 모델·AdamW46개 parameter 복원 감사도 통과했다. 원격 증적은 5090/instacart_completed/에 회수했다.

B→후보: RMSE5.8722169305→5.8680590985(-0.0708%), MAE3.9937810783→3.9854971992(-0.2074%), body3.4386015032→3.4338327972(-0.1387%), >p99 MAE21.9177947653→21.7819404093(-0.6198%), legacy clamped time loss3.2155228955→3.2148046068(-0.0007183)이다. 공통 성능 판정을 이어갈 필요조건을 넘었지만 개선 폭이 작고 단일 seed validation 결과이므로 최종 채택·통계적 우위를 확정하지 않는다.

큐는 Taxi로 자동 전환했다. 06:38 snapshot은 epoch19/best14이며 raw RMSE87.5322496669, MAE28.3410543490, legacy time loss1.3660060181이다. 현재 값은 학습 중 결과이며 Taxi full gate는 아직 판정하지 않았다. 평균11.41초/epoch, 이후 개선이 없으면 epoch54에서 약06:45 KST 종료하는 조건부 추정이다. 새 best에 따라 종료가 연장된다. GPU76%/2,930MiB와 CUDA PID1865022, tmux 생존을 확인했다.

5080은 기준선4개 complete, GPU0%/compute없음을 유지한다. 다음 순서는 Taxi 완료·gate → 통과할 때만 Intermittent → 최종 증적 정리다. 수동 재시작이나 학습 설정 변경 없이 기존 자동 큐와1시간 heartbeat를 유지한다. 다음 실패 시 B를 유지한다. 추가 seed·held-out은 실행하지 않았다.

## 2026-09-10 07:06 KST 최종 점검: Taxi 실패·조건부 평가 종료

Taxi는123epochs/best83으로 정상 종료했다. 수량 RMSE78.2554(-11.27%), 전체 MAE25.0483(-12.64%), body MAE15.7386(-17.15%), >p99 MAE303.4513(-8.81%)로 B보다 낮다. 같은 선택 epoch의 기존 시간 loss는1.4733906265→10.8258699686(+9.35248)으로 악화되어 유일하게 시간 guardrail을 실패했다. 실행 감사는passed이나 성능 gate는failed다.

독립 읽기 감사에서 선택 checkpoint·last checkpoint byte SHA,source/계약/summary/history/launch SHA, 표본 identity와 수량 hash, actual restore, finite 및patience40을 재확인했다. 큰 시간 loss는 선택 epoch83의 history, joint-logquantity 차이, 수량·이력 구간별 가중 합과 모두 일치한다. 같은legacy head/scale/cap이므로 정상화 정의가 달라 생긴 차이가 아니다. 수량<=p50 구간에 시간 오차가 집중됐다는 관측을 보존한다.

큐는stopped_seed42_gate_failed, final_model=TitanTPP(B)이며 Intermittent full screening 디렉터리가 없다. 새 학습·seed·held-out을 시작하지 않았다. 5090GPU0%/compute없음/campaign tmux종료, 5080은4/4완료/GPU유휴를 확인했다. 최종 증적은5090/final/과final_campaign_decision.json에 보관했다. 두 큐가 종료되어1시간 heartbeat를 PAUSED로 전환 완료했다. 이번 승인 범위의 남은 실행 작업은 없다.
