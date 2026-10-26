# 고정 위치 인코딩 실험 사전 QA 메모

## 상태

**방법론 검토 진행 중 — 신규 runner·계약·테스트가 준비되기 전이므로 실행 승인 판정이 아니다.**
프로젝트 역할 설정은 `qa-reviewer = gpt-6-astra/high`다. 도구로 실제 런타임 모델 식별자를 별도 확인한 것은 아니다.
소스 수정·원격 접속·Test 자료 열람은 하지 않았다. `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 현재 경로에 없어 읽지 못했으며, `.agents/`는 변경하지 않았다.

## 현재 확인한 기준선

- frozen source: `reports/phen_intermittent_best_last_20261026_v1/attempt3/bundle/source42`.
- `paper/scripts/run_shared_component_ablation.py:174–233`: 생성·optimizer·저장·joint 검증 hook을 설치하고 `finally`에서 원복한다.
- 같은 파일 235–242: maximum 300, minimum 40, patience 40을 설정한다.
- `paper/scripts/count_aware_tpp_backbone/training.py:613–627`: 실제 정지 조건은 현재 epoch >= minimum이며, 최초 strict 최저 선택 epoch 이후 patience 이상 경과한 경우다.
- 같은 파일 715–733: 별도 generator를 쓰는 셔플 Train loader와 generator=None인 비셔플 Validation loader를 만든다.
- 같은 파일 839–867: epoch 종료 모델의 Validation을 먼저 평가하고 선택 지표를 history에 기록한다. 964행에서 best checkpoint를 다시 평가하므로 추가 Train-monitor는 최종 best replay를 새 epoch로 기록하지 않아야 한다.
- `core.evaluate`는 `model.eval()`과 no_grad를 사용한다. 고정 모델의 전체 Train/Validation 비교에 같은 함수와 동일 분모를 사용하는 방향은 타당하다.

## 구현 검토 시 반드시 확인할 항목

1. **기존 Validation RNG 소비 보존.** `DataLoader(generator=None)`의 iterator 생성은 shuffle=False여도 Torch 전역 RNG를 소비한다. 추가 Train-eval만 상태를 저장·복원하고, 원래 Validation 소비는 유지해야 한다. 정상 Validation까지 복원하면 이후 dropout 난수 궤적이 바뀐다.
2. **추가 평가의 학습 비개입.** Python/NumPy/Torch/CUDA RNG, 모든 하위 module mode, 모델 state 및 학습 loader generator가 추가 평가 전후 동일해야 한다. 별도 비셔플 Train-eval loader를 쓰며, 생성/순회 모두 보존 범위에 포함하거나 전용 generator를 사용한다.
3. **순수 위치 표현 변경.** 기존 모델을 원래 순서로 구성한 후 pos_emb parameter만 deterministic persistent buffer로 교체한다. 다른 초기 state tensor와 초기화 후 RNG가 동일해야 한다. buffer는 optimizer에서 제외되고 checkpoint 왕복 후 동일해야 한다.
4. **epoch 및 최종 replay 구분.** epoch당 monitor 한 줄과 completed epoch 수가 일치해야 한다. 최종 best replay를 마지막 epoch의 Train/Validation 곡선으로 오인해서는 안 된다.
5. **source·input·scope 경계.** RAF/Taxi, seed42,5080,Train/Validation만 허용하고 Test는 materialize하지 않는다. 기존 output은 overwrite/자동재시도하지 않는다.
6. **단일 변수의 해석 범위.** 표준 sinusoidal은 기존 N(0,0.02²) 위치표와 표현함수·크기·학습가능성이 함께 다르다. 고정 위치 표현의 설계 비교이며 미노출 행 하나의 단독 인과효과 실험은 아니다.
7. **시간·비용 해석.** full Train-eval 추가 wall time은 기존 순수 학습/Validation 시간과 직접 비교하지 않는다.

## 로컬 자동 확인

`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3`의 Torch 2.14.0에서 비셔플 generator=None DataLoader를 순회한 전후 RNG가 달라지는 것을 확인했다. 같은 seed로 두 번 실행한 전후 상태가 각각 일치했다. 신규 runner의 RNG 보존 검증은 아직 수행하지 않았다.

## 심각도별 현재 판정

- CRITICAL: 신규 구현이 없어 확정 결함 없음.
- HIGH: 신규 구현이 없어 확정 결함 없음. 위 1–5의 위반은 최종 실행 전 차단 대상이다.
- MEDIUM: 신규 구현이 없어 확정 결함 없음.
- LOW: 실행 프로토콜 문서 경로 누락. 검토 범위와 별도로 기록하며 소스 결함으로 간주하지 않는다.

최종 PRELAUNCH_QA는 runner·테스트·실행 계약·봉인된 소스/입력 증적을 확인한 뒤 작성한다.
