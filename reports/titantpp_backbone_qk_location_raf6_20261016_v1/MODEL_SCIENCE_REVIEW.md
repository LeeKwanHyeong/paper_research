# 루프5 모델 구현–과학명세 대조

판정: **PASS_static_science_alignment**. 지정한 두 신규 모델의 핵심 과학 조건과 구현 사이에 불일치를 발견하지 않았습니다.

검토자는 모델·실행기·테스트 구현 저자와 다르지만 과학명세 작성자입니다. 따라서 독립적인 과학 설계 심사가 아니라, 고정 명세에 대한 정적 구현 대조와 합성검사 재실행입니다.

- **capacity_and_physical_removal — PASS**: Each arm retains only its selected original 3x64 parameter: 192 local coefficients. The other Q/K kernel and V local kernel are absent, not frozen zero/dummy parameters. Relative original CNN576: -384; relative QK-only384: -192. Historical inactive GRU registration remains unchanged.
- **common_qkv_and_v_projection — PASS**: Common qkv object and all projector rows remain normally trainable; removing the local V filter does not remove or freeze the common V projection. Synthetic gradient tests cover the retained192 coefficients and common V projection.
- **initialization_rng — PASS**: One original CNN constructor initializes the common tensors and original kernels. Replacement reuses those objects and introduces only a deterministic uint8 identity buffer; no new random initialization or checkpoint warmstart. Direct wrapper is state/RNG-equivalent to the QK-only-then-single-location design inheritance. Synthetic tests compare Torch/Python/NumPy RNG and all retained tensors, all3 seeds and contexts32/84/256.
- **forward_algebra — PASS**: Independent stdlib AST comparison: resolving the candidate q/k branch makes its complete forward AST identical to original ObservedCausalQKVMemoryAttention.forward after deleting only the two unselected local-residual assignment statements. All subsequent memory concatenation, scoring, masking, softmax, dropout, output projection and query masking are unchanged.
- **observed_boundaries_and_target_isolation — PASS**: Inherited observed_context and causal_depthwise_residual preserve current/previous/second-previous observed taps, padding skip, withheld boundary reset and unavailable taps. Synthetic tests include NaN padding, future/target perturbation, short histories, batch consistency and unchanged evaluation state.
- **memory_asymmetry — PASS**: Memory tokens bypass the retained local filter. Query filtering changes query scores against both event and memory keys; Key filtering changes event keys before memory concatenation. Either can change normalized memory mass. This is the specified position asymmetry, not identical functional intervention on memory.
- **common_training_conditions — PASS**: Runner reuses original common training/replay functions and model input/head implementations. It validates hidden64, quantity loss1/tail0, unchanged recorded-time head, AdamW .001/.01/clip1, loader128, max300/min40/patience40, strict earliest finite full-Validation rawRMSE selection and no warmstart/force rerun. No added gate/norm/nonlinearity/RevIN/GRU activation.
- **arm_identity_and_baseline_guard — PASS**: Distinct metadata and state identity bind both selected and terminal states to the physical single-kernel arm. Complete state validation remains strict even load_state_dict(strict=False); forged cross-arm identity with wrong physical kernel, baseline relabel, missing/extra/dummy tensors and changed split/head/selector are rejected in synthetic tests. training_args allows only the two new candidates, not the CNN alias. Historical baseline comparison/reuse is an operational responsibility outside this model review; no historical CNN checkpoint/prediction was read or run here.

독립 재실행한 합성검사는 `63 passed in 9.77s`, 실제 종료코드0입니다. 로컬 torch2.14.0/pytest9.0.3 환경이며 실제 데이터와 기존 CNN checkpoint는 읽지 않았습니다. 원래 CNN의 무작위 초기 모델을 이용한 합성 비교는 기존 기준선 재평가가 아닙니다.

전체 forward의 AST를 별도로 대조한 결과, 각 후보의 분기를 확정하면 원래 함수에서 선택하지 않은 지역 잔차 두 줄만 제거한 연산과 일치했습니다. 공통 V 투영은 계속 학습하며, V 지역 필터만 물리적으로 없습니다.

초기화의 QK-only 상속은 설계·상태 상속을 뜻합니다. 원래 CNN을 한 번 만든 뒤 해당 Q 또는 K kernel만 남기는 구현은 중간 모델을 추가 생성하지 않으며, 합성검사에서 QK-only를 거치는 구성과 초기 텐서·난수 상태가 일치했습니다.

Q-only와 K-only는 각각192계수로 같지만, 메모리 점수에 영향을 주는 위치는 다릅니다. 기존 CNN576 및 QK-only384와의 비교에는 용량 감소와 공동 최적화 차이가 남습니다. 성능 개선이나 원인 규명을 확인한 결과는 아닙니다.

**남은 작업 — 다음 작업**
- 별도 운영 검증에서 source145·입력 SHA·승인·기한·기존 결과 재사용을 확인합니다.
- 5090의 고정 Runtime에서 실제 합성 qualification이 통과한 뒤 승인된 신규6개만 학습합니다. 이 검토는 원격 실행이나 학습 완료를 확인하지 않았습니다.

모든 검토 입력 SHA와 코드 위치는 같은 이름의 JSON에 기록했습니다. 기존 source와 과학명세는 수정하지 않았습니다.
