# 5090 실제 native 및 최초 학습 독립 QA

상태: `5090_NATIVE_PASS_THREE_INITIAL_FITS_VERIFIED_BEFORE_FIRST_SAVE`. QA 직접 SSH/GPU0, 로컬 증거 확인만 수행했다.

## Review Result: PASS — native/동시성 gate

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

원관측: 2026-10-12 18:39:32.206 KST. `control/qualification_cache/5090/snapshot.json` raw SHA `1d86b4bb56117a0e747ef2fe6dffae35f154e14130f7deb1b88dfbb95848eb49`, embedded26개 raw SHA를 독립 확인했다. 봉인 `qualification_binding` 재실행 PASS다.

- qualification receipt raw SHA `4e0c035f12f8dd3cc75d8b8ddde349f46375bd8e5709a275a4fdc38986914fdb`.
- native child guard1058774/parent1058760 실제 exit0.
- FULL/CNN-only/GRU-only의 동시 GPU PID1058927/1058930/1058931와 정확한 guard·argv·PPID·process group/start ticks 확인.
- capacity guard1058926/1058928/1058929 모두 actual child exit0.
- 마지막 GPU PID는 비었고 최대 관측 memory fraction0.2584107706934094.
- qualify campaign PID1058760/PPID1058751 actual OS exit0 및 complete phase 확인.
- 고정 GPU UUID `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`, 전용 Python3.12.13/Torch2.11.0+cu130/CUDA13.0/cuDNN92000 일치.
- host5090 start permit `1791797850.3020022`→deadline `1792402650.3020022`로168시간 lease. 기존5080 lease는 변경하지 않았다.

Train/Validation interface, 초기화 및3seed 합성·인과성 검증을 확인했다. 합성 native/admission 통과는 scientific fit 완료 건수가 아니며 실제 Test 또는 Validation 성과를 계산하지 않는다. pipeline 자체 OS exit는 직접 관측하지 않았다는 구분을 유지한다.

## 완료 조건

- [x] 5090 실제 native·capacity·qualifier exit0
- [x] 세 동시 GPU owner 및 종료 후 GPU idle
- [x] host5090 source/contract/Runtime/lease 일치
- [x] 최초12조건 중 실제 scientific fit PID/argv/GPU·저장 상태 확인
- [ ] 12조건 자연종료와 원본 회수

## 최초 scientific fit 확인: 2026-10-12 18:41:08.359957 KST

snapshot `control/status/5090/af75041fcbc646c7aa05a4285af5f787.json` raw SHA `418f4d65854d6869a1c89a5d38742ce705e6605b76b0f83c9ffb192d030af3f2` 및 embedded raw SHA를 확인했다. `training_status` 독립 실행 결과 `training_running`, identity_verified true, active3/complete0/total12다.

| Intermittent seed42 구조 | GPU PID / guard PID | 저장 상태 |
|---|---|---|
| 보정 없음 | 1059784 / 1059783 | 첫 checkpoint 전 |
| CNN만 | 1059786 / 1059785 | 첫 checkpoint 전 |
| GRU만 | 1059788 / 1059787 | 첫 checkpoint 전 |
| CNN+GRU | 없음 | 앞의3개 종료 후 대기 |

controller1059740과 각 guard/worker의 exact argv, PPID/process group, lease, 고정GPU 소유를 확인했다. 학습 phase 시작 응답은18:39:56.292 KST이며 시작 응답과 실제 fit 관측을 구분한다. 아직 saved/best epoch는 없고 전체 queue ETA미확정이다. 첫 저장 전 상태를 학습 실패로 해석하지 않는다. 5080 deployment/permit 두 원본 SHA도 계속 불변이다.

## 마지막 초기 소유 확인: 2026-10-12 18:44:38.910305 KST

snapshot `control/status/5090/b31279c0ae28408682b6e59f5e890a05.json` raw SHA `1c21e5693ea7ce25373d923490ae9f817b3547194f709ab1faf1b8ab329c6f88` 및 embedded raw SHA를 검증했다. 동일 controller1059740과 GPU1059784/1059786/1059788, 정확 guard/argv/PPID/process group/lease가 재확인됐고 `training_running`, identity_verified true다.

Intermittent seed42 NONE/CNN-only/GRU-only 세 조건 모두 첫 checkpoint 전이다. 따라서 저장 epoch와 best epoch는 미확정으로 유지한다. 완료0/진행3/전체12, FULL 및 뒤의 seed52/62는 대기다. 완료된 epoch timing이 없어 자체 속도 ETA도 계산할 수 없다. Taxi 속도를 옮겨 사용하지 않는다.

학습 phase 시작 후 약4분43초의 원관측이며, 첫 저장 전을 학습 실패로 바꾸지 않는다. 이 회차의 상태 조회는 여기서 종료하고 추가 polling을 하지 않는다. 전체 종료·최종 성과·원본 회수는 미완료다.
