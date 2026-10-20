### 측정 완료 — <mention-date start="2026-10-20" startTime="15:43" timeZone="Asia/Seoul"/>
RAF·Taxi·Intermittent의 Shared15와 Titans-MAC를 동일한 5090에서 측정했다. 36개 비용 측정, 합성 검증 4개, 입력 준비 1개의 자식 프로세스 41개와 controller의 실제 OS exit0를 확인했고, 원본 230개 파일 SHA 및 source 985개·운영 소스 2개 SHA를 검증했다. 종료 시 소유 프로세스 부재와 GPU compute process 없음도 확인했다. supervisor 자신의 OS exit는 직접 관측하지 않았다.

#### 속도: 동일 batch128에서 학습과 평가 모두 단축
각 반복의 예열 5 batch 뒤 측정 32 batch의 계산 시간 중앙값을 구했다. 아래 값은 fresh process 반복 3회의 평균 ± 표본표준편차다. 속도비는 같은 반복 번호의 MAC/Shared15 비율을 집계했다. 모델별 CPU→GPU 전송은 계산 시간에서 제외했고 별도로 보존했다.
<table header-row="true">
<tr><td>데이터</td><td>경로</td><td>Shared15 ms/batch</td><td>MAC ms/batch</td><td>MAC/Shared15 속도비</td></tr>
<tr><td>RAF</td><td>학습 step</td><td>8.542 ± 0.162</td><td>67.137 ± 0.465</td><td>7.862 ± 0.192배</td></tr>
<tr><td>RAF</td><td>평가 batch</td><td>3.605 ± 0.030</td><td>25.177 ± 0.339</td><td>6.984 ± 0.118배</td></tr>
<tr><td>Taxi</td><td>학습 step</td><td>18.712 ± 0.108</td><td>175.800 ± 3.043</td><td>9.395 ± 0.142배</td></tr>
<tr><td>Taxi</td><td>평가 batch</td><td>5.215 ± 0.029</td><td>63.186 ± 0.810</td><td>12.116 ± 0.220배</td></tr>
<tr><td>Intermittent</td><td>학습 step</td><td>19.260 ± 0.076</td><td>178.008 ± 1.475</td><td>9.243 ± 0.104배</td></tr>
<tr><td>Intermittent</td><td>평가 batch</td><td>5.472 ± 0.048</td><td>63.623 ± 0.419</td><td>11.627 ± 0.115배</td></tr>
</table>

학습 step에는 forward·공통 head·target loss·backward·gradient clip·AdamW가 포함된다. 평가 batch에는 eval/no-grad forward·head·target loss와 MAC의 온라인 메모리 갱신이 포함된다. 따라서 평가 수치를 순수 inference 시간으로 표현하지 않는다. MAC 메모리 갱신을 비활성화해서 얻은 속도 차이가 아니다.

#### 메모리: 학습 peak는 감소하지만 평가 peak는 증가
아래 값은 측정 구간의 최대 allocated GPU 메모리다. reserved 메모리와 각 반복의 불리한 결과도 원본 CSV와 보고서에 모두 보존했다.
<table header-row="true">
<tr><td>데이터</td><td>경로</td><td>Shared15 MiB</td><td>MAC MiB</td><td>Shared15 상대 변화</td></tr>
<tr><td>RAF</td><td>학습</td><td>540.76</td><td>2873.66</td><td>−81.18%</td></tr>
<tr><td>RAF</td><td>평가</td><td>94.48</td><td>86.24</td><td>+9.55%</td></tr>
<tr><td>Taxi</td><td>학습</td><td>1877.71</td><td>7631.00</td><td>−75.39%</td></tr>
<tr><td>Taxi</td><td>평가</td><td>389.24</td><td>97.83</td><td>+297.87% (약 3.98배)</td></tr>
<tr><td>Intermittent</td><td>학습</td><td>1877.71</td><td>7631.00</td><td>−75.39%</td></tr>
<tr><td>Intermittent</td><td>평가</td><td>389.24</td><td>97.83</td><td>+297.87% (약 3.98배)</td></tr>
</table>

전체 모델 parameter는 RAF Shared15 88,451개 대 MAC 111,302개(−20.53%), Taxi·Intermittent 각각 99,459개 대 122,310개(−18.68%)다. parameter 수 감소를 모든 실행 경로의 메모리 감소로 확대하지 않는다.

#### 고정 조건과 해석 범위
- GPU: RTX5090, Python3.12.13, torch2.11.0+cu130, cuDNN92000, threads4. 두 모델 모두 이번 5090에서 새로 측정했으며 이전 5080 수치와 섞지 않았다.
- 원본 Train의 고정 target 4,736개/데이터셋을 공유했다. RAF padded 길이84, Taxi·Intermittent256이고, 측정 target의 평균 유효 길이는 각각4.56·111.16·52.93이다. 모든 길이에 대한 복잡도 검증은 아니다.
- 최종 Shared15 구조를 사용하되 fresh seed42 초기화의 짧은 scratch 비용 측정이다. scientific checkpoint 정확도 평가·저장·수정 및 Validation/Test 재평가는 하지 않았다. 반복 3회는 학습 3seed가 아니다.
- profiling optimizer step666개와 평가 batch666개(예열 포함), 별도 합성 qualification optimizer step12개(CPU6/CUDA6)를 구분한다. 전체 장치 scratch optimizer step 합계678개이며 새 scientific fit은0이다.
- 이번 근거는 해당 조건의 학습 step·평가 batch 속도와 peak 메모리다. 수렴까지 총학습비용, 동일 정확도 비용, 실측 초/epoch 또는 모든 GPU에서의 우세를 입증하지 않는다.
- 결과 판단: 세 데이터에서 실행 속도 및 학습 peak 메모리 절감 근거가 확보됐다. 평가 메모리 증가는 함께 보고한다.
- 5080의 추가 seed72·82 실험과 기존 확정 3seed 결과는 그대로 유지했다.

#### 재현 증적
- 보고서: `reports/titantpp_shared_efficiency_5090_20261020_v2/report.md`
- 집계: `analysis.json`, `verification.json`, `repetitions.csv`, `per_batch.csv`, `inputs.csv`
- contract canonical SHA: `e61b7c4b9dcded33f103ffaa116b3a53f8f304650ff2b3c79bbb332ac22eb81c`
- archive SHA: `77b66f62a44c2bd15c5e6fdc76bcfc921b4f02680ab86c2ee8d999e115f72464`
- terminal manifest SHA: `4afe56a41ee32e2ebde86330b2ae927405fba27390ab62adf54122f42341e25d`
- 전체 native 실행은 준비를 포함해407.181초였다. 첫 경로의 packaging 실패 기록과 원본도 별도로 보존했다.
