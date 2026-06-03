# simple_lab_test

이 폴더는 `paper_research` 안에서 실험 스크립트, 검증 노트북, 탐색 러너를
정리해 둔 실험 작업 공간입니다. 단순 보관 폴더가 아니라, 현재까지의 설계
변화와 다음 실험 우선순위를 이어받는 기준점 역할을 하도록 관리합니다.

## 구조

- `common/`
  - 공통 경로 해석 유틸
- `search/`
  - 재현 가능한 하이퍼파라미터 탐색, A/B 테스트, 후속 ablation 설계 문서
- `demos/`
  - 빠른 스모크 테스트와 예시 실행 스크립트
- `notebooks/preprocessing/`
  - 데이터 전처리 및 mark 설계 탐색 노트북
- `notebooks/baselines/`
  - Poisson/Hawkes 같은 전통적 점과정 베이스라인 노트북
- `notebooks/experiments/`
  - RMTPP/TitanTPP 실험용 노트북
- `notebooks/validation/`
  - 모델 검증 및 분포 시각화 노트북

## 실험 설계 히스토리

아래 순서는 지금 실험 코드가 어떤 배경으로 만들어졌는지, 그리고 어떤
설계가 이미 시도되었는지를 빠르게 복기하기 위한 기록입니다.

### 1. 고정 주기 회귀에서 이벤트 기반 TPP 해석으로 전환

처음 문제의식은 intermittent demand가 긴 zero 구간과 드문 positive-demand
event를 갖는다는 점이었습니다. 이 특성 때문에 주간/월간 고정 길이 회귀보다
`positive-demand event sequence`로 해석하는 편이 더 자연스럽다고 보고,
다음 이벤트의

- mark
- inter-event time
- quantity

를 예측하는 TPP 프레임으로 방향을 옮겼습니다.

### 2. 초기 mark 설계: quantity bin + representative quantity

초기 설계에서는 `mark`를 quantity bin으로 두고, 예측된 mark를
`rep_qty(mark)`로 복원했습니다. 이 방식은 구현이 단순했지만, 다음 문제가
분명했습니다.

- bin 경계 설정에 지나치게 민감함
- 같은 bin 내부의 값 차이를 잃어버림
- representative quantity 하나로 복원할 때 정보 손실이 큼

이 문제 때문에 quantity 복원을 더 직접적으로 다루는 설계가 필요해졌습니다.

### 3. magnitude-factorized 설계 도입

그 다음 단계에서 quantity를 한 번에 직접 분류하지 않고,

- `mark = coarse magnitude class`
- `value residual = within-class continuous scale`

로 분해하는 magnitude-factorized 설계를 도입했습니다.

현재 핵심 복원식은 다음과 같습니다.

```text
log_qty = mark + residual
qty = base^(log_qty)
```

이 설계의 목적은 두 가지였습니다.

- mark의 해석성을 유지하기
- representative quantity 없이 연속 수량을 직접 복원하기

### 4. week-lookback 데이터셋 정착

초기에는 event-count 기반 lookback도 함께 유지했지만, 현재 주요 학습 경로는
`최근 W주 안의 이벤트를 모아 left-pad`하는 week-lookback 방식으로
정리되었습니다. 이 구조는 sparse demand에 더 자연스럽고, RMTPP와 TitanTPP가
같은 split과 padding 규칙을 공유하도록 만들기 쉬웠습니다.

### 5. RMTPP에 residual head 추가

현재 RMTPP 계열은 다음 세 head를 동시에 학습합니다.

- mark head
- time head
- value head

여기서 value head는 최종 qty가 아니라 `log-scale residual`을 예측합니다.
즉 지금의 supervised quantity 경로는

```text
residual regression -> reconstructed qty -> validation qty_mae
```

형태이고, `qty_mae`는 검증 지표이지만 아직 직접 학습 loss는 아닙니다.

### 6. TitanTPP 구축

그 다음 단계에서 RMTPP의 recurrent backbone을 Titan encoder로 대체한
`TitanTPP`를 구축했습니다. 이 과정에서

- Titan encoder
- optional LMM
- RMTPP형 mark/time/value head

를 같은 TPP 프레임 안에서 결합했습니다.

핵심 비교 질문은 이후부터

- RMTPP vs TitanTPP
- 동일한 magnitude-factorized quantity 복원에서 어떤 encoder가 더 유리한가

로 정리되었습니다.

### 7. log base 탐색

`mark = floor(log_base(qty))`에서 base가 너무 크면 head class 쏠림이 심해지고,
너무 작으면 클래스 수가 과도하게 늘어납니다. 이를 확인하기 위해

- `log10`
- `log4`
- `log2`

를 비교했고, 실험적으로는 `log4`가 균형 면에서 유력했고, 이후 전체 탐색과
A/B 테스트에서는 `log10`도 함께 유지하며 dataset별 최적 조합을 찾았습니다.

### 8. Titan 하이퍼파라미터 탐색 자동화

`search/titan_hparam_search.py`는 다음 조합을 자동 탐색하도록 만들었습니다.

- dataset: `intermittent`, `yellow_trip`
- scale base: `10`, `4`, `2`
- Titan preset: depth, width, LMM, memory 계열 조합

이 스크립트는 stage1 coarse search와 stage2 refinement로 나누어,
중간 캐시와 leaderboard를 남기도록 설계했습니다.

### 9. RMTPP vs TitanTPP A/B 테스트

탐색 결과를 바탕으로 `search/titan_rmtpp_ab_test.py`에서 dataset별 최적 Titan
조합을 사용해 30 epoch 기준 A/B 테스트를 수행했습니다.

현재까지의 요약은 다음과 같습니다.

- `intermittent`: TitanTPP가 score, qty_mae, dt_mae 쪽에서 소폭 우세
- `yellow_trip`: TitanTPP가 score, validation NLL, qty_mae에서 더 분명한 우세
- 전체적으로 30 epoch 이후에는 TitanTPP가 overall winner

### 10. 현재 열린 문제: qty metric의 출렁임

지금 가장 중요한 후속 질문은 `qty_mae`의 epoch별 흔들림입니다. 특히 TitanTPP는
quantity reconstruction이 좋아지더라도 curve가 RMTPP보다 더 출렁여 보일 수
있습니다.

원인은 현재 구조상 다음 두 가지가 겹치기 때문입니다.

- loss는 `residual`에 직접 걸리고, `qty_mae`는 간접 지표임
- 작은 log-scale 오차가 `base^(mark + residual)` 복원에서 확대될 수 있음

따라서 다음 단계는 `qty` 자체를 loss로 직접 다루는 실험입니다.

## 다음 실험 우선순위

다음 우선순위는 `qty loss` 설계 비교입니다.

- 현재 baseline: residual-only supervision
- 실험안 1: qty direct loss
- 실험안 2: residual loss + qty loss hybrid

상세 설계는 [`search/README.md`](search/search_experiment_guide.md)에 정리합니다.

## 참고

- 새 Python 스크립트들은 `simple_lab_test/common/pathing.py`를 통해
  프로젝트 루트를 자동으로 찾습니다.
- 기존 노트북들은 호환성을 위해 bootstrap 셀을 추가하는 방식으로
  경로를 안정화했습니다.

## Jupyter 시작 셀

서버 Jupyter에서 가장 안정적인 시작 패턴은 아래와 같습니다.

```python
from pathlib import Path
import sys

SERVER_PROJECT_ROOT = Path('~/workspace/paper_research').expanduser().resolve()
if str(SERVER_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_PROJECT_ROOT))

from utils.notebook_bootstrap import bootstrap_notebook

PROJECT_ROOT, DIR = bootstrap_notebook(preferred_root=SERVER_PROJECT_ROOT)
```

이제 노트북마다 경로 탐색 로직을 직접 복사하지 않고 이 유틸을 재사용하면 됩니다.
