# simple_lab_test

실험용 스크립트와 노트북을 기능별로 정리한 폴더입니다.

## 구조

- `common/`
  - 공통 경로 해석 유틸
- `search/`
  - 재현 가능한 하이퍼파라미터 탐색 및 비교 실험 스크립트
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
