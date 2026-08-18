# TPP 모델 패키지 구조

## 결정

`models.TPPs`를 temporal point process 모델의 canonical package로 사용한다.
기존 `models.RMTPPs`는 과거 코드의 import 호환성을 위한 legacy namespace로
유지한다.

`RMTPPs` 아래에 NHP, SAHP, THP, TitanTPP를 함께 두면 패키지 이름이 실제
소유 범위를 표현하지 못한다. 반대로 모델마다 최상위 package를 만들면 공통
TPP head와 factory가 여러 package에 분산된다. TPP family를 하나의 package로
모으는 방식이 공통 인터페이스와 실험 확장에 가장 적합하다.

## 모듈 책임

| 파일 | 책임 |
|---|---|
| `RMTPP.py` | RMTPP public import |
| `TitanTPP.py` | Titan encoder 기반 TPP public import |
| `TransformerHawkesTPP.py` | Transformer Hawkes Process public import |
| `NeuralHawkesTPP.py` | adapted continuous-time LSTM NHP |
| `SelfAttentiveHawkesTPP.py` | adapted self-attentive SAHP |
| `CountAwareTPP.py` | 공통 count-aware time·quantity head와 controlled wrappers |
| `CountAwareFactory.py` | backbone 선택과 metadata 생성 |
| `config.py` | TPP configuration public import |

## 의존 방향

```text
paper experiment runner
    -> paper experiment helpers
        -> models.TPPs public package
            -> shared encoder/head implementations
```

새 코드에서는 `models.TPPs`만 import한다. `models.RMTPPs`에서 새 package를
다시 import하는 호환 wrapper는 허용하지만, `models.TPPs`의 실제 NHP·SAHP
구현이 실험 스크립트를 import해서는 안 된다.

## 이전 경로 호환성

`models.RMTPPs`는 즉시 삭제하지 않는다. 오래된 notebook과 저장된 실험
코드가 다음 경로를 계속 사용할 수 있도록 wrapper와 lazy export를 제공한다.

```python
from models.RMTPPs.NeuralHawkesTPP import CountAwareNHP
from models.RMTPPs.SelfAttentiveHawkesTPP import CountAwareSAHP
```

신규 코드는 다음 경로를 사용한다.

```python
from models.TPPs import CountAwareNHP, CountAwareSAHP
```

기존 RMTPP·TitanTPP·THP 구현 파일의 물리적 이동은 전체 notebook과 외부
checkpoint 소비 경로를 별도로 감사한 뒤 진행한다.
