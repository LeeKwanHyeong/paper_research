# TPP 모델 패키지 구조

## 결정

`models.TPPs`를 temporal point process 모델의 단일 package로 사용한다.

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

모든 실행 코드에서는 `models.TPPs`만 import한다. 모델 구현은 실험
스크립트를 import하지 않으며 의존 방향을 model에서 runner 방향으로
역전시키지 않는다.

## Import 경로

모든 모델은 다음 단일 경로에서 가져온다.

```python
from models.TPPs import CountAwareNHP, CountAwareSAHP
```

`models.RMTPPs` compatibility package는 유지하지 않는다. 저장 checkpoint는
모델 객체 전체가 아니라 `state_dict`를 저장하는 현재 실험 계약을 사용한다.
