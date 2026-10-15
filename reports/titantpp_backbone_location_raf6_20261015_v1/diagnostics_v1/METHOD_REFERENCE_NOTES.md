# 다음 CNN 가설 검토를 위한 선행 방법 확인

아직 새 방법론 채택이나 학습 계약이 아니다. 기존 Train·Validation의 관측 시간 간격과 오류를 먼저 확인하며, 공통 입력·손실·head·메모리는 유지한다.

- [CKConv: Continuous Kernel Convolution For Sequential Data](https://arxiv.org/abs/2102.02611): 연속 좌표의 함수로 convolution kernel을 표현해 불규칙 샘플링을 다루는 선행 방법이다. 우리 모델의 성능 개선을 보장하지 않으며, 시간 좌표 CNN이라는 발상 자체를 새로운 기여로 주장할 수 없다.
- [Time-Parameterized Convolutional Neural Networks for Irregularly Sampled Time Series](https://arxiv.org/abs/2308.03210): 시간 함수를 convolution weight에 반영한 선행 방법이다. 논문의 보간·분류 실험 결과를 RAF 수량·시간 예측에 그대로 옮겨 주장하지 않는다.

현재 코드에도 시간 입력은 이미 존재한다. 검토할 수 있는 차이는 시간 정보 유무가 아니라 CNN 필터가 관측 순서와 실제 경과시간을 처리하는 방식이다. 기존 예측을 이용한 조건별 분석은 상관 근거이며 인과적 결함이나 새로운 구조의 효과를 증명하지 않는다. 간격 변화 또는 비교 지원이 부족하면 이 축을 근거 없는 다음 학습으로 연결하지 않는다.
