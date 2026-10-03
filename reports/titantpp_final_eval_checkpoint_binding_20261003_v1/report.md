# 최종 평가 checkpoint 원본 연결

원래84조건과 구조대조24조건의 selected checkpoint 총108개를 모두 로컬 원본·파일SHA·tensorSHA·선택epoch·동결소스·계약에 연결했다.

설계 manifest의 누락36개는 실제 유실이 아니었다. 2026-10-01 준비 작업에서 이미 회수·감사된 원본이 다른 로컬 경로에 있었으며, 이번에는 두 보존 archive와755개파일을 SHA 재확인하고36개 selected weight를 원래6개 동결소스로 CPU strict-load 재검증했다. 네 모델(RMTPP·THP·NHP·SAHP) × Taxi·Intermittent·Instacart ×3seed에 해당한다. 원격 호출·추가 다운로드는0회다.

기존48개 원래모델과24개 구조대조는 이전 CPU 감사 및 현재 원본 파일SHA를 재사용했다.36개 재감사는 selected weight/tensor/route/epoch/validation 선택과 기록된exposure 범위이며, 없는 optimizer/last-state를 검증했다고 주장하지 않는다. 신규forward·학습·재평가·held-out 열람은0회다.

실행기는 `evaluation_registry.json`의 각 row `evaluator_source_bundle`을 `bundles`에 연결해 `contract_path`, `source_root`, `source_files`, `source_closure_sha256`, `datasets`, `factory_module`, `factory_function`을 사용한다. 공통 CountAwareFactory에 계약의 model 설정과 train통계 및 max_seq_len을 적용한다. 모델별 원래 selected epoch와 tensor는 변경되지 않았다. 서로 다른 frozen source는 별도 프로세스로 import한다. 모든 path는 프로젝트 root 기준이다.

`path_resolutions.json`은 낡은36개 alias와 보존원본 경로의 대응표이고, `checkpoint_audit.json`은 이번36개 CPU감사, `verification.json`은 archive·SHA 연결 검증이다. `condition_registry.csv`는108개 행을 제공한다.

이 연결 완료는 평가자료의 독립성 확립이나held-out 실행승인을 뜻하지 않는다. 원본데이터 적격성 검토와 공통평가실행기 검증은 별도 작업이다.
