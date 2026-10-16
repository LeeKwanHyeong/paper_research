# Cap36 격리 import 검사 — 완료

**결과: 로컬 import 범위에서 통과했습니다.** 검사 대상은 **v1 배포 압축본의 Python 코드 941개에 임시 빈 `sample_data` 디렉터리 하나를 추가한 묶음**입니다. 실제 v2 압축본을 검사한 결과가 아닙니다.

- 실제 subprocess 종료 코드: 0. 프로젝트 모듈 90개가 모두 임시 추출 경로에서 로드됐습니다.
- runner의 `install_hooks()`, 공통 학습·평가 도우미와 Cap36 runtime import가 통과했습니다.
- 저장소의 추가 비Python 리소스 읽기는 0이었습니다. `.git`과 과거 `paper/contracts` 디렉터리가 없는 상태로 통과했습니다.
- 원 압축본의 선언 source 941개와 입력 9개가 모두 포함돼 있습니다. source 파일의 SHA도 재검증했습니다. 입력 내용은 열지 않았습니다.
- 실제 데이터 접근, 모델 생성·추론, 학습, 원격 호출, source 변경은 모두 0입니다. Matplotlib의 임시 폰트 캐시 생성 메시지만 있었습니다.

`simple_lab_test/common/pathing.py:16,38`은 `models`, `utils`, `sample_data`의 존재만 검사합니다. 이번 확인 범위에서 `sample_data` 안의 실제 데이터나 다른 추가 프로젝트 리소스는 필요하지 않았습니다.

| 결속 대상 | SHA-256 |
| --- | --- |
| v1 deployment.tar.gz | `959c05ef02fb280dab2dbc8c34ceb080735327877ac3ea5919d8b88958b30e72` |
| v1 execution_contract.json 원문 | `266122f1a71ae8104b42ae1433ff442a78776ac865d8c3b7395653f4d0d5d1b2` |
| source 파일 목록 | `83d888784103749f11c699f4122ade59e8a22018027a4502186b96309d86c148` |

source revision은 `ab8d2f20068e526c4696b6e71250adf2aed3b847`입니다. 실행한 전체 명령과 캡처 stdout/stderr, 도구 증적 식별자는 [ISOLATED_IMPORT_QA.json](ISOLATED_IMPORT_QA.json)에 보존했습니다. 별도 raw transport 파일은 생성하지 않았고, 문서화를 위해 검사를 반복하지 않았습니다.

**다음 확인 — 다음 작업**

실제 v2 압축본을 그대로 임시 추출한 뒤 추가 디렉터리를 수동 생성하지 않고 같은 import 검사를 실행합니다. 세 표식 경로가 있고, 판정된 root와 로드된 프로젝트 모듈 경로가 모두 추출 source 안에 있으면 이 패키징 회귀 검사를 통과합니다. 이 검사는 native GPU 검증이나 실제 학습·평가 검증을 대신하지 않습니다.
