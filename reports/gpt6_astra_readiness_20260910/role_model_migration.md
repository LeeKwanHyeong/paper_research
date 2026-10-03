# paper_research 역할별 모델 전환

2026-09-10. 사용자 승인: 연구 판단 역할을 Astra, 일반 역할을 Terra로 전환.
대상 저장소: `paper_research`, 작업 브랜치: `codex/hard-lmm-causal-qkv`.

**프로젝트 역할 설정 적용 — 완료**
- [.codex/config.toml](../../.codex/config.toml)에 12개 역할을 명시적으로 등록하고,
  각 역할을 프로젝트의 `.codex/agents/<role>.toml`에 연결했다.
- 사용자 공용 `~/.codex/config.toml`, `~/.codex/agents/`와 OMA의 `.agents/`
  원본은 변경하지 않았다. 다른 프로젝트의 역할 정책은 이번 변경 대상이 아니다.
- 기존 역할 파일의 이름·설명·지침·sandbox·skill 설정은 보존하고 모델과 필요한
  추론 수준만 지정했다. 주 작업의 Astra/high 설정도 유지한다.

| 역할 | 모델 | 추론 수준 |
| --- | --- | --- |
| architecture-reviewer | gpt-6-astra | high |
| debug-investigator | gpt-6-astra | high |
| qa-reviewer | gpt-6-astra | high |
| research-explorer | gpt-6-astra | medium |
| docs-curator | gpt-5.6-terra | medium |
| pm-planner | gpt-5.6-terra | medium |
| refactor-engineer | gpt-5.6-terra | medium |
| backend-engineer | gpt-5.6-terra | medium |
| db-engineer | gpt-5.6-terra | medium |
| frontend-engineer | gpt-5.6-terra | medium |
| mobile-engineer | gpt-5.6-terra | medium |
| tf-infra-engineer | gpt-5.6-terra | medium |

내장 default/worker/explorer의 정책은 이번 전환에서 변경하지 않았다. 그 역할에
대해서는 상속된 모델을 사용할 수 있으므로 위 표의 명명된 역할과 구분한다.

**설정 연결 및 보존 검증 — 완료**
- 12개 역할의 TOML 구문·필수 필드·모델별 지정값과 비모델 필드 보존을 확인했다.
- Codex CLI 0.153.4의 새 app-server에서 `config/read(cwd=paper_research)`를
  실행해 12개 역할의 `config_file`이 프로젝트 파일을 가리키는 것을 확인했다.
  연결된 파일의 모델/effort도 대조했다. 주 모델은 Astra/high였다.
- `codex debug prompt-input` 진단도 정상 종료했다. 이 명령의 결과에는 역할별
  모델 목록이 없으므로 이를 역할 모델 검증 근거로 사용하지 않았다.
- [runtime 설정 확인](role_config_runtime_check.json)과
  [파일 보존 및 모델 매핑 검증](role_migration_verification.json)에 증적을 남겼다.
- 서브에이전트의 실제 추론, GPU 실험, 학습 코드 테스트는 수행하지 않았다.
  설정 로딩 검증이며 역할별 품질·속도·계정 가용성 실험을 대신하지 않는다.

Codex는 프로젝트의 standalone agent 파일과 역할별 `config_file`을 지원한다.
명시적 모델·추론 설정은 custom agent 파일에서 적용된다.
[공식 custom agent 문서](https://learn.chatgpt.com/docs/agent-configuration/subagents#custom-agents),
[공식 설정 참조](https://learn.chatgpt.com/docs/config-file/config-reference#configtoml).

**기존 작업의 새 설정 적용 확인 — 다음 실제 사용 시**
- 이번 확인은 새 app-server가 로드한 설정이다. 이미 열려 있는 작업의 역할 목록이
  실시간으로 갱신되었는지는 확인하지 않았다.
- 다음 역할 작업에서 실제 모델을 확인한다. 기존 목록에 GPT-5.4가 남아 있다면
  Codex를 재시작해 설정을 다시 로드한 뒤 사용한다.
- 프로젝트 지침과 파일을 변경했다고 해서 현재 실행 중인 에이전트의 모델이
  소급 변경되는 것은 아니다. 이 작업에서는 별도 에이전트를 실행하지 않았다.

**유지 관리와 복원**
- 프로젝트 모델 정책은 `.codex/config.toml` 및 `.codex/agents/`에서 관리한다.
  OMA 생성 도구를 실행할 때 이 파일들이 재생성되는지 확인하고, 이번 정책을
  의도치 않게 덮어쓰지 않는다. `.agents/` 원본을 직접 수정하지 않는다.
- 복원할 때는 이번에 추가한 프로젝트 역할 등록과 12개 역할 파일만 제거하면
  공용 역할 설정으로 돌아간다. 단, 공용 파일에는 이전 GPT-5.4 설정이 남아 있어
  로그인 방식에 따른 모델 지원 여부를 먼저 확인해야 한다.
- `paper_research`의 `codex/hard-lmm-causal-qkv` 작업 트리에 변경을 남겼으며,
  커밋·Push·MR은 수행하지 않았다.
