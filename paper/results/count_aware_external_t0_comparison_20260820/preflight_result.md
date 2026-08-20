# Count-aware External T0 NHP/SAHP Preflight

- 실행 서버: 5080
- source revision: `8ec1f42fe01f9e436296914cb4d8d2a950528732`
- tmux: `external_t0_nhp_sahp_e300_0820`
- 시작 시각: 2026-08-20 20:18:24 KST

## 판정

- source checksum: 로컬과 5080 일치
- CUDA: PyTorch `2.11.0+cu130`, RTX 5080 사용 가능
- dataset SHA-256: `85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f`
- split manifest SHA-256: `393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04`
- focused tests: `20 passed`
- runner dry-run: 통과
- launch contract: `t0_common_control`, NHP/SAHP, direct log-MSE,
  `legacy_clamped_rmtpp`, seeds 42/52/62, validation-only
- tmux 및 CUDA process: 정상 진입

Held-out test는 실행하지 않는다. 완료 확인은 사용자 요청 시 단회 수행한다.
