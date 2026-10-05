# CNN+GRU54 3seed 매시간 관측

기존 자동화 titantpp-gru-3seed를 새6조건 관측으로 전환합니다. 현재 전환 중이며 실제학습확인 후 활성화합니다. 저장된 scheduler_prompt.txt와 contract/current/receipts를 기준으로 source/GPU/PID·실제저장epoch·selected/lastfullValidation을확인하고 terminal원본SHA회수까지관리합니다. 종료된 이전33조건 및 A100은다시관측하지않습니다. Test3seed와독립평가·CPU재추론감사는별도입니다.
