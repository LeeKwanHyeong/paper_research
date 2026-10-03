# 완료 실험과 외부 비교군 전체 비교

대표 History MLP, 외부6종, 활성분기정규화 seed42, 기존 Full/내부대안/Gate를 구분했다. 완료114조건,34개완료3seed그룹,Taxi/Intermittent의36개동일seed외부비교를검증했다. 진행중조건은완료표에제외하고comparison.json에미완료상태를남겼다.

보고서: report.md / 수치: selected_conditions.csv / 집계: comparison.json / 검증: verification.json

로컬앱 http://127.0.0.1:4181/?view=1 은원본수치에기반하며공개게시하지않았다. 브라우저에서Taxi·Instacart표·데이터전환·완료표시·시각적레이아웃을확인했다.

추가TPP와5080Instacart정규화의최종binaryCPU감사는별도미완료다. 기존감사와소형기록SHA확인을그감사의완료로표시하지않는다.
