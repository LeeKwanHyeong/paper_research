"""Observe/finalize Instacart only after the user moved history27 to A100.

No activation path is imported: the old RTX history/retrieval queue is reserved.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    from paper.scripts.observe_titantpp_contribution_campaigns import observe
    from paper.scripts.finalize_titantpp_contribution_campaigns import finalize
    from paper.scripts.prepare_titantpp_contribution_campaigns import REPORT, write
    reservation = ROOT / 'reports/titantpp_history_a100_20261008_v1/ownership_reservation.json'
    r = json.loads(reservation.read_text())
    assert r['old_history_activation_allowed'] is False
    assert r['old_retrieval_activation_allowed'] is False
    result = {'stage': 'instacart', 'observations': observe('instacart'),
              'finalization': finalize('instacart'),
              'history_owner': r['new_history_bundle'],
              'retrieval_status': 'waiting_for_same_runtime_baseline_contract',
              'activation_calls': 0, 'created_unix': time.time()}
    write(REPORT / 'rounds' / f'instacart_only_{time.time_ns()}.json', result)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
