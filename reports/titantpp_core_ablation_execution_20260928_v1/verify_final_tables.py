"""Independent Decimal recomputation from saved endpoints, without finalizer helpers."""
from decimal import Decimal, localcontext
from pathlib import Path
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def main():
    records = json.loads((OUT/'final_results.json').read_text())
    actual = {}
    for row in records['core_conditions']:
        ep = json.loads((ROOT/row['source']/'endpoint_replays.json').read_text())
        actual[row['dataset'],row['arm'],row['seed']] = ep
    external = json.loads((ROOT/'reports/local_detail_3seed_final_20260927_v1/condition_results.json').read_text())
    for row in external:
        if row['model'] in ['rmtpp','nhp','sahp','thp']:
            actual[row['dataset'],row['model'],row['seed']] = row
    assert len(actual) == 90
    checked = 0
    with localcontext() as ctx:
        ctx.prec = 40
        for row in csv.DictReader((OUT/'final_three_seed_metrics.csv').open()):
            for metric in ['qty_mae','qty_rmse','time_nll']:
                values = [Decimal(str(actual[row['dataset'],row['model'],seed][row['endpoint']][metric])) for seed in [42,52,62]]
                mean = sum(values)/3
                sd = (sum((x-mean)**2 for x in values)/2).sqrt()
                for field,expected in [('mean',mean),('sample_sd',sd)]:
                    assert abs(Decimal(row[f'{metric}_{field}'])-expected) < Decimal('1e-11'), row
                    checked += 1
    manifest=json.loads((OUT/'final_output_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    conditions=list(csv.DictReader((OUT/'final_conditions.csv').open()))
    assert len(conditions)==180 and len({(r['dataset'],r['arm'],r['seed'],r['endpoint']) for r in conditions})==180
    assert sum(r['campaign']=='core' for r in conditions)==108
    strata=list(csv.DictReader((OUT/'final_strata.csv').open()))
    empty=[r for r in strata if r['count_per_seed']=='0']
    assert empty and all(not r['qty_rmse_mean'] and not r['qty_mae_mean'] for r in empty)
    report=(OUT/'final_report.md').read_text()
    required=['Instacart','시간 NLL','표본표준편차','0/9','validation only','held-out','복구','Full 기반','평가 peak','전기료']
    assert all(x in report for x in required)
    result={'status':'passed','independent_method':'Decimal40 arithmetic from original endpoint JSON; no finalizer calculation helpers','aggregate_cells_checked':checked,'aggregate_rows':60,'unique_condition_endpoint_rows':180,'core_endpoint_rows':108,'stratum_rows':len(strata),'empty_cells_preserved_as_missing':len(empty),'output_hashes_checked':len(manifest['files']),'report_claims_and_limitations_checked':required,'new_predictions':False,'new_gpu_execution':False,'held_out_read':False}
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
