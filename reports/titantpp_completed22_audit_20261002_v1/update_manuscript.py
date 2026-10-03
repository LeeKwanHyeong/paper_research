"""Update the existing LNCS draft from the pinned, audited validation comparison."""
import json,shutil,hashlib,difflib
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;p=R/'paper/titantpp_pakdd_2027_draft/main.tex'
j=json.loads((D/'comparison.json').read_text());s=p.read_text();backup=D/'main.before.tex'
assert not backup.exists();shutil.copyfile(p,backup)
# Quoted user constraint: 인접 상태 결합의 효과와 단계적 분기 활성화의 효과를 분리.
# Quoted scope: Deep Renewal 원고의 비교에서만 제외하고 학습은 유지.
g={(x['dataset'],x['model']):x for x in j['groups']}
names={'titantpp_history_mlp':'Original MLP','titantpp_current_only_param_matched':'Current-only','titantpp_all_available_history_mlp':'All-available'}
MLP='titantpp_history_mlp';CUR='titantpp_current_only_param_matched';ALL='titantpp_all_available_history_mlp'
metrics=['qty_mae','qty_rmse','time_nll']
full=[]
for d,title,models in [('yellow_trip_hourly','Taxi',[MLP,CUR,ALL]),('intermittent_frozen_5000','Intermittent',[MLP,CUR]),('raf_spare_parts','RAF',[MLP,CUR,ALL])]:
 if full:full.append(r'\midrule')
 full.append(r'\multicolumn{4}{l}{\textit{'+title+r'}}\\')
 for m in models:
  a=g[d,m];assert a['three_seed_complete']
  full.append(names[m]+' & '+' & '.join(f"${a['mean'][k]:.4f}\\pm{a['sample_sd'][k]:.4f}$" for k in metrics)+r'\\')
lookup={(x['dataset'],x['model'],x['seed']):x for x in j['reused_MLP']+j['completed']}
partial=[]
for d,title,models in [('intermittent_frozen_5000','Intermittent',[MLP,ALL]),('insta_market_basket','Instacart',[MLP,CUR,ALL])]:
 for seed in [42,52]:
  if partial:partial.append(r'\midrule')
  partial.append(r'\multicolumn{4}{l}{\textit{'+title+', seed '+str(seed)+r'}}\\')
  for m in models:
   a=lookup[d,m,seed];partial.append(names[m]+' & '+' & '.join(f'{a[k]:.6f}' for k in metrics)+r'\\')
appendix=r'''\section{Completed Structural Comparisons}\label{app:extension}
This appendix reports 21 completed structural conditions at the 2 October
2026, 21:57 KST observation cutoff: six on Taxi, five on Intermittent,
six on RAF, and four on Instacart. Each condition completed training and
selected/last validation re-evaluation; archived checkpoints and frozen
source were verified against the selected epochs and recorded metrics.
Table~\ref{tab:extension} contains the five complete three-seed control
groups alongside the reused original MLP results. The six conditions in
incomplete groups are reported individually in Table~\ref{tab:extension-partial}.

The current-only control removes the explicit predecessor input from the
correction while retaining history in the causal encoder. Its eight
64--6--64 branches match the original correction's 6,144 parameters and
retain the availability thresholds and fixed divisor. The all-available
control retains the original current--predecessor inputs and branch widths,
but activates all eight branches wherever a predecessor exists, still
dividing by eight. It differs from active-count normalization, which changes
the divisor rather than availability.

Taxi and RAF controls ran on RTX 5080, and Instacart controls on RTX 5090.
Intermittent seeds 42 and 52 ran on RTX 5080; the current-only seed-62 fit
ran on RTX PRO 4500 after transfer of its unstarted condition. Its frozen
training and selection rules were retained, and native qualification and
baseline replay passed. The Intermittent current-only aggregate therefore
spans two GPU types, while its original MLP reference ran on RTX 5080.

\begin{table}[t]
\caption{Complete structural-control groups on development validation data,
mean $\pm$ sample standard deviation over seeds 42, 52, and 62. Original MLP
results are reused. Each run contributes MAE, RMSE, and time NLL from the
same RMSE-selected checkpoint. The Intermittent current-only group includes
one RTX PRO 4500 run, as described in the text.}
\label{tab:extension}\centering\small
\begin{tabular}{lrrr}
\toprule
Configuration & MAE & RMSE & Time NLL\\
\midrule
FULL_ROWS
\bottomrule
\end{tabular}
\end{table}

On Taxi, the original MLP improves mean MAE/RMSE over current-only by
11.07\%/11.48\%, with lower quantity errors in every paired seed.
All-available improves mean MAE/RMSE over the original by 1.15\%/1.99\%,
but increases mean time NLL; its MAE and RMSE gains occur in one and two
of the three paired seeds, respectively. These comparisons support
explicit predecessor correction on Taxi and identify a quantity--timing
trade-off between branch-availability rules.

On Intermittent, the original MLP reduces mean MAE/RMSE relative to
current-only by 4.78\%/4.11\%. The quantity gains occur in seeds 42 and 52;
seed 62 favors current-only on all three metrics. Current-only has lower
mean time NLL. Thus the mean quantity benefit is less consistent across
seeds than on Taxi, and the seed-62 comparison also differs in GPU type.

On RAF, the original MLP reduces mean MAE/RMSE by 0.58\%/0.75\% relative
to current-only, with lower RMSE in all three paired seeds. All-available
has nearly the same mean quantity errors as the original: MAE is 0.02\%
higher and RMSE is 0.13\% higher. Its RMSE is nevertheless lower in two
paired seeds. Both controls have lower mean time NLL than the original.
The RAF results indicate a small quantity benefit from the direct
predecessor path and a metric-dependent comparison of availability rules.

\begin{table}[t]
\caption{Completed runs from structural-control groups with seed 62 still
pending at the observation cutoff. Individual development-validation
results are paired with the original MLP at the same seed; no partial-group
mean or standard deviation is reported. Intermittent current-only has a
complete three-seed group and appears in Table~\ref{tab:extension}.}
\label{tab:extension-partial}\centering\small
\begin{tabular}{lrrr}
\toprule
Configuration & MAE & RMSE & Time NLL\\
\midrule
PARTIAL_ROWS
\bottomrule
\end{tabular}
\end{table}

Intermittent's all-available control has higher errors than the original
on all three metrics at seed 42 and lower errors at seed 52. Instacart's
two controls reduce RMSE by 0.05--0.14\% at the completed seeds, with
mixed changes in MAE and time NLL. At this cutoff, the Intermittent
all-available seed-62 fit and the Instacart current-only seed-62 fit were
in progress; Instacart all-available seed 62 was queued. Three-seed
conclusions for these groups remain pending. Table~\ref{tab:main} retains
the completed common-head panel.


'''.replace('FULL_ROWS','\n'.join(full)).replace('PARTIAL_ROWS','\n'.join(partial))
a=s.index(r'\section{Completed Structural Comparisons}');b=s.index(r'\section{Data Definitions and Training Distributions}',a)
s=s[:a]+appendix+s[b:]
s=s.replace('% Appendix A reports 10 structural conditions from the 13-condition audited\n% subset at the 2026-10-02 10:51:58 KST cutoff; the original audit is preserved\n% in reports/titantpp_extension_completed13_audit_20261002_v1.', '% Appendix A reports 21 structural conditions at the 2026-10-02 21:57 KST\n% cutoff, using reports/titantpp_completed22_audit_20261002_v1 and the reused\n% reports/titantpp_extension_completed13_audit_20261002_v1 audit.\n% A100 exploratory candidates and Deep Renewal remain in separate research records.')
s=s.replace("Appendix~\\ref{app:extension} reports the verified completed subset, separating\nTaxi's three-seed groups from single-seed results on other datasets.","Appendix~\\ref{app:extension} reports the verified completed subset, separating\ncomplete three-seed groups from individual runs in incomplete groups.")
old=r'''The completed parameter-matched Taxi comparison further examines the
adjacent-state input (Appendix~\ref{app:extension}). Relative to a current-only
correction with the same parameter count, \method\ reduces mean MAE/RMSE
by 11.07\%/11.48\%, with lower quantity errors in all three paired seeds.
Because the current-only control retains the causal encoder's history,
this comparison supports directly combining adjacent contextual states
beyond the history already encoded in the current state.

Branch availability has a different effect. Activating all eight branches
wherever a predecessor exists reduces Taxi mean MAE/RMSE by 1.15\%/1.99\%
relative to \method, while increasing time NLL. In the completed seed-42
comparisons, the original correction achieves lower errors on all three
metrics than either structural control on Intermittent; Instacart's RMSE
differences slightly favor the controls. These results distinguish the
benefit of adjacent-state coupling on Taxi from the dataset-dependent
trade-offs of its branch-availability rule.'''
new=r'''The completed parameter-matched controls examine the adjacent-state input
(Appendix~\ref{app:extension}). Relative to a current-only correction with
the same parameter count, \method\ reduces mean MAE/RMSE by
11.07\%/11.48\% on Taxi, 4.78\%/4.11\% on Intermittent, and
0.58\%/0.75\% on RAF. RMSE is lower in all three paired Taxi and RAF
seeds, but only two Intermittent seeds. Current-only has lower mean time
NLL on all three datasets; its Intermittent seed-62 run also uses a different
GPU. Because this control retains the causal encoder's history, the
comparison assesses the additional direct predecessor path, with the
largest quantity benefit on Taxi.

Branch availability has a different effect. Activating all eight branches
wherever a predecessor exists reduces Taxi mean MAE/RMSE by 1.15\%/1.99\%
relative to \method, while increasing time NLL. On RAF, mean quantity
errors remain close and time NLL decreases. Intermittent's completed
all-available seeds have opposing rankings, while both controls slightly
reduce Instacart RMSE at seeds 42 and 52. These results distinguish the
benefit of adjacent-state coupling from the dataset-dependent trade-offs
of its branch-availability rule.'''
assert old in s;s=s.replace(old,new)
old=r'''The B comparison changes capacity as well as the correction path. The
completed parameter-matched Taxi controls strengthen the evidence for
adjacent-state coupling, while the all-available alternative achieves lower
mean quantity errors. This separates the correction's empirical benefit
from superiority of the original availability thresholds. The effects of
bottleneck rank, insertion position, and initialization remain unresolved.
Appendix~\ref{app:extension} distinguishes the extension's completed
three-seed groups from its seed-42 results; the other groups remain in
progress or queued at the stated cutoff. Likewise, the static-retrieval
removal comparison available for a different internal variant does not
isolate that component's necessity in the representative architecture.'''
new=r'''The B comparison changes capacity as well as the correction path.
Parameter-matched controls provide the strongest evidence for direct
adjacent-state coupling on Taxi, a less consistent quantity benefit on
Intermittent, and a small benefit on RAF. The all-available alternative
achieves lower mean quantity errors on Taxi, so these results do not
establish superiority of the original availability thresholds. The effects
of bottleneck rank, insertion position, and initialization remain unresolved.
Appendix~\ref{app:extension} separates completed three-seed groups from
individual runs in incomplete groups and records the transferred
Intermittent run's GPU provenance. Likewise, the static-retrieval removal
comparison available for a different internal variant does not isolate
that component's necessity in the representative architecture.'''
assert old in s;s=s.replace(old,new)
# Reuse Appendix B/C and the heatmap; update only two prose links to the new control evidence.
s=s.replace(r'''(Fig.~\ref{fig:train-lag-correlation}). Together with the completed Taxi
structural controls, this contrast motivates the hypothesis that directly''',r'''(Fig.~\ref{fig:train-lag-correlation}). Together with the stronger quantity
benefits of direct predecessor input on Taxi and Intermittent than on RAF,
this contrast motivates the hypothesis that directly''')
s=s.replace(r'''carry substantial information within a series. The completed Taxi
parameter-matched control supports that architectural interpretation
(Appendix~\ref{app:extension}). Data characteristics alone, however, do''',r'''carry substantial information within a series. The completed
parameter-matched controls provide stronger quantity gains on Taxi and
Intermittent than on RAF, although the Intermittent ranking varies by seed
(Appendix~\ref{app:extension}). Data characteristics alone, however, do''')
assert 'Deep Renewal &' not in s and '10 completed structural' not in s and 'tab:extension-seed42' not in s
p.write_text(s)
(D/'manuscript.diff').write_text(''.join(difflib.unified_diff(backup.read_text().splitlines(True),s.splitlines(True),fromfile=str(backup),tofile=str(p))))
(D/'generated_tables.json').write_text(json.dumps({'full_rows':full,'partial_rows':partial,'comparison_sha256':hashlib.sha256((D/'comparison.json').read_bytes()).hexdigest()},indent=2)+'\n')
print('Updated existing main.tex: 21 structural conditions, 5 complete groups; pending seeds preserved.')
