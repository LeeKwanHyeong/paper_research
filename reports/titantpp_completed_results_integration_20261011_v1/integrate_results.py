import pathlib, json, hashlib, statistics, math, re, difflib, shutil

ROOT = pathlib.Path('/Users/igwanhyeong/PycharmProjects/paper_research')
TARGET = ROOT/'paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1'
OUT = ROOT/'reports/titantpp_completed_results_integration_20261011_v1'
HB = ROOT/'search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1'
IB = ROOT/'search_artifacts/titantpp_instacart_completion_recovery_20261010_v1'

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def verified(p, digest):
    assert sha(p)==digest, str(p)
    return json.loads(pathlib.Path(p).read_text())

h = verified(HB/'analysis/selected_conditions.json', '41ddf10b5379b2ba647e1c8500b7e02e2214bfa127e797d0ee293c1050668742')
hc = verified(HB/'analysis/completion_receipt.json', '656eef38d3192d57f89e86aaa2ed65e69ae5c256288668c1819db7e838c48362')
ic = verified(IB/'analysis/completion_receipt.json', '459ae9f15f6f4247584a8d47bd8e3901ab6e7ba91d6a98df3f64b09dde95cea6')
i = verified('/private/tmp/instacart_manuscript_evidence_20261011.json', '1a29e935601183797eebfb0167682549dfd5f1345c2c61ac367e0f13aad52c98')
assert h['evaluation_scope']=='validation_only' and h['held_out_test_evaluated'] is False
assert hc['distinct_execution_ids']==27 and hc['status']=='complete'
assert i['condition_count']==9 and i['evaluation_scope']=='validation_only'
for entry in i['source_evidence'].values(): assert sha(entry['path'])==entry['rawSHA']
before = (OUT/'before/main.tex').read_text()
assert (TARGET/'main.tex').read_text()==before, 'Concurrent manuscript change; preserve it and reconcile'
s = before
edits=[]

def replace(old, new, name):
    global s
    assert s.count(old)==1, (name, s.count(old))
    s=s.replace(old,new,1); edits.append(name)

def replace_span(start, end, new, name):
    global s
    assert s.count(start)==1 and s.count(end)==1, name
    a=s.index(start); b=s.index(end,a)
    s=s[:a]+new+s[b:]; edits.append(name)

arm_order=['titantpp_cnn_gru54','titantpp_cnn_gru54_correction_disabled','titantpp_cnn_gru54_current_only']
ilabel={'titantpp_history_mlp_width8':'MLP8','titantpp_history_mlp_width12':'MLP12','titantpp_cnn_gru54':'CNN+GRU54'}
hlabel=dict(zip(arm_order,['Full','Correction bypass','Current-only']))
datasets=[('yellow_trip_hourly','Taxi'),('intermittent_frozen_5000','Intermittent'),('raf_spare_parts','RAF')]
hs={(x['dataset'],x['arm']):x['metrics'] for x in h['seed_summary']}
metric_order=['qty_mae','qty_rmse','time_nll']
table_cells=[]

def pm(v, bold=False):
    m, sd=v['mean'],v['sample_sd']
    assert math.isfinite(m) and math.isfinite(sd)
    if 0<abs(m)<.001 or 0<abs(sd)<.00001:
        def sci(x):
            mant,exp=f'{x:.3e}'.split('e')
            return mant+r'\!\times\!10^{'+str(int(exp))+'}'
        text=r'\begin{gathered}'+sci(m)+r'\\{\scriptstyle\pm\,'+sci(sd)+r'}\end{gathered}'
    else:
        text=f'{m:.4f}'+r'\pm'+f'{sd:.4f}'
    return '$'+(r'\mathbf{'+text+'}' if bold else text)+'$'

def metric_cell(metrics,key, source, bold=False):
    text=pm(metrics[key],bold)
    table_cells.append({'source':source,'metric':key,'mean':metrics[key]['mean'],'sample_sd':metrics[key]['sample_sd'],'latex':text})
    return text

def hist_main():
    lines=[r'\begin{table}[t]',r'\caption{Completed CNN+GRU54 correction study. Entries are mean $\pm$ sample SD over seeds 42, 52, and 62 at each fit\textquotesingle s earliest minimum overall quantity-RMSE checkpoint. These full-model fits form a separate campaign from Table~\ref{tab:external-cnngru}. Bold marks the lowest mean within a dataset and column. RAF seed 42 includes physical A100 migration.}',r'\label{tab:history-cnngru}',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Arm & Quantity MAE & Quantity RMSE & Time NLL\\',r'\midrule']
    for n,(ds,label) in enumerate(datasets):
        if n: lines.append(r'\midrule')
        lines.append(r'\multicolumn{4}{l}{\textit{'+label+r'}}\\')
        for arm in arm_order:
            metrics=hs[(ds,arm)]
            cells=[metric_cell(metrics,'overall_'+m,ds+'/'+arm,min(hs[(ds,a)]['overall_'+m]['mean'] for a in arm_order)==metrics['overall_'+m]['mean']) for m in metric_order]
            lines.append(hlabel[arm]+' & '+' & '.join(cells)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(lines)

def insta_main():
    lines=[r'\begin{table}[t]',r'\caption{Completed Instacart architecture comparison on 503,733 validation targets. Each architecture contributes three seeds and all columns inherit its quantity-RMSE-selected checkpoint. Entries are mean $\pm$ sample SD. Bold marks the lowest mean in each column. These variants have different parameter counts.}',r'\label{tab:instacart-architecture}',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Architecture & Quantity MAE & Quantity RMSE & Time NLL\\',r'\midrule']
    for arm,label in ilabel.items():
        metrics=i['seed_summary'][arm]
        cells=[metric_cell(metrics,'overall_'+m,'Instacart/'+arm,min(i['seed_summary'][a]['overall_'+m]['mean'] for a in ilabel)==metrics['overall_'+m]['mean']) for m in metric_order]
        lines.append(label+' & '+' & '.join(cells)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(lines)

replace_span(r'\begin{abstract}',r'\keywords{',r'''\begin{abstract}
Event forecasting requires predicting how much will occur as well as when it will occur. We examine an explicit history path within a causal attention model. TitanTPP's representative CNN+GRU54 architecture corrects attention queries, keys, and values with three observed positions, then adds a recurrent residual between attention blocks. In completed three-seed validation comparisons, CNN+GRU54 attains the lowest mean quantity MAE and RMSE among six common-head encoder adaptations on Taxi and Intermittent, while RAF favors S2P2 by RMSE. A separate 27-condition study compares full correction with bypass and an equal-active-parameter current-only control. Full correction lowers overall quantity RMSE against both controls in every seed on Taxi and Intermittent, but fails this consistency criterion on RAF. Nine Instacart fits yield smaller mean quantity errors for CNN+GRU54 than for MLP8 and MLP12, with a seed-level exception and worse mean overall timing NLL. Large-quantity and timing losses qualify the correction gains. Prototype-retrieval controls remain pending, and these development-validation results do not establish independent generalization.
''','abstract_completed_scope')

replace('Removing the history path also removes parameters, so that comparison alone cannot identify the benefit of its input.','Bypassing the correction removes its active trainable capacity, so that comparison alone cannot identify the benefit of explicit history input.','disabled_active_capacity_intro')
replace(r'\item We formulate controlled comparisons for explicit history input and prototype retrieval, separating these questions from extra parameter count and from the choice of output distribution.',r'\item We evaluate explicit history correction against bypass and a current-only control with the same active parameter count, while retaining one quantity-based selection rule and reporting timing and large-quantity tradeoffs.','contribution_correction_completed')
replace(r'\item We report completed CNN+GRU54 comparisons with six common-head encoder adaptations across three datasets and seeds. Quantity gains on Taxi and Intermittent coexist with timing costs and a different ranking on RAF. Earlier MLP4 and MLP8 findings provide supporting evidence in the appendices.',r'\item We report completed CNN+GRU54 comparisons with six common-head encoder adaptations across three datasets and seeds, followed by a three-seed Instacart architecture comparison. Quantity rankings and timing costs depend on the dataset. Earlier MLP4 and MLP8 findings remain supporting evidence in the appendices.','contribution_instacart_completed')
replace('The completed representative-model comparison establishes its performance under the common evaluation contract. The correction and retrieval controls address attribution separately and remain under evaluation, as does the supplementary three-seed Instacart comparison.','The external-encoder panel describes performance under a common evaluation contract, and the completed correction controls test the combined local and recurrent path separately. Instacart provides a supplementary architecture comparison. Matched prototype-retrieval controls remain a planned development study.','introduction_status')
replace('The correction-removal variant bypasses both optional paths.','The correction-removal variant bypasses both optional paths. Its correction modules remain registered but frozen, contributing zero active trainable correction parameters.','methods_disabled_registered')
replace(r'\caption{The matched retrieval comparison. Both rules select four of 64 prototypes and use softmax weights.',r'\caption{The planned matched retrieval comparison. Both rules select four of 64 prototypes and use softmax weights.','retrieval_figure_planned')

replace_span(r'\paragraph{Correction and retrieval controls.}',r'\paragraph{Decision criteria and computational cost.}',r'''\paragraph{Completed correction and architecture studies.}
The correction study compares full CNN+GRU54 correction, correction bypass, and the current-only control on each core dataset and seed. Its 27 distinct conditions are complete. Fifteen conditions completed on A100, three RAF seed-62 conditions on RTX 5080, and nine Intermittent conditions on RTX 5090. The RAF seed-42 full and current-only fits resumed across two physical A100 GPUs after an operational interruption, whereas correction bypass had already completed. Frozen initialization, inputs, loss, optimizer, and checkpoint selection were retained. This migration limits hardware matching for that seed. Appendix~\ref{app:history-completed} records the selected and final saved epochs.

Within each RTX dataset--seed group, the three arms shared one GPU and Runtime. Seeds nevertheless span hardware and software environments, including cuDNN versions 91900 on RTX 5080 and 92000 on RTX 5090. The nine full-correction fits in this study are kept separate from the nine representative-model fits in Table~\ref{tab:external-cnngru}; their means are neither pooled nor substituted between panels.

The supplementary Instacart study also completed nine conditions, comprising MLP8, MLP12, and CNN+GRU54 for seeds 42, 52, and 62. Seeds 42 and 62 ran on RTX 5080, and seed 52 on RTX 5090, with all three architectures sharing one GPU within each seed. The architectures contain 89,859, 96,003, and 102,879 trainable parameters, respectively. Their frozen inputs, common heads, objective, and selection rule agree, but their correction modules and parameter budgets differ. This is an architecture comparison rather than an equal-budget correction ablation. Appendix~\ref{app:instacart-completed} retains all six metrics and checkpoint records.

\paragraph{Pending retrieval comparison.}
The retrieval design compares mean, weighted, fixed, and general-query variants. Fixed and general query each add 512 parameters, whereas the earlier exploratory pilot's general-query control contained 521. A preliminary implementation check on RTX 5080 established 83 bitwise agreements for initialization, synthetic sequences, and the first three updates on a shared Taxi training batch. Its scope does not extend to complete training trajectories.

The completed correction and Instacart studies account for 36 conditions. Retrieval remains pending a common-GPU mean baseline and a new resource contract. The completed A100 full runs are not treated as RTX-matched mean baselines. The additional fit count depends on that baseline contract, so the earlier 63-fit plan does not describe a completed or currently executing campaign.

''','experimental_design_actual_execution')

replace('This completed comparison does not yet include a three-seed CNN+GRU54 result for Instacart.','The supplementary Instacart architecture comparison appears separately in Section~\\ref{sec:instacart-results}, without extending this external-encoder panel to a new campaign.','external_panel_scope_preserved')

history_prose=r'''\subsection{Additional effect of explicit history correction}
\label{sec:history-results}
Table~\ref{tab:history-cnngru} reports the completed 27-condition correction study. Against both correction bypass and current-only, full correction lowers overall quantity RMSE in all three seeds on Taxi and Intermittent. These two datasets meet the predeclared consistency criterion. Although full correction also attains lower mean RMSE on RAF, seed 52 favors both controls: full RMSE is 34.4012, compared with 34.3415 for bypass and 34.3349 for current-only. RAF therefore fails the seedwise criterion. Across the 18 full-versus-control comparisons, 16 favor full correction by overall RMSE.

'''+hist_main()+r'''

The quantity advantage carries costs. Full correction raises overall time NLL in 11 of the 18 comparisons, including all six Taxi comparisons. Large-quantity RMSE and MAE each worsen in ten comparisons. On Intermittent, for example, full correction attains mean overall RMSE 1.6957 versus 1.7409 for bypass, but large-quantity RMSE rises to 8.2968 versus 7.8753. On RAF, the corresponding large-quantity means are 327.6237 and 324.7870. These metrics inherit the same quantity-selected checkpoint. Appendix~\ref{app:history-completed} retains all six metrics and the adverse seedwise differences rather than selecting a separate favorable timing or tail epoch.

The results support the tested combined correction path within development validation. Both full and current-only contain 25,308 active correction parameters, yet their input operations and recurrent depth differ. Because the current contextual state already includes prefix history, this control does not remove history from the entire model. The comparison also does not identify the contribution of the CNN or GRU separately. RAF seed 42's physical GPU migration and cross-seed Runtime differences limit hardware matching. Concurrent training durations do not establish structural efficiency.

The earlier MLP8 study remains a separate observation in Appendix~\ref{app:metrics}. Its 18 correction gains concern that MLP path and are not substituted for the CNN+GRU54 controls.

\subsection{Supplementary Instacart architecture comparison}
\label{sec:instacart-results}
The nine completed Instacart fits extend the architecture comparison to basket quantities on 503,733 validation targets. CNN+GRU54 attains the lowest mean overall quantity RMSE and MAE among the three variants in Table~\ref{tab:instacart-architecture}. Its mean RMSE is 5.8774, compared with 5.8854 for MLP8 and 5.8801 for MLP12. The gaps are small relative to the reported seed dispersions. For seed 52, MLP12 attains lower overall and large-quantity RMSE and MAE than CNN+GRU54. Each quantity metric consequently favors CNN+GRU54 in five of the six seedwise comparisons, rather than in every comparison.

'''+insta_main()+r'''

For the 6,036 targets with quantity above 35, CNN+GRU54 attains mean RMSE 24.4380 and MAE 21.7822, compared with RMSE 24.6355 and 24.6683 for MLP8 and MLP12. Its large-quantity time NLL is lower in all six seedwise comparisons. Mean overall time NLL, however, rises to 2.8108 from 2.8075 and 2.8072. The seed-62 CNN+GRU54 fit incurs higher overall time NLL than both MLP variants. Appendix~\ref{app:instacart-completed} reports both populations at the same selected checkpoints.

Instacart therefore supplies a modest architecture-level quantity advantage with explicit exceptions. Different parameter counts prevent attribution to an equal-budget history intervention, and this study does not test prototype retrieval. Its results are not merged with the earlier MLP4 external comparison in Appendix~\ref{app:legacy-external}.

'''
replace_span(r'\subsection{Additional effect of explicit history correction}',r'\subsection{Additional effect of prototype retrieval}',history_prose,'results_history_instacart')
replace('The matched CNN+GRU54 retrieval comparison is also under evaluation.','The matched CNN+GRU54 retrieval comparison remains pending a common-GPU baseline and resource contract.','results_retrieval_waiting')

replace_span(r'\section{Discussion}',r'\begin{thebibliography}{28}',r'''\section{Discussion}
The completed CNN+GRU54 results support two bounded findings. In the external-encoder panel, its mean quantity errors are lower than those of the six tested common-head adaptations on Taxi and Intermittent, while RAF favors S2P2 by mean RMSE. The separate correction study favors the full path against both controls in every seed on Taxi and Intermittent, with a seed-level failure on RAF. Retaining one representative architecture across datasets avoids a different best-variant choice for each result.

This correction finding concerns the combined local and recurrent computation. Equal active parameter counts narrow a capacity explanation, but current-only changes the nonlinear bases and repeated-input updates. Its current hidden state also contains prefix history. The controls consequently distinguish the tested computational paths without identifying a unique lag or separating CNN and GRU effects. Earlier MLP8 results provide context rather than the mechanism evidence for CNN+GRU54.

Instacart supplies a smaller quantity difference. CNN+GRU54's lower mean errors coexist with a seed-52 MLP12 advantage and higher mean overall timing NLL. These variants differ in parameter count, and their limited mean gaps do not establish a statistically significant architectural advantage. Training-history length and event-lag associations in the appendices describe data characteristics rather than causes of the observed model ranking.

Quantity gains do not imply uniformly better forecasting. The correction study includes 11 overall timing losses and ten losses each for large-quantity RMSE and MAE among 18 comparisons. All scores come from quantity-selected checkpoints. Reporting these exceptions prevents the primary quantity criterion from concealing timing and rare-target costs.

The retrieval hypothesis remains distinct. Fixed split-query scoring can be expressed as a concatenated dot product, and a future advantage over the budget-matched general query would support the tested separate-normalization constraint. No completed CNN+GRU54 retrieval effect is established here. A100 full fits also cannot substitute for a RTX common-GPU mean baseline.

The external panel and correction study combine preserved and separately trained runs across hardware. RAF seed 42's correction trajectories include physical A100 migration, and seed summaries span Runtime differences. Task and selection agreement permit descriptive comparisons but do not establish hardware efficiency. All reported performance derives from validation evidence already encountered during research. CPU binary re-inference audits and an independently unexposed evaluation remain outstanding, and three-seed summaries do not establish statistical significance.

\section{Conclusion}
TitanTPP augments a causal attention encoder with local and recurrent history correction. Completed CNN+GRU54 runs attain lower mean quantity errors than six common-head encoder adaptations on Taxi and Intermittent, while S2P2 attains lower mean RMSE on RAF. The completed correction study supports the full path against bypass and an equal-active-parameter current-only control in every seed on Taxi and Intermittent. RAF fails the same criterion, and timing and large-quantity losses remain substantial qualifications. Instacart supplies a smaller mean quantity advantage over MLP8 and MLP12 with a seed-level exception and worse mean overall timing NLL. Matched retrieval and independent evaluation remain pending. These results support a bounded development-validation claim for the combined correction path.

''','discussion_conclusion_completed_claims')

replace('These are distinct from the fresh full-correction fits in the subsequent contribution study.','These representative-model records remain separate from the full-correction campaign summarized in Appendix~\\ref{app:history-completed}.','appendix_baseline_distinct')
replace_span(r'\subsection{Archived Instacart observation}',r'\subsection{Full validation metrics}',r'''The completed Instacart architecture comparison is reported in Section~\ref{sec:instacart-results} and Appendix~\ref{app:instacart-completed}. It supersedes the earlier single-fit monitoring observation without replacing the core-dataset MLP8 study preserved here.

''','remove_obsolete_partial_instacart')

def six_table(records, label, caption, source_prefix):
    lines=[r'\begin{table}[htbp]',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.12}',r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Arm & Population & Quantity MAE & Quantity RMSE & Time NLL\\',r'\midrule']
    for n,(arm,name,metrics) in enumerate(records):
        if n: lines.append(r'\midrule')
        for pop,prefix in [('All','overall_'),('Large','tail_')]:
            cells=[metric_cell(metrics,prefix+m,source_prefix+'/'+arm) for m in metric_order]
            lines.append((name if pop=='All' else '')+' & '+pop+' & '+' & '.join(cells)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(lines)

appendix=[r'\clearpage',r'\section{Completed CNN+GRU54 correction records}',r'\label{app:history-completed}',
 r'This appendix records the separate 27-condition correction campaign. Full, correction bypass, and current-only each contribute seeds 42, 52, and 62 on every core dataset. Selection follows the earliest strictly improving finite minimum of full-validation raw-quantity RMSE. The large subsets contain 79 Taxi, 1,141 Intermittent, and 50 RAF targets, with identical target membership across arms and seeds. Neither time NLL nor large-quantity error determines an additional checkpoint.', '',
 r'\begin{table}[htbp]',r'\caption{Selected and final saved epochs for the correction study. F denotes full, B denotes correction bypass, and C denotes current-only. Each entry is selected/final. A100 denotes the preserved completed campaign records, not a common RTX cost baseline. RAF seed 42 full/current-only resumed across physical A100 GPUs.}',r'\label{tab:history-epochs}',r'\centering\small',r'\begin{tabular}{@{}lrcccl@{}}',r'\toprule',r'Dataset & Seed & F & B & C & Execution\\',r'\midrule']
hcmap={(x['dataset'],x['seed'],x['arm']):x for x in h['conditions']}
for n,(ds,label) in enumerate(datasets):
    if n: appendix.append(r'\midrule')
    for seed in [42,52,62]:
        cs=[hcmap[(ds,seed,a)] for a in arm_order]
        vals=[f"{c['best_epoch']}/{c['last_epoch']}" for c in cs]
        host='RTX 5090' if ds=='intermittent_frozen_5000' else ('RTX 5080' if ds=='raf_spare_parts' and seed==62 else 'A100')
        if ds=='raf_spare_parts' and seed==42: host=r'A100 $\to$ A100 (F/C)'
        appendix.append(f'{label if seed==42 else ""} & {seed} & '+ ' & '.join(vals)+' & '+host+r'\\')
appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
 r'Full and current-only each contain 25,308 active correction parameters. Bypass retains the registered modules but freezes them and does not execute either correction path. The recurrent output projection and local kernels are zero-initialized under the frozen source contract. Fifteen conditions completed on A100 and twelve on RTX. Operational interruptions and checkpoint transport do not create additional scientific conditions. RAF seed 42 full/current-only retained the saved state and selection history through migration. Hardware-segment timings remain separate and concurrent fit wall time is not a structural efficiency estimate.', '']
for ds,label in datasets:
    rec=[(a,hlabel[a],hs[(ds,a)]) for a in arm_order]
    appendix += [six_table(rec,'tab:history-six-'+ds,'Completed correction study on '+label+r'. Mean $\pm$ sample SD over three seeds at the same selected checkpoint for all and large-quantity populations. These campaign values do not replace Table~\ref{tab:external-cnngru}.','history/'+ds),'']
appendix += [r'\clearpage',r'\subsection{Adverse paired differences}',
 r'Table~\ref{tab:history-adverse} records how often full correction has a higher metric than each control. Counts refer to seedwise paired differences, not differences between the rounded means. A timing or large-quantity loss remains part of the result even when overall RMSE improves.', '',
 r'\begin{table}[htbp]',r'\caption{Number of adverse differences (full minus control $>0$) in the completed correction study. Each control contributes three seed comparisons per dataset.}',r'\label{tab:history-adverse}',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',r' & & \multicolumn{3}{c}{All} & \multicolumn{3}{c}{Large}\\',r'\cmidrule(lr){3-5}\cmidrule(l){6-8}',r'Dataset & Control & MAE & RMSE & NLL & MAE & RMSE & NLL\\',r'\midrule']
adverse=[]
for nd,(ds,label) in enumerate(datasets):
    if nd: appendix.append(r'\midrule')
    for a in arm_order[1:]:
        counts=[]
        for prefix in ['overall_','tail_']:
            for metric in metric_order:
                values=[]
                for seed in [42,52,62]:
                    full=hcmap[(ds,seed,arm_order[0])]['selected']; ctrl=hcmap[(ds,seed,a)]['selected']
                    delta=(full['tail'][metric]-ctrl['tail'][metric]) if prefix=='tail_' else (full[metric]-ctrl[metric])
                    values.append(delta)
                counts.append(sum(v>0 for v in values))
                adverse.append({'dataset':ds,'control':a,'metric':prefix+metric,'seed_order':[42,52,62],'deltas':values,'higher_full_count':counts[-1]})
        appendix.append(label+' & '+('Bypass' if a==arm_order[1] else 'Current-only')+' & '+' & '.join(map(str,counts))+r'\\')
appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
 r'These counts retain the RAF seed-52 RMSE failure against both controls, all six Taxi overall time-NLL losses, and large-quantity losses despite some overall quantity gains. The full six-metric records and first-minimum checkpoint histories are retained in the completed Validation ledger. The criterion in Section~4 is a development consistency rule, not a significance or independent-confirmation test.', '',
 r'\clearpage',r'\section{Completed Instacart architecture records}',r'\label{app:instacart-completed}',
 r'The nine fits compare MLP8, MLP12, and CNN+GRU54 over seeds 42, 52, and 62. The evaluation population contains 503,733 targets and 6,036 large-quantity targets with $q>35$. Each seed repeats the same target population. Recorded intervals are measured in days, top-coded at 30, and scored with time scale seven. These are recorded integer-bin NLLs, not the older clamped time-loss field. Selection and all six metrics inherit the same full-validation quantity-RMSE checkpoint.', '',
 r'\begin{table}[htbp]',r'\caption{Selected and final saved epochs for the completed Instacart architecture study. All three architectures share one GPU within each seed. Cross-seed Runtime differences and unequal parameter budgets remain.}',r'\label{tab:instacart-epochs}',r'\centering\small',r'\begin{tabular}{@{}lrcrl@{}}',r'\toprule',r'Architecture & Seed & Selected & Final & Execution\\',r'\midrule']
for n,(arm,label) in enumerate(ilabel.items()):
    if n: appendix.append(r'\midrule')
    for x in sorted((v for v in i['rows'] if v['arm']==arm),key=lambda v:v['seed']):
        appendix.append(f"{label if x['seed']==42 else ''} & {x['seed']} & {x['selected_epoch']} & {x['last_epoch']} & RTX {x['GPU_host']}"+r'\\')
appendix += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
 six_table([(a,l,i['seed_summary'][a]) for a,l in ilabel.items()],'tab:instacart-six',r'Completed Instacart six-metric comparison. Entries are mean $\pm$ sample SD over seeds 42, 52, and 62. All and large-quantity rows share the same selected checkpoint within each fit.','Instacart'),' ',
 r'CNN+GRU54 has lower overall and large-quantity RMSE and MAE than MLP8 in each seed. Against MLP12, seed 52 instead favors MLP12 on all four quantity metrics. CNN+GRU54 lowers large-quantity time NLL in every paired comparison, but seed 62 raises overall time NLL against both MLP variants. These exceptions remain in the aggregate and do not support a uniform architecture ranking.', '',
 r'The frozen Instacart science revision differs from the correction-campaign revision, but its 126 source files match the corresponding files in the correction study\textquotesingle s 129-file closure, including the CNN+GRU and common-head implementations. Native checks record 89,859, 96,003, and 102,879 trainable parameters for MLP8, MLP12, and CNN+GRU54. Binary checkpoint bytes and complete endpoint records were collected and SHA-verified. No CPU binary re-inference or independent evaluation is inferred from that integrity check.', '']
replace(r'\end{document}','\n'.join(appendix)+r'\end{document}','appendices_complete_records')

# Exact preservation checks for established panels and definitions.
def table_with_label(text,label):
    for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',text,re.S):
        if r'\label{'+label+'}' in m.group(): return m.group()
    raise AssertionError(label)
for label in ['tab:external-cnngru','tab:cnngru-epochs','tab:cnngru-six','tab:rmse','tab:allmetrics','tab:data','tab:train-characteristics']:
    assert table_with_label(s,label)==table_with_label(before,label), label
method_before=before[before.index(r'\section{Method}'):before.index(r'\section{Experimental Design}')]
method_after=s[s.index(r'\section{Method}'):s.index(r'\section{Experimental Design}')]
assert method_after.replace(' Its correction modules remain registered but frozen, contributing zero active trainable correction parameters.','').replace('The planned matched retrieval comparison.','The matched retrieval comparison.')==method_before
assert s[s.index(r'\begin{thebibliography}'):s.index(r'\end{thebibliography}')]==before[before.index(r'\begin{thebibliography}'):before.index(r'\end{thebibliography}')]
assert s[s.index(r'\section{Data definitions and training distributions}'):s.index('\\clearpage\n\\section{Completed CNN+GRU54 correction records}')].strip()==before[before.index(r'\section{Data definitions and training distributions}'):before.index(r'\end{document}')].strip()

(TARGET/'main.tex').write_text(s)
(OUT/'main_tex.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='before/main.tex',tofile='paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex')))
for name, source in [('instacart_evidence.json','/private/tmp/instacart_manuscript_evidence_20261011.json'),('methods_alignment.md','/private/tmp/cnn_gru54_methods_alignment_20261011.md')]: shutil.copy2(source, OUT/name)
hist_audit=ROOT/'reports/titantpp_history_rtx_handoff_20261010_v1/FINAL_VALIDATION_AUDIT_20261011.json'
assert sha(hist_audit)=='849a1986577cf87aa896650bc547ab1f906c6655e34e1d8e9fef9df63839e397'
evidence={'evaluation_scope':'validation_only','source_files':{str(HB/'analysis/selected_conditions.json'):sha(HB/'analysis/selected_conditions.json'),str(HB/'analysis/completion_receipt.json'):sha(HB/'analysis/completion_receipt.json'),str(hist_audit):sha(hist_audit)},'Instacart_source_evidence':i['source_evidence'],'counts':{'history':27,'Instacart':9},'table_cells':table_cells,'history_adverse_deltas':adverse,'history_decision':h['decision'],'separate_external_panel_preserved':True,'historical_MLP_numeric_tables_preserved':True,'method_equations_algorithm_preserved':True,'bibliography_preserved':True,'train_and_strata_appendices_preserved':True,'CPU_binary_reinference_audit':False,'held_out_test_evaluated':False,'new_independent_evaluation':False,'operations':{'training':0,'inference':0,'SSH':0,'automation_changes':0},'changes':edits,'before_sha256':sha(OUT/'before/main.tex'),'after_sha256':sha(TARGET/'main.tex')}
(OUT/'integration_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'edited','target':str(TARGET/'main.tex'),'changes':edits,'new_metric_cells':len(table_cells),'sha256':sha(TARGET/'main.tex'),'protected_tables_and_method_equations':'PASS'},ensure_ascii=False))
