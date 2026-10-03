"""Generate paper tables and scientific figures from reconciled saved evidence."""
from pathlib import Path
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DRAFT = ROOT / 'paper/titantpp_pakdd_2027_draft/main.tex'
FIG = OUT / 'figures'
DATA = ['yellow_trip_hourly', 'intermittent_frozen_5000', 'raf_spare_parts', 'insta_market_basket']
NAMES = dict(zip(DATA, ['Taxi', 'Intermittent', 'RAF', 'Instacart']))
ANCHORS = dict(zip(DATA, ['RMTPP', 'RMTPP', 'S2P2', 'S2P2']))


def read_csv(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


P = {r['dataset']: r for r in read_csv('train_profiles.csv')}
R = read_csv('anchor_strata.csv')
A = json.loads((OUT / 'analysis.json').read_text())
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'axes.labelcolor': '#263746', 'xtick.color': '#263746', 'ytick.color': '#263746'})
BLUE, RED = '#28658a', '#b5553f'


def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.png'), dpi=210, bbox_inches='tight')
    plt.close(fig)


def figures():
    fig, ax = plt.subplots(figsize=(7.4, 3.0), layout='constrained')
    for j, ds in enumerate(DATA):
        y = 3 - j; p = P[ds]
        vals = [float(p['history_' + k]) for k in ['p05', 'p25', 'p50', 'p75', 'p95']]
        lo, q1, med, q3, hi = vals
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.4)
        ax.plot([lo, lo], [y-.08, y+.08], color=BLUE, lw=1.2)
        ax.plot([hi, hi], [y-.08, y+.08], color=BLUE, lw=1.2)
        ax.plot([q1, q3], [y, y], color=BLUE, lw=7, solid_capstyle='butt')
        ax.scatter([med], [y], color='white', edgecolor=BLUE, zorder=4, s=38)
        ax.annotate(f'median {med:g}', (med, y), xytext=(6, 10),
                    textcoords='offset points', color=BLUE, fontsize=8)
    ax.set_yticks(range(4), [NAMES[ds] for ds in DATA[::-1]])
    ax.set_xlim(0, 192); ax.set_ylim(-.5, 3.55)
    ax.set_xticks(range(0, 193, 32)); ax.set_xlabel('Usable observed events per train target')
    ax.set_axisbelow(True); ax.grid(axis='x', color='#dddddd', lw=.6)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    handles = [Line2D([0], [0], color=BLUE, lw=1.4, label='5th-95th percentiles'),
               Line2D([0], [0], color=BLUE, lw=7, label='25th-75th percentiles'),
               Line2D([0], [0], marker='o', color=BLUE, markerfacecolor='white', lw=0, label='Median')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.32), ncol=3,
              frameon=False, fontsize=8)
    save(fig, 'train_history_distribution')

    rows = [r for r in R if r['dataset'] == DATA[-1] and r['partition'] == 'quantity']
    labels = ['q ≤ 8', '8 < q ≤ 20', '20 < q ≤ 25', '25 < q ≤ 35', 'q > 35', 'Overall']
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), layout='constrained')
    for ax, field, label, xlim in zip(axes, ['mae_contribution', 'mse_contribution'],
            ['Contribution to ΔMAE (quantity units)', 'Contribution to ΔMSE (squared units)'],
            [(-.09, .075), (-1.25, 1.1)]):
        values = [float(r[field]) for r in rows]
        values.append(sum(values))
        ax.barh(range(6), values, height=.6, color=[BLUE if v < 0 else RED for v in values[:5]] + ['#263746'])
        ax.set_yticks(range(6), labels); ax.invert_yaxis(); ax.set_xlim(*xlim)
        ax.axvline(0, color='#263746', lw=.9); ax.axhline(4.5, color='#bbbbbb', lw=.6)
        ax.set_xlabel(label); ax.set_axisbelow(True); ax.grid(axis='x', color='#dddddd', lw=.6)
        ax.set_xticks([-.08, -.04, 0, .04] if field == 'mae_contribution' else [-1, -.5, 0, .5, 1])
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
        ax.set_title(f'Overall: {values[-1]:+.4f}', fontsize=10, loc='left')
    fig.supxlabel('TitanTPP − S2P2; negative values favor TitanTPP', fontsize=9)
    save(fig, 'instacart_error_contributions')


def tex_interval(r):
    s = r['interval']
    variable = 'q' if r['partition'] == 'quantity' else 'h'
    if s.startswith('<='): return f'${variable}\\le {s[2:]}$'
    if s.startswith('>'): return f'${variable}> {s[1:]}$'
    lo, hi = s.strip('()]').split(',')
    return f'${lo}<{variable}\\le {hi}$'


def stratum_table(partition):
    label = 'quantity' if partition == 'quantity' else 'history'
    caption = ('Quantity-stratum validation errors at the selected checkpoints. '
        if partition == 'quantity' else 'Usable-history-stratum validation errors at the selected checkpoints. ')
    caption += ('Entries are arithmetic means over seeds 42, 52, and 62; $N$ is the number '
        'of targets per seed. The external comparator is fixed per dataset by the overall '
        'RMSE in Table~\\ref{tab:main}, not selected within each bin. '
        'Empty strata have no estimated error.')
    lines = [r'\begin{table}[p]', '\\caption{' + caption + '}',
             '\\label{tab:strata-' + label + '}\\centering\\scriptsize',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Stratum & $N$ & \multicolumn{2}{c}{TitanTPP} & \multicolumn{2}{c}{External comparator}\\',
             r' & & MAE & RMSE & MAE & RMSE\\', r'\midrule']
    for ds in DATA:
        lines.append('\\multicolumn{6}{l}{\\textit{' + NAMES[ds] + ' vs. ' + ANCHORS[ds] + '}}\\\\')
        for r in R:
            if r['dataset'] != ds or r['partition'] != partition: continue
            vals = [f'{float(r[k]):.4f}' if r[k] else '--' for k in ['mlp_mae','mlp_rmse','anchor_mae','anchor_rmse']]
            lines.append(' & '.join([tex_interval(r), f"{int(r['count']):,}"] + vals) + r'\\')
        if ds != DATA[-1]: lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def profile_table():
    items = [('Train series', 'train_series', ',.0f'), ('Train event rows', 'train_rows', ',.0f'),
        ('Train targets', 'train_targets', ',.0f'), ('Quantity: median', 'quantity_p50', ',.0f'),
        ('Quantity: 95th percentile', 'quantity_p95', ',.0f'), ('Quantity: 99th percentile', 'quantity_p99', ',.0f'),
        ('Quantity: maximum', 'quantity_max', ',.0f'), ('Recorded gap: median', 'gap_p50', '.0f'),
        ('Recorded gap: 95th percentile', 'gap_p95', '.0f'), ('Usable history: median', 'history_p50', '.0f'),
        ('Usable history: 95th percentile', 'history_p95', '.0f'),
        ('History $\\le7$ events (\\%)', 'history_le7_percent', '.2f'),
        ('Within-series log correlation', 'within_entity_log_lag1_corr', '.3f'),
        ('Equal adjacent quantities (\\%)', 'adjacent_equal_percent', '.2f')]
    lines = [r'\begin{table}[t]',
        r'\caption{Train-only event characteristics. Quantity summaries use all train event rows; gap, usable-history, and adjacent-pair summaries use train targets after excluding each series\textquotesingle{} first event. Quantiles use the nearest observed value. Instacart gaps are top-coded at 30 days.}',
        r'\label{tab:train-characteristics}\centering\scriptsize',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Characteristic & Taxi & Intermittent & RAF & Instacart\\', r'\midrule',
        r'Time unit & Hour & Week & Month & Day\\']
    for title, key, fmt in items:
        lines.append(' & '.join([title] + [format(float(P[ds][key]), fmt) for ds in DATA]) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def history_tikz():
    lines = [r'\begin{figure}[t]', r'\centering',
        r'\begin{tikzpicture}[x=.047cm,y=.75cm,font=\scriptsize]',
        r'\definecolor{plotblue}{RGB}{40,101,138}']
    for tick in range(0, 193, 32):
        lines += [f'\\draw[gray!25] ({tick},-.4)--({tick},3.55);',
            f'\\node[below] at ({tick},-.4) {{{tick}}};']
    for j, ds in enumerate(DATA):
        y = 3-j; p=P[ds]
        lo,q1,med,q3,hi = [float(p['history_'+k]) for k in ['p05','p25','p50','p75','p95']]
        lines += [f'\\node[anchor=east] at (-3,{y}) {{{NAMES[ds]}}};',
            f'\\draw[plotblue,line width=.6pt] ({lo},{y})--({hi},{y});',
            f'\\draw[plotblue,line width=4pt] ({q1},{y})--({q3},{y});',
            f'\\draw[plotblue] ({lo},{y-.08})--({lo},{y+.08}) ({hi},{y-.08})--({hi},{y+.08});',
            f'\\filldraw[draw=plotblue,fill=white] ({med},{y}) circle[radius=2pt];',
            f'\\node[anchor=west,text=plotblue] at ({med+2},{y+.27}) {{median {med:g}}};']
    lines += [r'\node at (96,-1.05) {Usable observed events per train target};',
        r'\end{tikzpicture}',
        r'\caption{Train-history distributions under the frozen input windows. Thin lines span the 5th--95th percentiles, thick lines the 25th--75th percentiles, and circles mark medians. These are event-weighted train summaries, not validation performance curves.}',
        r'\label{fig:train-history}', r'\end{figure}']
    return '\n'.join(lines)


def contribution_tikz():
    rows = [r for r in R if r['dataset']==DATA[-1] and r['partition']=='quantity']
    labels = [tex_interval(r) for r in rows] + ['Overall']
    lines = [r'\begin{figure}[t]', r'\centering',
        r'\begin{tikzpicture}[x=1cm,y=.52cm,font=\scriptsize]',
        r'\definecolor{plotblue}{RGB}{40,101,138}', r'\definecolor{plotred}{RGB}{181,85,63}']
    for field, origin, xmin, xmax, ticks, axis in [
        ('mae_contribution',0.,-.09,.075,[-.08,-.04,0,.04],r'$\Delta$MAE (quantity units)'),
        ('mse_contribution',6.2,-1.25,1.1,[-1,-.5,0,.5,1],r'$\Delta$MSE (squared units)')]:
        width=4.3; tx=lambda x:origin+(x-xmin)/(xmax-xmin)*width
        values=[float(r[field]) for r in rows];values.append(sum(values))
        for t in ticks:
            lines += [f'\\draw[gray!25] ({tx(t):.5f},-.5)--({tx(t):.5f},5.4);',
                f'\\node[below] at ({tx(t):.5f},-.5) {{{t:g}}};']
        lines += [f'\\draw[ink] ({tx(0):.5f},-.4)--({tx(0):.5f},5.4);',
            f'\\node[anchor=west] at ({origin},6.1) {{Overall: ${values[-1]:+.4f}$}};']
        for j,(v,label) in enumerate(zip(values,labels)):
            y=5-j; color='ink' if j==5 else ('plotblue' if v<0 else 'plotred')
            lines += [f'\\node[anchor=east] at ({origin-.07},{y}) {{{label}}};',
                f'\\fill[{color}] ({tx(0):.5f},{y-.27}) rectangle ({tx(v):.5f},{y+.27});']
        lines += [f'\\draw[gray!50] ({origin},.5)--({origin+width},.5);',
            f'\\node at ({origin+width/2},-1.35) {{{axis}}};']
    lines += [r'\end{tikzpicture}',
        r'\caption{Instacart quantity-stratum contributions to the overall error difference, TitanTPP minus S2P2. Each bar includes its stratum\textquotesingle{}s target share; negative values favor TitanTPP. The five bars sum to the overall difference in each panel. The panels have different units and horizontal scales.}',
        r'\label{fig:instacart-contributions}', r'\end{figure}']
    return '\n'.join(lines)


METHOD = r'''
\section{Data Definitions and Training Distributions}\label{app:data-characteristics}
Taxi events are pickup counts in active spatial-cell hours; Intermittent
events are positive site--part order quantities; RAF events are positive
monthly spare-parts demands. Instacart events aggregate product rows for a
user on the same recorded activity day. Its quantity is a basket-size proxy,
not a count of physical units. In these event sequences, empty calendar
periods contribute to the recorded gap rather than separate zero-quantity
targets. Gaps retain their dataset-specific units (Table~\ref{tab:train-characteristics}).

We describe each frozen training population using the same definitions.
Quantity quantiles include all train event rows. Gap and usable-history
statistics describe the canonical prediction targets, which exclude the
first event of each series. For a target, usable history is the number of
observed predecessor events retained by the dataset's recorded-time window
and sequence-length cap. It excludes the target and padding. Thus it differs
from both elapsed calendar time and the total number of events in a series.
Train quantities and histories characterize available learning evidence;
the validation counts in Appendix~\ref{app:strata} describe a different
population later in each sequence.

For adjacent quantities, let $x_{e,j}=\log(1+q_{e,j-1})$ and
$y_{e,j}=\log(1+q_{e,j})$ for train pairs in series $e$. We compute
\begin{equation}
 r_{\mathrm{within}}=\operatorname{Corr}
 \bigl(x_{e,j}-\bar x_e,\;y_{e,j}-\bar y_e\bigr),
 \label{eq:within-correlation}
\end{equation}
pooling the centered pairs across series. The two means are computed
separately over each series' predecessor and successor members. This
pair-weighted statistic removes differences in average scale between series;
it is not an average of series-specific correlation coefficients.

PROFILE_TABLE

HISTORY_FIGURE

Taxi and Intermittent retain substantially longer histories and stronger
within-series adjacent association than the other datasets.
RAF's median usable train history is three events, and 93.58\% of targets
have at most seven; the corresponding Instacart values are five and 72.13\%.
RAF's centered correlation is $-0.105$, compared with Instacart's $0.037$.
The RAF estimate summarizes very short sequences after mean removal, so it
should not be read as a stable negative dependence law. These distributions
make history correction's dataset dependence plausible, while the observed
error comparisons below establish where its gains and losses occur.

\section{Validation Errors by Quantity and Usable History}\label{app:strata}
\subsection{Partitions and Aggregation}
The analysis uses the completed selected-checkpoint validation records for
all seven models in Table~\ref{tab:main}: four datasets and three seeds,
84 runs in total. Every stratum uses the same earliest minimum-RMSE
checkpoint as the overall results. The tables focus on TitanTPP and the
external model with the lowest overall mean validation RMSE on each dataset:
RMTPP on Taxi and Intermittent, and S2P2 on RAF and Instacart. This comparator
is held fixed across strata, including strata where another model performs
better.

Quantity boundaries are the frozen train-derived 50th, 90th, 95th, and
99th percentiles: $(7,686,1562,3449)$ for Taxi, $(2,31,46,187)$ for
Intermittent, $(2,30,60,200)$ for RAF, and $(8,20,25,35)$ for Instacart.
Upper boundaries belong to the lower bin. History boundaries reuse the
stored partitions: $(64,128)$ observed events for Taxi, Intermittent, and
RAF, and $(1,3,7,15,31)$ for Instacart. The finer Instacart partition
resolves its shorter histories. These dataset-specific bins support
within-dataset diagnosis; they are not identical history-length groups
across all four datasets. RAF's validation targets all fall in the first
stored history bin, so that partition cannot resolve its short-history
differences. We retain the empty cells in Table~\ref{tab:strata-history}.

Within each seed, MAE averages absolute error over targets and RMSE is the
square root of mean squared error. Tables~\ref{tab:strata-quantity}
and~\ref{tab:strata-history} report arithmetic means of those metrics over
seeds 42, 52, and 62. Each $N$ counts targets once per seed, not three times;
the target populations are identical across models and seeds. Duration NLL
in the overall comparison also uses the same selected checkpoints.

QUANTITY_TABLE

HISTORY_TABLE

\subsection{Contributions to the Aggregate Difference}
For stratum $b$ with $N_b$ of the $N$ validation targets, define the
contribution to the MSE difference as
\begin{equation}
 C_b=\frac{1}{3N}\sum_{s\in\{42,52,62\}}
       \left(\mathrm{SSE}_{\mathrm{TitanTPP},s,b}
             -\mathrm{SSE}_{\mathrm{external},s,b}\right).
 \label{eq:mse-contribution}
\end{equation}
Then $\sum_b C_b$ equals the mean of the three seeds' overall MSE
differences. It is not the difference of squared mean RMSEs. For MAE,
the analogous contribution is $N_b/N$ times the mean within-stratum
MAE difference. Figure~\ref{fig:instacart-contributions} shows both
decompositions for Instacart.

On Taxi, the bins at or below 686 contribute $+200.886$ squared quantity
units, whereas the bins above 686 contribute $-2822.868$, yielding a net
$-2621.982$ relative to RMTPP. Intermittent has the same direction of
concentration: the bins at or below 31 contribute $+0.05969$ and those
above 31 contribute $-3.77546$. These gains are concentrated rather than
uniform: RMTPP has lower RMSE in the two lower-quantity bins of both datasets.
Their history partitions also differ. Taxi's gain is concentrated above
128 observed events, while Intermittent improves in the first two history
bins and worsens above 128. Longer history alone therefore does not order
the benefits consistently.

RAF exhibits a near cancellation against S2P2. The bins at or below 60
contribute $-31.463$ squared quantity units and those above 60 contribute
$+28.463$, leaving $-3.000$ overall. This explains why appreciable
stratum-level differences coexist with a small aggregate RMSE improvement.
The comparison is specific to S2P2; RMTPP retains the lowest overall RAF
MAE in Table~\ref{tab:main}.

For Instacart, the bins at or below 20 contribute $-1.58816$ squared
quantity units and those above 20 contribute $+1.94581$, leaving an MSE
increase of $0.35765$. The corresponding MAE difference is $-0.00712$.
Thus the low-quantity majority yields a small MAE advantage, while the
larger high-quantity errors reverse the RMSE ranking. S2P2 also has lower
mean RMSE in every nonempty stored history bin. The observed disadvantage
cannot therefore be assigned solely to a single short-history group.

CONTRIBUTION_FIGURE
'''

MAIN = r'''\subsection{Where the Quantity Gains Occur}
The quantity gains are concentrated in different parts of each dataset.
Relative to the fixed external comparator with the best overall RMSE,
TitanTPP improves all three upper-quantity bins on Taxi and Intermittent,
but has higher RMSE in their two lower-quantity bins. RAF's improvements
at quantities up to 60 are largely offset by losses above 60. The complete
stratum counts and errors appear in Appendix~\ref{app:strata}.

Instacart shows the opposite aggregate trade-off. Targets at most 20 account
for 89.37\% of validation examples, and TitanTPP has lower mean MAE and RMSE
in both corresponding bins. S2P2 performs better in all three bins above 20.
Their contributions to the overall MSE difference are $-1.5882$ below or at
20 and $+1.9458$ above 20, explaining why a small MAE gain accompanies a
higher RMSE (Fig.~\ref{fig:instacart-contributions}). S2P2's lower RMSE
in every nonempty history bin also shows that short histories alone do not
isolate this disadvantage.

Train distributions provide context for these differences
(Appendix~\ref{app:data-characteristics}). Median usable histories are
129, 40, 3, and 5 events for Taxi, Intermittent, RAF, and Instacart.
After within-series centering, adjacent log-quantity correlations are
$0.749$, $0.928$, $-0.105$, and $0.037$, respectively. Stronger adjacent
association may help history correction on Taxi and Intermittent; the
stratum errors establish where gains occur, while this proposed explanation
remains a hypothesis rather than an identified causal effect.

'''


def main():
    figures()
    appendices = METHOD.replace('PROFILE_TABLE', profile_table()).replace('HISTORY_FIGURE', history_tikz())
    appendices = appendices.replace('QUANTITY_TABLE', stratum_table('quantity')).replace('HISTORY_TABLE', stratum_table('history'))
    appendices = appendices.replace('CONTRIBUTION_FIGURE', contribution_tikz())
    (OUT / 'appendices_B_C.tex').write_text(appendices)
    text = DRAFT.read_text()
    begin=text.index(r'\subsection{Where the Quantity Gains Occur}')
    end=text.index(r'\section{Computational Efficiency}', begin)
    text=text[:begin]+MAIN+text[end:]
    if r'\section{Data Definitions and Training Distributions}' in text:
        start=text.index(r'\section{Data Definitions and Training Distributions}')
        end=text.index(r'\begin{thebibliography}', start)
        text=text[:start]+text[end:]
    text=text.replace(r'\begin{thebibliography}', appendices+'\n'+r'\begin{thebibliography}',1)
    DRAFT.write_text(text)
    print(json.dumps({'tables':text.count(r'\begin{table}'),'figures':text.count(r'\begin{figure}'),
        'appendices':['A preserved','B train','C validation'],'plots':4}))


if __name__ == '__main__': main()
