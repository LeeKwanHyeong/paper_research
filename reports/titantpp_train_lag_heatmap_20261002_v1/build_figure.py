"""Produce an embedded TikZ figure and matching PNG/SVG previews from reviewed rows."""
from pathlib import Path
import json
import os
import tempfile

OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='titantpp-mpl-'))
os.environ.setdefault('XDG_CACHE_HOME', tempfile.mkdtemp(prefix='titantpp-font-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable

DATA = json.loads((OUT / 'analysis.json').read_text())
ROWS = DATA['rows']
LABELS = ['Taxi', 'Intermittent', 'RAF', 'Instacart']
LAGS = [1, 2, 4, 8, 16]
NEG, ZERO, POS, MISSING = '#b66b23', '#f8f8f8', '#28658a', '#e1e1e1'
CMAP = LinearSegmentedColormap.from_list('lag_signed', [NEG, ZERO, POS], N=1001)


def rgb_hex(value):
    from matplotlib.colors import to_hex
    return to_hex(CMAP((value + 1) / 2)).lstrip('#').upper() if value is not None else MISSING[1:].upper()


def text_color(value):
    h = rgb_hex(value)
    srgb = [int(h[i:i+2],16)/255 for i in (0,2,4)]
    linear = [c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in srgb]
    luminance = sum(c*w for c,w in zip(linear,[.2126,.7152,.0722]))
    black, white = (luminance+.05)/.05, 1.05/(luminance+.05)
    assert max(black,white) >= 4.5
    return 'black' if black>=white else 'white'


def cell(label, lag):
    return next(r for r in ROWS if r['label'] == label and r['event_lag'] == lag)


def main():
    lines = [r'% Generated from reports/titantpp_train_lag_heatmap_20261002_v1/analysis.json.',
             r'\begin{figure}[t]', r'\centering',
             r'\begin{tikzpicture}[x=1.65cm,y=1.10cm,font=\scriptsize]',
             r'\node at (2.5,4.70) {Event lag $\ell$};']
    for j, lag in enumerate(LAGS):
        lines.append(r'\node at (' + f'{j+.5},4.30' + r') {' + str(lag) + '};')
    for i, label in enumerate(LABELS):
        y = 3-i
        lines.append(r'\node[anchor=east] at (-.10,'+f'{y+.5}'+') {'+label+'};')
        for j, lag in enumerate(LAGS):
            row = cell(label, lag); value = row['correlation']; name = f'lagcell{i}{j}'
            color = text_color(value)
            lines.append(r'\definecolor{'+name+'}{HTML}{'+rgb_hex(value)+'}')
            lines.append(r'\filldraw[fill='+name+r',draw=white,line width=.6pt] ('+f'{j},{y}'+') rectangle ('+f'{j+1},{y+1}'+');')
            val = f'${value:.3f}$' if value is not None else 'NA'
            for yy, text, font in [(y+.73, val, r'\bfseries\small'),
                                    (y+.44, f"P: {row['pairs']:,}", r'\scriptsize'),
                                    (y+.20, f"S: {row['series_with_pairs']:,}", r'\scriptsize')]:
                lines.append(r'\node[text='+color+',font='+font+'] at ('+f'{j+.5},{yy:.2f}'+') {'+text+'};')
    lines.append(r'\node at (2.5,-.35) {P: event pairs; S: series with at least one pair};')
    # Explicit shared [-1,1] color scale; missing cell never shares zero's color.
    for k in range(100):
        lines.append(r'\definecolor{lagbar'+str(k)+'}{HTML}{'+rgb_hex(-1+2*(k+.5)/100)+'}')
        lines.append(r'\fill[lagbar'+str(k)+'] ('+f'{1.25+2.5*k/100:.4f},-.80'+') rectangle ('+f'{1.25+2.5*(k+1)/100:.4f},-.62'+');')
    for xx, text in [(1.25,'$-1$'),(2.5,'$0$'),(3.75,'$1$')]:
        lines.append(r'\node[below] at ('+f'{xx},-.81'+') {'+text+'};')
    lines.extend([r'\node[anchor=east] at (1.08,-.71) {$r_{\ell}$};', r'\end{tikzpicture}',
        r'\caption{Train-only within-series log-quantity correlation at event lags 1, 2, 4, 8, and 16. Each cell reports $r_{\ell}$, the number of event pairs (P), and the number of contributing series (S). All cells share the color scale $[-1,1]$; gray NA denotes no available pairs. Pair counts include constant and single-pair series, which contribute zero to one or both centered sums of squares. These are descriptive event offsets, not the correction module\textquotesingle{}s branch indices.}',
        r'\label{fig:train-lag-correlation}',
        r'\Description{A four-by-five heatmap. Taxi correlation declines from 0.749 at lag one to negative values at lags eight and sixteen. Intermittent remains positive, declining from 0.928 to 0.473. RAF has small negative correlations through lag eight and no pairs at lag sixteen. Instacart correlations remain close to zero. Pair and series counts are printed in every cell.}',
        r'\end{figure}', ''])
    (OUT / 'figure.tex').write_text('\n'.join(lines))

    plt.rcParams.update({'font.family':'DejaVu Serif', 'font.size':9, 'svg.fonttype':'none'})
    fig = plt.figure(figsize=(7.2, 4.6), facecolor='white')
    ax = fig.add_axes([.19,.26,.77,.61])
    for i,label in enumerate(LABELS):
        for j,lag in enumerate(LAGS):
            r = cell(label,lag); v = r['correlation']; y=3-i
            ax.add_patch(Rectangle((j,y),1,1,facecolor='#'+rgb_hex(v),edgecolor='white',linewidth=.7))
            color=text_color(v)
            ax.text(j+.5,y+.73,f'{v:.3f}' if v is not None else 'NA',ha='center',va='center',fontsize=11,fontweight='bold',color=color)
            ax.text(j+.5,y+.44,f"P: {r['pairs']:,}",ha='center',va='center',fontsize=8.4,color=color)
            ax.text(j+.5,y+.20,f"S: {r['series_with_pairs']:,}",ha='center',va='center',fontsize=8.4,color=color)
    ax.set(xlim=(0,5),ylim=(0,4),xticks=[.5+i for i in range(5)],xticklabels=LAGS,
           yticks=[3.5-i for i in range(4)],yticklabels=LABELS)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top');ax.set_xlabel(r'Event lag $\ell$',labelpad=10)
    ax.tick_params(length=0,pad=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.575,.21,'P: event pairs; S: series with at least one pair',ha='center',fontsize=9)
    cbax=fig.add_axes([.37,.12,.38,.025])
    fig.colorbar(ScalarMappable(norm=Normalize(-1,1),cmap=CMAP),cax=cbax,orientation='horizontal',ticks=[-1,0,1])
    cbax.set_title(r'$r_{\ell}$',loc='left',fontsize=10,pad=5);cbax.tick_params(length=2)
    fig.text(.575,.025,'Train only · within-series centered log(1 + quantity)\nGray NA: no available pairs',ha='center',fontsize=8)
    fig.savefig(OUT/'heatmap.png',dpi=240,facecolor='white')
    fig.savefig(OUT/'heatmap.svg',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
