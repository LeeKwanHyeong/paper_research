import os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/titantpp_efficiency_figure_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent
D=json.loads((R/'analysis.json').read_text())
rows=D['comparison'];x=np.arange(len(rows));w=.35
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
colors={'mlp':'#2878B5','mac':'#E08E36'}
for model,offset,label in [('mlp',-w/2,'TitanTPP (History MLP)'),('mac',w/2,'Titans-MAC adapter')]:
 vals=[r[model+'_median_compute_ms_mean'] for r in rows];errs=[r[model+'_median_compute_ms_sd'] for r in rows]
 b=axs[0].bar(x+offset,vals,w,label=label,color=colors[model],yerr=errs,capsize=3)
 axs[0].bar_label(b,labels=[f'{v:.1f}' for v in vals],padding=4,fontsize=8)
 vals=[r[model+'_peak_allocated_mib_mean'] for r in rows]
 b=axs[1].bar(x+offset,vals,w,label=label,color=colors[model]);axs[1].bar_label(b,labels=[f'{v:.0f}' for v in vals],padding=4,fontsize=8)
labels=[('Taxi' if r['dataset']=='yellow_trip_hourly' else 'Intermittent')+'\n'+('Train' if r['mode']=='train' else 'Evaluation') for r in rows]
for ax in axs:
 ax.set_xticks(x,labels);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.margins(y=.17)
axs[0].set_ylabel('Synchronized compute time (ms / batch128)');axs[0].set_title('Paired batch processing cost')
axs[1].set_ylabel('Peak allocated GPU memory (MiB)');axs[1].set_title('GPU memory trade-off')
axs[0].set_ylim(0,max(r['mac_median_compute_ms_mean']+r['mac_median_compute_ms_sd'] for r in rows)*1.4)
axs[0].legend(loc='upper left',fontsize=8)
fig.suptitle('RTX 5080 · same train inputs and shared heads/loss · 3 timing repetitions',fontsize=12)
fig.supxlabel('32 measured batches after 5 warm-ups; evaluation includes target loss; no accuracy or full-epoch measurement.',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(R/f'paired_efficiency.{ext}',dpi=180)
