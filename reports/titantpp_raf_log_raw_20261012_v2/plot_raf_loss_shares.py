#!/usr/bin/env python3
"""Plot six already-verified RAF sufficient-statistic entries; no inference."""
from pathlib import Path
import hashlib
import json
import math
import os

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'search_artifacts/titantpp_raf_log_raw_20261012_v2/analysis/raf_log_raw_conditions.json'
SOURCE_SHA='ada3e47b6bbfc3e87d17496297dadae3751d759b451b705f318c618596b75b7d'
CONTRACT='0c75bdd4d70adf8569ea2602c942e93f6b27b1206a4595711b2673997ee0758a'
OPERATION='ac834cd38fbb5f2ce693148711e7c187117c190923e78562eb5cb3975bfd8cd1'
OUTPUT=Path(__file__).resolve().parent/'figures'
SEEDS=(42,52,62)
SPLITS=('train','validation')
METRICS=('tail_target_share','tail_share_of_quantity_log_loss','tail_share_of_raw_quantity_sse')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_exclusive(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def extract_rows():
    require(sha(SOURCE)==SOURCE_SHA,'Pinned H1 analytical source changed')
    data=json.loads(SOURCE.read_bytes())
    require(data['status']=='complete' and data['contract_sha256']==CONTRACT
            and data['operation_integrity_sha256']==OPERATION
            and data['actual_worker_exit0_count']==18 and data['new_fits']==data['test_inference']==0,
            'H1 completion or scientific scope differs')
    rows=[]
    identity={}
    for row in data['rows']:
        if row['arm']!='titantpp_cnn_gru54':continue
        require(row['split'] in SPLITS and row['seed'] in SEEDS,'Foreign full-model chart entry')
        require(row['dataset']=='raf_spare_parts','Foreign chart dataset')
        aggregate=row['log_raw'];whole=aggregate['cells']['all']['model'];tail=aggregate['cells']['4']['model']
        require(aggregate['quantity_boundaries']==[2.,30.,60.,200.] and aggregate['tail_key']=='4', 'Tail definition differs')
        require(whole['count']>0 and tail['count']>0 and whole['log_native_loss_sum']>0 and whole['raw_error_square_sum']>0,
                'Share denominator is unavailable')
        calculated={'tail_target_share':tail['count']/whole['count'],
            'tail_share_of_quantity_log_loss':tail['log_native_loss_sum']/whole['log_native_loss_sum'],
            'tail_share_of_raw_quantity_sse':tail['raw_error_square_sum']/whole['raw_error_square_sum']}
        require(all(0<=v<=1 and math.isfinite(v) for v in calculated.values()),'Invalid part-to-whole share')
        for key,value in calculated.items():
            require(math.isclose(value,row['tail_loss_contributions'][key],rel_tol=1e-12,abs_tol=1e-12), 'Displayed share differs: '+key)
        require(identity.setdefault(row['split'],(whole['count'],tail['count']))==(whole['count'],tail['count']), 'Seed target denominators differ')
        rows.append({'split':row['split'],'seed':row['seed'],'target_count':whole['count'],'tail_count':tail['count'],
            **calculated,'quantity_log_loss_sum':whole['log_native_loss_sum'],'tail_quantity_log_loss_sum':tail['log_native_loss_sum'],
            'raw_quantity_sse':whole['raw_error_square_sum'],'tail_raw_quantity_sse':tail['raw_error_square_sum'],
            'checkpoint_sha256':row['checkpoint_sha256'],'state_sha256':row['state_sha256'],
            'target_identity_sha256':row['target_identity_sha256'],'truth_history_sha256':row['truth_history_sha256']})
    require(len(rows)==6 and {(r['split'],r['seed']) for r in rows}=={(p,s) for p in SPLITS for s in SEEDS},'Full-model six-entry chart matrix differs')
    return sorted(rows,key=lambda r:(SPLITS.index(r['split']),r['seed']))


def main():
    rows=extract_rows()
    require(OUTPUT.is_dir(),'Figure directory is not prepared')
    paths={ext:OUTPUT/('raf_tail_loss_shares.'+ext) for ext in ('png','svg')}
    require(not any(p.exists() for p in paths.values()) and not (OUTPUT/'figure_receipt.json').exists()
            and not (OUTPUT/'figure_data.json').exists(),'Existing figure outputs must be preserved')
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/raf_h1_figure_matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/raf_h1_figure_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titleweight':'semibold',
        'axes.labelcolor':'#292D32','text.color':'#292D32','xtick.color':'#50555C','ytick.color':'#292D32',
        'svg.fonttype':'none','savefig.facecolor':'white'})
    colors={42:'#276A9B',52:'#B88619',62:'#A34974'}
    markers={42:'o',52:'s',62:'^'}
    offsets={42:-.13,52:0.,62:.13}
    labels=['Tail events / all events','Tail log loss / all quantity log loss','Tail raw SSE / all raw quantity SSE']
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.31,right=.985,bottom=.29,top=.71,wspace=.18)
    fig.text(.035,.96,'RAF: tail shares in selected-checkpoint evaluation',fontsize=16,weight='semibold',va='top')
    fig.text(.035,.90,'Full CNN+GRU54  |  Tail: true quantity > 200  |  Same selected model per seed',fontsize=11.5,va='top')
    legend=[Line2D([0],[0],marker=markers[s],color=colors[s],markerfacecolor=colors[s],linestyle='None',
        markersize=7,label='seed '+str(s)) for s in SEEDS]
    fig.legend(handles=legend,loc='upper right',bbox_to_anchor=(.99,.90),frameon=False,ncol=3,fontsize=10,
               handletextpad=.3,columnspacing=1.3)
    for ax,split in zip(axes,SPLITS):
        selected=[r for r in rows if r['split']==split]
        first=selected[0]
        ax.set_title(('Train' if split=='train' else 'Validation')+'\n'+f"{first['tail_count']:,} tail / {first['target_count']:,} targets",loc='left',fontsize=12,pad=15)
        ax.set_xlim(0,100);ax.set_ylim(-.4,2.45)
        ax.set_xticks([0,20,40,60,80,100]);ax.set_xticklabels(['0','20','40','60','80','100'])
        ax.set_xlabel('Share of the corresponding split total (%)',fontsize=10,labelpad=9)
        ax.set_yticks([2,1,0]);ax.set_yticklabels(labels,fontsize=10.5)
        ax.tick_params(axis='y',length=0,pad=12)
        ax.tick_params(axis='x',length=3,width=.7)
        ax.xaxis.grid(True,color='#DDE1E5',linewidth=.65);ax.set_axisbelow(True)
        for side in ('top','right','left'):ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#808891');ax.spines['bottom'].set_linewidth(.75)
        for position,metric in zip((2,1,0),METRICS):
            values=[100*r[metric] for r in selected]
            ax.hlines(position,min(values),max(values),color='#68727B',linewidth=1.1,zorder=2)
            for row,value in zip(selected,values):
                seed=row['seed']
                ax.scatter(value,position+offsets[seed],s=44,marker=markers[seed],facecolor=colors[seed],
                           edgecolor='white',linewidth=.65,zorder=3)
            if metric=='tail_target_share':annotation=f"{values[0]:.4f}% (all seeds)"
            else:annotation=f"{min(values):.2f}–{max(values):.2f}%"
            # Numeric labels sit at a fixed y above all three seed markers.
            x=10 if max(values)<15 else max(values)-1
            ha='left' if max(values)<15 else 'right'
            ax.text(x,position+.31,annotation,fontsize=10.5,ha=ha,va='center',color='#292D32')
    axes[1].tick_params(labelleft=False)
    caption=('Each point is one seed; ranges show observed seed spread, not confidence intervals.\n'
        'Denominators differ by row: event count, sum of native quantity log-MSE, or raw quantity SSE.\n'
        'These are evaluation-time quantity-error shares—not joint time/quantity loss or training-gradient contributions.')
    fig.text(.035,.165,caption,fontsize=10.2,va='top',linespacing=1.5,color='#4E5660')
    fig.text(.035,.025,'Source: frozen RAF H1 Train/Validation sufficient statistics (12 Oct 2026). No new training or Test evaluation.',
             fontsize=9,va='bottom',color='#68727B')
    fig.savefig(paths['png'],dpi=180)
    fig.savefig(paths['svg'])
    plt.close(fig)
    write_exclusive(OUTPUT/'figure_data.json',{'source_path':str(SOURCE),'source_sha256':SOURCE_SHA,
        'contract_sha256':CONTRACT,'operation_integrity_sha256':OPERATION,'rows':rows,
        'marker_offsets_are_display_only':True,'shares_percent_multiplier':100})
    write_exclusive(OUTPUT/'figure_receipt.json',{'status':'rendered','source_sha256':SOURCE_SHA,
        'contract_sha256':CONTRACT,'plot_script_sha256':sha(__file__),
        'figure_files':{p.name:sha(p) for p in paths.values()},'figure_data_sha256':sha(OUTPUT/'figure_data.json'),
        'source_rows':6,'splits':list(SPLITS),'seeds':list(SEEDS),'tail_quantity_gt':200,
        'x_axis_limits_percent':[0,100],'independent_share_denominators':True,
        'spread_is_confidence_interval':False,'evaluation_quantity_error_shares_only':True,
        'joint_objective_or_training_gradient_contribution':False,'new_training':0,'new_inference':0,'test_access':0,
        'visual_inspection_recorded_separately':True})
    print(json.dumps({'status':'rendered','files':[str(p) for p in paths.values()],'source_rows':len(rows)}))


if __name__=='__main__':main()
