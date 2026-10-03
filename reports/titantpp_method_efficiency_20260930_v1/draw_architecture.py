from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(14,11),dpi=150)
fig.patch.set_facecolor('#f8fafc');ax.set_facecolor('#f8fafc')
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def box(x,y,w,h,title,detail='',color='#e8eff9',edge='#446388',fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007,rounding_size=0.01',
                              linewidth=1.2,facecolor=color,edgecolor=edge,zorder=2))
    ax.text(x+w/2,y+h*.65 if detail else y+h/2,title,ha='center',va='center',fontsize=fs,fontweight='bold',color='#14283f')
    if detail:ax.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=9.5,color='#314b66',linespacing=1.35)
def arrow(a,b,label='',dashed=False,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.4,color='#425b75',
        connectionstyle=f'arc3,rad={rad}',linestyle='--' if dashed else '-',zorder=1))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2,label,fontsize=9,color='#425b75',ha='center',va='bottom')
ax.text(.035,.969,'TitanTPP',fontsize=24,fontweight='bold',color='#14283f')
ax.text(.035,.937,'Observed-event encoding with a bottleneck history correction',fontsize=13,color='#425b75')
box(.27,.836,.36,.064,'Observed gaps and quantities','log1p → projection + learned position')
box(.27,.737,.36,.064,'Causal encoder 1','64 hidden units · 4 attention heads')
box(.27,.609,.36,.062,'Add history residual','current state + correction')
box(.71,.615,.25,.15,'History correction', 'Current + immediate predecessor\n8 × (128 → 4 → 64), GELU\nAvailability mask · fixed ÷8',color='#e1f2ed',edge='#42816e')
box(.27,.492,.36,.064,'Causal encoder 2','pre-LN attention + feed-forward residuals')
box(.27,.377,.36,.064,'Static prototype retrieval','cosine top-4 → mean raw vectors → residual')
box(.27,.276,.36,.061,'Last observed state','shared by time and quantity heads')
box(.042,.68,.17,.105,'Persistent vectors','16 per block, separate\nfixed at inference',fs=10)
box(.042,.37,.17,.08,'Static bank','64 learned prototypes',fs=10)
box(.15,.133,.27,.076,'Quantity head','softplus → expm1\nlog1p quantity MSE',color='#fff1d9',edge='#b78940')
box(.49,.133,.33,.076,'Duration head','conditional lognormal\nrecorded integer-bin NLL',color='#fff1d9',edge='#b78940')
for top,bottom in [(.836,.801),(.737,.671),(.609,.556),(.492,.441),(.377,.337)]:arrow((.45,top),(.45,bottom))
ax.plot([.637,.675,.675,.835],[.77,.77,.796,.796],color='#425b75',lw=1.4,zorder=1)
arrow((.835,.796),(.835,.773))
arrow((.71,.644),(.63,.64))
arrow((.212,.733),(.27,.77));arrow((.14,.673),(.265,.525))
arrow((.212,.410),(.265,.410))
arrow((.39,.276),(.285,.213));arrow((.51,.276),(.655,.213))
ax.text(.72,.59,'All branches read lag one.\nThresholds govern availability,\nnot different retrieval lags.',fontsize=10,color='#28624f',linespacing=1.45,va='top')
ax.text(.71,.447,'No online associative update\nNo learned Gate\nOutput projections initialize to zero',fontsize=10,color='#425b75',linespacing=1.55)
ax.text(.5,.078,'The next-event target is withheld from the encoder and enters the loss only.',ha='center',fontsize=11,color='#14283f')
ax.text(.5,.040,'Persistent and static memory remain; inference does not update their parameters.',ha='center',fontsize=10,color='#425b75')
fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.02)
for ext in ('svg','png','pdf'):fig.savefig(OUT/f'architecture.{ext}',facecolor=fig.get_facecolor())
