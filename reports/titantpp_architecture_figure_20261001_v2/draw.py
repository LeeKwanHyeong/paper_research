"""Publication vector schematic; reproduces frozen Equation (4), not a model run."""
import os
from pathlib import Path
OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','svg.fonttype':'none','pdf.fonttype':42,'font.size':10.5})
fig,ax=plt.subplots(figsize=(7.2,4.6),dpi=300)
fig.patch.set_facecolor('white');ax.set_xlim(0,150);ax.set_ylim(-4,100);ax.axis('off')
fig.subplots_adjust(left=.008,right=.992,bottom=.015,top=.985)
ink='#20303e';line='#4d5c68';blue='#e9eff5';teal='#dcefe8';green='#266758';orange='#fff0da';gray='#f5f6f8'
def txt(x,y,s,size=10.5,color=ink,weight='normal',**kwargs):
 return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha='center',va='center',zorder=5,**kwargs)
def box(x,y,w,h,s,fill=blue,edge=line,size=10.5,weight='normal'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=.8',linewidth=.85,facecolor=fill,edgecolor=edge,zorder=3))
 return txt(x+w/2,y+h/2,s,size,weight=weight)
def arrow(a,b,color=line,dash=False):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8.5,linewidth=.85,color=color,linestyle='--' if dash else '-',zorder=2,shrinkA=0,shrinkB=0))
def path(points,color=line,dash=False,head=True):
 ax.plot(*zip(*points[:-1] if head else points),color=color,lw=.85,linestyle='--' if dash else '-',zorder=2)
 if head:arrow(points[-2],points[-1],color,dash)
def node(x,y,s):
 ax.add_patch(Circle((x,y),1.6,facecolor='white',edgecolor=line,linewidth=.9,zorder=4));txt(x,y,s,size=11.5)
# The side-by-side layout is intended for 122-mm printing: 10.5 pt -> 7 pt.
ax.text(1,97,'(a) TitanTPP',fontsize=11.6,fontweight='bold',color=ink)
ax.text(55,97,'(b) History correction',fontsize=11.6,fontweight='bold',color=ink)
ax.plot([51,51],[1,94],color='#d6dce1',lw=.65)
box(2,83,33,8,r'Observed $(\Delta_i,q_i)$')
box(2,70,33,8,'log1p + projection\n+ position')
box(2,57,33,8,'Causal encoder 1')
box(2,44,33,8,'History correction',teal,green,weight='bold')
box(2,31,33,8,'Causal encoder 2')
box(2,18,33,8,'Static retrieval\ncosine top-4')
for lo,hi in [(83,78),(70,65),(57,52),(44,39),(31,26)]:arrow((18.5,lo),(18.5,hi))
# Distinct learned banks; per-block P banks and a final static M bank.
for yy,label in [(58,r'$P^{(1)}$'),(32,r'$P^{(2)}$'),(19,r'$M$')]:
 box(40,yy,8,6,label,gray,size=10.8);arrow((40,yy+3),(35,yy+3))
arrow((18.5,18),(18.5,14.5));txt(18.5,12.5,r'last observed $z_n$',10.5)
path([(18.5,10.3),(10,10.3),(10,9.7)]);path([(18.5,10.3),(35,10.3),(35,9.7)])
box(1,.5,19,9,'Quantity\n'+r'$\widehat q_{n+1}$',orange,size=10.5)
box(25,.5,23,9,'Duration\n'+r'$p(D_{n+1}\mid\mathcal{H}_n)$',orange,size=10.5)
arrow((35.5,48),(53.5,48),green,True)
# Input pair shared by all eight branches.
box(63,82,33,9,'Current\n'+r'$h_i^{(1)}\in\mathbb{R}^{64}$',gray,size=10.5)
box(110,82,37,9,'Predecessor\n'+r'$h_{\pi(i)}^{(1)}\in\mathbb{R}^{64}$',gray,size=10.5)
box(81,70,44,7,'Concatenate  '+r'$[h_i^{(1)};h_{\pi(i)}^{(1)}]$',teal,green,size=10.5)
path([(79.5,82),(79.5,79),(94,79),(94,77)])
path([(128.5,82),(128.5,79),(113,79),(113,77)])
# Bypass is the current state, added only after the branch aggregate.
path([(63,86.5),(56,86.5),(56,7.5),(134.4,7.5)])
txt(57.9,22,r'$h_i^{(1)}$',10.8,rotation=90)
# Replicated branch detail; ellipsis retains individual U_b and V_b weights.
path([(103,70),(103,67),(62,67),(62,28)],head=False)
txt(89,63.5,r'Eight branches, $b=1,\ldots,8$',10.5,color=green)
for y,b in [(56,1),(44,2),(28,8)]:
 arrow((62,y),(66,y));box(66,y-4,17,8,fr'$U_{b}$'+'\n128 → 4',teal,green,size=10.5)
 arrow((83,y),(87,y));box(87,y-4,14,8,'GELU',teal,green,size=10.5)
 arrow((101,y),(105,y));box(105,y-4,17,8,fr'$V_{b}$'+'\n4 → 64',teal,green,size=10.5)
 arrow((122,y),(130.7,y));node(132.5,y,'×');txt(132.5,y+6,fr'$a_{{i,{b}}}$',10.5)
 arrow((132.5,y+4.2),(132.5,y+1.7));path([(134.2,y),(146,y)],head=False)
for x in [74.5,94,113.5]:txt(x,36,r'$\vdots$',13)
path([(146,56),(146,20.7),(137,20.7),(137,19.5)])
box(128,13,18,6.5,'Sum ÷ 8',teal,green,size=10.5)
arrow((137,13),(137,9.2));txt(145.3,11.0,r'$r_i$',10.5)
node(137,7.5,'+');arrow((137,5.8),(137,3.2));txt(137,0.0,r'$\widetilde h_i^{(1)}$',11.6)
txt(94,19.3,r'$a_{i,b}=o_i\,\mathbf{1}[c_i>\tau_b]$',11.4)
txt(93,13.0,r'$\tau=(1,2,4,8,16,32,64,128)$',10.5)
txt(87,3.5,r'$c_i$: observed-history count',10.5)
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'architecture.{ext}',dpi=400,facecolor='white')
# A print-size preview for checking labels at the intended 122-mm width.
from PIL import Image
im=Image.open(OUT/'architecture.png');im.resize((960,round(im.height*960/im.width)),Image.Resampling.LANCZOS).save(OUT/'architecture_preview.png')
print(OUT/'architecture.png')
