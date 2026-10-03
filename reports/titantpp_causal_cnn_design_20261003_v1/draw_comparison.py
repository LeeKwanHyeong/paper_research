"""Source-backed architecture proposal; no model, dataset, or training execution."""
import os
from pathlib import Path
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/titantpp_cnn_diagram_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(18, 10), dpi=160)
fig.patch.set_facecolor('white')
ax.set(xlim=(0, 180), ylim=(0, 100)); ax.axis('off')
fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.985)
ink, line = '#172d3e', '#526b7b'
blue, teal, amber, gray = '#e8f0f8', '#e0f2eb', '#fff0d6', '#f3f5f7'
green, new = '#23725d', '#a65b08'

def text(x, y, s, fs=13, color=ink, weight='normal', **kwargs):
    return ax.text(x, y, s, ha='center', va='center', fontsize=fs,
                   color=color, fontweight=weight, zorder=5, **kwargs)
def box(x, y, w, h, s, fill=blue, edge=line, fs=13):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.05,rounding_size=.6',
                                fc=fill, ec=edge, lw=1.3, zorder=3))
    text(x+w/2,y+h/2,s,fs)
def arrow(a,b,color=line,dash=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,
                                color=color,lw=1.4,linestyle='--' if dash else '-',
                                shrinkA=0,shrinkB=0,zorder=2))
def path(points, color=line):
    ax.plot(*zip(*points[:-1]),color=color,lw=1.4,zorder=2)
    arrow(points[-2],points[-1],color)
def plus(x,y):
    ax.add_patch(Circle((x,y),1.7,fc='white',ec=line,lw=1.3,zorder=4))
    text(x,y,'+',18)

text(90,97,'First change: replace the history-correction module',23,weight='bold')
text(90,92.5,'Design proposal  |  Existing encoder, memory and prediction paths remain in place',13,color=line)
for x in (52,116): ax.plot([x,x],[17,87],color='#d8e0e6',lw=1)
text(25,85,'(a) Shared model path',16,weight='bold')
text(84,85,'(b) Existing History MLP',16,weight='bold')
text(148,85,'(c) Proposed Causal CNN',16,weight='bold',color=new)

# The green box is the only replacement point.
backbone = [(72,8,'Observed gap–quantity prefix\nlog1p + 64-D embedding',blue),
            (60,8,'Causal encoder 1\n+ persistent bank',blue),
            (48,8,'History correction\nMLP → proposed CNN',teal),
            (36,8,'Causal encoder 2\n+ persistent bank',blue),
            (24,8,'Static top-4 retrieval',blue),
            (12,8,'Last observed state\nQuantity + duration heads',amber)]
for y,h,s,c in backbone: box(3,y,44,h,s,c,green if c==teal else line,13)
for top, bottom in ((72,68),(60,56),(48,44),(36,32),(24,20)): arrow((25,top),(25,bottom))

# Existing correction: same pair for eight distinct small MLPs.
box(64,71,40,9,'Current + previous observed state\n64 + 64 = 128 features',gray,fs=12.5)
text(84,67.5,'Same pair for all 8 branches',11.5,color=green)
for x,label in ((60,'Branch 1'),(91,'Branch 8')):
    box(x,50,22,12,label+'\n128 → 4 → 64\nGELU',teal,green,12.5)
    path([(84,71),(84,64.5),(x+11,64.5),(x+11,62)])
    arrow((x+11,50),(x+11,47))
    text(x+11,45.5,'× availability',11.5)
    path([(x+11,43.5),(x+11,40.5),(84,40.5),(84,39)])
text(86.5,56,'···',20)
box(65,32,38,7,'Sum branches ÷ 8',teal,green,13)
arrow((84,32),(84,27.7)); plus(84,26)
path([(64,75.5),(56.5,75.5),(56.5,26),(82.3,26)])
text(58.5,35,'current state',10.5,rotation=90)
arrow((84,24.3),(84,21.5));text(84,19.5,'Corrected 64-D state',12.5)

# Candidate: the time axis comprises observed events, not tensor padding rows.
box(128,71,40,9,'Current + previous 2 observed states\n3 positions × 64 channels',gray,fs=12.5)
arrow((148,71),(148,67))
box(126,56,44,11,'Dense causal Conv1d\n64 → 24 channels  |  kernel 3\nGELU',amber,new,12.5)
arrow((148,56),(148,53))
box(126,44,44,9,'Pointwise Conv1d\n24 → 64  |  kernel 1\nOutput weights initialized to zero',amber,new,11.8)
arrow((148,44),(148,40.5))
box(126,32,44,8,'Observed-history gate × output ÷ 8\nActive with ≥1 previous event',amber,new,11.8)
arrow((148,32),(148,27.7));plus(148,26)
path([(128,75.5),(121,75.5),(121,26),(146.3,26)])
text(123,35,'current state',10.5,rotation=90)
arrow((148,24.3),(148,21.5));text(148,19.5,'Corrected 64-D state',12.5)

# Evidence/interpretation labels kept away from the flow.
text(84,13.7,'8 × (128×4 + 4×64)',13,color=green)
text(148,13.7,'64×24×3 + 24×64',13,color=new)
text(116,9.5,'6,144 correction parameters in either design',15,weight='bold')
text(90,4.3,'Local correction window only: encoder attention still sees the permitted prefix.\nThe proposal changes context, feature layout and gating together; it does not isolate a CNN-only effect.',11.5,color=line)
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'architecture_comparison.{ext}',dpi=160,facecolor='white')
print(OUT/'architecture_comparison.png')
