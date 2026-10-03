"""Integrate the reviewed train-only heatmap into the existing standalone source."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MAIN=ROOT/'paper/titantpp_pakdd_2027_draft/main.tex'

OLD_MAIN=r'''Train distributions provide context for these differences
(Appendix~\ref{app:data-characteristics}). Median usable histories are
129, 40, 3, and 5 events for Taxi, Intermittent, RAF, and Instacart.
After within-series centering, adjacent log-quantity correlations are
$0.749$, $0.928$, $-0.105$, and $0.037$, respectively. Stronger adjacent
association may help history correction on Taxi and Intermittent; the
stratum errors establish where gains occur, while this proposed explanation
remains a hypothesis rather than an identified causal effect.'''

NEW_MAIN=r'''Train distributions provide context for these differences
(Appendix~\ref{app:data-characteristics}). Median usable histories are
129, 40, 3, and 5 events for Taxi, Intermittent, RAF, and Instacart.
The train-only lag profiles distinguish strong short-range association on
Taxi from more persistent association on Intermittent, whereas RAF and
Instacart exhibit much weaker within-series correlations
(Fig.~\ref{fig:train-lag-correlation}). Together with the completed Taxi
structural controls, this contrast motivates the hypothesis that directly
coupling adjacent contextual states is useful when recent quantities carry
substantial within-series information. The error decomposition identifies
the realized gains and losses; the four-dataset association provides a
post hoc interpretation of their variation.'''

OLD_APPENDIX=r'''Taxi and Intermittent retain substantially longer histories and stronger
within-series adjacent association than the other datasets.
RAF's median usable train history is three events, and 93.58\% of targets
have at most seven; the corresponding Instacart values are five and 72.13\%.
RAF's centered correlation is $-0.105$, compared with Instacart's $0.037$.
The RAF estimate summarizes very short sequences after mean removal, so it
should not be read as a stable negative dependence law. These distributions
make history correction's dataset dependence plausible, while the observed
error comparisons below establish where its gains and losses occur.'''

NEW_APPENDIX=r'''\subsection{Event-Lag Association and Applicability}
We extend Eq.~\eqref{eq:within-correlation} to event lags
$\ell\in\{1,2,4,8,16\}$ by pairing
$x^{(\ell)}_{e,j}=\log(1+q_{e,j-\ell})$ with
$y^{(\ell)}_{e,j}=\log(1+q_{e,j})$ and recomputing each series'
two means at every lag. Pairs remain within the same training series.
The lag counts observed events rather than calendar units, and the
descriptive calculation covers the full train sequence without the model's
input-window restriction. The lag-one column reproduces
Table~\ref{tab:train-characteristics}.

FIGURE_PLACEHOLDER

Taxi and Intermittent exhibit different forms of serial association.
On Taxi, correlation declines from $0.749$ at lag one to $0.546$ at lag two
and $0.136$ at lag four, then becomes negative at lags eight and sixteen.
Intermittent retains a correlation of $0.473$ even at lag sixteen.
RAF's available-lag estimates range from $-0.127$ to $-0.043$, while
Instacart remains close to zero throughout the displayed range
($|r_{\ell}|<0.05$). Thus the two datasets with larger quantity gains
share strong immediate association, although its persistence differs.

The sample counts qualify the longer-lag comparison. RAF supplies 25,779
pairs at lag one, only 1,655 at lag eight, and none at lag sixteen.
After within-series centering, a series with one pair contributes zero
to both sums of squares; short-series estimates can also be affected by
mean removal. RAF's small negative values therefore describe this sample
rather than a stable negative dependence law. A sensitivity analysis
restricts every lag to the same series with at least 20 train events:
131 on Taxi, 4,968 on Intermittent, and 31,020 on Instacart; RAF supplies
none. In this fixed cohort, Instacart's correlations at lags one and two
increase to $0.074$ and $0.092$, respectively, while Taxi and
Intermittent retain their stronger immediate association.

These profiles suggest an applicability hypothesis for history correction:
combining adjacent contextual states can be useful when recent quantities
carry substantial information within a series. The completed Taxi
parameter-matched control supports that architectural interpretation
(Appendix~\ref{app:extension}). Data characteristics alone, however, do
not determine the error trade-off. Intermittent's last-quantity baseline
achieves lower MAE than \method, while \method\ achieves lower RMSE;
RAF and Instacart exhibit offsetting gains and losses across quantity bins
(Appendix~\ref{app:strata}). The weak Instacart lag profile also persists
among longer series, complementing the observation that S2P2's RMSE
advantage spans every nonempty history bin. This post hoc analysis relates
the observed results to train characteristics without assigning their
causes to a single distributional property. All eight correction branches
read the immediate predecessor; the displayed lags describe the data and
do not denote different branch inputs.'''


def main():
    before=MAIN.read_text()
    assert before.count(OLD_MAIN)==1 and before.count(OLD_APPENDIX)==1
    backup=OUT/'main_before.tex'
    assert not backup.exists(), 'Preserve earlier revision receipt; do not overwrite backup.'
    backup.write_text(before)
    fig=(OUT/'figure.tex').read_text().strip()
    after=before.replace(OLD_MAIN,NEW_MAIN).replace(OLD_APPENDIX,NEW_APPENDIX.replace('FIGURE_PLACEHOLDER',fig))
    MAIN.write_text(after)
    (OUT/'main.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='main_before.tex',tofile='paper/titantpp_pakdd_2027_draft/main.tex')))
    receipt=dict(source=str(MAIN.relative_to(ROOT)),before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                 after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                 changed_blocks=['Results 5.4 train-context paragraph','Appendix B final interpretation replaced by lag-analysis subsection'],
                 preserved=['authors','abstract','contributions','methods','main comparison tables','efficiency','Appendix A 13-condition cutoff','Appendix C','references'])
    (OUT/'manuscript_edit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__': main()
