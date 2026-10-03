"""Apply the reviewed editorial revision from the preserved manuscript baseline."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BACKUP = OUT / 'before_reference_expansion'
MAIN = ROOT / 'paper/titantpp_history_mlp_manuscript_20261001_v1.md'
old = (BACKUP / MAIN.name).read_text()
additions = json.loads((OUT / 'reference_additions.json').read_text())['records']
original_registry = json.loads((BACKUP / 'references.json').read_text())
original = original_registry['records']
records = original + additions
by_key = {r['key']: r for r in records}
labels = {
    'du2016rmtpp': 'Du et al., 2016', 'mei2017nhp': 'Mei and Eisner, 2017',
    'zuo2020thp': 'Zuo et al., 2020', 'zhang2020sahp': 'Zhang et al., 2020',
    'turkmen2021renewal': 'Türkmen et al., 2021', 'yang2022attnhp': 'Yang et al., 2022',
    'behrouz2025titans': 'Behrouz et al., 2025', 'chang2025s2p2': 'Chang et al., 2025',
    'draxler2025flextpp': 'Draxler et al., 2025',
}


def cite(key):
    r = by_key[key]
    return '[' + r.get('citation_label', labels.get(key)) + '](' + r['url'] + ')'


related = f'''## 2. Related work

### 2.1 Event-history representations and duration models

Hawkes processes describe event dependence through self-excitation and mutual excitation. {cite('hawkes1971spectra')}. Neural temporal point processes learn history representations and conditional event distributions. These two design choices provide a useful distinction between the history encoder and the prediction head. {cite('shchur2021review')}.

RMTPP summarizes event history with a recurrent neural network, whereas the Neural Hawkes Process maintains a continuous-time LSTM state that controls evolving event intensities. {cite('du2016rmtpp')}; {cite('mei2017nhp')}. Attention-based alternatives expose interactions between observed events directly. THP applies self-attention to temporal dependencies, SAHP incorporates time intervals into attention-based encoding, and AttNHP constructs attention-based embeddings for irregularly spaced events. {cite('zuo2020thp')}; {cite('zhang2020sahp')}; {cite('yang2022attnhp')}.

State-space representations offer another way to compress history. HiPPO develops polynomial projections for online history compression; S2P2 combines continuous-time state evolution, stochastic jumps, and nonlinear transformations for marked event sequences. Its temporal inductive bias complements the recurrent and attention encoders in our comparison. Section 4 specifies the S2P2 operations retained in the common-head adaptation. {cite('gu2020hippo')}; {cite('chang2025s2p2')}.

The time distribution also affects likelihood evaluation. Omi et al. model integrated intensity with a neural network and obtain intensity by differentiation. Intensity-free TPPs instead parameterize the conditional inter-event-time distribution through flows or tractable mixtures. TitanTPP follows the direct-distribution approach with a conditional lognormal duration head and an observation model for recorded integer durations. The shared head lets the experiments focus on event-history encoding. {cite('omi2019fully')}; {cite('shchur2020intensityfree')}.

### 2.2 Numerical event attributes and demand forecasting

Intermittent-demand forecasting has long separated demand occurrence from demand size. Croston estimates these components separately, and subsequent work evaluates Croston-style estimators and develops demand-probability updates that respond to zero-demand periods and obsolescence. These methods motivate the distinction between when demand occurs and how much occurs. {cite('croston1972forecasting')}; {cite('syntetos2005accuracy')}; {cite('teunter2011obsolescence')}.

Deep renewal processes connect that distinction to neural arrival-and-size models, including continuous-time formulations. FlexTPP broadens event modeling to heterogeneous attributes, with discrete heads and normalizing flows for continuous values. Numerical event attributes therefore fit within an established research setting. TitanTPP addresses the representation of gap–quantity histories within this setting, pairing a quantity point estimate with a distribution over the next recorded duration. {cite('turkmen2021renewal')}; {cite('draxler2025flextpp')}.

Neural time-series forecasting provides related approaches to learning across demand histories. DeepAR trains an autoregressive recurrent model across related series for probabilistic forecasting. PatchTST organizes temporal observations into patches with channel-independent processing, while iTransformer attends across variate tokens. These methods illustrate different choices of forecasting target and tokenization. Our event representation concentrates on positive-demand observations and their gaps, and the task predicts the next event's size and duration. Extending it to a fixed calendar horizon would require a rule for accumulating intervening events. {cite('salinas2020deepar')}; {cite('nie2023patchtst')}; {cite('liu2024itransformer')}.

### 2.3 Residual correction and memory access

Residual learning adds a learned transformation to a shortcut path. Bottleneck adapters apply down-projection, a nonlinearity, and up-projection within such a residual, allowing compact task-specific additions to a pretrained network. TitanTPP specializes this structure to event histories by combining the current and immediately preceding contextual states between two jointly trained encoder blocks. Its branches participate according to observed-history availability. {cite('he2016residual')}; {cite('houlsby2019adapters')}.

Initialization determines the residual's initial effect. ReZero starts each residual path with a zero-valued scalar gate. TitanTPP initializes the branch output matrices to zero, so the correction initially preserves the base encoder's function. The availability mask is deterministic, and the correction learns together with the encoder and prediction heads. {cite('bachlechner2021rezero')}.

Memory reading and online parameter adaptation provide distinct computational paths. End-to-end memory networks perform attention-based reads over external memory, whereas Titans introduces neural memory that learns from incoming context at test time. TitanTPP retains learned persistent vectors and a static prototype bank alongside its feed-forward correction, with all parameters fixed during inference. This design motivates the matched comparison against a Titans-MAC event adapter in Section 6, which measures the complete training and evaluation paths. {cite('sukhbaatar2015memory')}; {cite('behrouz2025titans')}.

'''

new = old.split('## 2. Related work', 1)[0] + related + '## 3. Method' + old.split('## 3. Method', 1)[1]
# Keep the introduction's argument and empirical claims; add its classical source.
intro_anchor = 'Renewal-process forecasting connects this event-based view'
new = new.replace(intro_anchor, 'Separating demand size from occurrence has a long history in intermittent-demand forecasting. ' + cite('croston1972forecasting') + '. Renewal-process forecasting connects this event-based view', 1)

replacements = {
    'TitanTPP contains two pre-normalized causal attention blocks.':
        'TitanTPP contains two pre-normalized causal attention blocks, built from multi-head attention and layer normalization. ' + cite('vaswani2017attention') + '; ' + cite('ba2016layernorm') + '.',
    'GELU follows the exact error-function formulation.':
        'GELU follows the exact error-function formulation. ' + cite('hendrycks2016gelu') + '.',
    'We optimize with AdamW at a learning rate':
        'We optimize with AdamW (' + cite('loshchilov2019adamw') + ') at a learning rate',
    'We compare the event representations of RMTPP, NHP, THP, SAHP, S2P2, and AttNHP under a shared duration–quantity objective.':
        'Reproducible TPP benchmarking benefits from explicit data interfaces and evaluation programs, as developed in EasyTPP. ' + cite('xue2024easytpp') + '. Our comparison isolates event representations through a shared duration–quantity objective for RMTPP, NHP, THP, SAHP, S2P2, and AttNHP.',
    'S2P2 is pinned to `UCIDataLab/':
        'The retained HiPPO component follows the polynomial-projection framework of ' + cite('gu2020hippo') + '. S2P2 is pinned to `UCIDataLab/',
    'All three metrics are evaluated at the same RMSE-selected checkpoint.':
        'All three metrics are evaluated at the same RMSE-selected checkpoint. MAE and RMSE retain the quantity scale, so comparisons are made within each dataset rather than by pooling raw errors across datasets. ' + cite('hyndman2006accuracy') + '. Duration NLL evaluates the probability assigned to the recorded observation through the logarithmic score. ' + cite('gneiting2007proper') + '.',
    'The measurement runtime uses Python 3.12.13, PyTorch 2.11.0+cu130,':
        'The measurement runtime uses Python 3.12.13, PyTorch 2.11.0+cu130 (' + cite('paszke2019pytorch') + '),',
}
for before, after in replacements.items():
    assert new.count(before) == 1, before
    new = new.replace(before, after, 1)

# One stable URL for each cited work across the body and bibliography.
new = new.replace('](https://arxiv.org/abs/2412.19634)', '](https://arxiv.org/abs/2412.19634v2)')
body = new.split('## References', 1)[0]
refs_ordered = sorted(records, key=lambda r: body.index('](' + r['url'] + ')'))
assert len(refs_ordered) == 32
old_bib = (BACKUP / 'references.bib').read_text()
for r in refs_ordered:
    if 'citation_label' not in r:
        r['citation_label'] = labels[r['key']]
    if 'entry_type' not in r:
        m = re.search(r'@(\w+)\{' + re.escape(r['key']) + r',', old_bib)
        r['entry_type'] = m.group(1)
        for field in ['volume', 'number', 'pages', 'series']:
            entry = old_bib[m.start():].split('\n}', 1)[0]
            value = re.search(r'^  ' + field + r' = \{(.*)\}', entry, re.M)
            if value:
                r[field] = value.group(1)
    if r in additions:
        r['source_type'] = 'primary_paper_or_official_proceedings'
        r['verified_on'] = '2026-10-01'

ref_lines = []
for index, r in enumerate(refs_ordered, 1):
    authors = ', '.join(r['authors'][:-1]) + ', and ' + r['authors'][-1] if len(r['authors']) > 2 else ' and '.join(r['authors'])
    details = r['venue']
    if r.get('volume'):
        details += ' ' + r['volume']
    if r.get('number'):
        details += '(' + r['number'] + ')'
    if r.get('pages'):
        details += ':' + r['pages'].replace('--', '–')
    ref_lines.append(f"{index}. {authors} ({r['year']}). *{r['title']}*. {details}. [Source]({r['url']}).")
new = body + '## References\n\n' + '\n'.join(ref_lines) + '\n\n## Internal' + new.split('## Internal', 1)[1]
MAIN.write_text(new)

registry = {**original_registry, 'updated_utc': datetime.now(timezone.utc).isoformat(),
    'records': refs_ordered, 'reference_count': len(refs_ordered),
    'scope': 'Primary-source-backed references attached to manuscript passages; targeted coverage expansion, not an exhaustive novelty review',
    'prior_registry': str((BACKUP / 'references.json').relative_to(ROOT))}
(OUT / 'references.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n')

# BibTeX keeps UTF-8 author names and protected titles for modern bibliography tools.
entries = []
for r in refs_ordered:
    fields = {'title': '{' + r['title'] + '}', 'author': ' and '.join(r['authors']), 'year': str(r['year'])}
    fields['journal' if r['entry_type']=='article' else ('howpublished' if r['entry_type']=='misc' else 'booktitle')] = r['venue']
    for field in ['volume', 'number', 'pages', 'series', 'doi', 'url']:
        if r.get(field):
            fields[field] = str(r[field])
    entries.append('@' + r['entry_type'] + '{' + r['key'] + ',\n' + ',\n'.join('  ' + k + ' = {' + v + '}' for k,v in fields.items()) + '\n}')
(OUT / 'references.bib').write_text('\n\n'.join(entries) + '\n')

change = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'written_pending_verification',
    'before_count': len(original), 'after_count': len(refs_ordered),
    'added_keys': [r['key'] for r in additions],
    'manuscript_before_sha256': hashlib.sha256(old.encode()).hexdigest(),
    'manuscript_after_sha256': hashlib.sha256(new.encode()).hexdigest(),
    'sections_rewritten': ['2.1', '2.2', '2.3'],
    'citation_or_context_additions': ['1', '3.2', '3.3', '3.6', '4.2', '4.3', '6.1'],
    'literal_replacements': replacements,
    'new_experiments_or_remote_observations': False}
(OUT / 'reference_expansion_changes.json').write_text(json.dumps(change, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'reference_count':len(refs_ordered),'added':len(additions),'written':str(MAIN)},ensure_ascii=False))
