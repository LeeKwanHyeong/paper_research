# Introduction

Intermittent demand, urban mobility, and online basket activity produce sequences of positive
events separated by periods with no observed demand. Conventional grid-based forecasting retains
every empty interval, even when the operational question concerns the timing and size of the next
positive demand. An event-based representation instead records each occurrence as
\(e_i=(t_i,q_i)\), where \(t_i\) is the event time and \(q_i>0\) is its quantity. This view is
consistent with renewal-based treatments of intermittent demand, which model nonzero arrivals and
their sizes rather than treating the intervening zeros as independent targets (Turkmen et al.,
2021). It also leads to a direct forecasting task: estimate when the next event will arrive and how
large it will be.

Marked temporal point processes provide a natural probabilistic model for this task, but demand
quantities expose two limitations in common formulations. RMTPP encodes the observed history in a
fixed-dimensional recurrent state and derives the next-mark distribution and conditional time
density from that state (Du et al., 2016). Such compression is efficient, yet distant interactions
must remain accessible through a single recurrent summary. Attention-based TPPs were introduced in
part to address this restriction (Zuo et al., 2020). A separate problem arises at the output: the
standard categorical mark identifies an event type but does not preserve variation within a demand
category. Mixed-type event research likewise distinguishes discrete event attributes from
continuous measurements and assigns them different predictive heads (Shchur et al., 2021;
*Transformers for Mixed-type Event Sequences*, 2025). Demand forecasting therefore requires both
an adequate history representation and a quantity interface that does not collapse continuous
magnitude into a class label.

We formulate positive demand as a magnitude-factorized marked temporal point process. For a base
\(b>1\), each quantity is decomposed into a coarse magnitude mark
\(m_i=\min(\lfloor\log_b q_i\rfloor,M)\) and a continuous residual
\(r_i=\log_b q_i-m_i\). The original quantity follows exactly from
\(q_i=b^{m_i+r_i}\), including observations assigned to the clipped tail mark. This factorization
compresses a wide quantity range in the logarithmic domain while retaining an interpretable demand
scale. More importantly, residual regression preserves differences among quantities that share the
same mark. The model can then predict the next mark, inter-event time, and continuous quantity
without forcing a single output type to serve two distinct roles.

TitanTPP couples this quantity formulation with a causal memory encoder. Each observed event is
represented by its mark, transformed inter-event time, and quantity residual. Two causal
memory-attention blocks encode the sequence, and a learnable local memory bank retrieves vectors
that match the current event representation. Mark, time, and residual heads operate on the resulting
history state. The residual and mark outputs are also combined in a differentiable expected-quantity
decoder, which permits direct quantity supervision without replacing the marked point-process
likelihood. On the Taxi dataset, a scale-aware variant assigns one residual expert to each predicted
mark and stops the quantity-loss gradient at the mark-probability gate. This routing preserves the
categorical objective while allowing the residual experts to model scale-dependent quantity
patterns. The design draws on memory-based sequence modeling (Behrouz et al., 2025), although the
frozen TitanTPP configuration employs static learned memory rather than test-time parameter updates.

We evaluate the formulation on three datasets with different sequence and quantity distributions.
The Intermittent dataset contains many short part-level histories and a small number of extreme
demand events. Taxi provides the longest histories and a broad quantity range, whereas Instacart
contains many user sequences with a narrower basket-size distribution. The comparison separates
three questions. RMTPP-original tests the legacy quantity interface, RMTPP-matched controls for the
new quantity supervision under a recurrent encoder, and THP-matched supplies an attention-based
point-process baseline. TitanTPP then tests the memory encoder under the same split and quantity
contract. A Taxi V2 control isolates the effect of the V3b expert head and gradient routing. Final
performance statements will be inserted after all strict validation runs, continuation decisions,
and the single held-out evaluation are complete.

This study makes three contributions. First, it defines a magnitude-factorized event formulation
that combines categorical demand scale with continuous within-scale variation. Second, it introduces
TitanTPP, a causal memory-augmented TPP with differentiable quantity reconstruction and a scale-aware
Taxi variant. Third, it establishes a fixed-split comparison across complementary demand regimes,
with separate analyses of event likelihood, timing, quantity reconstruction, history length, and
model specialization. These components permit the effect of quantity reformulation to be separated
from the effect of the sequence encoder.

## Draft status

- Result-dependent performance claims are intentionally absent.
- Final citations will be converted to the paper's bibliography format after the template is fixed.
- Figure F1 follows the formulation paragraph; F2 can open the methodology section, and F3 can open
  the experiment section.
