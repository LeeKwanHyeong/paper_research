# Raw-RMSE baseline completion: seed42 on RTX 5080

This completion campaign fills only the four missing validation rows from the
frozen selector-alignment campaign. The original Instacart result remains a
failed all-dataset gate; this authorization does not relabel or rerun it.

The queue is fixed to Intermittent RMTPP, Intermittent THP, Taxi RMTPP, and
Taxi THP. Every job uses seed 42, at most 300 epochs, at least 40 epochs,
patience 40, and the earliest strict finite validation raw-RMSE minimum. The
training loss, model definitions, time head, quantity head, optimizer, and
data splits are checksum-bound to the original source.

All four jobs run serially on RTX 5080. A finite CUDA forward/backward smoke
for both RMTPP and THP must pass before the first full fit. The queue stops on
the first failure and resumes only when the source, job, data, selector, and
optimizer/RNG checkpoint identity match exactly. The output registry is
separate from the original campaign. Seeds 52/62 and held-out test data remain
outside this campaign.

Expected wall time is dominated by Intermittent. Historical RTX 5080 runs at
the same sample counts and sequence lengths suggest roughly 25–35 minutes for
Intermittent RMTPP and 1.8–2.2 hours for Intermittent THP under the expected
raw-RMSE stopping trajectory. Taxi contributes about 10–25 minutes combined.
Allow 2.5–3.5 hours normally and up to about 9 hours if both Intermittent jobs
consume the full 300-epoch budget.
