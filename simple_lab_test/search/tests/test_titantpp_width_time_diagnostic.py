"""Guard against tail omission and misleading independent time attribution."""
import numpy as np
import pytest
from paper.scripts.diagnose_titantpp_width_time_train_validation import (
    stratified_sample, width_parameter_decomposition,
)


def test_sample_keeps_equality_lower_cell_and_sparse_long_gap_tail():
    q=np.array([2.,2.,3.,4.,9.,9.]);dt=np.array([1.,1.,2.,2.,60.,60.])
    ids,w,cells=stratified_sample(q,dt,[2.,4.],seed=12,cap=1)
    assert len(ids)==3 and len(set(ids))==3 and np.all(np.diff(ids)>0)
    assert w.sum()==6
    assert any(q[i]==9. and dt[i]==60. for i in ids)
    assert next(c for c in cells if c['quantity_bin']==0 and c['gap_bin']==0)['population_n']==2
    assert next(c for c in cells if c['quantity_bin']==1 and c['gap_bin']==1)['population_n']==2
    assert next(c for c in cells if c['quantity_bin']==2 and c['gap_bin']==4)['population_n']==2
    repeat=stratified_sample(q,dt,[2.,4.],seed=12,cap=1)
    assert np.array_equal(ids,repeat[0]) and np.array_equal(w,repeat[1])


def test_complete_strata_have_unit_weights_and_keep_all_targets():
    ids,w,_=stratified_sample(np.arange(6.),np.ones(6),[2.,4.],seed=99,cap=10)
    np.testing.assert_array_equal(ids,np.arange(6))
    np.testing.assert_array_equal(w,np.ones(6))


def test_location_scale_interaction_is_retained_in_exact_delta():
    out=width_parameter_decomposition([1.,2.],[8.,5.],[4.,3.],[2.,4.])
    np.testing.assert_allclose(out['total_delta'],[7.,3.])
    np.testing.assert_allclose(out['location_at_sigma4'],[3.,1.])
    np.testing.assert_allclose(out['scale_at_mu4'],[1.,2.])
    np.testing.assert_allclose(out['interaction'],[3.,0.])


@pytest.mark.parametrize('quantity,dt', [([1.],[0.]),([1.],[1.2]),([-1.],[1.]),([np.nan],[1.])])
def test_invalid_population_is_rejected(quantity,dt):
    with pytest.raises(ValueError):stratified_sample(quantity,dt,[2.],seed=1,cap=1)


def test_nonfinite_cross_mass_is_not_clipped():
    with pytest.raises(ValueError):width_parameter_decomposition([1.],[2.],[np.inf],[3.])


def test_design_se_respects_finite_population_and_is_not_naive_event_se():
    from paper.scripts.diagnose_titantpp_width_time_train_validation import stratified_mean_design_se
    # Two sampled events represent four finite-population members in one cell.
    assert stratified_mean_design_se([1.,3.],[2.,2.],[0,0])==pytest.approx(2**-.5)
    assert stratified_mean_design_se([1.,3.],[1.,1.],[0,0])==0.
    assert stratified_mean_design_se([1.],[2.],[0]) is None
