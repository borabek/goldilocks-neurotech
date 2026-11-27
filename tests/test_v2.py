import json
import numpy as np,joblib
from pipeline import ROOT
from v2_model import temporal,probabilities,conformal_q,OUT

def test_temporal_context_cannot_cross_a_signal_gap():
    x=np.zeros((9,28),dtype=float);x[:4]=1;x[5:]=100
    good=np.ones(9,bool);good[4]=False
    t=temporal(x,good)
    assert np.max(t[:4])<=1 and np.min(t[5:,:,:14])==100
    assert np.all(t[4]==0)

def test_calibration_model_selection_and_test_people_are_distinct():
    spec=json.loads((ROOT/'v2_protocol.json').read_text());records=json.loads((ROOT/'protocol.json').read_text())['records']
    a=set(spec['model_selection_pids']);b=set(spec['probability_calibration_pids']);c={r['pid'] for r in records if r['split'] in ['summary_calibration','test','fit']}
    assert len(a)==len(b)==6 and not a&b and not (a|b)&c

def test_conformal_small_sample_rule_does_not_invent_finite_interval():
    assert conformal_q([1,2],alpha=.1) is None
    assert conformal_q(list(range(20)),alpha=.1)==18

def test_channel_order_does_not_change_shared_predictions():
    frozen=joblib.load(OUT/'frozen_model.joblib');z=np.load(ROOT/'data/features/sub-1.npz');x=z['X'][:30];good=z['good'][:30]
    flipped=x.reshape(len(x),2,14)[:,::-1].reshape(len(x),28)
    def predict(v):return probabilities(frozen['model'],v,good,frozen['transition'],frozen['prior'],frozen['weight'],frozen['temperature'])
    np.testing.assert_allclose(predict(x),predict(flipped),atol=1e-12)
