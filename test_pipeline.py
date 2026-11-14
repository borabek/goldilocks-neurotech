import json
import numpy as np
from pipeline import ROOT,freeze,epoch_features,predict_summary

def test_person_independence_and_repeats():
    spec=freeze();r=spec['records']; groups={}
    for row in r:groups.setdefault(row['pid'],set()).add(row['split'])
    assert len(groups)==100 and all(len(g)==1 for g in groups.values())
    assert len(r)==128
    assert sum(x['confirmatory_night'] for x in r)==100

def test_unknown_is_not_wake_or_sleep():
    y=np.array([0,1,8,4]);p=np.array([0,1,2,4]);mask=y<5
    assert predict_summary(y,p,mask)==(1.0,1.0)

def test_spectral_features_detect_frequency_and_are_finite():
    fs=256;t=np.arange(fs*30)/fs
    x=np.stack([np.sin(2*np.pi*2*t),np.sin(2*np.pi*10*t)])
    feat=epoch_features(np.stack([np.stack([a,a]) for a in x]),fs)
    assert np.isfinite(feat).all() and feat.shape==(2,28)
    # channel 1 delta relative power is index 2; alpha relative is index 6.
    assert feat[0,2]>feat[1,2] and feat[1,6]>feat[0,6]
