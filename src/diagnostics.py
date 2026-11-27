"""Reporting diagnostics only; does not fit or select models."""
import json
import numpy as np,pandas as pd,joblib
from scipy.special import softmax
from sklearn.metrics import log_loss
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared core modules sit at the top level
from pipeline import ROOT,OUT,stage_metrics
spec=json.loads((ROOT/'protocol.json').read_text()); saved=joblib.load(OUT/'model.joblib')
confidence_rows=[]; availability=[];full=[];errors=[]; calibration=[]
for r in spec['records']:
    if r['split']!='test' or not r['confirmatory_night']:continue
    z=np.load(ROOT/'data/features'/f"{r['subject']}.npz");valid=z['y']<5;m=valid&z['good']
    p=saved['model'].predict_proba(z['X']);q=softmax(np.log(p+1e-12)/saved['temperature'],axis=1);guess=q.argmax(axis=1)
    availability.append({'subject':r['subject'],'human_valid_epochs':int(valid.sum()),'usable_epochs':int(m.sum()),'signal_coverage_of_valid_reference':float(m.sum()/valid.sum())})
    masked=guess.copy();masked[~z['good']]=-2
    for name,pred in [('feature_model',masked),('released_bitbrain',z['vendor'])]:full.append({'subject':r['subject'],'method':name,**stage_metrics(z['y'][valid],pred[valid])})
    for name,prob in [('raw',p),('temperature_scaled',q)]:
        yy=z['y'][m]; pp=prob[m]; hot=np.eye(5)[yy]; conf=pp.max(axis=1);correct=pp.argmax(axis=1)==yy
        ece=0.0
        for j,(lo,hi) in enumerate(zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:])):
            b=(conf>=lo)&((conf<=hi) if j==9 else (conf<hi))
            if b.any():ece+=float(b.mean()*abs(conf[b].mean()-correct[b].mean()))
        calibration.append({'subject':r['subject'],'probabilities':name,'log_loss':float(log_loss(yy,pp,labels=list(range(5)))),'multiclass_brier':float(np.mean(np.sum((pp-hot)**2,axis=1))),'confidence_ece_10bins':ece})
    for threshold in [.0,.5,.6,.7,.8,.9]:
        keep=m&(q.max(axis=1)>=threshold)
        confidence_rows.append({'subject':r['subject'],'threshold':threshold,'coverage_of_usable':float(keep.sum()/m.sum()),'accuracy_retained':float(np.mean(guess[keep]==z['y'][keep])) if keep.any() else None,'retained_epochs':int(keep.sum())})
    for i in np.flatnonzero(m&(guess!=z['y'])):
        key=(int(z['y'][i]),int(guess[i]))
        if len(errors)<3 and key not in [(e['reference'],e['prediction']) for e in errors]:errors.append({'subject':r['subject'],'epoch':int(i),'onset_seconds':float(z['onset'][i]),'reference':key[0],'prediction':key[1],'confidence':float(q[i].max()),'interpretation':'Observed stage disagreement; no causal artefact explanation has been established.'})
pd.DataFrame(availability).to_csv(OUT/'availability.csv',index=False);pd.DataFrame(full).to_csv(OUT/'full_reference_stage.csv',index=False);pd.DataFrame(calibration).to_csv(OUT/'probability_calibration.csv',index=False);pd.DataFrame(confidence_rows).to_csv(OUT/'confidence_coverage.csv',index=False)
(OUT/'failure_gallery.json').write_text(json.dumps(errors,indent=2))
df=pd.DataFrame(calibration);a=pd.DataFrame(availability)
diagnostic={'description':'Secondary diagnostics of frozen predictions, not a new tuned model. Full-reference metrics count unavailable model epochs as abstentions.','mean_signal_coverage_of_valid_reference':float(a.signal_coverage_of_valid_reference.mean()),'probability_metrics':df.groupby('probabilities')[['log_loss','multiclass_brier','confidence_ece_10bins']].mean().to_dict('index'),'failure_selection':'First three distinct confusion types in ascending source-record order, not worst/best case selection.'}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2));print(json.dumps(diagnostic,indent=2))
