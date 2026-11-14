"""Channel-shared retrospective temporal EEG model. V1 artefacts are never overwritten."""
from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd,joblib
from scipy.special import logsumexp,softmax
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score,f1_score,log_loss
from pipeline import ROOT,stage_metrics

OUT=ROOT/'results/v2';OUT.mkdir(parents=True,exist_ok=True)
SEED=20261108

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))

def segments(good,onset=None):
    idx=np.flatnonzero(good)
    if len(idx)==0:return []
    breaks=np.diff(idx)>1
    if onset is not None:breaks|=np.abs(np.diff(onset[idx])-30)>1e-5
    return np.split(idx,np.flatnonzero(breaks)+1)

def temporal(x,good,onset=None):
    """Channel-shared 14 features + centered 5-epoch mean/std + 15-epoch mean.
    Future context makes this retrospective. No reference labels enter this function.
    """
    x=x.reshape(len(x),-1,14);out=np.zeros((len(x),x.shape[1],56),dtype='float32')
    for ids in segments(good,onset):
        z=x[ids];parts=[z]
        for width in [5,15]:
            pad=np.pad(z,((width//2,width//2),(0,0),(0,0)),mode='edge')
            win=np.lib.stride_tricks.sliding_window_view(pad,width,axis=0)
            parts.append(win.mean(axis=-1))
            if width==5:parts.append(win.std(axis=-1))
        out[ids]=np.concatenate(parts,axis=-1)
    return out

def smooth(p,good,trans,prior,weight,onset=None):
    if weight==0:return p
    result=p.copy();logt=np.log(trans)*weight
    for ids in segments(good,onset):
        emission=np.log(p[ids]+1e-12)-weight*np.log(prior[None,:])
        a=np.empty_like(emission);b=np.zeros_like(emission);a[0]=np.log(prior)+emission[0]
        for t in range(1,len(ids)):a[t]=emission[t]+logsumexp(a[t-1][:,None]+logt,axis=0)
        for t in range(len(ids)-2,-1,-1):b[t]=logsumexp(logt+emission[t+1][None,:]+b[t+1][None,:],axis=1)
        result[ids]=softmax(a+b,axis=1)
    return result

def probabilities(model,x,good,trans,prior,weight,temperature=1,onset=None):
    xx=temporal(x,good,onset);n,c,f=xx.shape
    p=model.predict_proba(xx.reshape(n*c,f)).reshape(n,c,5).mean(axis=1)
    p=smooth(p,good,trans,prior,weight,onset)
    return softmax(np.log(p+1e-12)/temperature,axis=1)

def scale(p,mask):
    # A heteroscedastic scale proxy, not a claim of independent epoch errors.
    return max(5.0,float(.5*np.sqrt(np.sum(p[mask,0]*(1-p[mask,0])))))

def conformal_q(errors,alpha=.1):
    rank=int(np.ceil((len(errors)+1)*(1-alpha)))
    return float(np.sort(errors)[rank-1]) if rank<=len(errors) else None

def train():
    v1=json.loads((ROOT/'protocol.json').read_text());rows=v1['records']
    val_ids=sorted({r['pid'] for r in rows if r['split']=='probability_validation'},key=int)
    shuffled=np.random.default_rng(SEED).permutation(val_ids).tolist();selection=set(shuffled[:6]);probcal=set(shuffled[6:])
    selection_data=json.loads((ROOT/'data/external_sleepedf/selection.json').read_text())
    protocol={'version':'2','seed':SEED,'created_unix':time.time(),'fit_people':48,'model_selection_pids':sorted(selection),'probability_calibration_pids':sorted(probcal),'summary_calibration':'original 20 BOAS people, first night','historical_test':'original 20 BOAS people, already inspected at v1; not independent confirmation','new_external_test':selection_data,'external_scope':'Sleep-EDF Fpz-Cz only, different device/montage/R&K labels; external domain-shift test, not ADHD or in-domain headband confirmation','candidate_sequence_weights':[0,.5,1.0],'classifier':{'max_iter':180,'max_leaf_nodes':31,'learning_rate':.08,'l2_regularization':2.0,'max_bins':128},'features':'channel-shared 14 spectral/statistical inputs with centered 5/15 epoch context, 56 values per channel; channel probabilities averaged','selection':'mean person kappa on six selection participants; no old test, summary calibration or external outcomes used','intervals':'both fixed global and normalized 90% split-conformal absolute residual intervals, 20 BOAS calibration people; no tuning to attain test coverage','external_windows':['all valid scored epochs','secondary sleep span with 30-minute margins, defined using reference for evaluation only'],'external_label_mapping':'W=0,1=1,2=2,3/4=3,R=4; movement/unscored excluded, no Wake imputation'}
    path=ROOT/'v2_protocol.json'
    if path.exists():raise RuntimeError('V2 protocol already exists; do not silently rerun/reselect. Use saved artefacts or explicitly version a new development experiment.')
    path.write_text(json.dumps(protocol,indent=2))
    record={r['subject']:dict(np.load(ROOT/'data/features'/f"{r['subject']}.npz")) for r in rows}
    X=[];Y=[];trans=np.ones((5,5),float);prior=np.ones(5,float)
    for r in rows:
        if r['split']!='fit':continue
        z=record[r['subject']];good=z['good'];mask=(z['y']<5)&good
        xx=temporal(z['X'],good,z['onset']);X.append(xx[mask].reshape(-1,56));Y.append(np.repeat(z['y'][mask],2))
        prior+=np.bincount(z['y'][mask],minlength=5)
        adjacent=mask[1:]&mask[:-1]
        np.add.at(trans,(z['y'][:-1][adjacent],z['y'][1:][adjacent]),1)
    prior/=prior.sum();trans/=trans.sum(axis=1,keepdims=True)
    model=HistGradientBoostingClassifier(**protocol['classifier'],random_state=SEED,early_stopping=False)
    model.fit(np.concatenate(X),np.concatenate(Y))
    print('TEMPORAL_FIT_COMPLETE',flush=True)
    candidates=[]
    for weight in protocol['candidate_sequence_weights']:
        scores=[]
        for r in rows:
            if r['pid'] not in selection or not r['confirmatory_night']:continue
            z=record[r['subject']];mask=(z['y']<5)&z['good'];p=probabilities(model,z['X'],z['good'],trans,prior,weight,onset=z['onset'])
            scores.append(float(cohen_kappa_score(z['y'][mask],p[mask].argmax(axis=1))))
        candidates.append({'sequence_weight':weight,'mean_person_kappa':float(np.mean(scores))})
    chosen=max(candidates,key=lambda r:r['mean_person_kappa'])['sequence_weight']
    probs=[];targets=[]
    for r in rows:
        if r['pid'] not in probcal or not r['confirmatory_night']:continue
        z=record[r['subject']];mask=(z['y']<5)&z['good'];probs.append(probabilities(model,z['X'],z['good'],trans,prior,chosen,onset=z['onset'])[mask]);targets.append(z['y'][mask])
    def objective(t):return np.mean([log_loss(y,softmax(np.log(p+1e-12)/t,axis=1),labels=list(range(5))) for p,y in zip(probs,targets)])+.02*np.log(t)**2
    temperature=float(minimize_scalar(objective,bounds=(.5,3),method='bounded').x)
    residual=[];normalized=[]
    for r in rows:
        if r['split']!='summary_calibration' or not r['confirmatory_night']:continue
        z=record[r['subject']];mask=(z['y']<5)&z['good'];p=probabilities(model,z['X'],z['good'],trans,prior,chosen,temperature,z['onset']);e=abs(float(np.sum((p.argmax(axis=1)>0)&mask)*.5-np.sum((z['y']>0)&mask)*.5));residual.append(e);normalized.append(e/scale(p,mask))
    frozen={'model':model,'transition':trans,'prior':prior,'weight':chosen,'temperature':temperature,'global_q':conformal_q(residual),'normalized_q':conformal_q(normalized),'protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_class':'channel-shared retrospective temporal feature model'}
    joblib.dump(frozen,OUT/'frozen_model.joblib')
    model_hash=hashlib.sha256((OUT/'frozen_model.joblib').read_bytes()).hexdigest()
    save('freeze_record.json',{'frozen_unix':time.time(),'model_sha256':model_hash,'protocol_sha256':frozen['protocol_sha256'],'external_reference_labels_accessed':False,'sequence_candidates':candidates,'selected_weight':chosen,'temperature':temperature,'global_q_minutes':frozen['global_q'],'normalized_q':frozen['normalized_q']})
    print(json.dumps({'selected':chosen,'temperature':temperature,'candidates':candidates,'model_sha256':model_hash}),flush=True)

if __name__=='__main__':train()
