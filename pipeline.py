"""Reproducible BOAS feature floor, participant-separated calibration and evaluation."""
from pathlib import Path
import argparse, json, hashlib, time, sys, platform
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix, log_loss
import joblib

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/boas'
OUT=ROOT/'results'
SEED=20261107
LABELS=['Wake','N1','N2','N3','REM']

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def freeze():
    people=pd.read_csv(DATA/'participants.tsv',sep='\t',dtype={'pid':str})
    ids=sorted(people.pid.unique(),key=int)
    rng=np.random.default_rng(SEED); shuffled=rng.permutation(ids).tolist()
    assert len(ids)==100
    groups={pid:('fit' if i<48 else 'probability_validation' if i<60 else 'summary_calibration' if i<80 else 'test') for i,pid in enumerate(shuffled)}
    records=[]
    for row in people.to_dict('records'):
        sub=row['participant_id']; records.append({'subject':sub,'pid':row['pid'],'split':groups[row['pid']]})
    records.sort(key=lambda x:int(x['subject'].split('-')[1]))
    first={}
    for r in records:first.setdefault(r['pid'],r['subject'])
    for r in records:r['confirmatory_night']=r['subject']==first[r['pid']]
    spec={'seed':SEED,'pin':'face09565c4337ddbce7a048c9b02060ad20df45','records':records,
          'model':{'max_iter':100,'max_leaf_nodes':15,'learning_rate':0.08,'l2_regularization':1.0,'max_bins':128},
          'probability_calibration':'single temperature fitted on probability_validation people only',
          'summary_interval':'90% split conformal absolute TST residual; first night per person; empirical coverage/width reported',
          'headline':'first-night-per-test-person total-sleep-time MAE, same valid reference/signal mask',
          'stage_metrics':['person-averaged kappa','person-averaged macro F1','pooled metrics secondary'],
          'stage_target':'PSG stage_hum only','baseline':'released headband stage_ai and training-majority',
          'time_definition':'recording-window quantities; no clinical lights-off latency',
          'tuning':'fixed parameters, no final-test tuning','created_unix':time.time()}
    path=ROOT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text());assert existing['records']==records,'Frozen split changed';return existing
    write_json(path,spec)
    return spec

def epoch_features(x,fs):
    """x: epochs x channels x samples, physical microvolts. No labels/context."""
    x=x-np.mean(x,axis=-1,keepdims=True)
    freq,p=welch(x,fs=fs,nperseg=int(fs*4),axis=-1)
    keep=(freq>=0.5)&(freq<=30)
    total=np.trapezoid(p[...,keep],freq[keep],axis=-1)+1e-12
    feats=[np.log(total)]
    for lo,hi in [(0.5,4),(4,8),(8,12),(12,16),(16,30)]:
        mask=(freq>=lo)&(freq<hi); power=np.trapezoid(p[...,mask],freq[mask],axis=-1)+1e-12
        feats.extend([np.log(power),power/total])
    prob=p[...,keep]/(p[...,keep].sum(axis=-1,keepdims=True)+1e-12)
    feats += [-(prob*np.log(prob+1e-12)).sum(axis=-1)/np.log(prob.shape[-1]),np.log(np.std(x,axis=-1)+1e-9),np.log(np.mean(np.abs(np.diff(x,axis=-1)),axis=-1)+1e-9)]
    return np.stack(feats,axis=-1).reshape(len(x),-1).astype('float32')

def extract(sub):
    cache=ROOT/'data/features'/f'{sub}.npz'
    if cache.exists() and (ROOT/'data/loader_audits'/f'{sub}.json').exists() and (ROOT/'data/eeg_digital'/f'{sub}.npz').exists():return
    base=DATA/sub/'eeg'/f'{sub}_task-Sleep_acq-'
    eeg=Path(str(base)+'headband_eeg.edf')
    human=pd.read_csv(str(base)+'psg_events.tsv',sep='\t')
    vendor=pd.read_csv(str(base)+'headband_events.tsv',sep='\t')
    assert len(human)==len(vendor)
    assert np.allclose(human.onset,vendor.onset) and np.allclose(human.duration,30)
    raw=mne.io.read_raw_edf(eeg,preload=False,verbose='ERROR')
    fs=float(raw.info['sfreq']); names=raw.ch_names
    picks=[names.index(n) for n in ['HB_1','HB_2']]
    assert fs==256,(sub,fs)
    starts=np.rint(human.onset.to_numpy()*fs).astype(int)
    assert np.max(np.abs(starts/fs-human.onset))<1/fs
    assert np.all(starts>=0) and np.all(starts+int(30*fs)<=raw.n_times)
    # BIDS sample columns are one-based inclusive; EDF reader indices are zero-based.
    assert np.array_equal(human.begsample.to_numpy()-1,starts)
    assert np.array_equal(human.endsample.to_numpy(),starts+int(30*fs))
    values=[]; quality=[]
    for i in range(0,len(human),32):
        epochs=np.stack([raw.get_data(picks=picks,start=int(a),stop=int(a+30*fs))*1e6 for a in starts[i:i+32]])
        good=np.isfinite(epochs).all(axis=(1,2))&(np.std(epochs,axis=-1).min(axis=1)>1e-6)
        quality.extend(good.tolist());values.append(epoch_features(np.nan_to_num(epochs),fs))
    features=np.concatenate(values)
    audit={'subject':sub,'channels':[names[p] for p in picks],'sampling_hz':fs,'samples':int(raw.n_times),'duration_seconds':float(raw.n_times/fs),'epochs':len(human),'alignment':'onsets, durations, one-based begin/end sample checked','input_unit':'MNE-calibrated volts converted to microvolts','finite_nonflat_epochs':int(sum(quality))}
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,X=features,y=human.stage_hum.to_numpy().astype('int8'),vendor=vendor.stage_ai.to_numpy().astype('int8'),onset=human.onset.to_numpy(),good=np.array(quality))
    write_json(ROOT/'data/loader_audits'/f'{sub}.json',audit)
    compact_edf(eeg,ROOT/'data/eeg_digital'/f'{sub}.npz')
    print(json.dumps(audit),flush=True)

def compact_edf(source,destination):
    """Lossless task-channel digital samples plus EDF calibration; original URL/hash in acquisition manifest."""
    with source.open('rb') as f:
        header=f.read(256);ns=int(header[252:256]);nrecords=int(header[236:244]);duration=float(header[244:252]);header_bytes=int(header[184:192]);h=f.read(ns*256)
    pos=0
    def fields(width):
        nonlocal pos
        a=[h[pos+i*width:pos+(i+1)*width].decode('ascii').strip() for i in range(ns)];pos+=ns*width;return a
    names=fields(16);fields(80);units=fields(8);pmin=list(map(float,fields(8)));pmax=list(map(float,fields(8)));dmin=list(map(float,fields(8)));dmax=list(map(float,fields(8)));fields(80);samples=list(map(int,fields(8)));fields(32)
    assert source.stat().st_size==header_bytes+nrecords*sum(samples)*2,'EDF byte layout mismatch'
    mmap=np.memmap(source,dtype='<i2',mode='r',offset=header_bytes,shape=(nrecords,sum(samples)))
    pick=[names.index('HB_1'),names.index('HB_2')];offset=np.cumsum([0]+samples)
    digital=np.stack([np.array(mmap[:,offset[i]:offset[i+1]]).reshape(-1) for i in pick])
    assert all(units[i] in ['uV','µV'] for i in pick)
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,digital=digital,physical_min=np.array(pmin)[pick],physical_max=np.array(pmax)[pick],digital_min=np.array(dmin)[pick],digital_max=np.array(dmax)[pick],sfreq=samples[pick[0]]/duration,labels=np.array(['HB_1','HB_2']))
    z=np.load(destination);assert np.array_equal(z['digital'],digital)
    reference=mne.io.read_raw_edf(source,preload=False,verbose='ERROR')
    for at in [0,digital.shape[1]//2,digital.shape[1]-20]:
        values=(digital[:,at:at+20]-z['digital_min'][:,None])*(z['physical_max']-z['physical_min'])[:,None]/(z['digital_max']-z['digital_min'])[:,None]+z['physical_min'][:,None]
        np.testing.assert_allclose(values,reference.get_data(picks=['HB_1','HB_2'],start=at,stop=at+20)*1e6,atol=1e-7)

def stage_metrics(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),'kappa':float(cohen_kappa_score(y,pred,labels=list(range(5))+[-2])),'macro_f1':float(f1_score(y,pred,labels=list(range(5)),average='macro',zero_division=0)),'per_stage_f1':f1_score(y,pred,labels=list(range(5)),average=None,zero_division=0).tolist(),'coverage':float(np.mean(pred>=0))}

def predict_summary(y,pred,mask):
    # Model/reference share exactly the same known-reference, usable-signal mask.
    return float(np.sum((pred>0)&(pred<5)&mask)*0.5),float(np.sum((y>0)&(y<5)&mask)*0.5)

def run():
    spec=freeze(); rows=spec['records']; parts={}; record={}
    for r in rows:
        z=np.load(ROOT/'data/features'/f"{r['subject']}.npz")
        record[r['subject']]={k:z[k] for k in z.files}
        mask=(z['y']>=0)&(z['y']<5)&z['good']
        if r['split']=='fit':parts.setdefault('X',[]).append(z['X'][mask]);parts.setdefault('y',[]).append(z['y'][mask])
    X=np.concatenate(parts['X']); y=np.concatenate(parts['y'])
    model=HistGradientBoostingClassifier(**spec['model'],random_state=SEED,early_stopping=False)
    model.fit(X,y)
    majority=int(np.bincount(y,minlength=5).argmax())
    val=[r for r in rows if r['split']=='probability_validation' and r['confirmatory_night']]
    vp=[];vy=[]
    for r in val:
        z=record[r['subject']];m=(z['y']<5)&z['good'];vp.append(model.predict_proba(z['X'][m]));vy.append(z['y'][m])
    vp=np.concatenate(vp);vy=np.concatenate(vy)
    temp=minimize_scalar(lambda t:log_loss(vy,softmax(np.log(vp+1e-12)/t,axis=1),labels=list(range(5))),bounds=(0.25,4.0),method='bounded').x
    cal_errors=[]; test_rows=[]; predictions={}; stage_rows=[]
    for r in rows:
        if r['split'] not in ['test','summary_calibration'] or not r['confirmatory_night']:continue
        z=record[r['subject']];m=(z['y']>=0)&(z['y']<5)&z['good'];prob=softmax(np.log(model.predict_proba(z['X'])+1e-12)/temp,axis=1);pred=prob.argmax(axis=1)
        tst,ref=predict_summary(z['y'],pred,m)
        if r['split']=='summary_calibration':cal_errors.append(abs(tst-ref));continue
        row={**r,'epochs':int(m.sum()),'unknown_reference_minutes':float(np.sum(z['y']==8)*0.5),'unusable_signal_minutes':float(np.sum(~z['good'])*0.5),'reference_tst_minutes':ref,'model_tst_minutes':tst,'model_absolute_error_minutes':abs(tst-ref)}
        comparisons={'feature_model':pred,'released_bitbrain':z['vendor'],'majority':np.full(len(pred),majority)}
        for name,p in comparisons.items():
            stat=stage_metrics(z['y'][m],p[m]);stage_rows.append({'subject':r['subject'],'pid':r['pid'],'method':name,**stat})
            covered=m&(p>=0)&(p<5)
            row[name+'_conditional_tst_error_minutes']=abs(predict_summary(z['y'],p,covered)[0]-predict_summary(z['y'],p,covered)[1])
            row[name+'_coverage']=float(covered.sum()/m.sum())
        common=m&(z['vendor']>=0)&(z['vendor']<5)
        row['common_mask_model_tst_error_minutes']=abs(predict_summary(z['y'],pred,common)[0]-predict_summary(z['y'],pred,common)[1])
        row['common_mask_vendor_tst_error_minutes']=abs(predict_summary(z['y'],z['vendor'],common)[0]-predict_summary(z['y'],z['vendor'],common)[1])
        predictions[r['subject']]={'pid':r['pid'],'onset':z['onset'].tolist(),'reference':z['y'].tolist(),'model':pred.tolist(),'bitbrain':z['vendor'].tolist(),'confidence':prob.max(axis=1).round(4).tolist(),'good_signal':z['good'].tolist()}
        test_rows.append(row)
    m=len(cal_errors); rank=int(np.ceil((m+1)*0.90));q=float(np.sort(cal_errors)[rank-1]) if rank<=m else None
    for row in test_rows:
        lower=max(0,row['model_tst_minutes']-q) if q is not None else None
        upper=min(row['epochs']*.5,row['model_tst_minutes']+q) if q is not None else None
        row.update(interval_lower=lower,interval_upper=upper,interval_covered=bool(lower<=row['reference_tst_minutes']<=upper) if q is not None else None)
    summary={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'seed':SEED,'fit_people':48,'probability_validation_people':12,'summary_calibration_people':m,'test_people':len(test_rows),'fit_epochs':len(y),'features':X.shape[1],'temperature':float(temp),'nominal_summary_coverage':0.9,'summary_interval_halfwidth_minutes':q,'observed_summary_coverage':float(np.mean([r['interval_covered'] for r in test_rows])) if q is not None else None,'model_tst_mae_minutes':float(np.mean([r['model_absolute_error_minutes'] for r in test_rows])),'mean_interval_width_minutes':float(np.mean([r['interval_upper']-r['interval_lower'] for r in test_rows])) if q is not None else None,'methods':{},'limitations':['Adult BOAS cohort only; no ADHD/medication-response validation.','Retrospective research analysis, not clinical sleep diagnosis.','Intervals calibrated on 20 independent people; empirical coverage has sampling uncertainty.','Vendor training provenance differs; released predictions are a reference comparator.','YASA not run; no fabricated fourth prediction lane.','Quality rule rejects only nonfinite or flat signal; it is not a comprehensive artefact detector.']}
    for name in ['feature_model','released_bitbrain','majority']:
        ss=[r for r in stage_rows if r['method']==name]
        summary['methods'][name]={k:float(np.mean([r[k] for r in ss])) for k in ['accuracy','kappa','macro_f1','coverage']}
    paired=np.array([r['common_mask_model_tst_error_minutes']-r['common_mask_vendor_tst_error_minutes'] for r in test_rows])
    rng=np.random.default_rng(SEED)
    boot=np.mean(rng.choice(paired,size=(2000,len(paired)),replace=True),axis=1)
    summary['common_mask_tst']={'model_mae_minutes':float(np.mean([r['common_mask_model_tst_error_minutes'] for r in test_rows])),'vendor_mae_minutes':float(np.mean([r['common_mask_vendor_tst_error_minutes'] for r in test_rows])),'paired_model_minus_vendor_minutes':float(paired.mean()),'paired_bootstrap_95ci':np.quantile(boot,[0.025,0.975]).tolist(),'mask':'human-valid, both EEG channels nonflat/finite, vendor nonabstaining; same epochs for both'}
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(test_rows).to_csv(OUT/'per_person_sleep.csv',index=False)
    pd.DataFrame(stage_rows).to_csv(OUT/'per_person_stage.csv',index=False)
    write_json(OUT/'metrics.json',summary);write_json(OUT/'predictions.json',predictions)
    joblib.dump({'model':model,'temperature':temp,'feature_protocol':spec,'interval_halfwidth_minutes':q},OUT/'model.joblib')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','extract','train']);parser.add_argument('--subject');args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='extract':
        if args.subject:extract(args.subject)
        else:
            for r in freeze()['records']:extract(r['subject'])
    else:run()
