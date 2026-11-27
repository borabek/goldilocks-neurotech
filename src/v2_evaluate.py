"""Frozen v2 evaluation. Never fits a model/calibrator or changes thresholds."""
import argparse,json,time,hashlib
import numpy as np,pandas as pd,joblib,mne
from scipy.signal import resample_poly
from sklearn.metrics import log_loss
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared core modules sit at the top level
from pipeline import ROOT,epoch_features,stage_metrics
from v2_model import probabilities,scale,OUT

def extract_external(rec):
    folder=ROOT/'data/external_sleepedf';cache=folder/'features'/f"{rec['subject']}.npz"
    if cache.exists():return dict(np.load(cache))
    raw=mne.io.read_raw_edf(folder/rec['psg'],preload=False,verbose='ERROR')
    assert raw.info['sfreq']==100
    assert 'EEG Fpz-Cz' in raw.ch_names
    annotations=mne.read_annotations(folder/rec['hypnogram']);raw.set_annotations(annotations,verbose='ERROR')
    mapping={'Sleep stage W':0,'Sleep stage 1':1,'Sleep stage 2':2,'Sleep stage 3':3,'Sleep stage 4':3,'Sleep stage R':4}
    events,_=mne.events_from_annotations(raw,event_id=mapping,chunk_duration=30,verbose='ERROR')
    event_starts=events[:,0]-raw.first_samp
    assert np.all(event_starts%3000==0),'Scoring grid must align to raw 30-second epochs'
    starts=np.arange(raw.n_times//3000)*3000
    y=np.full(len(starts),8,dtype='int8')
    event_index=event_starts//3000;within=(event_index>=0)&(event_index<len(starts));y[event_index[within]]=events[within,2]
    features=[];quality=[]
    assert np.all(starts>=0) and np.all(starts+3000<=raw.n_times)
    for i in range(0,len(starts),32):
        ep=np.stack([raw.get_data(picks=['EEG Fpz-Cz'],start=int(a),stop=int(a+3000))*1e6 for a in starts[i:i+32]])
        good=np.isfinite(ep).all(axis=(1,2))&(np.std(ep,axis=-1).min(axis=1)>1e-6)
        ep=resample_poly(np.nan_to_num(ep),up=64,down=25,axis=-1)
        assert ep.shape[-1]==7680
        features.append(epoch_features(ep,256));quality.extend(good.tolist())
    result={'X':np.concatenate(features),'y':y,'good':np.array(quality),'onset':starts/100.0}
    cache.parent.mkdir(exist_ok=True);np.savez_compressed(cache,**result)
    print(json.dumps({'external_features':rec['subject'],'raw_grid_epochs':len(y),'valid_reference_epochs':int(np.sum(y<5)),'channel':'EEG Fpz-Cz','source_hz':100,'feature_hz':256,'min_onset':float(result['onset'].min()),'max_onset':float(result['onset'].max())}),flush=True)
    return result

def assess(z,frozen,mask):
    p=probabilities(frozen['model'],z['X'],z['good'],frozen['transition'],frozen['prior'],frozen['weight'],frozen['temperature'],z['onset']);guess=p.argmax(axis=1)
    m=mask&z['good']&(z['y']<5);assert m.any()
    stat=stage_metrics(z['y'][m],guess[m]);tst=float(np.sum((guess>0)&m)*.5);ref=float(np.sum((z['y']>0)&m)*.5)
    stat.update(reference_tst_minutes=ref,model_tst_minutes=tst,tst_error_minutes=abs(tst-ref),available_epochs=int(m.sum()),signal_coverage_of_scored=float(m.sum()/mask.sum()),log_loss=float(log_loss(z['y'][m],p[m],labels=list(range(5)))))
    for name,half in [('global',frozen['global_q']),('normalized',frozen['normalized_q']*scale(p,m))]:
        lo=max(0,tst-half);hi=min(m.sum()*.5,tst+half)
        stat.update({name+'_lower':float(lo),name+'_upper':float(hi),name+'_covered':bool(lo<=ref<=hi),name+'_width':float(hi-lo)})
    return stat,p,guess,m

def summarize(rows):
    keys=['accuracy','kappa','macro_f1','tst_error_minutes','log_loss','global_covered','global_width','normalized_covered','normalized_width','signal_coverage_of_scored']
    result={'people':len(rows)}
    for key in keys:result[key]=float(np.mean([r[key] for r in rows]))
    for method in ['global','normalized']:
        success=sum(r[method+'_covered'] for r in rows);n=len(rows);ph=success/n;z=1.95996398454;den=1+z*z/n;center=(ph+z*z/(2*n))/den;half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        result[method+'_coverage_wilson95']=[float(center-half),float(center+half)]
    return result

def main(external):
    frozen=joblib.load(OUT/'frozen_model.joblib'); freeze=json.loads((OUT/'freeze_record.json').read_text());assert hashlib.sha256((OUT/'frozen_model.joblib').read_bytes()).hexdigest()==freeze['model_sha256']
    output=[];predictions={};secondary=[]
    if external:
        folder=ROOT/'data/external_sleepedf';selection=json.loads((folder/'selection.json').read_text())['records'];label_manifest=json.loads((folder/'label_acquisition.json').read_text())
        assert all(r['acquired_unix']>=freeze['frozen_unix'] for r in label_manifest),'References accessed before freeze'
        records=[{'subject':r['subject'],'record':r} for r in selection]
    else:records=[r for r in json.loads((ROOT/'protocol.json').read_text())['records'] if r['split']=='test' and r['confirmatory_night']]
    for r in records:
        z=extract_external(r['record']) if external else dict(np.load(ROOT/'data/features'/f"{r['subject']}.npz"))
        mask=z['y']<5;stats,p,guess,valid=assess(z,frozen,mask);stats['subject']=r['subject']
        if external:
            baseline=np.full(valid.sum(),int(np.argmax(frozen['prior'])));stats['training_majority_kappa']=stage_metrics(z['y'][valid],baseline)['kappa']
            sleep=z['onset'][(z['y']>0)&(z['y']<5)]
            if len(sleep):
                crop=mask&(z['onset']>=sleep.min()-1800)&(z['onset']<=sleep.max()+1800);s,_,_,_=assess(z,frozen,crop);secondary.append({'subject':r['subject'],**s})
        else:
            common=valid&(z['vendor']>=0)&(z['vendor']<5)
            stats['common_model_tst_error_minutes']=float(abs(np.sum((guess>0)&common)*.5-np.sum((z['y']>0)&common)*.5))
            stats['common_vendor_tst_error_minutes']=float(abs(np.sum((z['vendor']>0)&common)*.5-np.sum((z['y']>0)&common)*.5))
            stats['vendor_kappa_on_same_usable_mask']=stage_metrics(z['y'][valid],z['vendor'][valid])['kappa']
        output.append(stats);predictions[r['subject']]={'onset':z['onset'].tolist(),'reference':z['y'].tolist(),'model':guess.tolist(),'confidence':p.max(axis=1).round(4).tolist(),'good':z['good'].tolist()}
        print(json.dumps({'evaluated':r['subject'],'kappa':stats['kappa'],'tst_error':stats['tst_error_minutes']}),flush=True)
    name='external' if external else 'historical_boas'
    report={'cohort':name,'scope':'Independent cohort, different montage/scorer/recording window; a domain-shift evaluation, not same-device or ADHD validation.' if external else 'Previously inspected v1 cohort; development comparison, not new independent confirmation.','model_sha256':freeze['model_sha256'],'model_frozen_unix':freeze['frozen_unix'],'results':summarize(output),'nominal_interval_coverage':.9,'interval_warning':'No external exchangeability guarantee; report empirical coverage and width, including failure.','reference_labels_opened_after_freeze':external}
    if external:report['secondary_reference_defined_sleep_window']=summarize(secondary)
    else:
        report['same_mask_tst']={'model':float(np.mean([r['common_model_tst_error_minutes'] for r in output])),'vendor':float(np.mean([r['common_vendor_tst_error_minutes'] for r in output]))}
    (OUT/(name+'_metrics.json')).write_text(json.dumps(report,indent=2));pd.DataFrame(output).to_csv(OUT/(name+'_people.csv'),index=False);(OUT/(name+'_predictions.json')).write_text(json.dumps(predictions))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--external',action='store_true');args=parser.parse_args();main(args.external)
