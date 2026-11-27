"""Audit authors' release measurements; separate source-held-out PLGA experiment."""
from pathlib import Path
import json,hashlib,io,time
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/release';OUT=ROOT/'results/release';OUT.mkdir(parents=True,exist_ok=True)
audit={'scope':'PLGA microparticle in-vitro release only. Not hydrogel, brain exposure or ADHD response.','sources':[]}
for p in sorted(DATA.glob('*')):
    if p.suffix not in ['.tsv','.xlsx']:continue
    blob=p.read_bytes();df=pd.read_excel(io.BytesIO(blob),engine='openpyxl')
    item={'file':p.name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'rows':len(df),'columns':df.columns.tolist(),'actual_format':'xlsx ZIP container','missing_cells':int(df.isna().sum().sum())}
    if p.name.startswith('bannigan'):
        item['md5']=hashlib.md5(blob).hexdigest();item['profiles']=int(df.Experimental_index.nunique());item['drug_polymer_groups']=int(df.DP_Group.nunique())
        item['provenance']='Authors GitHub mirror pinned at 0df9c4aafab2c2a41ed3f28752c4ebcb0daf63ab; bytes match Zenodo published MD5.'
        item['license']='Authors repository MIT license, retained locally.'
        item['warning']='Filename says TSV but bytes are Excel. 14/17 variants contain the same outcomes; never append them as independent samples.'
    else:item['license']='CC BY 4.0; DOI 10.17632/zzvtdrcy76.2; original SHA256 verified against Mendeley file manifest.'
    audit['sources'].append(item)

raw=pd.read_excel(DATA/'mp_dataset_initial.xlsx');processed=pd.read_excel(DATA/'mp_dataset_processed.xlsx')
raw['DOI']=raw['DOI'].astype(str).str.strip().str.lower()
ids=raw[['Formulation Index','DOI']].drop_duplicates()
assert ids['Formulation Index'].is_unique,'Ambiguous source mapping'
df=processed.merge(ids,on='Formulation Index',validate='many_to_one')
assert len(df)==len(processed)
sources=sorted(df.DOI.unique());rng=np.random.default_rng(20261107);rng.shuffle(sources)
test_sources=set(sources[:int(np.ceil(.2*len(sources)))])
test=df.DOI.isin(test_sources);features=[c for c in processed.columns if c not in ['Release','Formulation Index']]
protocol={'seed':20261107,'target':'fractional in-vitro release at specified time','features':features,'test_source_dois':sorted(test_sources),'split':'whole DOI, no shared formulation/source','baseline':'training median release in fixed time bins','domain':'small-molecule emulsion PLGA microparticles; not transferable to hydrogels','created_unix':time.time(),'model':{'max_iter':100,'max_leaf_nodes':15,'learning_rate':0.08,'l2_regularization':1.0}}
proto=ROOT/'release_protocol.json'
if proto.exists():
    existing=json.loads(proto.read_text());assert existing['test_source_dois']==protocol['test_source_dois'];protocol=existing
else:proto.write_text(json.dumps(protocol,indent=2))
assert not set(df.loc[test,'Formulation Index'])&set(df.loc[~test,'Formulation Index'])
model=HistGradientBoostingRegressor(**protocol['model'],random_state=20261107,early_stopping=False)
model.fit(df.loc[~test,features],df.loc[~test,'Release'])
pred=model.predict(df.loc[test,features])
bins=np.array([0,.25,1,3,7,14,30,60,90,180,np.inf])
train_bin=np.digitize(df.loc[~test,'Time'],bins);med={int(k):float(np.median(df.loc[~test,'Release'].to_numpy()[train_bin==k])) for k in np.unique(train_bin)}
base=np.array([med.get(int(k),float(df.loc[~test,'Release'].median())) for k in np.digitize(df.loc[test,'Time'],bins)])
evaluation=df.loc[test,['Formulation Index','DOI','Time','Release']].copy();evaluation['model']=pred;evaluation['baseline']=base
evaluation['model_error']=np.abs(pred-evaluation.Release);evaluation['baseline_error']=np.abs(base-evaluation.Release)
curve=evaluation.groupby(['DOI','Formulation Index'])[['model_error','baseline_error']].mean();study=curve.groupby('DOI').mean()
report={'rows':len(df),'formulations':int(df['Formulation Index'].nunique()),'sources':int(df.DOI.nunique()),'train_sources':int(df.loc[~test,'DOI'].nunique()),'test_sources':int(df.loc[test,'DOI'].nunique()),'test_formulations':int(df.loc[test,'Formulation Index'].nunique()),'study_averaged_model_mae_fraction':float(study.model_error.mean()),'study_averaged_baseline_mae_fraction':float(study.baseline_error.mean()),'outside_0_1_observations':int(((df.Release<0)|(df.Release>1)).sum()),'limitations':['Separate PLGA in-vitro experiment; not the neural-stream core.','Literature measurements/digitisation and reporting quality limit interpretation.','No hydrogel, protein-payload, brain exposure, toxicity or clinical validation.','No cross-corpus pooling; overlap with Bannigan has not been ruled out.']}
audit['plga_experiment']=report
(OUT/'audit.json').write_text(json.dumps(audit,indent=2));(OUT/'metrics.json').write_text(json.dumps(report,indent=2))
evaluation.to_csv(OUT/'heldout_release.csv',index=False);df.to_csv(OUT/'plga_measurements_with_source.csv',index=False)
joblib.dump({'model':model,'features':features,'protocol':protocol},OUT/'plga_release_model.joblib')
print(json.dumps(report,indent=2))
