"""Package source/results and independent original data without modifying v1 archives."""
from pathlib import Path
import hashlib,json,zipfile

root=Path(__file__).resolve().parent
out=root.parent/'outputs';out.mkdir(exist_ok=True)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in list(root.glob('*.py'))+list(root.glob('*.mjs'))+list(root.glob('*protocol.json'))+[root/'requirements.txt']}
(root/'source_sha256_v2.json').write_text(json.dumps(manifest,indent=2))
paths=[]
with zipfile.ZipFile(out/'Goldilocks_Build_v2.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(root)
        if any(t.startswith('.') or t in ['__pycache__','eeg_digital'] for t in rel.parts):continue
        if p.suffix.lower() in ['.edf','.log','.exit','.pyc','.zip']:continue
        if p.name.startswith(('explorer_','v2_desktop','v2_mobile')):continue
        z.write(p,'goldilocks/'+str(rel))
    z.write(out/'NEURORELEASE_Stream2_ADHD_Nanoparticle_Plan.md','goldilocks/MASTER_PLAN_REV14.md')
paths.append(out/'Goldilocks_Build_v2.zip')
folder=root/'data/external_sleepedf';selected=json.loads((folder/'selection.json').read_text())['records']
for part in [0,1]:
    path=out/f'Goldilocks_External_SleepEDF_part_{part+1:02d}_of_02.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for record in selected[part*10:(part+1)*10]:
            for key in ['psg','hypnogram']:
                p=folder/record[key];z.write(p,'goldilocks/data/external_sleepedf/'+p.name)
        for name in ['SHA256SUMS.txt','LICENSE.html','source_page.html','selection.json','preprocessing_protocol.json','signal_acquisition.json','label_acquisition.json']:
            z.write(folder/name,'goldilocks/data/external_sleepedf/'+name)
        z.writestr('README.txt','Sleep-EDF Expanded 1.0.0, DOI 10.13026/C2X676, Open Data Commons Attribution License v1.0. Original PSG and expert reference EDF files, SHA256 checked against publisher manifest. Independent ZIPs: extract both into the same folder as Goldilocks Build v2. Read README_V2.md for attribution, licence, protocol and limits. Domain-shift research, not ADHD/clinical validation.')
    assert path.stat().st_size<512*1024*1024
    paths.append(path)
for p in paths:
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
summary={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
(out/'Goldilocks_v2_package_checksums.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

