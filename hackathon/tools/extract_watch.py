import time
from pipeline import ROOT,DATA,freeze,extract
deadline=time.time()+3600
for row in freeze()['records']:
    sub=row['subject'];edf=DATA/sub/'eeg'/f'{sub}_task-Sleep_acq-headband_eeg.edf'
    compact=ROOT/'data/eeg_digital'/f'{sub}.npz'
    while not compact.exists() and not edf.exists():
        if time.time()>deadline:raise TimeoutError(sub)
        time.sleep(1)
    extract(sub)
    if sub!='sub-1' and edf.exists():edf.unlink()
print('EXTRACTION_COMPLETE',flush=True)

