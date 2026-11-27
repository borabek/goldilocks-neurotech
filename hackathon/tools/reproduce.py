"""Reproduce the frozen feature-based experiment offline from included features."""
from pathlib import Path
import os,subprocess,sys
root=Path(__file__).resolve().parent
env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','LOKY_MAX_CPU_COUNT':'4'}
for args in [['-m','pytest','test_pipeline.py','-q'],['pipeline.py','train'],['diagnostics.py'],['release_audit.py'],['build_report.py']]:
    subprocess.run([sys.executable,*args],cwd=root,env=env,check=True)
