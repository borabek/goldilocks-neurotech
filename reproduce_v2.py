"""Evaluate saved v2 weights offline; never refit or change the frozen protocol."""
from pathlib import Path
import os, subprocess, sys

root = Path(__file__).resolve().parent
env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
for args in [
    ['-m', 'pytest', 'test_pipeline.py', 'test_v2.py', '-q'],
    ['v2_evaluate.py'],
    ['v2_evaluate.py', '--external'],
    ['v2_report.py'],
]:
    subprocess.run([sys.executable, *args], cwd=root, env=env, check=True)
