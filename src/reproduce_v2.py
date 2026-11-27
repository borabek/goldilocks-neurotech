"""Evaluate saved v2 weights offline; never refit or change the frozen protocol."""
from pathlib import Path
import os, subprocess, sys

root = Path(__file__).resolve().parents[1]
env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
for args in [
    ['-m', 'pytest', 'tests/test_pipeline.py', 'tests/test_v2.py', '-q'],
    ['src/v2_evaluate.py'],
    ['src/v2_evaluate.py', '--external'],
    ['src/v2_report.py'],
]:
    subprocess.run([sys.executable, *args], cwd=root, env=env, check=True)
