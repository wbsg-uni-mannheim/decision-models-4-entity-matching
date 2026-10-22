"""One cache-only export on the original CPU node; networking disabled."""
import json
import os
from pathlib import Path
import platform
import runpy
import socket
import ssl  # noqa: F401 - load socket subclass before disabling construction
import sys

ROOT=Path(__file__).resolve().parents[2]
print(json.dumps({'hostname':platform.node(),'node':os.environ.get('SLURMD_NODENAME'),'omp':os.environ.get('OMP_NUM_THREADS'),'blas':os.environ.get('OPENBLAS_NUM_THREADS')}),flush=True)
if os.environ.get('SLURMD_NODENAME')!='dws-13' or os.environ.get('OPENBLAS_NUM_THREADS')!='2' or os.environ.get('OMP_NUM_THREADS')!='2':
    raise RuntimeError('Use original node and thread configuration')


def no_network(*args, **kwargs):
    raise RuntimeError('Network disabled for cache-only export')


socket.socket=no_network
receipt=ROOT/('artifacts/operations/cascade_export_original_node_receipt_'+os.environ['SLURM_JOB_ID']+'.json')
with receipt.open('x') as f:
    json.dump({'job_id':os.environ['SLURM_JOB_ID'],'host':platform.node(),'threads':2,'network_disabled':True,'purpose':'Read-only original-node replay, no inference or result overwrite'},f)
sys.argv=['export_run_metrics.py','--root',str(ROOT),'--run-id','1','--stages','cascades',
          '--output',str(ROOT/'artifacts/results/three_runs/run01_verified_cascades_v1')]
runpy.run_path(str(ROOT/'artifacts/operations/export_run_metrics.py'),run_name='__main__')
