"""Install the frozen package versions only inside the new recovery Pod."""
import json
from pathlib import Path
import subprocess
import sys

assert sys.version_info[:2] == (3, 12), sys.version
target = Path('/opt/titantpp-history27-venv')
assert not target.exists(), 'Preserve any earlier bootstrap attempt'
subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True)
python = str(target / 'bin/python')
subprocess.run([python, '-m', 'pip', 'install', '-q', 'torch==2.11.0',
                '--index-url', 'https://download.pytorch.org/whl/cu130'],
               check=True, timeout=900)
subprocess.run([python, '-m', 'pip', 'install', '-q', 'numpy==2.4.4',
                'polars==1.39.3', 'pandas', 'matplotlib', 'scipy', 'pyarrow',
                'PyYAML', 'tqdm', 'scikit-learn'], check=True, timeout=360)
subprocess.run([python, '-c',
    'import torch,numpy,polars,json;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=numpy.__version__,polars=polars.__version__)))'], check=True)
subprocess.run(['which', 'tmux', 'timeout'], check=True)
print(json.dumps({'status': 'installed_not_scientifically_qualified'}), flush=True)
