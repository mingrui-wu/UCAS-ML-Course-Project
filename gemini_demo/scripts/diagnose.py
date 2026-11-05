#!/usr/bin/env python3
import importlib.util, os, platform, shutil, subprocess, sys
from pathlib import Path
print('Python:',sys.version.split()[0], 'Platform:',platform.platform())
for env in ['GEMINI_CODE','GEMINI_PRETRAIN','GEMINI_DATA_IN1','GEMINI_DATA_OUT','MODEL_DIR']:
    print(env,'=',os.environ.get(env,'(unset)'))
for folder in ['/gemini/code','/gemini/pretrain','/gemini/data-1','/gemini/output']:
    p=Path(folder);print(folder,'exists:',p.exists(),'writable:',os.access(p,os.W_OK) if p.exists() else False)
for dep in ['torch','transformers','llamafactory','peft','av','cv2','qwen_vl_utils']:
    print(dep,':',bool(importlib.util.find_spec(dep)))
if shutil.which('nvidia-smi'):
    subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],check=False)
try:
    import torch
    print('Torch:',torch.__version__,'CUDA:',torch.version.cuda,'available:',torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0),'bf16:',torch.cuda.is_bf16_supported())
except ImportError:pass
