#!/usr/bin/env python3
"""Inspect mounted UCAS Gemini model/data and runtime before training (no GPU allocation)."""
import json
import os
from pathlib import Path
from resolve_config import find_model, ROOT

print('=== Model ===')
try:
    model = find_model(os.getenv('MODEL_DIR'))
    print('Model directory:', model)
    cfg = json.loads((model / 'config.json').read_text(encoding='utf8'))
    print('model_type:', cfg.get('model_type'), 'architectures:', cfg.get('architectures'))
    weights = sorted(model.glob('*.safetensors'))
    print('safetensors shards:', len(weights), 'total GB:', round(sum(p.stat().st_size for p in weights)/1e9, 2))
except Exception as e:
    print('Model NOT found:', type(e).__name__, str(e))

print('\n=== Data ===')
root = Path(os.getenv('DATA_DIR') or '/gemini/code/demo_data')
if not (root/'dataset_info.json').is_file():
    root = ROOT/'data'
print('Data directory:', root, 'exists:', root.is_dir())
try:
    info = json.loads((root/'dataset_info.json').read_text(encoding='utf8'))
    for key, spec in info.items():
        rows = json.loads((root/spec['file_name']).read_text(encoding='utf8'))
        print(key, ':', len(rows), 'QA')
    print('MP4:', len(list((root/'videos').glob('*.mp4'))))
except Exception as e:
    print('Data NOT found:', type(e).__name__, str(e))

print('\n=== Runtime ===')
try:
    import torch
    print('torch', torch.__version__, 'GPU accessible:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU name:', torch.cuda.get_device_name(0), 'visible VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9,2))
except Exception as e:
    print('Torch error:', type(e).__name__, str(e))
try:
    import bitsandbytes as bnb
    print('bitsandbytes:', bnb.__version__)
except Exception as e:
    print('bitsandbytes unavailable:', type(e).__name__, str(e))
try:
    import av
    print('PyAV:', av.__version__)
except Exception as e:
    print('PyAV unavailable:', type(e).__name__, str(e))
