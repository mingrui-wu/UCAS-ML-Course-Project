#!/usr/bin/env python3
"""Resolve paths for UCAS Gemini, without requiring PyTorch to be installed."""
import argparse
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"

def find_model(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"MODEL_DIR must contain config.json: {path}")
        return path
    # Gemini Platform versions use different mount directory names.
    # Prefer the path exported by the running platform, then check known defaults.
    env_roots = [Path(os.environ[key]) for key in
                 ('GEMINI_PRETRAIN', 'GEMINI_PRETRAIN2', 'GEMINI_PRETRAIN3')
                 if os.environ.get(key)]
    roots = env_roots + [
        Path('/gemini/pretrain'), Path('/gemini/pretrain2'), Path('/gemini/pretrain3'),
        Path('/gemini/predata'), Path('/gemini/predata2'), Path('/gemini/predata3'),
        ROOT / 'model_export' / MODEL_NAME, ROOT / 'models' / MODEL_NAME
    ]
    for base in roots:
        if (base / 'config.json').is_file():
            return base
        if base.exists() and base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir() and (child / 'config.json').is_file():
                    return child
    raise FileNotFoundError(
        'Base model not found. Mount your model into /gemini/pretrain, '\
        'or run python scripts/download_base_model.py, or set MODEL_DIR=/absolute/path/to/model'
    )

def build(preset, model_dir=None, data_dir=None, output_dir=None, dest=None):
    config_path = ROOT / 'configs' / f'{preset}.yaml'
    if preset not in ('smoke','one_epoch','qlora_smoke_8g','qlora_one_epoch_8g'):
        raise ValueError(f'Unknown preset: {preset}')
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    model = find_model(model_dir or os.environ.get('MODEL_DIR'))
    dataset = Path(data_dir or os.environ.get('DATA_DIR') or ROOT/'data').expanduser().resolve()
    if not (dataset/'dataset_info.json').is_file():
        raise FileNotFoundError(f'Missing {dataset / "dataset_info.json"}')
    out = Path(output_dir or os.environ.get('DEMO_OUTPUT_DIR') or (ROOT/'outputs'/preset)).expanduser().resolve()
    cfg.update(model_name_or_path=str(model), dataset_dir=str(dataset), media_dir=str(dataset), output_dir=str(out))
    if os.environ.get('DEMO_DATASET'):
        cfg['dataset']=os.environ['DEMO_DATASET']
    if os.environ.get('DEMO_MAX_STEPS'):
        cfg['max_steps'] = int(os.environ['DEMO_MAX_STEPS'])
    if os.environ.get('DEMO_VIDEO_MAXLEN'):
        cfg['video_maxlen'] = int(os.environ['DEMO_VIDEO_MAXLEN'])
    if os.environ.get('DEMO_VIDEO_FPS'):
        cfg['video_fps'] = float(os.environ['DEMO_VIDEO_FPS'])
    out_yaml = Path(dest or ROOT/'configs'/'generated'/f'{preset}_resolved.yaml')
    out_yaml.parent.mkdir(parents=True,exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(f'Model:      {model}\nDataset:    {dataset}\nOutput:     {out}\nConfig:     {out_yaml}')
    return out_yaml

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--preset', choices=['smoke','one_epoch','qlora_smoke_8g','qlora_one_epoch_8g'], default='qlora_smoke_8g')
    p.add_argument('--model-dir')
    p.add_argument('--data-dir')
    p.add_argument('--output-dir')
    p.add_argument('--dest')
    args=p.parse_args()
    build(args.preset,args.model_dir,args.data_dir,args.output_dir,args.dest)
