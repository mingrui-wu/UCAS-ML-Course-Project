#!/usr/bin/env python3
"""Download model into folder suitable for Gemini model-asset export (needs internet)."""
import argparse
from pathlib import Path
from resolve_config import ROOT, MODEL_NAME

MODEL_ID='Qwen/Qwen2.5-VL-3B-Instruct'
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', choices=['modelscope','huggingface'], default='modelscope')
    p.add_argument('--output', type=Path, default=ROOT/'model_export'/MODEL_NAME)
    args=p.parse_args();dest=args.output.expanduser().resolve()
    dest.mkdir(parents=True,exist_ok=True)
    print('Downloading:', MODEL_ID, '->',dest)
    if args.source=='modelscope':
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(MODEL_ID, local_dir=str(dest))
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID,local_dir=str(dest))
    if not (dest/'config.json').exists():
        raise RuntimeError('Download did not produce config.json; check network and errors.')
    print('Ready:',dest)
    print('Next: export model folder as Gemini asset, then mount it at /gemini/pretrain.')
if __name__ == '__main__':main()
