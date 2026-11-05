#!/usr/bin/env python3
"""Read Hugging Face Trainer's actual training speed and loss."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);args=p.parse_args()
state=args.output_dir/'trainer_state.json'
if not state.exists():raise SystemExit('No trainer_state.json: '+str(state))
d=json.loads(state.read_text(encoding='utf-8'))
print('Global steps:',d.get('global_step'))
for row in d.get('log_history',[]):
    keys=['step','loss','epoch','train_runtime','train_samples_per_second','train_steps_per_second','train_loss']
    fields={k:row[k] for k in keys if k in row}
    if fields:print(fields)
