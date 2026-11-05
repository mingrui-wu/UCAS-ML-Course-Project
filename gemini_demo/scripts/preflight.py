#!/usr/bin/env python3
"""CPU-only static checks for Video-QA package."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def check(root, dataset='demo_train'):
    info=json.loads((root/'dataset_info.json').read_text(encoding='utf-8'))
    if dataset not in info:raise KeyError(f"Dataset {dataset!r} is absent from {root/'dataset_info.json'}")
    checks=[dataset]
    if dataset=='demo_train' and 'demo_eval' in info:checks.append('demo_eval')
    if dataset=='course_train' and 'course_eval' in info:checks.append('course_eval')
    total=0;split_videos={}
    for split in checks:
        spec=info[split]
        assert spec['formatting']=='sharegpt'
        samples=json.loads((root/spec['file_name']).read_text(encoding='utf-8'))
        vids=set()
        for i,row in enumerate(samples):
            msgs=row.get('messages',[])
            assert len(msgs)==2 and msgs[0]['role']=='user' and msgs[1]['role']=='assistant',(split,i)
            assert msgs[0]['content'].count('<video>')==1,(split,i)
            assert len(row.get('videos',[]))==1,(split,i)
            video=(root/row['videos'][0]).resolve()
            assert video.is_file() and video.stat().st_size>1000,video
            vids.add(video)
        split_videos[split]=vids;total+=len(samples)
        print(f'{split}: {len(samples)} samples, {len(vids)} unique videos')
    if len(checks)>1:
        assert not split_videos[checks[0]] & split_videos[checks[1]], 'Train/eval video leakage'
    print('Preflight OK:',total,'examples, ffmpeg=',bool(shutil.which('ffmpeg')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--dataset',default='demo_train')
    a=p.parse_args();check(a.data.resolve(),a.dataset)
