#!/usr/bin/env python3
"""Convert self-provided video QA data to LLaMA-Factory schema.
Input JSON list[{'video_path':..., 'question':..., 'answer':..., 'split':'train'|'eval'}].
"""
import argparse,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input_json',type=Path)
    p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.video_root.resolve();dest=args.out.resolve()
    (dest/'videos').mkdir(parents=True,exist_ok=True)
    rows=json.loads(args.input_json.read_text(encoding='utf-8'))
    splits={'train':[],'eval':[]};cache={}
    for i,row in enumerate(rows):
        split=row.get('split','train')
        if split not in splits:raise ValueError(f'Invalid split {split} at row {i}')
        relative=Path(row['video_path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError(f'Unsafe path: {relative}')
        source=(root/relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():raise FileNotFoundError(source)
        if source not in cache:
            name=f'video_{len(cache):05d}{source.suffix.lower()}'
            shutil.copy2(source,dest/'videos'/name);cache[source]='videos/'+name
        splits[split].append({'messages':[
            {'role':'user','content':'<video>'+str(row['question'])},
            {'role':'assistant','content':str(row['answer'])}],
            'videos':[cache[source]]})
    metadata={}
    for split,items in splits.items():
        (dest/(split+'.json')).write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
        metadata['course_'+split]={
            'file_name':split+'.json','formatting':'sharegpt',
            'columns':{'messages':'messages','videos':'videos'},
            'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}}
    (dest/'dataset_info.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    tr={tuple(x['videos']) for x in splits['train']}
    ev={tuple(x['videos']) for x in splits['eval']}
    if tr & ev:print('WARNING: Train/eval contain identical videos; split by video!')
    print('Ready:',dest,'train',len(splits['train']),'eval',len(splits['eval']))
    print('Run: DATA_DIR='+str(dest)+' DEMO_DATASET=course_train bash scripts/train.sh one_epoch')
if __name__=='__main__':main()
