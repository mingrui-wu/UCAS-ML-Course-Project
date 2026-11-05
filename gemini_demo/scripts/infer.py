#!/usr/bin/env python3
"""Single real-MP4 inference, either base or trained LoRA adapter."""
import argparse
from pathlib import Path
from resolve_config import ROOT,find_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',default=str(ROOT/'data/videos/clip_00.mp4'))
    p.add_argument('--question',default='红色入口位于道路哪一侧？')
    p.add_argument('--model-dir');p.add_argument('--adapter')
    p.add_argument('--max-new-tokens',type=int,default=48)
    args=p.parse_args();video=Path(args.video).resolve()
    if not video.is_file():raise FileNotFoundError(video)
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    model_dir=str(find_model(args.model_dir))
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa')
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.adapter)
    model.eval();processor=AutoProcessor.from_pretrained(model_dir)
    messages=[{'role':'user','content':[
        {'type':'video','video':str(video),'fps':2.0,'max_pixels':65536},
        {'type':'text','text':args.question}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos=process_vision_info(messages)
    inputs=processor(text=[prompt],images=images,videos=videos,return_tensors='pt',padding=True).to(model.device)
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
    new_ids=ids[:,inputs['input_ids'].shape[1]:]
    print(processor.batch_decode(new_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0])
if __name__=='__main__':main()
