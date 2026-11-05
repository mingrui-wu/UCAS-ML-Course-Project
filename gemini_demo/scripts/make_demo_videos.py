#!/usr/bin/env python3
"""Reproducibly create synthetic MP4 Video-QA clips for pipeline tests."""
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np
ROOT=Path(__file__).resolve().parent.parent; DATA=ROOT/'data'
W,H=384,256
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if not Path(FONT).is_file():
    from subprocess import check_output
    try:FONT=check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    except Exception:FONT=''
def font(size):
    try:return ImageFont.truetype(FONT,size)
    except Exception:return ImageFont.load_default()
F16,F12=font(16),font(12)
def frame(scene,idx):
    img=Image.new('RGB',(W,H),(231,235,237));d=ImageDraw.Draw(img)
    d.rectangle((120,18,264,236),fill=(195,201,204))
    d.rectangle((2,18,116,236),fill=(207,220,201))
    d.rectangle((269,18,381,236),fill=(207,220,201))
    for y in range(34,236,42):d.rectangle((188,y,195,y+18),fill=(252,252,252))
    dx=64 if scene['side']=='左侧' else 320
    d.rectangle((dx-31,66,dx+31,101),fill=(214,60,59),outline=(130,40,37),width=2)
    d.text((dx,83),'入口',fill='white',font=F16,anchor='mm')
    yy=44 if scene['sign_pos']=='上方' else 156
    d.ellipse((dx-16,yy-16,dx+16,yy+16),fill=(245,202,48),outline=(104,83,22),width=2)
    d.text((dx,yy),'标',fill=(50,46,35),font=F16,anchor='mm')
    t=idx/23
    if t<.5:x=192;y=218-118*(t/.5)
    else:
        phase=(t-.5)/.5
        x=192+{'左转':-73,'右转':73,'直行':0}[scene['turn']]*phase
        y=100-(49 if scene['turn']=='直行' else 8)*phase
    xx=int(x);yy=int(y);r=10
    d.ellipse((xx-r,yy-r,xx+r,yy+r),fill=(45,110,214),outline=(15,55,137),width=2)
    d.polygon([(xx,yy-17),(xx-6,yy-6),(xx+6,yy-6)],fill=(21,70,159))
    d.rectangle((4,239,380,255),fill=(245,245,246))
    d.text((10,246),'蓝色圆点：行人   红色矩形：入口   黄色圆形：标志',fill=(35,40,45),font=F12,anchor='lm')
    return cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
def main():
    (DATA/'videos').mkdir(parents=True,exist_ok=True)
    train=[];ev=[];specs=[]
    for i in range(12):
        scene={'side':'左侧' if i%2==0 else '右侧',
               'turn':('左转','右转','直行')[i%3],
               'sign_pos':'上方' if (i//2)%2==0 else '下方'}
        specs.append({'clip':i,**scene})
        video=DATA/'videos'/f'clip_{i:02d}.mp4'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),6,(W,H))
        if not writer.isOpened():raise RuntimeError('OpenCV mp4v encoder unavailable')
        for t in range(24):writer.write(frame(scene,t))
        writer.release()
        if video.stat().st_size<1000:raise RuntimeError('Missing video data '+str(video))
        samples=[('红色入口位于道路哪一侧？',scene['side']),
                 ('蓝色行人在视频最后是左转、右转还是直行？',scene['turn']),
                 ('黄色圆形标志位于红色入口的上方还是下方？',scene['sign_pos'])]
        dest=ev if i>=9 else train
        for q,a in samples:
            dest.append({'messages':[{'role':'user','content':'<video>'+q},
                                     {'role':'assistant','content':a}],
                         'videos':[f'videos/clip_{i:02d}.mp4']})
    for split,items in [('train',train),('eval',ev)]:
        (DATA/f'{split}.json').write_text(json.dumps(items,indent=2,ensure_ascii=False),encoding='utf-8')
    (DATA/'scene_manifest.json').write_text(json.dumps(specs,indent=2,ensure_ascii=False),encoding='utf-8')
    grid=Image.new('RGB',(W*2,H*3),'white')
    for j in range(6):
        im=Image.fromarray(cv2.cvtColor(frame(specs[j],20),cv2.COLOR_BGR2RGB))
        grid.paste(im,((j%2)*W,(j//2)*H))
    grid.save(ROOT/'demo_preview.png')
    print(f'Created {len(train)} train QA, {len(ev)} eval QA, {len(specs)} videos')
if __name__=='__main__':main()
