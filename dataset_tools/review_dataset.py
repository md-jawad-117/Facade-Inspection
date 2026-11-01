"""Sample actual training records and draw their exported YOLO polygons on RGB."""
import argparse
import html
import json
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from dataset_utils import CLASSES,COLORS


def review(dataset,output,seed=2026101907,count=6):
    dataset=Path(dataset);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((dataset/'manifest.json').read_text())
    pool=[m for m in manifest if m['split']=='train' and m['yolo_eligible']]
    selected=random.Random(seed).sample(pool,min(count,len(pool)))
    records=[];tiles=[];sections=[]
    for index,m in enumerate(selected,1):
        stem=m['stem'];image=cv2.imread(str(dataset/'images'/f'{stem}.jpg'))
        exact=cv2.imread(str(dataset/'masks'/f'{stem}.png'),cv2.IMREAD_UNCHANGED)
        h,w=image.shape[:2];marked=image.copy();labels=[]
        rows=(dataset/'labels'/f'{stem}.txt').read_text().splitlines()
        for row,obj in zip(rows,m['instances']):
            values=row.split();kind=int(values[0]);points=np.array(list(map(float,values[1:]))).reshape(-1,2)
            points=np.rint(points*np.array([w,h])).astype(np.int32)
            mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[points],1)
            foreground=mask>0;truth=exact==obj['instance_id'];color=np.array(COLORS[kind])
            marked[foreground]=(.55*marked[foreground]+.45*color).astype(np.uint8)
            cv2.polylines(marked,[points],True,tuple(map(int,color)),1,cv2.LINE_AA)
            x,y=points.min(axis=0);text=f"D{obj['instance_id']:02d} {CLASSES[kind]}"
            cv2.putText(marked,text,(max(2,int(x)),max(15,int(y)-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),3,cv2.LINE_AA)
            cv2.putText(marked,text,(max(2,int(x)),max(15,int(y)-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,tuple(map(int,color)),1,cv2.LINE_AA)
            intersection=int(np.count_nonzero(foreground&truth));union=int(np.count_nonzero(foreground|truth))
            labels.append({'physical_id':obj['physical_id'],'class':CLASSES[kind],
                           'polygon_vs_exact_mask_iou':intersection/max(1,union)})
        if not rows:cv2.putText(marked,'Clean negative: no target labels',(20,40),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,100,20),2)
        original=f'{index:02d}-raw.jpg';annotated=f'{index:02d}-annotated.png';comparison=f'{index:02d}-comparison.jpg'
        shutil.copy2(dataset/'images'/f'{stem}.jpg',output/original)
        cv2.imwrite(str(output/annotated),marked)
        pair=np.hstack([image,marked]);header=np.full((56,pair.shape[1],3),247,np.uint8)
        cv2.putText(header,f'{index}. {stem} | TRAIN | RGB (left) / exported segmentation labels (right)',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,35,30),1,cv2.LINE_AA)
        cv2.putText(header,'Red = crack, orange = rust, blue = spall. D## = source defect ID.',(12,44),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,35,30),1,cv2.LINE_AA)
        pair=np.vstack([header,pair]);cv2.imwrite(str(output/comparison),pair)
        tiles.append(cv2.resize(pair,(768,418),interpolation=cv2.INTER_AREA))
        records.append({'sample':index,'stem':stem,'scene_id':m['scene_id'],'split':m['split'],
                        'image_file':original,'annotation_file':annotated,'labels':labels})
        sections.append(f'<section><h2>{index}. {html.escape(stem)}</h2><p>{len(labels)} labeled instances; training split. '+
            ('Clean negative example.' if not labels else f'Minimum polygon/mask agreement: {min(a["polygon_vs_exact_mask_iou"] for a in labels):.3f}.')+
            f'</p><a href="{comparison}"><img src="{comparison}" alt="Raw image beside exported annotation overlay for sample {index}"></a><p><a href="{original}">Raw image</a> · <a href="{annotated}">Annotation overlay</a> · <a href="{comparison}">Full-size comparison</a></p></section>')
    contact=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]) if len(tiles)%2==0 else np.vstack(tiles)
    cv2.imwrite(str(output/'random-samples.jpg'),contact)
    audit={'seed':seed,'sampling':'Uniform random sample without replacement from eligible TRAIN views; no samples replaced based on appearance.',
           'pool_size':len(pool),'samples':records,'annotation_source':'The exported labels/*.txt training polygons, not the cached preview drawings.',
           'review_status':'Pending joint visual review; numeric agreement alone does not establish semantic correctness.'}
    (output/'selection.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Random annotation review</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:24px;background:#f5f7f2;color:#203c31}p{line-height:1.6}img{max-width:100%}section{background:white;padding:18px;margin:24px 0}a{color:#176e55}</style><h1>Random training-annotation review</h1><p>Six views sampled uniformly without replacement from the eligible training set. Left: raw image. Right: exported polygon annotations rendered over the same image. Red = crack, orange = rust, blue = spall. Thin cracks and cropped edges deserve close inspection. The overlay does not change the RGB training input.</p><p>The exact instance masks are the reference for polygon conversion checks. A high overlap score checks alignment; it does not prove that the generated appearance is realistic. Source scenes stay separated across train/validation/test. Review these before approving bulk augmentation.</p>'+''.join(sections)+'<p><a href="selection.json">Sampling seed, selected files and per-object mask agreement</a></p>',encoding='utf-8')
    print(json.dumps(audit,indent=2))
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--seed',type=int,default=2026101907);p.add_argument('--count',type=int,default=6)
    args=p.parse_args();review(args.dataset,args.output,args.seed,args.count)
