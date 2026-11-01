"""Validate split isolation, exact annotations and all exported YOLO polygons."""
import argparse
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from dataset_utils import decode_rle


def validate(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    scenes=json.loads((root/'wall-identities.json').read_text());summary=json.loads((root/'summary.json').read_text())
    failures=[];scene_split={s['scene_id']:s['split'] for s in scenes};seen_hash={};min_iou=1.0
    records={m['image_id']:m for m in manifest};coco_annotations=0
    @lru_cache(maxsize=24)
    def load_mask(stem):
        return cv2.imread(str(root/'masks'/f'{stem}.png'),cv2.IMREAD_UNCHANGED)
    for split in ['train','val','test']:
        doc=json.loads((root/'annotations'/f'{split}.coco.json').read_text())
        for a in doc['annotations']:
            coco_annotations+=1;m=records[a['image_id']]
            labels=load_mask(m['stem'])
            truth=labels==a['instance_id'];decoded=decode_rle(a['segmentation'])>0
            if not np.array_equal(truth,decoded):failures.append('RLE mismatch: '+m['stem'])
            if a['area']!=int(truth.sum()):failures.append('Area mismatch: '+m['stem'])
            ys,xs=np.where(truth)
            box=[int(xs.min()),int(ys.min()),int(np.ptp(xs))+1,int(np.ptp(ys))+1]
            if box!=a['bbox']:failures.append('Bounding box mismatch: '+m['stem'])
            if a['physical_id'].split('-D')[0]!=m['scene_id']:failures.append('Physical ID mismatch')
            expected=next((o for o in m['instances'] if o['instance_id']==a['instance_id']),None)
            if expected is None or expected['class_id']+1!=a['category_id']:failures.append('COCO class mismatch: '+m['stem'])
        listed=(root/f'{split}.txt').read_text().splitlines()
        expected=[f"./images/{m['stem']}.jpg" for m in manifest if m['split']==split and m['yolo_eligible']]
        if listed!=expected:failures.append('Training list mismatch: '+split)
    for m in manifest:
        stem=m['stem'];image=root/'images'/f'{stem}.jpg';labels=load_mask(stem)
        if labels is None or labels.dtype!=np.uint16:failures.append('Invalid instance PNG: '+stem);continue
        if m['split']!=scene_split[m['scene_id']]:failures.append('Scene leaked across splits: '+stem)
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        if digest!=m['sha256']:failures.append('Image hash mismatch: '+stem)
        if digest in seen_hash and seen_hash[digest]!=m['split']:failures.append('Duplicate RGB across splits: '+stem)
        seen_hash[digest]=m['split']
        if set(np.unique(labels))-{0}!={o['instance_id'] for o in m['instances']}:failures.append('Mask instance list mismatch: '+stem)
        rows=(root/'labels'/f'{stem}.txt').read_text().splitlines()
        if m['yolo_eligible']:
            if len(rows)!=len(m['instances']):failures.append('YOLO object count mismatch: '+stem);continue
            for row,obj in zip(rows,m['instances']):
                tokens=row.split();category=int(tokens[0]);points=np.array(list(map(float,tokens[1:]))).reshape(-1,2)
                if category!=obj['class_id'] or len(points)<3 or not np.isfinite(points).all() or np.any((points<0)|(points>1)):
                    failures.append('Invalid polygon: '+stem);continue
                mask=np.zeros(labels.shape,np.uint8)
                cv2.fillPoly(mask,[np.rint(points*np.array([labels.shape[1],labels.shape[0]])).astype(np.int32)],1)
                truth=labels==obj['instance_id'];pred=mask>0
                iou=np.count_nonzero(truth&pred)/max(1,np.count_nonzero(truth|pred));min_iou=min(min_iou,iou)
                if iou<.95:failures.append('Lossy YOLO polygon below threshold: '+stem)
    result={'passed':not failures,'views_checked':len(manifest),'exact_instance_annotations_checked':coco_annotations,
            'scenes_per_split':dict(Counter(s['split'] for s in scenes)),'yolo_views_per_split':summary['yolo_images'],
            'minimum_exported_polygon_iou':min_iou,'cross_split_scene_leakage':False if not any('leak' in e for e in failures) else True,
            'failures':failures,'scope':'Format/consistency tests, not visual realism or trained-model accuracy.'}
    (root/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args()
    raise SystemExit(0 if validate(a.dataset)['passed'] else 1)
