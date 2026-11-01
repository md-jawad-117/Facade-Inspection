"""Verify every variant against its parent, beyond self-consistent mask labels."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from augment_dataset import TRANSFORMS,transform_array,transform_matrix,transform_labels


def check(source,output):
    source=Path(source);output=Path(output)
    originals=json.loads((source/'manifest.json').read_text());manifest=json.loads((output/'manifest.json').read_text())
    lookup={m['stem']:m for m in originals};variants={};failures=[];checked=0
    # Evaluation inputs and original data must remain byte-for-byte identical.
    for m in originals:
        for kind,ext in [('images','.jpg'),('labels','.txt'),('masks','.png')]:
            path=Path(kind)/(m['stem']+ext)
            if (source/path).read_bytes()!=(output/path).read_bytes():failures.append('Changed original: '+str(path))
    for split in ['val','test']:
        for path in [Path(f'{split}.txt'),Path('annotations')/f'{split}.coco.json']:
            if (source/path).read_bytes()!=(output/path).read_bytes():failures.append('Changed evaluation artifact: '+str(path))
    parent_mask=None;parent_stem=None;parent_rows=None
    for m in manifest:
        augmentation=m.get('augmentation')
        if not augmentation:continue
        stem=augmentation['source_stem'];kind=augmentation['name'];parent=lookup[stem]
        variants.setdefault(stem,[]).append(kind)
        if m['split']!='train' or parent['split']!='train':failures.append('Nontraining augmentation')
        if m['scene_id']!=parent['scene_id']:failures.append('Scene ID changed')
        if [(o['physical_id'],o['instance_id'],o['class_id'],o['area']) for o in m['instances']]!=[(o['physical_id'],o['instance_id'],o['class_id'],o['area']) for o in parent['instances']]:
            failures.append('Physical identity, class or area changed: '+m['stem'])
        if stem!=parent_stem:
            parent_stem=stem;parent_mask=cv2.imread(str(source/'masks'/f'{stem}.png'),cv2.IMREAD_UNCHANGED)
            parent_rows=(source/'labels'/f'{stem}.txt').read_text().splitlines()
        actual=cv2.imread(str(output/'masks'/f"{m['stem']}.png"),cv2.IMREAD_UNCHANGED)
        if not np.array_equal(actual,transform_array(parent_mask,kind)):failures.append('Wrong mask transform: '+m['stem'])
        h,w=parent_mask.shape;matrix=transform_matrix(kind,w,h)
        expected=matrix@np.array(parent['homography_source_to_image'])
        if not np.allclose(expected,m['homography_source_to_image']):failures.append('Incorrect view homography: '+m['stem'])
        rows=(output/'labels'/f"{m['stem']}.txt").read_text().splitlines()
        if rows!=transform_labels(parent_rows,kind,w,h):failures.append('Wrong polygon transform: '+m['stem'])
        checked+=1
    expected_parents={m['stem'] for m in originals if m['split']=='train' and m['yolo_eligible']}
    if set(variants)!=expected_parents:failures.append('Augmentation parent list mismatch')
    for stem,kinds in variants.items():
        if sorted(kinds)!=sorted(TRANSFORMS[1:]):failures.append('Variant list mismatch: '+stem)
    result={'passed':not failures,'original_views_verified_unchanged':len(originals),'transformed_views_verified':checked,
            'parent_training_views':len(variants),'exact_masks_and_polygon_transforms':not failures,
            'validation_test_unchanged':not any('evaluation' in f or 'original' in f for f in failures),
            'failures':failures}
    (output/'augmentation-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raise SystemExit(0 if check(a.source,a.output)['passed'] else 1)
