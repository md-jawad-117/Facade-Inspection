"""Six-way training augmentation, preserving instance geometry and split groups.

Copies the original dataset; appends five isometric variants of eligible TRAIN
images only. Validation/test remain unchanged. Classes and physical IDs persist.
"""
import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from dataset_utils import CLASSES,rle,write_json

TRANSFORMS=['original','rotate90cw','rotate180','rotate270cw','flip_horizontal','flip_vertical']


def transform_array(array,kind):
    if kind=='original':return array.copy()
    if kind=='rotate90cw':return np.ascontiguousarray(np.rot90(array,-1))
    if kind=='rotate180':return np.ascontiguousarray(np.rot90(array,2))
    if kind=='rotate270cw':return np.ascontiguousarray(np.rot90(array,1))
    if kind=='flip_horizontal':return np.ascontiguousarray(array[:,::-1])
    if kind=='flip_vertical':return np.ascontiguousarray(array[::-1])
    raise ValueError('Unknown transform: '+kind)


def transform_matrix(kind,width,height):
    matrices={
        'original':[[1,0,0],[0,1,0],[0,0,1]],
        'rotate90cw':[[0,-1,height-1],[1,0,0],[0,0,1]],
        'rotate180':[[-1,0,width-1],[0,-1,height-1],[0,0,1]],
        'rotate270cw':[[0,1,0],[-1,0,width-1],[0,0,1]],
        'flip_horizontal':[[-1,0,width-1],[0,1,0],[0,0,1]],
        'flip_vertical':[[1,0,0],[0,-1,height-1],[0,0,1]]}
    if kind not in matrices:raise ValueError('Unknown transform: '+kind)
    return np.array(matrices[kind],dtype=float)


def transform_labels(rows,kind,width,height):
    matrix=transform_matrix(kind,width,height)
    new_width,new_height=(height,width) if kind in ['rotate90cw','rotate270cw'] else (width,height)
    result=[]
    for row in rows:
        tokens=row.split();points=np.array(list(map(float,tokens[1:]))).reshape(-1,2)
        # Dataset polygons encode pixel-center coordinates as x/width,y/height.
        # Convert to pixels before transforming, using width-1 rather than 1-x.
        pixels=np.rint(points*np.array([width,height]))
        homogeneous=np.column_stack([pixels,np.ones(len(pixels))])
        mapped=(homogeneous@matrix.T)[:,:2]/np.array([new_width,new_height])
        result.append(tokens[0]+' '+' '.join(f'{v:.8f}' for v in mapped.flatten()))
    return result


def build(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists() or output.is_relative_to(source):raise ValueError('Choose a new output outside the source dataset.')
    validation=json.loads((source/'validation.json').read_text())
    if not validation['passed']:raise ValueError('Source validation must pass before augmentation.')
    source_manifest=json.loads((source/'manifest.json').read_text())
    if any(m.get('augmentation',{}).get('name') not in [None,'original'] for m in source_manifest):
        raise ValueError('Use the original dataset, not an already augmented dataset.')
    manifest=copy.deepcopy(source_manifest)
    documents={s:json.loads((source/'annotations'/f'{s}.coco.json').read_text()) for s in ['train','val','test']}
    summary=json.loads((source/'summary.json').read_text())
    annotation_id=max(a['id'] for doc in documents.values() for a in doc['annotations'])
    image_id=max(m['image_id'] for m in manifest)
    shutil.copytree(source,output)
    write_json(output/'validation.json',{'passed':False,'status':'Augmentation built; full validation pending.'})
    eligible=[m for m in source_manifest if m['split']=='train' and m['yolo_eligible']]
    for index,m in enumerate(eligible,1):
        stem=m['stem'];rgb=cv2.imread(str(source/'images'/f'{stem}.jpg'))
        labels=cv2.imread(str(source/'masks'/f'{stem}.png'),cv2.IMREAD_UNCHANGED)
        height,width=labels.shape
        rows=(source/'labels'/f'{stem}.txt').read_text().splitlines()
        for kind in TRANSFORMS[1:]:
            name=stem+'__'+kind;image_id+=1
            transformed_rgb=transform_array(rgb,kind);transformed_mask=transform_array(labels,kind)
            nh,nw=transformed_mask.shape;matrix=transform_matrix(kind,width,height)
            cv2.imwrite(str(output/'images'/f'{name}.jpg'),transformed_rgb,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(output/'masks'/f'{name}.png'),transformed_mask)
            new_rows=transform_labels(rows,kind,width,height)
            (output/'labels'/f'{name}.txt').write_text('\n'.join(new_rows)+('\n' if new_rows else ''),encoding='utf-8')
            record=copy.deepcopy(m)
            record.update(image_id=image_id,stem=name,
                          augmentation={'name':kind,'source_stem':stem,'matrix_source_view_to_variant':matrix.tolist()},
                          sha256=hashlib.sha256((output/'images'/f'{name}.jpg').read_bytes()).hexdigest(),
                          homography_source_to_image=(matrix@np.array(m['homography_source_to_image'])).tolist())
            for row,obj in zip(new_rows,record['instances']):
                mask=(transformed_mask==obj['instance_id']).astype(np.uint8)
                ys,xs=np.where(mask);area=int(mask.sum())
                points=np.array(list(map(float,row.split()[1:]))).reshape(-1,2)
                predicted=np.zeros((nh,nw),np.uint8)
                cv2.fillPoly(predicted,[np.rint(points*np.array([nw,nh])).astype(np.int32)],1)
                truth=mask>0;prediction=predicted>0
                iou=np.count_nonzero(truth&prediction)/max(1,np.count_nonzero(truth|prediction))
                if iou<.95:raise ValueError('Transformed polygon failed mask agreement: '+name)
                if area!=obj['area']:raise ValueError('Isometric transform changed instance area.')
                obj['polygon_iou']=round(float(iou),5)
                annotation_id+=1
                documents['train']['annotations'].append({'id':annotation_id,'image_id':image_id,'category_id':obj['class_id']+1,
                    'segmentation':rle(mask),'area':area,'bbox':[int(xs.min()),int(ys.min()),int(np.ptp(xs))+1,int(np.ptp(ys))+1],
                    'iscrowd':0,'physical_id':obj['physical_id'],'instance_id':obj['instance_id']})
            manifest.append(record)
            documents['train']['images'].append({'id':image_id,'file_name':f'images/{name}.jpg','width':nw,'height':nh,'scene_id':m['scene_id']})
        if index%64==0:print(f'Augmented {index}/{len(eligible)} source training views',flush=True)
    for split,document in documents.items():
        write_json(output/'annotations'/f'{split}.coco.json',document)
        files=[f"./images/{m['stem']}.jpg" for m in manifest if m['split']==split and m['yolo_eligible']]
        (output/f'{split}.txt').write_text('\n'.join(files)+'\n',encoding='utf-8')
    counts=Counter(CLASSES[o['class_id']] for m in manifest for o in m['instances'])
    summary.update(views=len(manifest),instance_observations=dict(counts),negative_views=sum(not m['instances'] for m in manifest),
                   yolo_images={s:sum(m['split']==s and m['yolo_eligible'] for m in manifest) for s in documents},
                   augmentation={'version':'train-six-way-v1','source_views':len(source_manifest),'source_eligible_train_views':len(eligible),
                       'transform_names':TRANSFORMS,'applied_to':'eligible training views only','added_views':len(eligible)*5,
                       'validation_test_changed':False,'independent_wall_scenes_added':0,
                       'geometry':'Exact pixel-index isometries for masks/polygons; augmented JPGs are re-encoded at quality 95.'})
    write_json(output/'manifest.json',manifest);write_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();build(args.source,args.output)
    from validate_dataset import validate
    raise SystemExit(0 if validate(args.output)['passed'] else 1)
