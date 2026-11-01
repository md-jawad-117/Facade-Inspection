"""Record trained predictions during flight; no ground-truth labels are read."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .model_client import ensure_worker
from .room_vision import canonical_record
from .vision import wall_roi,detect,image_quality


def inspect_capture(directory,room,record):
    directory=Path(directory)
    wall=next((w for w in room['walls'] if w['id']==record.get('wall_id')),None)
    if wall is None:return
    config,canonical=canonical_record(room,wall,record)
    raw=(directory/record['file']).read_bytes();depth_raw=(directory/record['depth_file']).read_bytes()
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    record['inspection_quality']=image_quality(image)
    depth=np.frombuffer(depth_raw,np.float32).reshape(record['height'],record['width'])
    roi=wall_roi(config,canonical,depth);features=detect(image,roi,backend='trained',input_scale=room.get('model_input_scale',1.0))
    from .wall_support import partition
    features,excluded=partition(features,room.get('wall_support_policy'))
    from .frame_postprocess import apply as postprocess
    features=postprocess(features,room.get('frame_postprocess','recorded-v1'))
    filename=record['file'].replace('.png','-predictions.json')
    value={'frame_postprocess':room.get('frame_postprocess','recorded-v1'),'model':ensure_worker(),'method':'wall-plane-depth-roi-v1','model_input_scale':room.get('model_input_scale',1.0),'image_sha256':hashlib.sha256(raw).hexdigest(),
           'depth_sha256':hashlib.sha256(depth_raw).hexdigest(),'findings':features,'excluded_findings':excluded,'wall_support_policy':room.get('wall_support_policy')}
    (directory/filename).write_text(json.dumps(value))
    crops=directory/'defect-captures';crops.mkdir(exist_ok=True)
    for i,feature in enumerate(features):
        x,y,w,h=feature['bbox'];margin=24
        feature['inspection_quality']=image_quality(image[max(0,y-margin):min(image.shape[0],y+h+margin),max(0,x-margin):min(image.shape[1],x+w+margin)])
        target=f'{Path(record["file"]).stem}-{wall["id"]}-{feature["kind"]}-{i+1}.png'
        cv2.imwrite(str(crops/target),image[max(0,y-margin):min(image.shape[0],y+h+margin),max(0,x-margin):min(image.shape[1],x+w+margin)])
        feature['crop_file']='defect-captures/'+target
    # Save crops' references after writing them. These are observations, not
    # claimed unique defect IDs; the multi-view association runs afterward.
    (directory/filename).write_text(json.dumps(value))
    record.update(trained_prediction_file=filename,model_observation_count=len(features))
    return len(features)
