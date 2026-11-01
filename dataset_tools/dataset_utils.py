"""Shared exact-mask, polygon-export and preview helpers for the v2 dataset."""
import json
import cv2
import numpy as np
CLASSES=['crack','rust','spall']
COLORS=[(40,70,240),(20,160,245),(230,135,50)]


def write_json(path,value):
    path.write_text(json.dumps(value,separators=(',',':')),encoding='utf-8')


def rle(mask):
    values=(mask>0).flatten(order='F').astype(np.uint8)
    edges=np.flatnonzero(np.diff(np.r_[0,values,0]))
    counts=[];last=0
    for edge in edges:
        counts.append(int(edge-last));last=int(edge)
    # Last transition returns to background; include remaining zeros.
    counts.append(int(len(values)-last))
    return {'size':list(mask.shape),'counts':counts}


def decode_rle(value):
    flat=np.zeros(sum(value['counts']),np.uint8);offset=0
    for i,count in enumerate(value['counts']):
        if i%2:flat[offset:offset+count]=1
        offset+=count
    return flat.reshape(value['size'],order='F')


def polygon(mask):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    paths=[c[:,0,:] for c in contours if len(c)>=3]
    if not paths:return None,0
    joined=paths.pop(0)
    # Connect disconnected visible fragments with a doubled zero-area bridge.
    # Pixel agreement is checked below; an entire view is excluded if unsuitable.
    for other in paths:
        distances=np.sum((joined[:,None,:].astype(float)-other[None,:,:])**2,axis=2)
        a,b=np.unravel_index(np.argmin(distances),distances.shape)
        first=np.roll(joined,-a,axis=0);second=np.roll(other,-b,axis=0)
        joined=np.concatenate([first,first[:1],second,second[:1],first[:1]])
    reconstructed=np.zeros(mask.shape,np.uint8)
    cv2.fillPoly(reconstructed,[joined.astype(np.int32)],1)
    truth=mask>0;prediction=reconstructed>0
    iou=np.count_nonzero(truth&prediction)/max(1,np.count_nonzero(truth|prediction))
    return joined,float(iou)


def overlay(image,labels,objects):
    marked=image.copy()
    for obj in objects:
        mask=labels==obj['instance_id']
        if not np.any(mask):continue
        color=np.array(COLORS[obj['class_id']])
        marked[mask]=(.45*marked[mask]+.55*color).astype(np.uint8)
        ys,xs=np.where(mask)
        cv2.rectangle(marked,(int(xs.min()),int(ys.min())),(int(xs.max()),int(ys.max())),tuple(map(int,color)),1)
        cv2.putText(marked,f"D{obj['instance_id']:02d} {CLASSES[obj['class_id']]}",(int(xs.min()),max(14,int(ys.min())-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,tuple(map(int,color)),1)
    return marked
