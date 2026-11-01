"""Native tile coordinates and conservative same-frame mask stitching."""
import cv2
import numpy as np
from .association import mask_runs

NAMES=['crack','rust','spall']
LABELS={'crack':'Crack candidate','rust':'Rust candidate','spall':'Spall candidate'}


def tile_origins(length,size=640,overlap=128):
    if length<=0 or not 0<=overlap<size:raise ValueError('Invalid tile geometry')
    if length<=size:return [0]
    values=list(range(0,length-size+1,size-overlap))
    if values[-1]!=length-size:values.append(length-size)
    return values


def candidate(mask,kind,confidence,x=0,y=0,tile=None,tile_id=0):
    mask=np.asarray(mask)>0
    yy,xx=np.nonzero(mask)
    if not len(xx):return None
    lo_x,hi_x=int(xx.min()),int(xx.max())+1;lo_y,hi_y=int(yy.min()),int(yy.max())+1
    return {'kind':kind,'confidence':float(confidence),'box':[x+lo_x,y+lo_y,x+hi_x,y+hi_y],
            'mask':mask[lo_y:hi_y,lo_x:hi_x].copy(),'tiles':{tile_id},
            'tile_bounds':tile or [x,y,x+mask.shape[1],y+mask.shape[0]],'tile_ambiguous':False}


def crop(item,rectangle):
    x,y,xx,yy=rectangle;result=np.zeros((max(0,yy-y),max(0,xx-x)),bool)
    a,b,c,d=item['box'];left,top,right,bottom=max(x,a),max(y,b),min(xx,c),min(yy,d)
    if right>left and bottom>top:result[top-y:bottom-y,left-x:right-x]=item['mask'][top-b:bottom-b,left-a:right-a]
    return result


def common_agreement(a,b):
    if a['kind']!=b['kind']:return False
    aa,bb=a['tile_bounds'],b['tile_bounds']
    common=[max(aa[0],bb[0]),max(aa[1],bb[1]),min(aa[2],bb[2]),min(aa[3],bb[3])]
    if common[2]<=common[0] or common[3]<=common[1]:return False
    x,y=a['box'],b['box']
    if min(x[2],y[2])<=max(x[0],y[0]) or min(x[3],y[3])<=max(x[1],y[1]):return False
    ma,mb=crop(a,common),crop(b,common)
    na,nb=np.count_nonzero(ma),np.count_nonzero(mb)
    if min(na,nb)<6:return False
    intersection=np.count_nonzero(ma&mb);union=np.count_nonzero(ma|mb)
    return intersection/min(na,nb)>=.7 and intersection/max(1,union)>=.5


def stitch(items):
    # Build links before merging. A component with two detections from one tile is
    # ambiguous and stays separate, rather than silently merging nearby cracks.
    groups=[{i} for i in range(len(items))];ambiguous=set()
    links=[]
    for i,a in enumerate(items):
        for j in range(i+1,len(items)):
            b=items[j]
            if a['tiles'].isdisjoint(b['tiles']) and common_agreement(a,b):links.append((i,j))
    for i,j in links:
        ia=next(k for k,g in enumerate(groups) if i in g);ib=next(k for k,g in enumerate(groups) if j in g)
        if ia==ib:continue
        ga,gb=groups[ia],groups[ib]
        ta=set().union(*(items[k]['tiles'] for k in ga));tb=set().union(*(items[k]['tiles'] for k in gb))
        if ta&tb:ambiguous.update(ga|gb);continue
        groups[ia]=ga|gb;groups.pop(ib)
    output=[]
    for group in groups:
        members=[items[i] for i in sorted(group)]
        rectangle=[min(m['box'][0] for m in members),min(m['box'][1] for m in members),
                   max(m['box'][2] for m in members),max(m['box'][3] for m in members)]
        mask=np.zeros((rectangle[3]-rectangle[1],rectangle[2]-rectangle[0]),bool)
        for member in members:mask|=crop(member,rectangle)
        x,y,xx,yy=rectangle;area=int(np.count_nonzero(mask))
        if area<3:continue
        ys,xs=np.nonzero(mask);kind=members[0]['kind']
        output.append({'kind':kind,'label':LABELS[kind],'confidence':round(max(m['confidence'] for m in members),5),
              'score_type':'model confidence, not calibrated accuracy','bbox':[x,y,xx-x,yy-y],
              'center_px':[float(x+xs.mean()),float(y+ys.mean())],'pixel_area':area,
              'mask_runs':mask_runs(mask,x,y),'review_status':'unreviewed','source_tile_count':len(members),
              'tile_identity_status':'ambiguous' if group&ambiguous else 'overlap-supported' if len(members)>1 else 'single-tile'})
    return sorted(output,key=lambda f:f['confidence'],reverse=True)


def restore_findings(findings,source_shape,target_shape):
    """Restore binary model masks to capture coordinates before depth clipping."""
    sh,sw=source_shape[:2];th,tw=target_shape[:2]
    if (sh,sw)==(th,tw):return findings
    output=[]
    for finding in findings:
        mask=np.zeros((sh,sw),np.uint8)
        for y,a,b in finding['mask_runs']:
            if 0<=y<sh:mask[y,max(0,a):min(sw,b)]=1
        restored=cv2.resize(mask,(tw,th),interpolation=cv2.INTER_NEAREST)
        runs=mask_runs(restored,0,0)
        if not runs:continue
        item=dict(finding,mask_runs=runs)
        # Recompute native area, bounding box and centroid from restored pixels.
        output.extend(clip_findings([item],np.full((th,tw),255,np.uint8)))
    return output


def clip_findings(findings,roi):
    if roi is None:return findings
    height,width=roi.shape;output=[]
    for finding in findings:
        runs=[]
        for y,a,b in finding['mask_runs']:
            if not 0<=y<height:continue
            a,b=max(0,a),min(width,b)
            if b<=a:continue
            runs.extend(mask_runs((roi[y:y+1,a:b]>0),a,y))
        if not runs:continue
        xs=[a for _,a,_ in runs]+[b-1 for _,_,b in runs];ys=[y for y,_,_ in runs]
        area=sum(b-a for _,a,b in runs)
        item=dict(finding,mask_runs=runs,pixel_area=area,bbox=[min(xs),min(ys),max(xs)-min(xs)+1,max(ys)-min(ys)+1])
        item['pre_wall_pixel_area']=finding.get('pre_wall_pixel_area',sum(b-a for _,a,b in finding['mask_runs']))
        item['pre_wall_mask_runs']=finding.get('pre_wall_mask_runs',finding['mask_runs'])
        item['pre_wall_bbox']=finding.get('pre_wall_bbox',finding['bbox'])
        item['wall_support_fraction']=area/item['pre_wall_pixel_area']
        item['center_px']=[sum((a+b-1)/2*(b-a) for _,a,b in runs)/area,sum(y*(b-a) for y,a,b in runs)/area]
        output.append(item)
    return output
