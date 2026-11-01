"""World-unit procedural walls and native-resolution pinhole-camera tiles.

All physical geometry is in millimeters. This is an ideal frontal planar renderer,
not AirSim imagery, calibrated optics or real-world accuracy validation.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from dataset_utils import CLASSES,polygon,rle,overlay,write_json

VERSION='calibrated-walls-v2'
WALL_MM=1200.0
SOURCE_SIZE=2048
PATCH_SIZE=640
PROFILES=[
    {'id':'4k-2m-A','width':3840,'height':2160,'distance_mm':2000,'tile_center_fraction':[.4,.4]},
    {'id':'4k-2m-B','width':3840,'height':2160,'distance_mm':2000,'tile_center_fraction':[.6,.6]},
    {'id':'4k-3m','width':3840,'height':2160,'distance_mm':3000,'tile_center_fraction':[.5,.5]},
    {'id':'1440p-1.5m','width':2560,'height':1440,'distance_mm':1500,'tile_center_fraction':[.5,.5]},
    {'id':'current-640-2m-stress','width':640,'height':360,'distance_mm':2000,'stress_only':True},
]
FOV=89.90362548828125


def trace_length(points):
    return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())


def crack_geometry(rng):
    """Length includes half-width round caps at each visible tip; spread includes gap."""
    patterns=[[0,1,.8],[0,.8,.4,1],[0,.65,.15,.8,.4,1],[0,1,.7,.9],[0,.4,.2,1,.7]]
    width=5.0
    for _ in range(1000):
        length=float(rng.uniform(50,100));spread=float(rng.uniform(20,50))
        center_width=spread-width;target_arc=length-width
        feasible=[]
        for pattern in patterns:
            x=np.array(pattern)*center_width
            min_height=(len(x)-1)*width*1.2
            minimum=np.column_stack([x,np.linspace(0,min_height,len(x))])
            if trace_length(minimum)<target_arc:feasible.append((x,min_height))
        if not feasible:continue
        x,low=feasible[int(rng.integers(len(feasible)))];high=target_arc
        for _ in range(60):
            height=(low+high)/2
            points=np.column_stack([x,np.linspace(0,height,len(x))])
            if trace_length(points)<target_arc:low=height
            else:high=height
        points=np.column_stack([x,np.linspace(0,(low+high)/2,len(x))])
        if rng.random()<.5:points[:,0]*=-1
        if rng.random()<.5:points[:,1]*=-1
        points-=np.mean([points.min(axis=0),points.max(axis=0)],axis=0)
        return {'class_id':0,'polyline_mm':points.tolist(),'opening_width_mm':width,
                'visible_trace_length_mm':trace_length(points)+width,'centerline_arc_length_mm':trace_length(points),
                'zigzag_spread_mm':float(np.ptp(points[:,0])+width),
                'dimension_definition':'A: centerline trace between visible tips including round caps; B: nominal width normal to a straight segment; C: total horizontal opening extent in object-local frame.'}
    raise RuntimeError('Unable to construct a physically feasible crack.')


def rasterize_crack(obj,pitch,size):
    points=np.array(obj['polyline_mm']);radius=obj['opening_width_mm']/2
    lo=np.maximum(0,np.floor((points.min(axis=0)-radius-pitch)/pitch).astype(int))
    hi=np.minimum(size-1,np.ceil((points.max(axis=0)+radius+pitch)/pitch).astype(int))
    yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
    coordinates=np.stack([xx*pitch,yy*pitch],axis=-1)
    minimum=np.full(xx.shape,np.inf)
    for a,b in zip(points,points[1:]):
        direction=b-a;t=np.clip(np.sum((coordinates-a)*direction,axis=-1)/np.dot(direction,direction),0,1)
        distance=np.linalg.norm(coordinates-(a+t[:,:,None]*direction),axis=-1)
        minimum=np.minimum(minimum,distance)
    region=(minimum<=radius).astype(np.uint8)
    alpha=np.clip((radius-minimum)/pitch+.5,0,1)
    return (slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1)),region,alpha


def make_wall(seed,size=SOURCE_SIZE):
    rng=np.random.default_rng(seed);pitch=WALL_MM/size;base=float(rng.uniform(140,200))
    coarse=cv2.resize(rng.normal(0,rng.uniform(2,7),(48,48)).astype(np.float32),(size,size),interpolation=cv2.INTER_CUBIC)
    gray=np.clip(base+coarse+rng.normal(0,rng.uniform(.5,2),(size,size)),0,255).astype(np.uint8)
    image=np.stack([gray,gray,np.clip(gray.astype(np.int16)+rng.integers(-3,6),0,255).astype(np.uint8)],axis=-1)
    for _ in range(int(rng.integers(0,4))):
        p=int(rng.integers(40,size-40));shade=tuple(int(base-12) for _ in range(3))
        cv2.line(image,(p,0),(p,size-1),shade,max(1,int(round(.7/pitch))))
    ids=np.zeros((size,size),np.uint16);objects=[];negative=bool(rng.random()<.2)
    kinds=[] if negative else [0]*int(rng.integers(4,9))+[1]*int(rng.integers(1,4))+[2]*int(rng.integers(1,4))
    rng.shuffle(kinds)
    occupied=[]
    for kind in kinds:
        for attempt in range(100):
            center=rng.uniform(100,WALL_MM-100,2)
            if kind==0:
                obj=crack_geometry(rng);points=np.array(obj['polyline_mm'])+center
                obj['polyline_mm']=points.tolist();radius=2.5
                lower=points.min(axis=0)-radius;upper=points.max(axis=0)+radius
            else:
                # Auxiliary class dimensions are explicit design assumptions, not user targets.
                rx,ry=(rng.uniform(10,20),rng.uniform(25,45)) if kind==1 else (rng.uniform(20,35),rng.uniform(20,35))
                lower=center-[rx,ry];upper=center+[rx,ry]
                obj={'class_id':kind,'center_mm':center.tolist(),'half_extent_mm':[float(rx),float(ry)],
                     'physical_size_scope':'Auxiliary stain/spall sizes are synthetic design assumptions.'}
            if any(np.all(lower<hi+20) and np.all(upper>lo-20) for lo,hi in occupied):continue
            break
        else:continue
        occupied.append((lower,upper));ident=len(objects)+1
        if kind==0:
            region,mask,alpha=rasterize_crack(obj,pitch,size)
            color=rng.uniform(30,75);patch=image[region].astype(float)
            patch=patch*(1-alpha[:,:,None])+color*alpha[:,:,None]
            image[region]=np.clip(patch,0,255).astype(np.uint8)
            ids[region][mask>0]=ident
        else:
            lo=np.floor(lower/pitch).astype(int);hi=np.ceil(upper/pitch).astype(int)
            yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            relative=(np.stack([xx*pitch,yy*pitch],axis=-1)-center)/[rx,ry]
            radius=np.linalg.norm(relative,axis=-1)
            if kind==1:
                mask=radius<=1;color=np.array([45,rng.uniform(75,100),rng.uniform(120,155)])
            else:
                angle=np.arctan2(relative[:,:,1],relative[:,:,0])
                mask=radius<=.9+.08*np.sin(angle*7)+.04*np.sin(angle*13)
                color=np.full(3,rng.uniform(75,110))
            region=(slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1))
            patch=image[region];patch[mask]=np.clip(color+rng.normal(0,5,(int(mask.sum()),1)),0,255).astype(np.uint8)
            ids[region][mask]=ident
        obj.update(instance_id=ident,source_bbox_mm=[*map(float,lower),*map(float,upper)])
        objects.append(obj)
    return image,ids,objects,negative


def render(image,ids,profile,seed):
    rng=np.random.default_rng(seed);size=PATCH_SIZE;pitch=WALL_MM/image.shape[0]
    focal=profile['width']/(2*math.tan(math.radians(FOV)/2));gsd=profile['distance_mm']/focal
    principal=np.array([(profile['width']-1)/2,(profile['height']-1)/2])
    camera=np.array([WALL_MM/2,WALL_MM/2])
    if profile.get('stress_only'):
        origin=np.array([0,-(size-profile['height'])//2],dtype=int)
    else:
        desired=np.array(profile['tile_center_fraction'])*WALL_MM
        desired_pixel=(desired-camera)/gsd+principal
        origin=np.rint(desired_pixel-(size-1)/2).astype(int)
        origin=np.clip(origin,[0,0],[profile['width']-size,profile['height']-size])
    matrix=np.array([[pitch/gsd,0,-camera[0]/gsd+principal[0]-origin[0]],
                     [0,pitch/gsd,-camera[1]/gsd+principal[1]-origin[1]],[0,0,1]],float)
    reduction=gsd/pitch;sigma=.5*math.sqrt(max(0,reduction*reduction-1))
    # Approximate antialiasing, not an empirically calibrated optical response.
    filtered=cv2.GaussianBlur(image,(0,0),sigma) if sigma>.15 else image
    rgb=cv2.warpPerspective(filtered,matrix,(size,size),flags=cv2.INTER_LINEAR,borderValue=(140,140,140))
    labels=cv2.warpPerspective(ids,matrix,(size,size),flags=cv2.INTER_NEAREST,borderValue=0)
    yy,xx=np.mgrid[:size,:size]
    sensor=(xx+origin[0]>=0)&(xx+origin[0]<profile['width'])&(yy+origin[1]>=0)&(yy+origin[1]<profile['height'])
    rgb[~sensor]=110;labels[~sensor]=0
    blur=float(rng.choice([0,.3,.5]));
    if blur:rgb=cv2.GaussianBlur(rgb,(3,3),blur)
    rgb=np.clip(rgb.astype(float)*rng.uniform(.85,1.1)+rng.normal(0,rng.uniform(.4,1.2),rgb.shape),0,255).astype(np.uint8)
    metadata={'profile_id':profile['id'],'full_sensor_width_px':profile['width'],'full_sensor_height_px':profile['height'],
              'horizontal_fov_deg':FOV,'focal_length_px':focal,'standoff_mm':profile['distance_mm'],
              'world_mm_per_pixel':gsd,'tile_origin_in_sensor_px':origin.tolist(),'native_tile_size_px':[size,size],
              'wall_patch_size_mm':[WALL_MM,WALL_MM],'source_mm_per_pixel':pitch,
              'camera_wall_center_mm':camera.tolist(),'stress_only':bool(profile.get('stress_only')),
              'projection':'Ideal frontal pinhole. Native sensor detail preserved in tile; no full-frame-to-640 resize.'}
    return rgb,labels,matrix,metadata


def build(output,scenes=256,seed=2026102301):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Choose a new output directory; existing datasets are preserved.')
    if scenes<20:raise ValueError('At least 20 source walls are required.')
    output.mkdir(parents=True)
    for name in ['images','masks','labels','annotations','preview']:(output/name).mkdir()
    order=np.random.default_rng(seed).permutation(scenes)
    split_map={int(s):'train' if rank<int(scenes*.75) else 'val' if rank<int(scenes*.875) else 'test' for rank,s in enumerate(order)}
    docs={s:{'images':[],'annotations':[],'categories':[{'id':i+1,'name':k} for i,k in enumerate(CLASSES)]} for s in ['train','val','test']}
    manifest=[];walls=[];annotation_id=0;excluded=Counter()
    for scene in range(scenes):
        scene_id=f'wall-v2-{scene:05d}';split=split_map[scene];wall_seed=seed+1009*scene
        rgb,mask,objects,negative=make_wall(wall_seed)
        for o in objects:o['physical_id']=scene_id+f"-D{o['instance_id']:03d}"
        walls.append({'scene_id':scene_id,'split':split,'seed':wall_seed,'negative_wall':negative,
                      'wall_patch_size_mm':[WALL_MM,WALL_MM],'objects':objects})
        for index,profile in enumerate(PROFILES):
            name=f'{scene_id}-view-{index:02d}';image_id=len(manifest)+1
            image,labels,matrix,camera=render(rgb,mask,profile,wall_seed+index*43+7)
            cv2.imwrite(str(output/'images'/f'{name}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(output/'masks'/f'{name}.png'),labels)
            rows=[];instances=[];polygon_eligible=True
            for o in objects:
                foreground=(labels==o['instance_id']).astype(np.uint8);area=int(foreground.sum())
                if not area:continue
                ys,xs=np.where(foreground);outline,iou=polygon(foreground)
                if outline is None or iou<.95:polygon_eligible=False
                else:rows.append(str(o['class_id'])+' '+' '.join(f'{v:.8f}' for v in (outline/PATCH_SIZE).flatten()))
                annotation_id+=1
                docs[split]['annotations'].append({'id':annotation_id,'image_id':image_id,'category_id':o['class_id']+1,
                    'segmentation':rle(foreground),'area':area,'bbox':[int(xs.min()),int(ys.min()),int(np.ptp(xs))+1,int(np.ptp(ys))+1],
                    'iscrowd':0,'physical_id':o['physical_id'],'instance_id':o['instance_id']})
                instance={'physical_id':o['physical_id'],'instance_id':o['instance_id'],'class_id':o['class_id'],
                          'area':area,'polygon_iou':round(iou,5),'physical_geometry':o,
                          'touches_tile_border':bool(xs.min()==0 or ys.min()==0 or xs.max()==PATCH_SIZE-1 or ys.max()==PATCH_SIZE-1)}
                if o['class_id']==0:
                    instance['nominal_full_object_size_px']={'trace_length':o['visible_trace_length_mm']/camera['world_mm_per_pixel'],
                        'opening_width':o['opening_width_mm']/camera['world_mm_per_pixel'],
                        'zigzag_spread':o['zigzag_spread_mm']/camera['world_mm_per_pixel']}
                instances.append(instance)
            eligible=polygon_eligible and not camera['stress_only']
            reason='resolution-stress view reserved outside training' if camera['stress_only'] else 'polygon conversion below 0.95 IoU' if not polygon_eligible else None
            if not eligible:excluded[reason]+=1
            (output/'labels'/f'{name}.txt').write_text('\n'.join(rows)+('\n' if rows else ''),encoding='utf-8')
            manifest.append({'image_id':image_id,'stem':name,'scene_id':scene_id,'split':split,'view':index,
                'homography_source_to_image':matrix.tolist(),'camera':camera,'instances':instances,'yolo_eligible':eligible,
                'polygon_eligible':polygon_eligible,'exclusion_reason':reason,
                'sha256':hashlib.sha256((output/'images'/f'{name}.jpg').read_bytes()).hexdigest()})
            docs[split]['images'].append({'id':image_id,'file_name':f'images/{name}.jpg','width':PATCH_SIZE,'height':PATCH_SIZE,'scene_id':scene_id})
            if scene<12 and index in [0,4]:
                cv2.imwrite(str(output/'preview'/f'{name}.jpg'),overlay(image,labels,objects))
        if (scene+1)%16==0:print(f'Generated {scene+1}/{scenes} calibrated source walls',flush=True)
    for split,doc in docs.items():
        write_json(output/'annotations'/f'{split}.coco.json',doc)
        paths=[f"./images/{m['stem']}.jpg" for m in manifest if m['split']==split and m['yolo_eligible']]
        (output/f'{split}.txt').write_text('\n'.join(paths)+'\n',encoding='utf-8')
    stress=[f"./images/{m['stem']}.jpg" for m in manifest if m['split']=='test' and m['camera']['stress_only'] and m['polygon_eligible']]
    (output/'resolution-stress-test.txt').write_text('\n'.join(stress)+'\n',encoding='utf-8')
    (output/'data.yaml').write_text('path: .\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES)),encoding='utf-8')
    write_json(output/'manifest.json',manifest);write_json(output/'wall-identities.json',walls)
    summary={'version':VERSION,'seed':seed,'wall_scenes':scenes,'views':len(manifest),'image_size':PATCH_SIZE,
        'source':'World-unit procedural 2D wall patches and ideal frontal pinhole native camera tiles; NOT AirSim renders or real photographs.',
        'target_crack_dimensions_mm':{'visible_trace_length':[50,100],'opening_width':5,'zigzag_spread':[20,50]},
        'source_wall_patch_mm':[WALL_MM,WALL_MM],'source_raster_size_px':SOURCE_SIZE,
        'split_unit':'Entire physical source wall and every view stay in one split.',
        'camera_profiles':PROFILES,'yolo_images':{s:sum(m['split']==s and m['yolo_eligible'] for m in manifest) for s in docs},
        'yolo_excluded_views':sum(excluded.values()),'exclusion_reasons':dict(excluded),
        'resolution_stress_test_views':len(stress),'negative_views':sum(not m['instances'] for m in manifest),
        'instance_observations':dict(Counter(CLASSES[o['class_id']] for m in manifest for o in m['instances'])),
        'limitations':'Known synthetic dimensions and ideal projection, not calibrated camera optics or verified detection. Actual AirSim evaluation, real imagery, occlusion and latency validation remain required. The live simulator camera is unchanged; higher-resolution profiles are design candidates.'}
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',type=int,default=256);p.add_argument('--seed',type=int,default=2026102301)
    a=p.parse_args();build(a.output,a.scenes,a.seed)
