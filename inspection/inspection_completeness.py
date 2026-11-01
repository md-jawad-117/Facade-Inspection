"""Sampled visibility and declared inspection-quality checks, never clean-wall claims."""
import json,math
from pathlib import Path
import numpy as np
from .room import wall_point,to_ned
from .mission_scope import inspection_walls
from .vision import rotation

VERSION='wall-completeness-v1'

def quality_passes(quality):
    return (quality.get('sharpness_laplacian_variance',0)>=20 and
            quality.get('dark_fraction',1)<=.4 and quality.get('overexposed_fraction',1)<=.2)

def cell_status(seen,adequate,pending=False):
    if not seen:return 'not_inspected'
    return 'adequately_inspected' if adequate and not pending else 'needs_verification'

def assess(room,records,directory,step=.3):
    if not .1<=step<=1:raise ValueError('Unsupported completeness cell spacing')
    directory=Path(directory);analysis={}
    if (directory/'model-analysis.json').exists():analysis=json.loads((directory/'model-analysis.json').read_text())
    frames={f['file']:f for f in analysis.get('frames',[])}
    target_mm=room.get('inspection_target_opening_mm',5. if room.get('scene_profile')=='original' else 20.)
    if not math.isfinite(target_mm) or target_mm<=0:raise ValueError('Invalid declared opening target')
    max_gsd=target_mm/1000/3
    result=dict(version=VERSION,step_m=step,policy=dict(target_opening_mm=target_mm,minimum_model_pixels_across=3,
        maximum_model_gsd_m=max_gsd,minimum_laplacian_variance=20,maximum_dark_fraction=.4,maximum_overexposed_fraction=.2,
        note='Heuristic quality checks. Low texture and blur are not distinguished. Three model pixels is a sampling requirement, not validated detection accuracy.'),walls={},
        scope='Saved camera poses and planar depth at wall-cell centres. Adequate under these declared checks does not mean defect-free or structurally safe.')
    for wall in inspection_walls(room):
        nx=math.ceil(wall['length']/step);ny=math.ceil(wall['height']/step)
        uv=np.array([[(i+.5)*wall['length']/nx,(j+.5)*wall['height']/ny] for j in range(ny) for i in range(nx)])
        points=np.array([to_ned(room,wall_point(wall,u,h)) for u,h in uv]);seen=np.zeros(len(points),bool);good=seen.copy();framed=seen.copy()
        observed_by=[[] for _ in points];frame_quality=[]
        local_records=[r for r in records if r.get('wall_id')==wall['id']]
        for record in local_records:
            q=frames.get(record['file'],{}).get('quality') or record.get('inspection_quality')
            if not q:
                import cv2
                from .vision import image_quality
                q=image_quality(cv2.imread(str(directory/record['file'])))
            q=dict(q)
            # A mostly uniform plaster wall has low whole-image variance even
            # when its visible detail is sharp. Keep global exposure checks,
            # but permit focus evidence from a model-mask-local image patch.
            if q.get('sharpness_laplacian_variance',0)<20:
                features=frames.get(record['file'],{}).get('findings')
                if features is None and record.get('trained_prediction_file'):
                    features=json.loads((directory/record['trained_prediction_file']).read_text())['findings']
                local=[];image=None
                for feature in features or []:
                    detail=feature.get('inspection_quality')
                    if detail is None:
                        import cv2
                        from .vision import image_quality
                        if image is None:image=cv2.imread(str(directory/record['file']))
                        x,y,w,h=feature['bbox'];patch=image[max(0,y-24):min(image.shape[0],y+h+24),max(0,x-24):min(image.shape[1],x+w+24)]
                        detail=image_quality(patch)
                    local.append(detail.get('sharpness_laplacian_variance',0))
                if local:
                    q['whole_frame_sharpness_laplacian_variance']=q.get('sharpness_laplacian_variance',0)
                    q['sharpness_laplacian_variance']=max(q.get('sharpness_laplacian_variance',0),max(local))
                    q['sharpness_basis']='Strongest visible mask-local image detail; assumed representative of focus on the same wall plane'
            R=rotation(record['camera_orientation']);origin=np.array([record['camera_position'][k] for k in ['x_val','y_val','z_val']])
            cam=(points-origin)@R;forward=cam[:,0];f=record['width']/(2*math.tan(math.radians(record['fov_deg']/2)))
            safe=np.maximum(forward,1e-9);u=(record['width']-1)/2+f*cam[:,1]/safe;v=(record['height']-1)/2+f*cam[:,2]/safe
            inside=(forward>0)&(u>=0)&(u<record['width']-1)&(v>=0)&(v<record['height']-1);framed|=inside
            indices=np.flatnonzero(inside);depth=np.fromfile(directory/record['depth_file'],np.float32).reshape(record['height'],record['width'])
            measured=depth[np.rint(v[indices]).astype(int),np.rint(u[indices]).astype(int)]
            visible=indices[np.isfinite(measured)&(measured>0)&(np.abs(measured-forward[indices])<=.18)]
            seen[visible]=True
            for i in visible:
                if len(observed_by[i])<6:observed_by[i].append(record['file'])
            scale=room.get('model_input_scale',1.);margin=16/scale
            cosine=abs(float(np.dot(R[:,0],-np.array(wall['inward']))))
            gsd=forward[visible]/(f*scale*max(cosine,1e-9))
            accepted=(u[visible]>margin)&(u[visible]<record['width']-margin)&(v[visible]>margin)&(v[visible]<record['height']-margin)&(gsd<=max_gsd)&(cosine>=.85)
            if quality_passes(q):good[visible[accepted]]=True
            frame_quality.append(dict(file=record['file'],quality=q,quality_passed=quality_passes(q),visible_cells=len(visible),
                adequate_cells_in_frame=int(np.count_nonzero(accepted)) if quality_passes(q) else 0))
        pending=np.zeros(len(points),bool);pending_ids=[]
        for record in local_records:
            outcome=record.get('verification_outcome',{})
            if outcome.get('status') in ['class_disagreement','not_confirmed']:
                target=np.array(outcome['wall_u_height_m'])
                pending|=np.all(np.abs(uv-target)<=max(.2,step/2),axis=1)
                pending_ids.append('verification:'+record['file'])
        if analysis:
            from .room_vision import canonical_record
            from .association import project
            record_by_file={r['file']:r for r in local_records}
            for finding in [f for f in analysis.get('findings',[]) if f['wall_id']==wall['id']]:
                boxes=[];clipped=[]
                for frame in frames.values():
                    if frame['file'] not in record_by_file:continue
                    record=record_by_file[frame['file']];config,canonical=canonical_record(room,wall,record)
                    for feature in frame['findings']:
                        if feature['id']!=finding['id']:continue
                        x,y,w,h=feature['bbox'];margin=16/room.get('model_input_scale',1.)
                        clipped.append(x<=margin or y<=margin or x+w>=record['width']-margin or y+h>=record['height']-margin)
                        corners=project(config,canonical,[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],rotation(canonical['camera_orientation']))
                        if len(corners):boxes.append(corners)
                if boxes and (finding['identity_status']=='ambiguous' or clipped and all(clipped)):
                    corners=np.concatenate(boxes);low=corners.min(axis=0)-step/2;high=corners.max(axis=0)+step/2
                    pending|=np.all((uv>=low)&(uv<=high),axis=1);pending_ids.append(finding['id'])
        cells=[]
        for i,(u,h) in enumerate(uv):
            status=cell_status(bool(seen[i]),bool(good[i]),bool(pending[i]));reasons=[]
            if not seen[i]:reasons.append('Framed but not depth-visible / obstructed' if framed[i] else 'No recorded view')
            elif pending[i]:reasons.append('Unresolved edge-only or ambiguous model observation')
            elif not good[i]:reasons.append('Exposure, sharpness/texture, edge margin, angle or declared resolution check not satisfied')
            cells.append(dict(index=i,u_m=round(float(u),4),height_m=round(float(h),4),status=status,reasons=reasons,observed_by=observed_by[i]))
        counts={status:sum(c['status']==status for c in cells) for status in ['adequately_inspected','needs_verification','not_inspected']}
        result['walls'][wall['id']]=dict(length_m=wall['length'],height_m=wall['height'],grid_width=nx,grid_height=ny,
            counts=counts,percentages={k:round(100*v/len(cells),2) for k,v in counts.items()},cells=cells,
            unresolved_candidate_ids=pending_ids,frame_quality=frame_quality)
    return result

def gap_requests(room,completeness,records,scheduled):
    remaining=room.get('maximum_centred_rechecks',16)-len(scheduled)
    budget=min(room.get('maximum_gap_rechecks',4),remaining)
    chosen=[];queues={};parents_by_wall={}
    for key,detail in completeness['walls'].items():
        wall=next(w for w in room['walls'] if w['id']==key)
        parents=[r for r in records if r.get('wall_id')==key and not r.get('verification_parent_file')]
        if not parents:continue
        margin=room['clearance_m']+.3
        bad=[c for c in detail['cells'] if c['status']!='adequately_inspected' and margin<=c['u_m']<=wall['length']-margin and margin<=c['height_m']<=wall['height']-margin]
        candidates=bad[::max(1,len(bad)//80)]
        def gain(c):return sum((2 if p['status']=='not_inspected' else 1) for p in bad if abs(p['u_m']-c['u_m'])<1.5 and abs(p['height_m']-c['height_m'])<.8)
        queues[key]=sorted(candidates,key=lambda c:(-gain(c),c['u_m'],c['height_m']));parents_by_wall[key]=parents[-1]
    while budget>0 and any(queues.values()):
        for key,queue in queues.items():
            if budget<=0:return
            while queue:
                cell=queue.pop(0);u,h=cell['u_m'],cell['height_m'];token=(key,round(u/.3),round(h/.3),'coverage_gap')
                if token in scheduled or any(k==key and abs(a-u)<1.5 and abs(b-h)<.8 for k,a,b in chosen):continue
                scheduled.add(token);chosen.append((key,u,h));budget-=1
                wall=next(w for w in room['walls'] if w['id']==key)
                yield dict(id=f'{key}-gap-{len(scheduled):03d}',wall_id=key,
                    position=wall_point(wall,u,h,room['standoff_m']+room.get('camera_forward_offset_m',0)),yaw_deg=wall['yaw_deg'],
                    parent_file=parents_by_wall[key]['file'],parent_bbox=None,parent_kind='coverage_gap',reason='Inspect an unresolved completeness cell')
                break

def write_report(directory,result,output_directory=None):
    import html
    directory=Path(directory);out=Path(output_directory) if output_directory else directory;out.mkdir(parents=True,exist_ok=True)
    (out/'inspection-completeness.json').write_text(json.dumps(result,indent=2))
    palette={'adequately_inspected':'#287c57','needs_verification':'#e4a826','not_inspected':'#4c555e'};sections=[]
    for key,wall in result['walls'].items():
        nx,ny=wall['grid_width'],wall['grid_height'];height=900*wall['height_m']/wall['length_m'];marks=''
        for c in wall['cells']:
            i=c['index']%nx;j=c['index']//nx
            title=f"u {c['u_m']} m, height {c['height_m']} m: {c['status']}. {'; '.join(c['reasons'])}"
            marks+=f'<rect x="{i*900/nx}" y="{(ny-1-j)*height/ny}" width="{900/nx+.1}" height="{height/ny+.1}" fill="{palette[c["status"]]}"><title>{html.escape(title)}</title></rect>'
        numbers=' · '.join(k.replace('_',' ')+': '+str(v)+'%' for k,v in wall['percentages'].items())
        sections.append(f'<section><h2>Wall {key}</h2><p>{numbers}</p><svg viewBox="0 0 900 {height}" role="img" aria-label="Inspection quality map for wall {key}">{marks}</svg><p>Horizontal: along-wall metres; bottom to top: wall height. Hover cells for reasons.</p></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wall inspection completeness</title><style>body{font:16px system-ui;background:#edf1eb;color:#20372d;margin:auto;max-width:1250px;padding:28px}section{background:white;padding:22px;border-radius:12px;margin:20px 0}p{line-height:1.6}svg{width:100%;height:auto}a{color:#165d45}</style><a href="/assets/inspection-system/index.html">Inspection controls</a><h1>What was adequately inspected?</h1><p>Green: passed the declared checks. Amber: needs verification. Charcoal: not inspected or obstructed. Adequate does not mean defect-free. Low texture can trigger the same warning as blur.</p><p>Declared target opening: '''+str(result['policy']['target_opening_mm'])+' mm, with at least 3 model-input pixels across it. This sampling rule is not validated detection accuracy.</p>'+''.join(sections)+'<p>'+html.escape(result['scope'])+'</p><p><a href="inspection-completeness.json">Policy, cells and frame-quality evidence</a></p></html>'
    (out/'inspection-completeness.html').write_text(page,encoding='utf-8')
