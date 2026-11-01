"""Trained masks and wall-local association for recorded rectangular rooms."""
import hashlib
import html
import json
import math
from pathlib import Path
import cv2
import numpy as np
from .room import to_ned
from .vision import rotation,wall_roi,map_pixel,detect,annotate,image_quality
from . import association


def canonical_record(room,wall,record):
    """Rigid world->wall frame: forward into wall, right along wall, NED down."""
    origin=np.array(to_ned(room,wall['origin']),float)
    inward=np.array(wall['inward'],float);tangent=np.array(wall['tangent'],float)
    basis=np.array([-inward,tangent,[0,0,1]],float)
    if not np.allclose(basis@basis.T,np.eye(3)) or np.linalg.det(basis)<.99:
        raise ValueError('Wall frame must be an orthonormal right-handed vertical plane.')
    p=np.array([record['camera_position'][k] for k in ['x_val','y_val','z_val']])
    local=basis@(p-origin)
    angle=math.atan2(basis[1,0],basis[0,0])/2
    c,s=math.cos(angle),math.sin(angle)
    w,x,y,z=[record['camera_orientation'][k] for k in ['w_val','x_val','y_val','z_val']]
    q={'w_val':c*w-s*z,'x_val':c*x-s*y,'y_val':c*y+s*x,'z_val':c*z+s*w}
    transformed=dict(record,camera_position=dict(zip(['x_val','y_val','z_val'],map(float,local))),camera_orientation=q,wall_id=wall['id'])
    config={'wall_x_m':0.,'wall_y_start_m':0.,'width_m':wall['length'],
            'bottom_height_m':0.,'height_m':wall['height']}
    return config,transformed


def world_location(room,wall,u,height):
    local=[wall['origin'][0]+u*wall['tangent'][0],wall['origin'][1]+u*wall['tangent'][1],height]
    return {'wall_id':wall['id'],'along_wall_m':round(float(u),4),'height_m':round(float(height),4),
            'world_ned_m':to_ned(room,local)}


def analyze_room(directory,sensitivity='balanced'):
    from .model_client import ensure_worker
    directory=Path(directory);flight=json.loads((directory/'flight.json').read_text())
    from .mission_scope import inspection_walls
    room=json.loads((directory/'room.json').read_text());walls={w['id']:w for w in inspection_walls(room)}
    policies={}
    for key,wall in walls.items():
        transformed=[canonical_record(room,wall,r) for r in flight['images'] if r.get('wall_id')==key]
        if transformed:policies[key]=association.small_crack_policy(transformed[0][0],[item[1] for item in transformed],model_input_scale=room.get('model_input_scale',1.) if room.get('association_model_resolution',False) else 1.)
    meta=ensure_worker();by_wall={key:[] for key in walls};frames=[];images={}
    for index,record in enumerate(flight['images']):
        wall_id=record.get('wall_id')
        if wall_id not in walls:continue
        config,canonical=canonical_record(room,walls[wall_id],record)
        image=cv2.imread(str(directory/record['file']))
        if image is None:raise ValueError('Unreadable capture: '+record['file'])
        depth=np.fromfile(directory/record['depth_file'],np.float32).reshape(record['height'],record['width'])
        roi=wall_roi(config,canonical,depth)
        if record.get('trained_prediction_file'):
            cached=json.loads((directory/record['trained_prediction_file']).read_text())
            if cached.get('model_input_scale',1.0)!=room.get('model_input_scale',1.0) or cached['model']['sha256']!=meta['sha256'] or cached['image_sha256']!=hashlib.sha256((directory/record['file']).read_bytes()).hexdigest() or cached['depth_sha256']!=hashlib.sha256((directory/record['depth_file']).read_bytes()).hexdigest():
                raise RuntimeError('Recorded trained prediction input/model mismatch')
            if cached.get('wall_support_policy')!=room.get('wall_support_policy'):
                features=detect(image,roi,sensitivity,backend='trained',input_scale=room.get('model_input_scale',1.0))
            else:features=cached['findings']
        else:features=detect(image,roi,sensitivity,backend='trained',input_scale=room.get('model_input_scale',1.0))
        from .wall_support import partition
        features,excluded=partition(features,room.get('wall_support_policy'))
        if record.get('trained_prediction_file') and cached.get('wall_support_policy')==room.get('wall_support_policy'):
            excluded=cached.get('excluded_findings',[])
        from .frame_postprocess import apply as postprocess
        expected=room.get('frame_postprocess','recorded-v1')
        recorded=cached.get('frame_postprocess','recorded-v1') if record.get('trained_prediction_file') and cached.get('wall_support_policy')==room.get('wall_support_policy') else 'recorded-v1'
        if recorded!=expected:
            if recorded!='recorded-v1':raise RuntimeError('Recorded postprocessing version mismatch')
            features=postprocess(features,expected)
        common=cv2.erode(roi,np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        for feature in features:
            point=map_pixel(config,canonical,feature['center_px'])
            if point is None:continue
            feature['location']=world_location(room,walls[wall_id],point['y_m'],point['height_m'])
            by_wall[wall_id].append(association.observation(config,canonical,feature,common,rotation(canonical['camera_orientation']),policy=policies[wall_id]))
        frame={'file':record['file'],'wall_id':wall_id,'annotated_file':record['file'].replace('.png','-model.png'),
               'findings':features,'excluded_findings':excluded,'wall_roi_fraction':float(np.mean(roi>0)),'quality':image_quality(image)}
        frames.append(frame)
        if (index+1)%10==0:print(f'Trained room analysis {index+1}/{len(flight["images"])}',flush=True)
    findings=[];summaries={}
    for wall_id,observations in by_wall.items():
        backend=room.get('association_backend','legacy')
        if backend=='wall-footprint-map-v4':
            from .wall_tracks import associate
            groups,ambiguous,details=associate(observations)
        elif backend=='legacy':groups,ambiguous,details=association.associate(observations)
        else:raise ValueError('Unknown wall association backend')
        for group in groups:
            members=[observations[i] for i in sorted(group)]
            key=hashlib.sha256((wall_id+'|'+meta['sha256']+'|'+'|'.join(sorted(o['key'] for o in members))).encode()).hexdigest()
            support=set().union(*(o['support'] for o in members))
            if support:point=np.mean(list(support),axis=0)*members[0]['policy']['grid_m']
            else:point=[members[0]['feature']['location']['along_wall_m'],members[0]['feature']['location']['height_m']]
            feature=members[0]['feature']
            member_keys={m['key'] for m in members}
            resolutions=[v for v in details.get('class_resolutions',[]) if set(v['source_keys'])<=member_keys and set(v['target_keys'])<=member_keys]
            resolved_kind=resolutions[0]['resolved_kind'] if resolutions else feature['kind']
            finding={'id':wall_id+'-'+key[:10].upper(),'wall_id':wall_id,'kind':resolved_kind,
                     'confidence':max(m['feature']['confidence'] for m in members if m['kind']==resolved_kind),
                     'class_resolution':resolutions,
                     'location':world_location(room,walls[wall_id],point[0],point[1]),
                     'identity_status':'ambiguous' if group&ambiguous else 'multi-view-supported' if len(group)>1 else 'single-view',
                     'observations':[{'file':m['file'],'bbox':m['feature']['bbox'],'observation_key':m['key'],'raw_kind':m['kind']} for m in members]}
            for member in members:member['feature']['id']=finding['id']
            findings.append(finding)
        summaries[wall_id]={'observation_count':len(observations),'candidate_tracks':len(groups),'association':details}
    for frame in frames:annotate(cv2.imread(str(directory/frame['file'])),frame['findings'],directory/frame['annotated_file'])
    result={'trained_model':True,'frame_postprocess':room.get('frame_postprocess','recorded-v1'),'model_input_scale':room.get('model_input_scale',1.0),'model':meta,'sensitivity':sensitivity,'frames':frames,'findings':findings,'walls':summaries,
            'unique_physical_count_verified':False,
            'scope':'Trained masks from recorded room frames, using verified live-prediction caches when available. Depth-validated wall ROI and separate wall-local association. Candidate identities require independent physical-count evaluation.'}
    temporary=directory/'model-analysis.pending.json';temporary.write_text(json.dumps(result,indent=2));temporary.replace(directory/'model-analysis.json')
    write_room_model_report(directory,result)
    return result


def write_room_model_report(directory,value):
    directory=Path(directory);sections=''
    for wall_id,summary in value['walls'].items():
        tracks=[f for f in value['findings'] if f['wall_id']==wall_id]
        sections+=f'<section><h2>Wall {wall_id}</h2><p>{summary["observation_count"]} observations → {len(tracks)} candidate tracks.</p>'
        for track in tracks:
            for decision in track.get('class_resolution',[]):
                evidence=decision['evidence']
                descriptions=[]
                for item in evidence:
                    descriptions.append(html.escape(item['source_file'])+' originally '+html.escape(item['original_kind'])+'; supporting views: '+html.escape(', '.join(item['supporting_files']))+'; minimum exact common-support agreement '+str(round(100*item['minimum_exact_common_support_agreement'],2))+'%')
                sections+='<p><strong>Track '+html.escape(track['id'])+': provisional class consensus → '+html.escape(decision['resolved_kind'])+'.</strong> '+'; '.join(descriptions)+'. Original image classes, masks and per-frame errors remain unchanged. This is an automatic identity decision, not human confirmation.</p>'
        for frame in [f for f in value['frames'] if f['wall_id']==wall_id]:
            sections+=f'<details><summary>{html.escape(frame["file"])} · {len(frame["findings"])} detections</summary><img loading="lazy" src="{frame["annotated_file"]}" alt="Trained instance masks on saved camera image"></details>'
        rejected=[dict(file=f['file'],**item) for f in value['frames'] if f['wall_id']==wall_id for item in f.get('excluded_findings',[])]
        if rejected:
            sections+='<h3>Predictions excluded from wall identities</h3><p>Original model masks remain available in the JSON. These masks were mostly outside the depth-verified wall; they are not silently deleted or declared false physical defects.</p>'
            for item in rejected:
                sections+='<p>'+html.escape(item['file'])+': '+html.escape(item['kind'])+', wall-supported fraction '+str(round(100*item['wall_support_fraction'],2))+'%. '+html.escape(item['exclusion_reason'])+'.</p>'
        sections+='</section>'
    scene_note=('This run includes independently labeled texture defects. See the physical-count verification for matched IDs, misses, splits and incorrect merges.' if (directory/'truth/truth.json').exists() else 'No independent labeled defect scene is attached to this run; candidate counts are unverified.')
    count_link=('<a href="count-verification/index.html">Physical-count verification</a> · ' if (directory/'count-verification/results.json').exists() else '')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trained room inspection replay</title><style>body{{font:16px system-ui;background:#eef0ec;color:#243b35;margin:0}}main{{max-width:1050px;margin:auto;padding:28px}}p{{line-height:1.6}}section{{background:#fff;padding:20px;margin:20px 0;border-radius:12px}}details{{padding:10px}}img{{width:100%;height:auto}}a{{color:#126a52}}</style><main><p>TRAINED MODEL · RECORDED ROOM IMAGES</p><h1>Four walls, separate defect identities</h1><p>{len(value["frames"])} saved frames analyzed using your actual YOLO26 checkpoint. Only depth-verified wall pixels contribute to detections; machinery in front of a wall is excluded. Wall IDs remain separate when associating repeated masks.</p><p>{html.escape(value["scope"])} {scene_note}</p><p>{count_link}<a href="report.html">Original flight evidence</a> · <a href="model-analysis.json">Complete predictions</a> · <a href="/assets/inspection-system/index.html">Inspection controls</a></p>{sections}</main></html>'''
    (directory/'model-report.html').write_text(page,encoding='utf-8')
