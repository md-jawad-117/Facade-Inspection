"""Bounded alternative views from wall geometry and observed occupancy only."""
import math
from .room import wall_point
from .scene import segment_hits_box
from .available_view import route_to_view

VERSION='reachable-wall-verification-v1'
PLANNING_MARGIN_M=.5
MAX_RECHECK_SPEED_MPS=.35

def evidence_reasons(room,record,feature):
    x,y,w,h=feature['bbox'];margin=16/room.get('model_input_scale',1.)
    reasons=[]
    if x<=margin or y<=margin or x+w>=record['width']-margin or y+h>=record['height']-margin:reasons.append('Edge-clipped or edge-adjacent mask')
    if feature['confidence']<.5:reasons.append('Low model score; not a calibrated probability')
    quality=feature.get('inspection_quality',{})
    if quality.get('sharpness_laplacian_variance',100)>=0 and quality.get('sharpness_laplacian_variance',100)<20:
        reasons.append('Low local texture or sharpness')
    frame_quality=record.get('inspection_quality',{})
    if frame_quality.get('dark_fraction',0)>.4 or frame_quality.get('overexposed_fraction',0)>.2:reasons.append('Exposure warning')
    return reasons

def verification_outcome(room,record,features,target):
    from .room_vision import canonical_record
    from .vision import map_pixel
    wall=next(w for w in room['walls'] if w['id']==target['wall_id'])
    config,canonical=canonical_record(room,wall,record)
    point=target['inspection_point'];u=sum((p-o)*t for p,o,t in zip(point,wall['origin'],wall['tangent']));height=point[2]
    nearby=[];margin=16/room.get('model_input_scale',1.)
    for feature in features:
        location=map_pixel(config,canonical,feature['center_px'])
        if location is None:continue
        distance=math.hypot(location['y_m']-u,location['height_m']-height)
        if distance>.4:continue
        x,y,w,h=feature['bbox'];clipped=x<=margin or y<=margin or x+w>=record['width']-margin or y+h>=record['height']-margin
        nearby.append(dict(kind=feature['kind'],score=feature['confidence'],edge_clipped=clipped,distance_to_requested_point_m=round(distance,4),crop_file=feature.get('crop_file')))
    full=[f for f in nearby if not f['edge_clipped']];original=target.get('parent_kind')
    status='coverage_view_captured' if original=='coverage_gap' else 'class_disagreement' if any(f['kind']!=original for f in full) else 'clear_same_class_observation' if any(f['kind']==original for f in full) else 'not_confirmed'
    return dict(status=status,original_class=original,nearby_predictions=nearby,wall_u_height_m=[u,height],
        original_prediction_preserved=True,automatic_relabelling=False,note='Nearby class disagreement requires review; proximity alone does not prove identical physical damage.')

def choose_view(room,current,request,obstacles,remaining_distance_m=40.):
    wall=next(w for w in room['walls'] if w['id']==request['wall_id'])
    target=request['position'];offset=room.get('camera_forward_offset_m',0)
    # Requests originate at the measured wall location, at the standard body
    # standoff. Keep the camera normal to that wall; no yaw/pitch trick is used.
    u=sum((p-o)*t for p,o,t in zip(target,wall['origin'],wall['tangent']))
    h=target[2];surface=wall_point(wall,u,h,0)
    fov=room['camera']['horizontal_fov_deg'];half_width=room['standoff_m']*math.tan(math.radians(fov/2))
    half_height=half_width*room['camera']['height']/room['camera']['width']
    # Minimum contextual region around a clipped centre, plus image-edge margin.
    observed=request.get('observed_half_extent_m',[0.,0.])
    context_u=max(.2,observed[0]+.05);context_h=max(.2,observed[1]+.05)
    edge_m=half_width*2*16/(room['camera']['width']*room.get('model_input_scale',1.))
    offsets=[(0.,0.)]+[(du,dh) for du in [0.,-.5,.5,-.75,.75,-1.25,1.25] for dh in [0.,.25,.5,.75,1.,-.25,-.5,-.75]]
    seen=set();choices=[];rejected=[]
    c=room['clearance_m']+PLANNING_MARGIN_M
    for du,dh in offsets:
        if (du,dh) in seen:continue
        seen.add((du,dh));point=wall_point(wall,u+du,h+dh,room['standoff_m']+offset)
        reason=None
        if abs(du)+context_u+edge_m>=half_width or abs(dh)+context_h+edge_m>=half_height:reason='Insufficient image context or edge margin'
        elif not all(c<=p<=d-c for p,d in zip(point,room['size_m'])):reason='Outside buffered room bounds'
        elif any(segment_hits_box(point,point,o['lower'],o['upper'],c) for o in obstacles):reason='Buffered viewpoint occupied'
        camera=[p-offset*n for p,n in zip(point,wall['inward'])]
        region=[surface]+[wall_point(wall,u+a,h+b) for a in [-context_u,context_u] for b in [-context_h,context_h]]
        if reason is None and any(segment_hits_box(camera,p,o['lower'],o['upper'],.05) for p in region for o in obstacles):reason='Target region line of sight obstructed'
        if reason:
            rejected.append(dict(position=point,reason=reason));continue
        try:route,exact=route_to_view(room,current,point,obstacles,wall_id=wall['id'],allow_adjust=False,planning_margin=PLANNING_MARGIN_M)
        except (ValueError,RuntimeError) as error:
            rejected.append(dict(position=point,reason=str(error)));continue
        from .altitude_path import stage_path
        route['path']=stage_path(route['path'],obstacles,route['clearance_m'])
        route['path_length_m']=sum(math.dist(a,b) for a,b in zip(route['path'],route['path'][1:]))
        if route['path_length_m']>remaining_distance_m:
            rejected.append(dict(position=point,reason='Verification travel budget exceeded'));continue
        framing=abs(du)/half_width+abs(dh)/half_height
        score=route['path_length_m']+2*framing+.2*abs(dh)
        choice=dict(request,position=exact,requested_position=target,inspection_point=surface,
            view_type='exact' if du==dh==0 else 'alternative',planning_margin_m=PLANNING_MARGIN_M,
            maximum_speed_mps=MAX_RECHECK_SPEED_MPS,predicted_route=route,score=round(score,4),
            contextual_half_size_m=[context_u,context_h],wall_offset_u_height_m=[du,dh])
        choices.append(choice)
        if du==dh==0:break  # A safe exact view needs no search detour.
    selected=min(choices,key=lambda v:(v['score'],v['position'])) if choices else None
    return dict(version=VERSION,selected=selected,candidates_considered=len(seen),rejected=rejected,
        feasible_candidates=len(choices),new_observation_obtained=False,
        scope='Known/observed box geometry and ideal camera framing only; actual motion and capture still require flight guards.')
