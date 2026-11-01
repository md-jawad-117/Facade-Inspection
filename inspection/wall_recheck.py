"""Choose centred reinspection views from detections, never reference labels."""
import json,math
from pathlib import Path
from .room import wall_point,to_ned
from .room_vision import canonical_record
from .vision import map_pixel
from .coordinates import vehicle_position
from .camera import capture


def requests(room,record,features,scheduled):
    wall=next(w for w in room['walls'] if w['id']==record['wall_id'])
    config,canonical=canonical_record(room,wall,record)
    margin=16/room.get('model_input_scale',1.)
    for feature in sorted(features,key=lambda f:-f['confidence']):
        x,y,w,h=feature['bbox']
        reasons=['Centre an edge-clipped detection using its measured wall position']
        if room.get('adaptive_reinspection'):
            from .adaptive_view import evidence_reasons
            reasons=evidence_reasons(room,record,feature)
            if not reasons:continue
        elif x>margin and y>margin and x+w<record['width']-margin and y+h<record['height']-margin:continue
        location=map_pixel(config,canonical,feature['center_px'])
        if location is None:continue
        u,height=location['y_m'],location['height_m']
        # This cell only schedules repeat images; it does not merge identities.
        key=(wall['id'],round(u/.3),round(height/.3),feature['kind'])
        if key in scheduled:continue
        if len(scheduled)>=room.get('maximum_centred_rechecks',16):break
        margin_m=room['clearance_m']+.3
        u=max(margin_m,min(wall['length']-margin_m,u));height=max(margin_m,min(room['size_m'][2]-margin_m,height))
        corners=[map_pixel(config,canonical,p) for p in [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]]
        corners=[p for p in corners if p is not None]
        observed_half_extent=[max([abs(p['y_m']-u) for p in corners] or [0.]),max([abs(p['height_m']-height) for p in corners] or [0.])]
        scheduled.add(key)
        yield dict(id=f"{wall['id']}-recheck-{len(scheduled):03d}",position=wall_point(wall,u,height,room['standoff_m']+room.get('camera_forward_offset_m',0)),
                   yaw_deg=wall['yaw_deg'],wall_id=wall['id'],parent_file=record['file'],parent_bbox=feature['bbox'],parent_kind=feature['kind'],observed_half_extent_m=observed_half_extent,reason='; '.join(reasons))


def face_wall(pilot,yaw):
    for _ in range(30):
        state,_=pilot.observe('verification-camera-settle');q=state['kinematics_estimated']['orientation']
        w,x,y,z=[q[k] for k in ['w_val','x_val','y_val','z_val']]
        actual=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        if abs((actual-yaw+180)%360-180)<2:break
        altitude=vehicle_position(to_ned(pilot.room,pilot.last_target),pilot.summary['coordinate_calibration'])[2]
        pilot.client.call('moveByVelocityZ',0.,0.,altitude,.2,0,{'is_rate':False,'yaw_or_rate':yaw},'',timeout=4)
    else:raise RuntimeError('Verification camera did not face the inspection wall')
    pilot.client.call('hover','',timeout=10)


def inspect_edges(pilot,room,record,scheduled):
    if not record.get('trained_prediction_file') or not record.get('wall_id'):return
    directory=pilot.directory
    features=json.loads((directory/record['trained_prediction_file']).read_text())['findings']
    for target in requests(room,record,features,scheduled):
        perform_verification(pilot,room,record,target)


def perform_verification(pilot,room,record,target):
        directory=pilot.directory
        # A shifted viewpoint cannot centre the requested wall location. Skip
        # optional views before motion rather than sliding beside an obstacle.
        from .available_view import route_to_view
        _,current=pilot.observe('verification-preflight')
        obstacles=room['obstacles']+(pilot.sensor.map.boxes() if pilot.sensor else [])
        if room.get('adaptive_reinspection'):
            from .adaptive_view import choose_view
            remaining=room.get('adaptive_travel_budget_m',40.)-pilot.summary.get('adaptive_planned_travel_m',0.)
            decision=choose_view(room,current,target,obstacles,remaining)
            pilot.summary.setdefault('adaptive_view_decisions',[]).append(decision)
            if decision['selected'] is None:
                pilot.summary.setdefault('skipped_centred_verification_views',[]).append(dict(target,reason='No reachable view with sufficient context inside the verification budget'))
                (directory/'flight.json').write_text(json.dumps(pilot.summary,indent=2))
                print('Skipped optional verification '+target['id']+': no valid alternative view',flush=True)
                return
            target=decision['selected']
            pilot.summary['adaptive_planned_travel_m']=pilot.summary.get('adaptive_planned_travel_m',0.)+target['predicted_route']['path_length_m']
        try:
            route_to_view(room,current,target['position'],obstacles,
                          wall_id=target['wall_id'],allow_adjust=False,planning_margin=target.get('planning_margin_m',.25))
        except (ValueError,RuntimeError) as error:
            pilot.summary.setdefault('skipped_centred_verification_views',[]).append(
                dict(target,reason='Exact centred viewpoint unavailable: '+str(error)))
            (directory/'flight.json').write_text(json.dumps(pilot.summary,indent=2))
            print('Skipped optional verification '+target['id']+': '+str(error),flush=True)
            return
        print(('Adaptive verification ' if room.get('adaptive_reinspection') else 'Centred verification ')+target['id']+' from '+record['file'],flush=True)
        options=dict(allow_adjust=False)
        if room.get('adaptive_reinspection'):options.update(planning_margin=target['planning_margin_m'],max_speed=target['maximum_speed_mps'])
        pilot.move(target['position'],target['yaw_deg'],target['id'],target['wall_id'],**options);face_wall(pilot,target['yaw_deg'])
        name=Path(record['file']).stem+'-'+target['id']
        extra=capture(pilot.client,directory,name,with_depth=True)
        extra.update(wall_id=target['wall_id'],target=target,achieved_target=pilot.last_target,actual_position=pilot.observe('verification-capture')[1],
                     verification_parent_file=record['file'],verification_parent_bbox=target['parent_bbox'])
        wall=next(w for w in room['walls'] if w['id']==target['wall_id']);origin=room['airsim_origin_ned_m'];p=extra['camera_position']
        local=[p['x_val']-origin[0],p['y_val']-origin[1],origin[2]-p['z_val']]
        distance=sum((a-b)*n for a,b,n in zip(local,wall['origin'],wall['inward']))
        extra['measured_camera_standoff_m']=distance
        if abs(distance-room['standoff_m'])>.2:raise RuntimeError('Verification camera standoff outside tolerance')
        from .room_live_model import inspect_capture
        inspect_capture(directory,room,extra);pilot.observe('verification-inference-check')
        if room.get('adaptive_reinspection'):
            from .adaptive_view import verification_outcome
            predictions=json.loads((directory/extra['trained_prediction_file']).read_text())['findings']
            extra['verification_outcome']=verification_outcome(room,extra,predictions,target)
        pilot.summary['images'].append(extra);pilot.summary.setdefault('centred_verification_views',[]).append(target)
        if room.get('adaptive_reinspection'):
            decision['new_observation_obtained']=True
            decision['actual_capture_file']=extra['file']
            decision['verification_outcome']=extra['verification_outcome']
        (directory/'flight.json').write_text(json.dumps(pilot.summary,indent=2))
        print('Verification saved '+extra['file'],flush=True)


def inspect_gaps(pilot,room,scheduled):
    if not room.get('adaptive_reinspection') or room.get('maximum_gap_rechecks',4)<=0:return
    from .inspection_completeness import assess,gap_requests
    result=assess(room,pilot.summary['images'],pilot.directory)
    for request in gap_requests(room,result,pilot.summary['images'],scheduled):
        parent=next(r for r in pilot.summary['images'] if r['file']==request['parent_file'])
        perform_verification(pilot,room,parent,request)
