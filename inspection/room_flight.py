"""Controlled room flight with sampled telemetry and RGB/depth evidence.

Supports optional live trained inspection and depth stop/replanning. All movement is simulated.
"""
import array
import json
import math
import time
from pathlib import Path
from .navigation import plan_transit
from .room import to_ned,wall_point,in_free_space
from .scene import segment_hits_box,mesh_bounds
from .coverage import visible_samples,rotate_inverse
from .rpc import AirSimRPC
from .coordinates import measure_scene_offset,scene_position,vehicle_position
from .camera import capture,position,verify_takeoff,ground_contact


def from_ned(room,point):
    origin=room['airsim_origin_ned_m']
    return [point[0]-origin[0],point[1]-origin[1],origin[2]-point[2]]


def clearance(room,point):
    """Euclidean center distance to map surfaces, not vehicle-body clearance."""
    distances=[p for p in point]+[d-p for p,d in zip(point,room['size_m'])]
    for item in room['obstacles']:
        distances.append(math.sqrt(sum(max(a-p,0,p-b)**2 for p,a,b in zip(point,item['lower'],item['upper']))))
    return min(distances)


def velocity_command(current,target,max_speed=.75):
    delta=[b-a for a,b in zip(current,target)]
    norm=math.sqrt(sum(d*d for d in delta))
    factor=min(.85,max_speed/max(norm,1e-9))
    return [d*factor for d in delta]


def static_route(room,start,goal):
    # Extra 0.25 m planning margin leaves room for measured tracking error.
    c=room['clearance_m']+.25
    return plan_transit(start,goal,room['obstacles'],[[c]*3,[d-c for d in room['size_m']]],
                        resolution=.25,clearance=c)


def validate_launch_column(room,launch):
    ground=[launch[0],launch[1],.15]
    c=room['clearance_m']
    if not all(c<=p<=d-c for p,d in zip(ground[:2],room['size_m'][:2])):
        raise ValueError('Launch point is too close to a room boundary.')
    top=[launch[0],launch[1],max(launch[2],3.3)]
    if any(segment_hits_box(ground,top,o['lower'],o['upper'],c) for o in room['obstacles']):
        raise ValueError('Takeoff/landing column is blocked.')
    return ground


def short_targets(room):
    body_standoff=room['standoff_m']+room.get('camera_forward_offset_m',0)
    return [
        {'id':'south-check','position':[4,body_standoff,2.5],'yaw_deg':-90,'wall_id':'S'},
        {'id':'machine-detour-check','position':[4,9,2.5],'yaw_deg':180,'wall_id':None},
        {'id':'west-check-1','position':[body_standoff,9,2.5],'yaw_deg':180,'wall_id':'W'},
        {'id':'west-check-2','position':[body_standoff,13,3.2],'yaw_deg':180,'wall_id':'W'}]


def verify_scene(client,room):
    """Verify room actors and reject unrelated physical geometry in the room."""
    from .environment import pieces
    actual={};foreign=[]
    for mesh in client.call('simGetMeshPositionVertexBuffers',timeout=30):
        name=mesh['name'].decode() if isinstance(mesh['name'],bytes) else mesh['name']
        lo,hi=mesh_bounds(mesh);key=name.lower()
        if key.startswith('facaderoom_'):actual[key]=[lo,hi]
        elif not any(term in key for term in ['simpleflight','camera']):
            local=[from_ned(room,p) for p in [lo,hi]]
            # Strict interior overlap excludes underlying original ground plane.
            low=[min(a,b) for a,b in zip(*local)];high=[max(a,b) for a,b in zip(*local)]
            if all(a<d-.05 and b>.05 for a,b,d in zip(low,high,room['size_m'])):foreign.append(name)
    if foreign:raise ValueError('Unmapped scene geometry in room: '+', '.join(foreign))
    for piece in pieces(room):
        key=('FacadeRoom_'+piece['id']).lower()
        if key not in actual:raise ValueError('Missing room mesh '+key)
        center=to_ned(room,piece['center']);size=piece['size']
        expected=[[p-d/2 for p,d in zip(center,size)],[p+d/2 for p,d in zip(center,size)]]
        if any(abs(a-b)>.06 for aa,bb in zip(actual[key],expected) for a,b in zip(aa,bb)):
            raise ValueError('Scene bounds differ from route map: '+key)
    return {'matched_meshes':len(actual),'foreign_interior_meshes':foreign,'scope':'Indexed geometry bounds; collision response is checked during flight.'}


def depth_coverage(room,records,directory,step=.3):
    from .mission_scope import inspection_walls
    report={}
    for wall in inspection_walls(room):
        nu=math.ceil(wall['length']/step);nh=math.ceil(wall['height']/step)
        points=[to_ned(room,wall_point(wall,(i+.5)*wall['length']/nu,(j+.5)*wall['height']/nh))
                for j in range(nh) for i in range(nu)]
        hits=set();per_view=[]
        for record in records:
            if record.get('wall_id')!=wall['id']:continue
            depth=array.array('f')
            with (Path(directory)/record['depth_file']).open('rb') as stream:
                depth.fromfile(stream,record['width']*record['height'])
            found=visible_samples({'depth_tolerance_m':.18},record,depth,points)
            hits.update(found);per_view.append({'file':record['file'],'visible_samples':len(found)})
        report[wall['id']]={'sample_count':len(points),'visible_samples':len(hits),
             'sampled_visible_percent':100*len(hits)/len(points),'views':per_view,
             'grid_width':nu,'grid_height':nh,'covered_indices':sorted(hits)}
    return {'walls':report,'scope':'Actual saved planar depth + camera pose, wall-cell center visibility only; no defect detection.'}


class RoomPilot:
    def __init__(self,client,room,directory,summary):
        self.client=client;self.room=room;self.directory=Path(directory);self.summary=summary
        self.since=0;self.previous=None
        self.sensor=None
        if room.get('online_depth',False):
            from .live_depth import DepthSensor
            self.sensor=DepthSensor(client,room,directory,summary['coordinate_calibration'])
        self.stream=(self.directory/'telemetry.jsonl').open('a',encoding='utf-8')

    def observe(self,phase,guard=True):
        state=self.client.call('getMultirotorState','');local_ned=position(state)
        ned=scene_position(local_ned,self.summary['coordinate_calibration'])
        local=from_ned(self.room,ned);collision=self.client.call('simGetCollisionInfo','')
        row={'timestamp':state['timestamp'],'phase':phase,'position':local,'position_ned':ned,'position_vehicle_local_ned':local_ned,'position_frame':'scene-global-ned',
             'velocity':state['kinematics_estimated']['linear_velocity'],'clearance_m':clearance(self.room,local),
             'collision_timestamp':collision['time_stamp'],'landed_state':state['landed_state']}
        self.stream.write(json.dumps(row)+'\n');self.stream.flush()
        if guard:
            from .camera import check_stop
            check_stop()
            if (self.directory/'STOP').exists():raise RuntimeError('Operator stop requested.')
            if self.since and collision['time_stamp']>self.since:raise RuntimeError('New collision during room flight.')
            if not in_free_space(self.room,local):raise RuntimeError('Measured position violates map clearance.')
            if self.previous is not None:
                for obstacle in self.room['obstacles']:
                    if segment_hits_box(self.previous,local,obstacle['lower'],obstacle['upper'],self.room['clearance_m']):
                        raise RuntimeError('Sampled motion segment violates obstacle clearance.')
            self.summary['minimum_sampled_center_clearance_m']=min(self.summary.get('minimum_sampled_center_clearance_m',99),row['clearance_m'])
        self.previous=local
        return state,local

    def move(self,target,yaw,phase,wall_id=None,*,allow_adjust=True,planning_margin=.25,max_speed=.75):
        if not 0<max_speed<=.75:raise ValueError('Invalid movement speed limit')
        state,current=self.observe(phase)
        replans=0
        while True:
            obstacles=self.room['obstacles']+(self.sensor.map.boxes() if self.sensor else [])
            from .available_view import route_to_view
            route,available_target=route_to_view(self.room,current,target,obstacles,wall_id=wall_id,allow_adjust=allow_adjust and phase!='return',planning_margin=planning_margin)
            from .altitude_path import stage_path
            route['path']=stage_path(route['path'],obstacles,route['clearance_m'])
            route['path_length_m']=sum(math.dist(a,b) for a,b in zip(route['path'],route['path'][1:]))
            self.last_target=available_target
            if math.dist(available_target,target)>.01:
                self.summary.setdefault('view_adjustments',[]).append({'phase':phase,'requested':target,'available':available_target,'reason':'Requested viewpoint occupied by observed depth geometry.'})
                print('Adjusted blocked viewpoint '+phase+' to '+str(available_target),flush=True)
            self.summary['executed_transits'].append(dict(route,phase=phase,online_replan=replans))
            interrupted=False
            for goal in route['path'][1:]:
                deadline=time.monotonic()+max(90 if self.sensor else 25,math.dist(current,goal)/(.15 if self.sensor else .3)+20)
                while True:
                    state,current=self.observe(phase)
                    speed=state['kinematics_estimated']['linear_velocity']
                    speed=math.sqrt(sum(speed[k]**2 for k in ['x_val','y_val','z_val']))
                    if math.dist(current,goal)<.13 and speed<.16:break
                    if time.monotonic()>deadline:raise RuntimeError('Waypoint tracking timed out: '+phase)
                    if self.sensor:
                        q=state['kinematics_estimated']['orientation'];qw,qx,qy,qz=[q[k] for k in ['w_val','x_val','y_val','z_val']]
                        actual_yaw=math.degrees(math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
                        horizontal=math.hypot(goal[0]-current[0],goal[1]-current[1])
                        heading=actual_yaw if horizontal<.2 else math.degrees(math.atan2(goal[1]-current[1],goal[0]-current[0]))
                        sensor_pitch=math.degrees(math.atan2(goal[2]-current[2],max(.05,horizontal)))
                        if abs((heading-actual_yaw+180)%360-180)>3:
                            self.client.call('moveByVelocityZ',0.,0.,vehicle_position(to_ned(self.room,goal),self.summary['coordinate_calibration'])[2],.2,0,{'is_rate':False,'yaw_or_rate':heading},'',timeout=4)
                            continue
                        scan=self.sensor.scan(heading,q,sensor_pitch)
                        boxes=scan['occupied_boxes']
                        if any(segment_hits_box(current,goal,b['lower'],b['upper'],self.room['clearance_m']+.1) for b in boxes):
                            self.client.call('moveByVelocityZ',0.,0.,vehicle_position(to_ned(self.room,current),self.summary['coordinate_calibration'])[2],.3,0,{'is_rate':False,'yaw_or_rate':yaw},'',timeout=4)
                            self.client.call('hover','',timeout=10)
                            self.summary.setdefault('depth_replan_events',[]).append({'phase':phase,'position':current,'boxes':boxes,'sensor_sample':self.sensor.count-1})
                            print('Depth obstacle observed: stopping and replanning '+phase,flush=True)
                            replans+=1;interrupted=True
                            if replans>12:raise RuntimeError('Depth replanning limit reached; inspection stopped.')
                            state,current=self.observe('depth-stop')
                            break
                    command=velocity_command(current,goal,max_speed=max_speed)
                    altitude_setpoint=vehicle_position(to_ned(self.room,goal),self.summary['coordinate_calibration'])[2]
                    self.client.call('moveByVelocityZ',command[0],command[1],altitude_setpoint,.4 if self.sensor else .2,0,
                                     {'is_rate':False,'yaw_or_rate':heading if self.sensor else yaw},'',timeout=4)
                if interrupted:break
                self.client.call('moveByVelocityZ',0.,0.,vehicle_position(to_ned(self.room,current),self.summary['coordinate_calibration'])[2],.3,0,{'is_rate':False,'yaw_or_rate':yaw},'',timeout=4)
                # Final vertical position hold removes the steady velocity-mode
                # altitude bias before the camera/inference pause. The vehicle
                # is already within 13 cm of a clearance-checked waypoint.
                desired_z=vehicle_position(to_ned(self.room,goal),self.summary['coordinate_calibration'])[2]
                for _ in range(8):
                    state,current=self.observe('altitude-settle')
                    if abs(current[2]-goal[2])<.035:break
                    self.client.call('moveByVelocityZ',0.,0.,desired_z,.5,0,{'is_rate':False,'yaw_or_rate':yaw},'',timeout=4)
                else:raise RuntimeError('Altitude hold did not settle at waypoint.')
                self.client.call('hover','',timeout=10)
                self.observe(phase)
            if not interrupted:return current


def execute(room,targets,directory):
    from .environment import pose
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    summary={'status':'started','kind':'room-static-flight','scope':'Known-map planning with optional live depth stop/replan and native trained inspection; enabled capabilities are recorded in room.json.',
             'images':[],'executed_transits':[],'landing':'not-started',
             'scene_initialization':'Landed drone relocated to room floor before arming; no airborne teleport used.'}
    (directory/'room.json').write_text(json.dumps(room,indent=2))
    (directory/'targets.json').write_text(json.dumps(targets,indent=2))
    controlled=False;pilot=None;error=None
    launch=[room['launch_m'][0],room['launch_m'][1],3]
    try:
        with AirSimRPC(timeout=25) as client:
            if client.call('getMultirotorState','')['landed_state']!=0:raise RuntimeError('Start with the drone landed.')
            client.scene_coordinate_calibration=measure_scene_offset(client)
            summary['coordinate_calibration']=client.scene_coordinate_calibration
            summary['scene_check']=verify_scene(client,room)
            ground=validate_launch_column(room,launch)
            for a,b in zip([launch]+[t['position'] for t in targets],[t['position'] for t in targets]+[launch]):static_route(room,a,b)
            # Ground setup is explicit and logged separately from measured flight.
            client.call('simSetVehiclePose',pose(vehicle_position(to_ned(room,ground),client.scene_coordinate_calibration)),True,'')
            client.call('simSetCameraPose','0',pose([0,0,0]),'',False)
            time.sleep(1)
            origin=position(client.call('getMultirotorState',''))
            if math.dist(scene_position(origin,client.scene_coordinate_calibration)[:2],to_ned(room,launch)[:2])>.25:raise RuntimeError('Landed room placement did not converge.')
            summary['ground_start_ned']=origin
            state=client.call('getMultirotorState','');camera=client.call('simGetCameraInfo','0','',False)
            offset=rotate_inverse(state['kinematics_estimated']['orientation'],
                tuple(camera['pose']['position'][k]-p for k,p in zip(['x_val','y_val','z_val'],position(state))))
            if abs(offset[0]-room.get('camera_forward_offset_m',0))>.03 or max(abs(v) for v in offset[1:])>.03:
                raise RuntimeError('Camera offset differs from room plan: '+str(offset))
            if room.get('online_depth',False):
                client.call('simSetCameraPose','nav',pose([.46,0,0]),'',False)
                nav=client.call('simGetCameraInfo','nav','',False)
                nav_offset=rotate_inverse(state['kinematics_estimated']['orientation'],tuple(nav['pose']['position'][k]-p for k,p in zip(['x_val','y_val','z_val'],position(state))))
                if abs(nav_offset[0]-.46)>.03 or max(abs(v) for v in nav_offset[1:])>.03:raise RuntimeError('Navigation depth camera mount is invalid: '+str(nav_offset))
                summary['navigation_camera_mount_body_m']=nav_offset
            summary['measured_camera_offset_body_m']=offset
            client.call('enableApiControl',True,'');controlled=True
            if not client.call('armDisarm',True,''):raise RuntimeError('Arming failed.')
            print('Taking off inside the room...',flush=True)
            summary['takeoff_rpc_result']=client.call('takeoff',25.,'',timeout=30)
            client.call('hover','',timeout=10);time.sleep(.5)
            summary['takeoff_verified']=verify_takeoff(client.call('getMultirotorState',''),origin)
            pilot=RoomPilot(client,room,directory,summary)
            pilot.since=client.call('getMultirotorState','')['timestamp']
            summary['airborne_collision_baseline_timestamp']=pilot.since
            pilot.observe('takeoff-hover')
            verification_scheduled=set()
            for index,target in enumerate(targets):
                print('Flying to '+target['id']+'...',flush=True)
                pilot.move(target['position'],target['yaw_deg'],target['id'],target['wall_id'])
                # Rotate while holding position, then verify actual yaw before capture.
                for _ in range(30):
                    state,current=pilot.observe('camera-settle')
                    q=state['kinematics_estimated']['orientation'];w,x,y,z=[q[k] for k in ['w_val','x_val','y_val','z_val']]
                    actual_yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
                    difference=(actual_yaw-target['yaw_deg']+180)%360-180
                    if abs(difference)<2:break
                    client.call('moveByVelocityZ',0.,0.,vehicle_position(to_ned(room,pilot.last_target),client.scene_coordinate_calibration)[2],.2,0,{'is_rate':False,'yaw_or_rate':target['yaw_deg']},'',timeout=4)
                else:raise RuntimeError('Camera yaw did not settle.')
                client.call('hover','',timeout=10)
                record=capture(client,directory,f'room-{index+1:03d}',with_depth=True)
                record.update(wall_id=target['wall_id'],target=target,achieved_target=pilot.last_target,actual_position=pilot.observe('capture')[1])
                if target['wall_id']:
                    wall=next(w for w in room['walls'] if w['id']==target['wall_id'])
                    camera_local=from_ned(room,[record['camera_position'][k] for k in ['x_val','y_val','z_val']])
                    standoff=sum((a-b)*n for a,b,n in zip(camera_local,wall['origin'],wall['inward']))
                    record['measured_camera_standoff_m']=standoff
                    if abs(standoff-room['standoff_m'])>.2:raise RuntimeError('Camera standoff outside 20 cm tolerance.')
                if room.get('trained_inspection',False):
                    from .room_live_model import inspect_capture
                    found=inspect_capture(directory,room,record)
                    pilot.observe('trained-inference-check')
                    print('Trained camera observations: '+str(found or 0),flush=True)
                summary['images'].append(record)
                (directory/'flight.json').write_text(json.dumps(summary,indent=2))
                print(f"Captured {index+1}/{len(targets)}: {target['id']}",flush=True)
                if room.get('verify_edge_candidates',False):
                    from .wall_recheck import inspect_edges
                    inspect_edges(pilot,room,record,verification_scheduled)
            if room.get('adaptive_reinspection'):
                from .wall_recheck import inspect_gaps
                inspect_gaps(pilot,room,verification_scheduled)
            print('Returning to the room launch area...',flush=True)
            pilot.move(launch,0,'return')
            summary['return_verified']=math.dist(pilot.observe('return-hover')[1],launch)<.2
            summary['coverage']=depth_coverage(room,summary['images'],directory)
            summary['status']='completed'
    except BaseException as exc:
        error=exc;summary.update(status='failed',error=str(exc))
    finally:
        if controlled:
            print('Landing and verifying ground contact...',flush=True)
            try:
                with AirSimRPC(timeout=12) as recovery:
                    recovery.call('cancelLastTask','')
                    current=from_ned(room,scene_position(position(recovery.call('getMultirotorState','')),summary['coordinate_calibration']))
                    ground=[current[0],current[1],.15]
                    if any(segment_hits_box(current,ground,o['lower'],o['upper'],.4) for o in room['obstacles']+(pilot.sensor.map.boxes() if pilot and pilot.sensor else [])):
                        recovery.call('hover','',timeout=8)
                        raise RuntimeError('Landing column obstructed; vehicle left hovering.')
                    since=recovery.call('getMultirotorState','')['timestamp'];deadline=time.monotonic()+35
                    while time.monotonic()<deadline:
                        recovery.call('land',4.,'',timeout=6);time.sleep(.2)
                        state=recovery.call('getMultirotorState','');collision=recovery.call('simGetCollisionInfo','')
                        if ground_contact(state,collision,since):break
                    else:raise RuntimeError('Ground contact not verified; motors left armed.')
                    recovery.call('armDisarm',False,'');recovery.call('enableApiControl',False,'')
                    summary.update(landing='completed',landing_position_ned=position(state),api_control_released=True)
            except Exception as exc:
                summary.update(status='failed',landing='unconfirmed',landing_error=str(exc));error=error or exc
        if pilot:
            if pilot.sensor:pilot.sensor.close()
            pilot.stream.close()
        (directory/'flight.json').write_text(json.dumps(summary,indent=2))
    if error:raise error
    return summary
