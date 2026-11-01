"""Apply the wall textures and independently calibrate their UV mapping."""
import json,math,time,uuid
from pathlib import Path
import cv2
import numpy as np
from .environment import pose,vec
from .room import to_ned,wall_point
from .rpc import AirSimRPC
from .coordinates import measure_scene_offset,vehicle_position,camera_in_scene
from .vision import rotation
from .scene_geometry import source_projection
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/construction-room/scene'


def prepare(room,source=OUT,expected_version='whole-wall-sparse-demo-v4-proposal'):
    source=Path(source)
    truth=json.loads((source/'truth.json').read_text())
    if truth['version']!=expected_version:raise RuntimeError('Unexpected approved wall source')
    return truth


def set_camera(c,desired,yaw):
    state=c.call('getMultirotorState','');q=state['kinematics_estimated']['orientation'];R=rotation(q)
    body=np.array([state['kinematics_estimated']['position'][k] for k in ['x_val','y_val','z_val']])
    w,x,y,z=[q[k] for k in ['w_val','x_val','y_val','z_val']];body_yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    target=np.array(desired);offset=R.T@(target-body)
    for _ in range(5):
        c.call('simSetCameraPose','0',pose(offset.tolist(),yaw-body_yaw),'',False);time.sleep(.15)
        info=c.call('simGetCameraInfo','0','',False)
        actual=np.array([info['pose']['position'][k] for k in ['x_val','y_val','z_val']]);error=target-actual
        if np.linalg.norm(error)<.001:return info
        offset+=R.T@error
    raise RuntimeError('Landed calibration camera did not converge')


def atlas(panel,source=OUT):
    width,height=panel['source_size_px'];length,physical_h=panel['size_m'];ppm=height/physical_h
    image=np.full((height,width),225,np.uint8);dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100);centres={};side=round(.38*ppm)
    for row,v in enumerate(np.linspace(.65,physical_h-.65,4)):
        for col,u in enumerate(np.linspace(.65,length-.65,8)):
            identity=row*8+col;left=round(u*ppm-side/2);top=round(v*ppm-side/2)
            image[top:top+side,left:left+side]=cv2.aruco.generateImageMarker(dictionary,identity,side)
            centres[identity]=[left+(side-1)/2,top+(side-1)/2]
    path=Path(source)/(panel['id']+'-fiducials.png');cv2.imwrite(str(path),image);return path,centres


def spawn(room,truth,source=OUT):
    source=Path(source)
    checks=[]
    with AirSimRPC(timeout=60) as c:
        before=c.call('getMultirotorState','')
        if before['landed_state']!=0 or c.call('isApiControlEnabled',''):raise RuntimeError('Fixture calibration requires landed and API released')
        calibration=measure_scene_offset(c)
        try:
            for panel in truth['panels']:
                wall=next(w for w in room['walls'] if w['id']==panel['wall_id'])
                for existing in c.call('simListSceneObjects',panel['id']):c.call('simDestroyObject',existing.decode() if isinstance(existing,bytes) else existing)
                center=to_ned(room,wall_point(wall,panel['u_center_m'],panel['height_center_m'],-.008))
                name=c.call('simSpawnObject',panel['id']+'_'+uuid.uuid4().hex[:8],'Cube',pose(center,wall['yaw_deg']),vec([.02,*panel['size_m']]),False,False)
                if not name:raise RuntimeError('Whole-wall spawn failed')
                name=name.decode() if isinstance(name,bytes) else name;panel['sim_actor_name']=name;time.sleep(.25)
                try:
                    path,centres=atlas(panel,source)
                    if not c.call('simSetObjectMaterialFromTexture',name,str(path),0):raise RuntimeError('Calibration material failed')
                    distance=wall['length']*.55+panel['front_inward_m']
                    # The central W sightline is occluded by the scaffold; inspect
                    # from the clear southern side, with the same independent UV gate.
                    camera_u=wall['length']-3 if wall['id']=='W' else wall['length']/2
                    desired=vehicle_position(to_ned(room,wall_point(wall,camera_u,wall['height']/2,distance)),calibration)
                    info=set_camera(c,desired,wall['yaw_deg']);time.sleep(.5)
                    frame=c.call('simGetImages',[dict(camera_name='0',image_type=0,pixels_as_float=False,compress=True)],'',False)[0]
                    raw=bytes(frame['image_data_uint8']);(source/(panel['id']+'-calibration.png')).write_bytes(raw)
                    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    pars=cv2.aruco.DetectorParameters();pars.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
                    corners,ids,_=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),pars).detectMarkers(image)
                    if ids is None or len(ids)<16:raise RuntimeError('Too few whole-wall calibration markers')
                    source_pixels=np.float32([centres[int(i)] for i in ids.flatten()]);observed=np.float32([x.reshape(4,2).mean(axis=0) for x in corners]);training=np.array([(int(i)//8+int(i)%8)%2==0 for i in ids.flatten()])
                    if min(int(training.sum()),int((~training).sum()))<6:raise RuntimeError('Too few independent held-out markers')
                    fit,_=cv2.findHomography(source_pixels[training],observed[training],0)
                    holdout=np.linalg.norm(cv2.perspectiveTransform(source_pixels[None,~training],fit)[0]-observed[~training],axis=1)
                    record={k:frame[k] for k in ['width','height','camera_position','camera_orientation']};record['fov_deg']=info['fov'];record=camera_in_scene(record,calibration)
                    projection=source_projection(room,panel,record)
                    residual=np.linalg.norm(cv2.perspectiveTransform(source_pixels[None],projection)[0]-observed,axis=1)
                    check=dict(wall=wall['id'],markers=len(ids),heldout_markers=int((~training).sum()),heldout_p95_px=float(np.percentile(holdout,95)),world_projection_p95_px=float(np.percentile(residual,95)))
                    checks.append(check);print('WHOLE_WALL_ALIGNMENT',json.dumps(check),flush=True)
                    if max(check['heldout_p95_px'],check['world_projection_p95_px'])>1.5:raise RuntimeError('Whole-wall UV/world alignment exceeds 1.5 pixels')
                finally:
                    if not c.call('simSetObjectMaterialFromTexture',name,str(source/panel['texture']),0):raise RuntimeError('Could not restore approved concrete surface')
        finally:c.call('simSetCameraPose','0',pose([0,0,0]),'',False)
        after=c.call('getMultirotorState','')
        before_p=[before['kinematics_estimated']['position'][k] for k in ['x_val','y_val','z_val']];after_p=[after['kinematics_estimated']['position'][k] for k in ['x_val','y_val','z_val']]
        if after['landed_state']!=0 or math.dist(before_p,after_p)>.005:raise RuntimeError('Vehicle moved during landed fixture check')
    (source/'calibration.json').write_text(json.dumps(dict(passed=True,checks=checks,vehicle_displacement_m=math.dist(before_p,after_p),landed=True),indent=2))
    print('LABELED_ROOM_READY',json.dumps(truth['counts']),'whole-wall demonstration profile',flush=True)
