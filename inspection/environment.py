"""Spawn the construction room and verify its geometry."""
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
from .room import to_ned
from .rpc import AirSimRPC
from .scene import mesh_bounds
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/construction-room'


def vec(values):return dict(zip(['x_val','y_val','z_val'],map(float,values)))


def pose(position,yaw=0,pitch=0):
    y,p=math.radians(yaw)/2,math.radians(pitch)/2
    return {'position':vec(position),'orientation':{'w_val':math.cos(p)*math.cos(y),
            'x_val':-math.sin(p)*math.sin(y),'y_val':math.sin(p)*math.cos(y),
            'z_val':math.cos(p)*math.sin(y)}}


def textures():
    rng=np.random.default_rng(42);paths={}
    colors={'plaster':(190,193,195),'floor':(124,129,134),'blocks':(145,149,153),
            'pallets':(70,112,153),'machine':(28,162,213),'metal':(105,112,117),'rubber':(31,33,36)}
    for kind,color in colors.items():
        noise=cv2.resize(rng.normal(0,4,(32,32)).astype(np.float32),(1024,1024),interpolation=cv2.INTER_CUBIC)
        noise+=rng.normal(0,1,(1024,1024))
        image=np.clip(np.array(color)[None,None,:]+noise[:,:,None],0,255).astype(np.uint8)
        if kind=='plaster':
            for _ in range(70):
                x,y=rng.integers(0,1024,2);cv2.ellipse(image,(int(x),int(y)),(60,5),float(rng.uniform(-20,20)),0,360,(183,187,189),1,cv2.LINE_AA)
        if kind in ['pallets','blocks']:
            for row in range(0,1024,128):
                cv2.line(image,(0,row),(1023,row),tuple(max(0,c-28) for c in color),5)
                if kind=='blocks':
                    for x in range((row//128%2)*128,1024,256):cv2.line(image,(x,row),(x,min(row+128,1023)),(90,94,98),4)
        path=OUT/f'{kind}.png';cv2.imwrite(str(path),image);paths[kind]=path
    return paths


def pieces(room):
    parts=[]
    def add(name,center,size,material):parts.append({'id':name,'center':center,'size':size,'material':material})
    add('floor',[12,9,-.15],[24.5,18.5,.3],'floor')
    add('wall-S',[12,-.125,3],[24.5,.25,6],'plaster')
    add('wall-N',[12,18.125,3],[24.5,.25,6],'plaster')
    add('wall-W',[-.125,9,3],[.25,18,6],'plaster')
    add('wall-E',[24.125,9,3],[.25,18,6],'plaster')
    for item in room['obstacles']:
        lo,hi=item['lower'],item['upper'];x,y=(lo[0]+hi[0])/2,(lo[1]+hi[1])/2
        dx,dy=hi[0]-lo[0],hi[1]-lo[1];h=hi[2];key=item['id'];kind=item['kind']
        if kind=='scaffold':
            for ix in [-1,1]:
                for iy in [-1,1]:add(key+f'-post{ix}{iy}',[x+ix*(dx/2-.06),y+iy*(dy/2-.06),h/2],[.12,.12,h],'metal')
            for level in [1.5,3,4.35]:add(key+f'-deck{level}',[x,y,level],[dx,dy,.14],'pallets')
        elif kind=='machine':
            add(key+'-body',[x,y,1.1],[dx,dy,.9],'machine')
            add(key+'-motor',[x,y,h-.4],[dx*.6,dy*.65,.8],'metal')
            for side in [-1,1]:add(key+f'-track{side}',[x,y+side*(dy/2-.2),.3],[dx,.4,.6],'rubber')
        else:
            add(key,[x,y,h/2],[dx,dy,h],kind)
    return parts


def spawn(room):
    materials=textures();parts=pieces(room);names=[];material_results=[]
    with AirSimRPC(timeout=20) as c:
        if c.call('getMultirotorState','')['landed_state']!=0:raise RuntimeError('Drone must be landed before changing the scene.')
        # Remove only our own transient simulator actors, never original scene geometry.
        old=c.call('simListSceneObjects','FacadeRoom_.*')
        for name in old:c.call('simDestroyObject',name.decode() if isinstance(name,bytes) else name)
        time.sleep(.2)
        for part in parts:
            name=c.call('simSpawnObject','FacadeRoom_'+part['id'],'Cube',
                        pose(to_ned(room,part['center'])),vec(part['size']),False,False)
            if isinstance(name,bytes):name=name.decode()
            if not name:raise RuntimeError('Could not spawn '+part['id'])
            names.append(name)
            ok=c.call('simSetObjectMaterialFromTexture',name,str(materials[part['material']]),0)
            material_results.append({'object':name,'api_success':bool(ok)})
        time.sleep(.5)
        meshes=c.call('simGetMeshPositionVertexBuffers',timeout=30)
        actual={}
        for mesh in meshes:
            name=mesh['name'].decode() if isinstance(mesh['name'],bytes) else mesh['name']
            if name.lower() in {n.lower() for n in names}:actual[name.lower()]=mesh_bounds(mesh)
        failures=[]
        for part,name in zip(parts,names):
            if name.lower() not in actual:failures.append('No indexed mesh bounds for '+name);continue
            center=to_ned(room,part['center']);size=part['size']
            expected=[[p-d/2 for p,d in zip(center,size)],[p+d/2 for p,d in zip(center,size)]]
            if any(abs(a-b)>.06 for aa,bb in zip(actual[name.lower()],expected) for a,b in zip(aa,bb)):
                failures.append('Mesh bounds mismatch: '+name)
        # Move only the camera for documentary stills; the drone stays landed at home.
        try:
            for name,local,yaw,pitch in [('overview',[-5,-7,15],42,-31),('interior',[4,4,2.5],27,-4)]:
                c.call('simSetCameraPose','0',pose(to_ned(room,local),yaw,pitch),'',False)
                time.sleep(.5)
                frame=c.call('simGetImages',[{'camera_name':'0','image_type':0,'pixels_as_float':False,'compress':True}],'',False)[0]
                raw=bytes(frame['image_data_uint8'])
                if not raw.startswith(b'\x89PNG'):raise RuntimeError('Invalid camera still')
                (OUT/f'airsim-{name}.png').write_bytes(raw)
        finally:c.call('simSetCameraPose','0',pose([0,0,0]),'',False)
        report={'room_seed':room['seed'],'spawned':len(names),'names':names,'mesh_bounds_checked':len(actual),'geometry_passed':not failures,
                'geometry_failures':failures,'materials':material_results,'landed_after':c.call('getMultirotorState','')['landed_state']==0,
                'scope':'Primitive construction fixture rendered by AirSim; no flight, collision-response or depth-avoidance test.',
                'api_reference':'https://microsoft.github.io/AirSim/api_docs/html/#airsim.client.VehicleClient.simSpawnObject'}
        (OUT/'simulator-check.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ['names','materials']},indent=2))
        return report
