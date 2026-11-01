"""Online depth occupancy for previously unmapped stationary objects.

Keeps conservative occupied voxels during one mission. Does not predict moving
objects or infer free space behind visible surfaces. Flight controller stops and
replans when observed occupancy intersects its next route segment.
"""
import math
from pathlib import Path
import time
import json
import cv2
import numpy as np
from .coordinates import camera_in_scene
from .vision import rotation


class DepthMap:
    def __init__(self,room,resolution=.25):
        self.room=room;self.resolution=resolution;self.cells=set()

    def update(self,depth,record):
        h,w=depth.shape;v,u=np.mgrid[2:h:3,2:w:3];d=depth[v,u]
        f=w/(2*math.tan(math.radians(record['fov_deg'])/2))
        rays=np.stack([np.ones_like(u),(u-(w-1)/2)/f,(v-(h-1)/2)/f],axis=-1)
        local=rays*np.where(np.isfinite(d),d,0)[:,:,None]
        origin=np.array([record['camera_position'][k] for k in ['x_val','y_val','z_val']])
        world=local@rotation(record['camera_orientation']).T+origin
        base=np.array(self.room['airsim_origin_ned_m']);points=(world-base)*[1,1,-1]
        valid=np.isfinite(points).all(axis=2)&np.isfinite(d)&(d>.3)&(d<6.)
        valid &= np.all((points>.18)&(points<np.array(self.room['size_m'])-.18),axis=2)
        for o in self.room['obstacles']:
            valid &= ~np.all((points>=np.array(o['lower'])-.18)&(points<=np.array(o['upper'])+.18),axis=2)
        points=points[valid]
        voxels=set(map(tuple,np.floor(points/self.resolution).astype(int)))
        added=len(voxels-self.cells);self.cells|=voxels
        return {'new_cells':added,'observed_unmapped_points':len(points),'occupied_cells':len(self.cells)}

    def boxes(self):
        remaining=set(self.cells);output=[]
        while remaining:
            first=remaining.pop();component={first};pending=[first]
            while pending:
                a=pending.pop()
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        for dz in [-1,0,1]:
                            b=(a[0]+dx,a[1]+dy,a[2]+dz)
                            if b in remaining:remaining.remove(b);component.add(b);pending.append(b)
            if len(component)<3:continue
            points=np.array(list(component));lo=points.min(axis=0)*self.resolution;hi=(points.max(axis=0)+1)*self.resolution
            observed_bounds=[lo.tolist(),hi.tolist()]
            lo[2]=0.;hi[2]=self.room['size_m'][2]  # Unmeasured height is not assumed free.
            output.append({'id':f'observed-{len(output)+1:03d}','kind':'depth-occupied',
                           'lower':lo.tolist(),'upper':hi.tolist(),'occupied_voxels':len(component),'observed_bounds':observed_bounds,'height_policy':'Full room-height conservative column; no unobserved height assumed free.'})
        return output


class DepthSensor:
    def __init__(self,client,room,directory,calibration):
        self.client=client;self.room=room;self.calibration=calibration
        self.map=DepthMap(room);self.directory=Path(directory)/'navigation-depth';self.directory.mkdir(exist_ok=True)
        self.stream=(self.directory/'observations.jsonl').open('a',encoding='utf-8');self.count=0

    def scan(self,heading_deg,body_q,pitch_deg=0):
        from .environment import pose
        w,x,y,z=[body_q[k] for k in ['w_val','x_val','y_val','z_val']]
        yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        started=time.perf_counter()
        self.client.call('simSetCameraPose','nav',pose([.46,0,0],0,pitch_deg),'',False)
        frame=self.client.call('simGetImages',[{'camera_name':'nav','image_type':1,'pixels_as_float':True,'compress':False}],'',False)[0]
        if (frame['width'],frame['height'])!=(320,180):raise RuntimeError('Navigation camera must return native 320×180 depth')
        depth=np.asarray(frame['image_data_float'],np.float32).reshape(180,320)
        record={k:frame[k] for k in ['width','height','camera_position','camera_orientation']}
        record.update(file=f'{self.count:05d}.f32',fov_deg=90,timestamp=frame['time_stamp'])
        record=camera_in_scene(record,self.calibration)
        result=self.map.update(depth,record);record.update(result,sensor_request_seconds=time.perf_counter()-started)
        depth.tofile(self.directory/record['file'])
        record['occupied_boxes']=self.map.boxes()
        self.stream.write(json.dumps(record)+'\n');self.stream.flush();self.count+=1
        return record

    def close(self):self.stream.close()
