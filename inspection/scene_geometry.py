"""Deterministic scene geometry and independent reference projection."""
import cv2
import numpy as np
from .room import wall_point,to_ned
from .vision import rotation


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


def stretch_crack(points,opening_width_mm,multiplier=2.5):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('A finite 2D polyline is required')
    if opening_width_mm<=0 or multiplier<1:raise ValueError('Invalid crack dimensions')
    delta=np.diff(points,axis=0)
    original=float(np.linalg.norm(delta,axis=1).sum())+opening_width_mm
    target=original*multiplier-opening_width_mm
    if np.ptp(points[:,1])==0:raise ValueError('A vertical component is required to preserve horizontal spread')
    low,high=1.,max(2.,multiplier)
    def arc(scale):return float(np.linalg.norm(delta*[1.,scale],axis=1).sum())
    while arc(high)<target:high*=2
    for _ in range(80):
        middle=(low+high)/2
        if arc(middle)<target:low=middle
        else:high=middle
    result=points.copy();center=(points[:,1].min()+points[:,1].max())/2
    result[:,1]=center+(points[:,1]-center)*(low+high)/2
    length=float(np.linalg.norm(np.diff(result,axis=0),axis=1).sum())
    return dict(polyline_mm=result.tolist(),opening_width_mm=float(opening_width_mm),
                visible_trace_length_mm=length+opening_width_mm,centerline_arc_length_mm=length,
                zigzag_spread_mm=float(np.ptp(result[:,0])+opening_width_mm),
                length_multiplier=float(multiplier),original_visible_trace_length_mm=original)


def source_world(room,panel):
    wall=next(w for w in room['walls'] if w['id']==panel['wall_id'])
    width,height=panel.get('size_m',[panel.get('side_m',1.2)]*2)
    if min(width,height)<=0:raise ValueError('Source surface dimensions must be positive')
    return np.array([to_ned(room,wall_point(wall,panel['u_center_m']+du,panel['height_center_m']+dh,panel['front_inward_m']))
                     for du,dh in [(-width/2,height/2),(width/2,height/2),(width/2,-height/2),(-width/2,-height/2)]],float)


def source_projection(room,panel,record):
    origin=np.array([record['camera_position'][k] for k in ['x_val','y_val','z_val']])
    camera=(source_world(room,panel)-origin)@rotation(record['camera_orientation'])
    if np.any(camera[:,0]<=.05):return None
    f=record['width']/(2*np.tan(np.deg2rad(record['fov_deg'])/2))
    uv=camera[:,1:]/camera[:,0,None]*f+[(record['width']-1)/2,(record['height']-1)/2]
    size=panel.get('source_size_px',2048)
    width,height=([size,size] if isinstance(size,(int,float)) else size)
    if min(width,height)<=0:raise ValueError('Source pixel dimensions must be positive')
    return cv2.getPerspectiveTransform(np.float32([[0,0],[width,0],[width,height],[0,height]]),uv.astype(np.float32))
