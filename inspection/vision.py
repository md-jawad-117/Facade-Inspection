"""Trained image segmentation and pose/depth-based wall localization."""
import math
import cv2
import numpy as np
COLORS={'crack':(62,76,239),'rust':(38,157,245),'spall':(235,139,73)}


def rotation(q):
    w,x,y,z=(float(q[k]) for k in ['w_val','x_val','y_val','z_val'])
    norm=math.sqrt(w*w+x*x+y*y+z*z)
    w,x,y,z=w/norm,x/norm,y/norm,z/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def image_quality(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    dark=float(np.mean(gray<20)); clipped=float(np.mean(gray>245))
    return {'sharpness_laplacian_variance':round(sharpness,2),'dark_fraction':round(dark,3),
            'overexposed_fraction':round(clipped,3),
            'warnings':(['Low image texture or blur; inspect manually.'] if sharpness<20 else [])+
                       (['Large dark region.'] if dark>.4 else [])}


def annotate(image,findings,path):
    canvas=image.copy()
    occupied=[]
    for i,finding in enumerate(findings):
        x,y,w,h=finding['bbox']; color=COLORS[finding['kind']]
        cv2.rectangle(canvas,(x,y),(x+w,y+h),color,2)
        for row,start,end in finding.get('mask_runs',[]):
            if 0<=row<canvas.shape[0]:
                a,b=max(0,start),min(canvas.shape[1],end)
                if b>a:canvas[row,a:b]=np.rint(canvas[row,a:b]*.65+np.array(color)*.35).astype(np.uint8)
        label=f'{finding.get("id",str(i+1))} {finding["kind"]}'
        if 'confidence' in finding:label+=f" {finding['confidence']:.2f}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.45,1)
        left=max(0,min(x,canvas.shape[1]-tw-7))
        top=max(th+6,min(y,canvas.shape[0]-1))
        # Keep close tracks' ID labels legible instead of painting over each other.
        for candidate in [top]+list(range(th+6,canvas.shape[0],th+9)):
            box=(left,candidate-th-6,min(canvas.shape[1]-1,left+tw+6),candidate)
            if not any(box[0]<b[2]+2 and box[2]>b[0]-2 and box[1]<b[3]+2 and box[3]>b[1]-2 for b in occupied):
                top=candidate
                break
        occupied.append((left,top-th-6,min(canvas.shape[1]-1,left+tw+6),top))
        if abs(top-y)>th+6:
            cv2.line(canvas,(left,top),(x,y),color,1)
        cv2.rectangle(canvas,(left,top-th-6),(min(canvas.shape[1]-1,left+tw+6),top),color,-1)
        cv2.putText(canvas,label,(left+3,top-4),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(path),canvas)


def wall_roi(config,record,depth):
    height,width=depth.shape
    focal=width/(2*math.tan(math.radians(record['fov_deg'])/2))
    vv,uu=np.mgrid[0:height,0:width]
    rays=np.stack([np.ones_like(uu),(uu-(width-1)/2)/focal,(vv-(height-1)/2)/focal],axis=-1)
    rays=rays@rotation(record['camera_orientation']).T
    p=record['camera_position']; origin=np.array([p[k] for k in ['x_val','y_val','z_val']])
    valid=rays[:,:,0]>.01
    expected=(config['wall_x_m']-origin[0])/np.where(valid,rays[:,:,0],1)
    world=origin+rays*expected[:,:,None]
    valid &= (expected>0)&np.isfinite(depth)&(np.abs(depth-expected)<.25)
    valid &= (world[:,:,1]>=config['wall_y_start_m'])&(world[:,:,1]<=config['wall_y_start_m']+config['width_m'])
    valid &= (world[:,:,2]<=-config['bottom_height_m'])&(world[:,:,2]>=-config['bottom_height_m']-config['height_m'])
    return valid.astype(np.uint8)*255


def map_pixel(config,record,pixel):
    width,height=record['width'],record['height']
    focal=width/(2*math.tan(math.radians(record['fov_deg'])/2))
    ray=rotation(record['camera_orientation'])@np.array([1,(pixel[0]-(width-1)/2)/focal,(pixel[1]-(height-1)/2)/focal])
    p=record['camera_position']; origin=np.array([p[k] for k in ['x_val','y_val','z_val']])
    if ray[0]<=0:
        return None
    world=origin+ray*((config['wall_x_m']-origin[0])/ray[0])
    return {'x_m':round(float(world[0]),3),'y_m':round(float(world[1]),3),'height_m':round(float(-world[2]),3)}


def detect(image,roi=None,sensitivity='balanced',backend='trained',input_scale=1.0):
    if image is None or image.ndim!=3:
        raise ValueError('A decoded color image is required.')
    if image.shape[0]*image.shape[1]>30_000_000:
        raise ValueError('Image exceeds 30 megapixels.')
    if sensitivity not in ['conservative','balanced','sensitive']:
        raise ValueError('Unknown detector sensitivity.')
    if backend!='trained':
        raise ValueError('This application requires the trained segmentation model.')
    from .model_client import detect as trained_detect
    return trained_detect(image,roi,sensitivity,input_scale=input_scale)
