"""Depth-verified sampled visibility of a known planar facade."""
import array
import math
from pathlib import Path


def rotate_inverse(q, vector):
    # q conjugate * vector * q; input quaternion rotates camera coordinates into NED.
    w, x, y, z = (q[k] for k in ['w_val','x_val','y_val','z_val'])
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    w,x,y,z = w/norm,-x/norm,-y/norm,-z/norm
    vx,vy,vz = vector
    tx,ty,tz = 2*(y*vz-z*vy),2*(z*vx-x*vz),2*(x*vy-y*vx)
    return vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx


def sample_grid(config):
    nx = math.ceil(config['width_m']/config.get('coverage_cell_m',0.1))
    ny = math.ceil(config['height_m']/config.get('coverage_cell_m',0.1))
    if nx*ny > 1000000:
        raise ValueError('Coverage grid too large.')
    points = [(config['wall_x_m'],config['wall_y_start_m']+(i+0.5)*config['width_m']/nx,
               -(config['bottom_height_m']+(j+0.5)*config['height_m']/ny))
              for j in range(ny) for i in range(nx)]
    return nx,ny,points


def visible_samples(config, record, depth, points):
    width,height = record['width'],record['height']
    focal = width/(2*math.tan(math.radians(record['fov_deg'])/2))
    camera = record['camera_position']
    origin = tuple(camera[k] for k in ['x_val','y_val','z_val'])
    hits = set()
    for index,point in enumerate(points):
        forward,right,down = rotate_inverse(record['camera_orientation'],tuple(a-b for a,b in zip(point,origin)))
        if forward <= 0:
            continue
        u,v = (width-1)/2+focal*right/forward,(height-1)/2+focal*down/forward
        if not (0 <= u < width-1 and 0 <= v < height-1):
            continue
        measured = depth[round(v)*width+round(u)]
        if math.isfinite(measured) and measured > 0 and abs(measured-forward) <= config.get('depth_tolerance_m',0.18):
            hits.add(index)
    return hits


def evaluate(config, records, directory):
    nx,ny,points = sample_grid(config)
    covered = set()
    counts = [0]*len(points)
    frames = []
    for record in records:
        depth = array.array('f')
        with (Path(directory)/record['depth_file']).open('rb') as stream:
            depth.fromfile(stream,record['width']*record['height'])
        hits = visible_samples(config,record,depth,points)
        covered.update(hits)
        for index in hits:
            counts[index] += 1
        frames.append({'file':record['file'],'visible_samples':len(hits),
                       'cumulative_coverage_percent':100*len(covered)/len(points)})
    return {'method':'Depth-verified visibility of grid-cell centers on the selected flat face',
            'grid_width':nx,'grid_height':ny,'sample_count':len(points),
            'covered_samples':len(covered),'coverage_percent':100*len(covered)/len(points),
            'estimated_visible_area_m2':len(covered)/len(points)*config['width_m']*config['height_m'],
            'target_area_m2':config['width_m']*config['height_m'],
            'depth_tolerance_m':config.get('depth_tolerance_m',0.18),
            'counts':counts,'frames':frames,
            'limitations':'Sampled simulator visibility, not continuous coverage or structural defect detection. Ground-truth depth and pose are used.'}


def next_gap_view(config, coverage, next_index):
    """Greedily center a new view on the largest group of unverified samples."""
    _,_,points=sample_grid(config)
    missing=[point for point,count in zip(points,coverage['counts']) if count==0]
    if not missing:
        return None
    fw=2*config['standoff_m']*math.tan(math.radians(config['horizontal_fov_deg'])/2)
    fh=fw*config['image_height_px']/config['image_width_px']
    # Evaluate a bounded set of candidates; count only a conservative inner footprint.
    candidates=missing[::max(1,len(missing)//250)]
    best=max(candidates,key=lambda p:sum(abs(q[1]-p[1])<fw*.4 and abs(q[2]-p[2])<fh*.4 for q in missing))
    return {'index':next_index,'x':config['wall_x_m']-config['standoff_m']-config.get('camera_offset_x_m',0),
            'y':best[1],'z':best[2],'yaw_deg':0,'purpose':'reinspect uncovered grid samples'}
