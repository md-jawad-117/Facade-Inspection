"""Preflight checks against actual Blocks geometry (Unreal cm to NED m)."""
import math


def mesh_bounds(mesh):
    vertices = mesh['vertices']
    # Only indexed vertices belong to the mesh; unused buffer entries may contain garbage.
    points = [vertices[i*3:i*3+3] for i in set(mesh['indices'])]
    if not points or any(len(p) != 3 or any(not math.isfinite(v) or abs(v) > 1e7 for v in p) for p in points):
        raise ValueError('Invalid referenced scene geometry.')
    points = [(p[0]/100, p[1]/100, -p[2]/100) for p in points]
    return [min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]


def segment_hits_box(start, end, lower, upper, clearance):
    low, high = 0.0, 1.0
    for a, b, lo, hi in zip(start, end, lower, upper):
        lo, hi = lo-clearance, hi+clearance
        delta = b-a
        if abs(delta) < 1e-10:
            if a < lo or a > hi:
                return False
        else:
            x, y = sorted(((lo-a)/delta, (hi-a)/delta))
            low, high = max(low, x), min(high, y)
            if low > high:
                return False
    return True


def preflight(client, result):
    config = result['config']
    if not config.get('target_object'):
        raise ValueError('A verified target_object is required for full inspection. Use blocks-wall.json.')
    meshes = client.call('simGetMeshPositionVertexBuffers', timeout=30)
    target = config['target_object'].lower()
    boxes = []
    target_box = None
    for mesh in meshes:
        name = mesh['name'].decode() if isinstance(mesh['name'], bytes) else mesh['name']
        bounds = mesh_bounds(mesh)
        boxes.append((name, *bounds))
        if name.lower() == target:
            target_box = bounds
    if target_box is None:
        raise ValueError('Configured wall object is absent from this scene.')
    lo, hi = target_box
    if (abs(lo[0]-config['wall_x_m']) > 0.05
            or config['wall_y_start_m'] < lo[1]+0.3
            or config['wall_y_start_m']+config['width_m'] > hi[1]-0.3
            or -config['bottom_height_m'] > hi[2]-0.2
            or -config['bottom_height_m']-config['height_m'] < lo[2]+0.2):
        raise ValueError('The inspection rectangle does not fit inside the verified target face.')
    # The facade is axis-aligned; reject a moved or rotated target.
    pose = client.call('simGetObjectPose', config['target_object'])
    orientation = pose['orientation']
    if any(abs(orientation[k]) > 0.001 for k in ['x_val','y_val','z_val']):
        raise ValueError('Target rotated; this demo requires an axis-aligned face.')
    from .coordinates import scene_position
    origin_calibration=getattr(client,'scene_coordinate_calibration',None)
    start=scene_position((0,0,-1.5),origin_calibration) if origin_calibration else (0,0,-1.5)
    end=scene_position((0,0,-3),origin_calibration) if origin_calibration else (0,0,-3)
    route = [start] + [(p['x'],p['y'],p['z']) for p in result['waypoints']] + [end]
    # Gap-filling views can use any point in this same rectangular scan plane.
    scan_x=result['waypoints'][0]['x']
    scan_lo=(scan_x,config['wall_y_start_m'],-config['bottom_height_m']-config['height_m'])
    scan_hi=(scan_x,config['wall_y_start_m']+config['width_m'],-config['bottom_height_m'])
    for name,lower,upper in boxes:
        if all(a-config.get('clearance_m',0.8)<=d and b+config.get('clearance_m',0.8)>=c
               for a,b,c,d in zip(lower,upper,scan_lo,scan_hi)):
            raise ValueError(f'Inspection scan plane intersects inflated obstacle: {name}')
    for start, end in zip(route, route[1:]):
        for name, lower, upper in boxes:
            if segment_hits_box(start,end,lower,upper,config.get('clearance_m',0.8)):
                raise ValueError(f'Planned flight corridor intersects inflated obstacle: {name}')
    return {'target': config['target_object'], 'target_bounds_ned_m': target_box,
            'mesh_count': len(meshes), 'clearance_m': config.get('clearance_m',0.8),
            'method': 'Conservative static mesh bounding boxes; approach, inspection and return corridors.'}
