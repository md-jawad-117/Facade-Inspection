"""Explicit AirSim vehicle-local to scene-global NED translation.

AirSim 1.8.1 camera/kinematic poses use the pawn's local origin. Scene object
poses use the global origin. Measure their difference while landed; never infer
an unknown historical run's origin from a different simulator session.
"""
import math

KEYS=('x_val','y_val','z_val')


def measure_scene_offset(client,actor='SimpleFlight'):
    first=client.call('getMultirotorState','')
    if first['landed_state']!=0:raise RuntimeError('Coordinate calibration requires a landed vehicle.')
    global_pose=client.call('simGetObjectPose',actor)
    second=client.call('getMultirotorState','')
    a=first['kinematics_estimated']['position'];b=second['kinematics_estimated']['position']
    values=[global_pose['position'][k] for k in KEYS]+[a[k] for k in KEYS]+[b[k] for k in KEYS]
    if not all(math.isfinite(v) for v in values):raise RuntimeError('Vehicle actor pose is unavailable.')
    if second['landed_state']!=0 or math.dist([a[k] for k in KEYS],[b[k] for k in KEYS])>.001:
        raise RuntimeError('Vehicle moved during coordinate calibration.')
    offset={k:global_pose['position'][k]-(a[k]+b[k])/2 for k in KEYS}
    if math.sqrt(sum(v*v for v in offset.values()))>5:raise RuntimeError('Unexpected coordinate-origin separation.')
    return {'version':'airsim-local-to-scene-v1','vehicle_actor':actor,
            'local_to_global_translation_m':offset,'landed_calibration':True,
            'source':'Same-session vehicle global actor pose minus vehicle-local kinematics.'}


def camera_in_scene(record,calibration):
    if record.get('camera_position_frame')=='scene-global-ned':raise ValueError('Camera is already in the scene frame.')
    if record.get('camera_position_frame') not in [None,'vehicle-local-ned']:raise ValueError('Unknown camera coordinate frame.')
    offset=calibration['local_to_global_translation_m'];position=record['camera_position']
    return dict(record,camera_position_local_ned=dict(position),
                camera_position={k:position[k]+offset[k] for k in KEYS},
                camera_position_frame='scene-global-ned',coordinate_calibration=calibration)


def scene_position(local,calibration):
    offset=calibration['local_to_global_translation_m']
    return tuple(v+offset[k] for k,v in zip(KEYS,local))


def vehicle_position(global_position,calibration):
    offset=calibration['local_to_global_translation_m']
    return tuple(v-offset[k] for k,v in zip(KEYS,global_position))
