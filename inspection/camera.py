"""Camera capture and flight-state checks shared by the controller."""
import array
import math
import os
from pathlib import Path
from .coordinates import measure_scene_offset,camera_in_scene


def check_stop():
    stop_file=os.environ.get('FACADE_STOP_FILE')
    if stop_file and Path(stop_file).exists():
        raise RuntimeError('Inspection stopped by operator. Landing requested.')


def capture(client, directory, name, with_depth=False):
    request = {'camera_name': '0', 'image_type': 0, 'pixels_as_float': False, 'compress': True}
    requests = [request]
    if with_depth:
        requests.append({'camera_name':'0','image_type':1,'pixels_as_float':True,'compress':False})
    images = client.call('simGetImages', requests, '', False)
    if not images or images[0]['width'] <= 0:
        raise RuntimeError('AirSim returned an empty camera image.')
    image = images[0]
    data = image['image_data_uint8']
    if isinstance(data, list):
        data = bytes(data)
    if not isinstance(data, bytes) or not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise RuntimeError('Camera response is not a PNG image.')
    path = directory / f'{name}.png'
    path.write_bytes(data)
    if not hasattr(client,'scene_coordinate_calibration'):
        client.scene_coordinate_calibration=measure_scene_offset(client)
    record = {'file': path.name, 'width': image['width'], 'height': image['height'],
            'timestamp': image['time_stamp'], 'camera_position': image['camera_position'],
            'camera_orientation': image['camera_orientation']}
    record=camera_in_scene(record,client.scene_coordinate_calibration)
    if with_depth:
        depth = images[1]
        if ((depth['width'],depth['height']) != (image['width'],image['height'])
                or len(depth['image_data_float']) != image['width']*image['height']):
            raise RuntimeError('Depth and RGB dimensions differ.')
        record['depth_file'] = f'{name}.depth.f32'
        record['depth_timestamp'] = depth['time_stamp']
        record['fov_deg'] = client.call('simGetCameraInfo','0','',False)['fov']
        with (directory/record['depth_file']).open('wb') as stream:
            array.array('f',depth['image_data_float']).tofile(stream)
    return record


def position(state):
    p = state['kinematics_estimated']['position']
    return (p['x_val'], p['y_val'], p['z_val'])


def verify_takeoff(state, origin):
    current = position(state)
    velocity = state['kinematics_estimated']['linear_velocity']
    speed = math.sqrt(sum(velocity[k] ** 2 for k in ['x_val', 'y_val', 'z_val']))
    if (state['landed_state'] != 1 or origin[2] - current[2] < 1.0
            or math.dist(current[:2], origin[:2]) > 0.75 or speed > 0.5):
        raise RuntimeError('Takeoff did not produce a stable airborne hover.')
    return {'rise_m': origin[2] - current[2], 'position_ned': current, 'speed_mps': speed}


def ground_contact(state, collision, since):
    velocity = state['kinematics_estimated']['linear_velocity']
    speed = math.sqrt(sum(velocity[k] ** 2 for k in ['x_val', 'y_val', 'z_val']))
    contact = collision['position']
    return (collision['time_stamp'] > 0 and collision['time_stamp'] >= since
            and collision['normal']['z_val'] < -0.8 and speed < 0.15
            and math.dist(position(state), tuple(contact[k] for k in ['x_val', 'y_val', 'z_val'])) < 0.3)
