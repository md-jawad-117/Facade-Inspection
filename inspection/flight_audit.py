"""Audit saved flight evidence independently of the controller."""
from collections import Counter
import json
import math
from pathlib import Path
import cv2
from .scene import segment_hits_box
from .room import in_free_space,to_ned


def capture_manifest_errors(targets,images):
    scans=[r for r in images if not r.get('verification_parent_file')]
    expected=Counter(t['id'] for t in targets);actual=Counter(r.get('target',{}).get('id') for r in scans)
    errors=[]
    if expected!=actual:errors.append('Missing or duplicated planned captures')
    names=[r['file'] for r in images]
    if len(set(names))!=len(names):errors.append('Duplicate capture filenames')
    parents={r['file']:r for r in scans}
    for record in images:
        parent=record.get('verification_parent_file')
        if not parent:continue
        if parent not in parents or record.get('wall_id')!=parents[parent].get('wall_id'):
            errors.append('Invalid verification parent or wall')
        if record.get('target',{}).get('id') in expected:errors.append('Verification reused planned target identity')
    return sorted(set(errors))


def verify(directory):
    directory=Path(directory);value=json.loads((directory/'flight.json').read_text())
    room=json.loads((directory/'room.json').read_text());targets=json.loads((directory/'targets.json').read_text())
    rows=[json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()]
    failures=[]
    if value['status']!='completed' or value['landing']!='completed':failures.append('Flight or landing incomplete')
    if not value.get('return_verified'):failures.append('Return not verified')
    failures.extend(capture_manifest_errors(targets,value['images']))
    withheld_path=directory/'withheld-obstacles.json'
    withheld=json.loads(withheld_path.read_text()) if withheld_path.exists() else []
    for obstacle in withheld:
        for row in rows:
            if segment_hits_box(row['position'],row['position'],obstacle['lower'],obstacle['upper'],room['clearance_m']):
                failures.append('Measured position violates withheld-obstacle clearance')
        for a,b in zip(rows,rows[1:]):
            if segment_hits_box(a['position'],b['position'],obstacle['lower'],obstacle['upper'],room['clearance_m']):
                failures.append('Measured segment violates withheld-obstacle clearance')
    if withheld and not value.get('depth_replan_events'):failures.append('No depth-based replan recorded in withheld-obstacle trial')
    baseline=value.get('airborne_collision_baseline_timestamp',rows[0]['timestamp'])
    if any(row['collision_timestamp']>baseline for row in rows):failures.append('New collision timestamp during sampled flight')
    if any(not in_free_space(room,row['position']) for row in rows):failures.append('Measured position outside free space')
    for a,b in zip(rows,rows[1:]):
        if b['timestamp']<a['timestamp']:failures.append('Telemetry clock moved backwards')
        if any(segment_hits_box(a['position'],b['position'],o['lower'],o['upper'],room['clearance_m']) for o in room['obstacles']):
            failures.append('Measured motion segment crosses inflated obstacle')
    for record in value['images']:
        path=directory/record['file'];image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(record['height'],record['width']):failures.append('Invalid RGB '+record['file'])
        if (directory/record['depth_file']).stat().st_size!=record['width']*record['height']*4:failures.append('Invalid depth '+record['file'])
        if record['wall_id'] and abs(record['measured_camera_standoff_m']-room['standoff_m'])>.2:
            failures.append('Camera distance outside tolerance '+record['file'])
    frame_calibration=value.get('coordinate_calibration')
    if frame_calibration:
        keys=['x_val','y_val','z_val'];offset=frame_calibration['local_to_global_translation_m']
        for row in rows:
            local=row.get('position_vehicle_local_ned')
            if row.get('position_frame')!='scene-global-ned' or local is None:
                failures.append('Telemetry coordinate frame not recorded');continue
            if any(abs(global_value-local_value-offset[k])>.0001 for k,global_value,local_value in zip(keys,row['position_ned'],local)):
                failures.append('Telemetry frame conversion differs from recorded calibration')
        for record in value['images']:
            if record.get('camera_position_frame')!='scene-global-ned' or 'camera_position_local_ned' not in record:
                failures.append('Camera coordinate frame not recorded');continue
            if any(abs(record['camera_position'][k]-record['camera_position_local_ned'][k]-offset[k])>.0001 for k in keys):
                failures.append('Camera frame conversion differs from recorded calibration')
            if abs(-record['camera_position']['z_val']-record['target']['position'][2])>.2:
                failures.append('Camera scene height differs from planned height')
    landing=value.get('landing_position_ned')
    if landing and math.dist(landing[:2],to_ned(room,room['launch_m'])[:2])>.3:failures.append('Landed outside launch XY tolerance')
    result={'planned_capture_count':len(targets),'verification_capture_count':sum(bool(r.get('verification_parent_file')) for r in value['images']),'passed':not failures,'capture_pairs_checked':len(value['images']),'telemetry_samples_checked':len(rows),
            'actual_motion_segments_checked':len(rows)-1,'withheld_obstacles_checked':len(withheld),'depth_replan_events':len(value.get('depth_replan_events',[])),'failures':sorted(set(failures)),
            'scope':'Saved simulator evidence: sampled center positions, collision timestamps, image/depth files, standoff and landing. Not continuous rotor clearance or live depth avoidance.'}
    (directory/'evidence-audit.json').write_text(json.dumps(result,indent=2))
    return result
