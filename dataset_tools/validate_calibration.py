import argparse
import json
import math
from pathlib import Path
import numpy as np
from build_calibrated_dataset import trace_length


def validate(root):
    root=Path(root);walls=json.loads((root/'wall-identities.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    errors=[];cracks=[];openings={};stress=0
    for wall in walls:
        for obj in wall['objects']:
            if obj['class_id']!=0:continue
            points=np.array(obj['polyline_mm']);width=obj['opening_width_mm']
            length=trace_length(points)+width;spread=float(np.ptp(points[:,0]))+width
            if not 50-1e-7<=length<=100+1e-7:errors.append('Length outside target: '+obj['physical_id'])
            if abs(width-5)>1e-9:errors.append('Width outside target: '+obj['physical_id'])
            if not 20-1e-7<=spread<=50+1e-7:errors.append('Spread outside target: '+obj['physical_id'])
            if abs(length-obj['visible_trace_length_mm'])>1e-7 or abs(spread-obj['zigzag_spread_mm'])>1e-7:
                errors.append('Stored physical dimensions disagree with geometry: '+obj['physical_id'])
            cracks.append([length,width,spread])
    for view in manifest:
        camera=view['camera'];gsd=camera['world_mm_per_pixel'];focal=camera['focal_length_px']
        expected_focal=camera['full_sensor_width_px']/(2*math.tan(math.radians(camera['horizontal_fov_deg'])/2))
        if abs(focal-expected_focal)>1e-8 or abs(gsd-camera['standoff_mm']/focal)>1e-8:errors.append('Invalid camera scale: '+view['stem'])
        if camera['stress_only']:
            stress+=1
            if view['yolo_eligible']:errors.append('Low-resolution stress view in ordinary training/evaluation list')
        if view.get('augmentation'):continue
        pitch=camera['source_mm_per_pixel'];offset=np.array(camera['tile_origin_in_sensor_px'])
        principal=np.array([(camera['full_sensor_width_px']-1)/2,(camera['full_sensor_height_px']-1)/2])
        expected=np.array([[pitch/gsd,0,-camera['camera_wall_center_mm'][0]/gsd+principal[0]-offset[0]],
                           [0,pitch/gsd,-camera['camera_wall_center_mm'][1]/gsd+principal[1]-offset[1]],[0,0,1]])
        if not np.allclose(expected,view['homography_source_to_image']):errors.append('Projection mismatch: '+view['stem'])
        openings[camera['profile_id']]=5/gsd
    values=np.array(cracks)
    report={'passed':not errors,'physical_crack_objects_checked':len(cracks),'views_checked':len(manifest),
            'actual_nominal_length_range_mm':[float(values[:,0].min()),float(values[:,0].max())],
            'actual_nominal_width_range_mm':[float(values[:,1].min()),float(values[:,1].max())],
            'actual_nominal_spread_range_mm':[float(values[:,2].min()),float(values[:,2].max())],
            'opening_width_pixels_by_camera_profile':openings,'resolution_stress_views':stress,'failures':errors,
            'scope':'Continuous synthetic geometry and ideal pinhole projection validated; real optics, camera performance and detection accuracy unverified.'}
    (root/'physical-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);a=p.parse_args()
    raise SystemExit(0 if validate(a.dataset)['passed'] else 1)
