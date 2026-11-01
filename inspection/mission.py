"""Execute the current selected-wall inspection and save its audited results."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3

from .environment import spawn
from .room import make_room
from .runtime import ensure_simulator,ready
from .rpc import AirSimRPC
from .room_flight import execute
from .room_report import write_room_report
from .room_vision import analyze_room,write_room_model_report
from .room_scene import prepare,spawn as spawn_defects,OUT as SCENE
from .mission_scope import normalize_wall_ids,select_room_walls,plan_selected_walls
from .model_client import ensure_worker
from .flight_audit import verify
from .evaluation import evaluate
from .inspection_completeness import assess,write_report as write_completeness
from .digital_twin import register_run
from .single_wall_gate import make_gate,require_passed
from .worker_review import write_report as write_worker_report

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Start the simulator and fly the inspection')
    parser.add_argument('--walls',nargs='+',choices=['S','E','N','W'],default=['S'])
    parser.add_argument('--adaptive',action='store_true',help='Seek clearer views of uncertain or incomplete areas')
    args=parser.parse_args()
    selected=normalize_wall_ids(args.walls)
    single_wall=selected[0] if len(selected)==1 else None
    room=select_room_walls(make_room(),selected)
    room.update(online_depth=True,trained_inspection=True,scene_profile='sparse-demo-v4',model_input_scale=.25,
        adaptive_reinspection=args.adaptive,adaptive_travel_budget_m=40.,maximum_gap_rechecks=4,
        inspection_target_opening_mm=20.,wall_support_policy='majority-depth-wall-v1',
        frame_postprocess='instance-mask-nms-v2',association_model_resolution=True,
        association_backend='wall-footprint-map-v4',verify_edge_candidates=True,maximum_centred_rechecks=16)
    if not args.execute:
        print('Use the worker interface, or add --execute to start the inspection.')
        return
    if not (ROOT/'weights/best.pt').is_file():
        raise RuntimeError('Missing weights/best.pt. No simulator was started.')
    if not (ROOT/'.venv-model/Scripts/python.exe').is_file():
        raise RuntimeError('Create .venv-model first; see README.md. No simulator was started.')
    if not (SCENE/'truth.json').is_file():
        raise RuntimeError('Run python run.py --prepare first. No simulator was started.')
    if len(selected)>1:
        require_passed(ROOT,room['scene_profile'],adaptive_reinspection=args.adaptive)
    ensure_worker()
    if ready():
        with AirSimRPC() as client:
            if client.call('getMultirotorState','')['landed_state']!=0:
                raise RuntimeError('Drone must be landed')
            try:
                response=client.call('simGetImages',[{'camera_name':'nav','image_type':1,'pixels_as_float':True,'compress':False}],'',False)[0]
                nav_ready=(response['width'],response['height'])==(320,180)
            except RuntimeError:
                nav_ready=False
            if not nav_ready:
                try:client.call('simRunConsoleCommand','quit')
                except ConnectionError:pass
                import time
                time.sleep(2)
    ensure_simulator(json.loads((ROOT/'configs/camera.json').read_text()))
    spawn(room)
    truth=prepare(room)
    spawn_defects(room,truth)
    directory=ROOT/'assets/construction-room/flights'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    full_plan=plan_selected_walls(room,selected)
    print('RUN_DIR',directory,flush=True)
    try:
        value=execute(room,full_plan['views'],directory)
    finally:
        if directory.exists():
            shutil.copytree(SCENE,directory/'truth',dirs_exist_ok=True)
            (directory/'withheld-obstacles.json').write_text('[]')
            (directory/'planned-coverage.json').write_text(json.dumps(full_plan,indent=2))
            if (directory/'flight.json').exists():
                write_room_report(directory)
                (ROOT/'assets/construction-room/latest-integrated.json').write_text(json.dumps({
                    'run':directory.name,'flight_report':f'flights/{directory.name}/report.html',
                    'count_report':f'flights/{directory.name}/count-verification/index.html'}))
    audit=verify(directory)
    if not audit['passed']:
        raise RuntimeError('Independent flight audit failed: '+str(audit['failures']))
    result=analyze_room(directory)
    evaluated=evaluate(directory)
    write_completeness(directory,assess(room,value['images'],directory))
    try:
        twin_sync=register_run(ROOT,directory)
    except (OSError,ValueError,sqlite3.Error) as error:
        twin_sync={'status':'failed','error':str(error)}
    (directory/'twin-sync.json').write_text(json.dumps(twin_sync,indent=2))
    print('DIGITAL_TWIN_SYNC',json.dumps(twin_sync),flush=True)
    gate=make_gate(directory,room,result,evaluated) if single_wall else None
    write_room_model_report(directory,result)
    write_worker_report(directory)
    write_room_report(directory)
    category='wall-'+single_wall if single_wall else 'walls-'+''.join(selected) if len(selected)<4 else 'full'
    manifest=dict(run=directory.name,status='completed',count_ready=True,
        flight_report=f'flights/{directory.name}/report.html',count_report=f'flights/{directory.name}/count-verification/index.html',
        worker_report=f'flights/{directory.name}/worker-report.html',worker_review_json=f'flights/{directory.name}/worker-review.json',
        adaptive_reinspection=args.adaptive,completeness_report=f'flights/{directory.name}/inspection-completeness.html',
        scenario=category,inspection_wall_id=single_wall,inspection_wall_ids=selected,
        association_backend=room['association_backend'],single_wall_gate_passed=gate['passed'] if gate else None,
        centred_verification_views=len(value.get('centred_verification_views',[])),model_input_scale=room['model_input_scale'],
        scene_profile=room['scene_profile'],capture_pairs=len(value['images']),depth_replans=len(value.get('depth_replan_events',[])))
    for filename in ['latest-integrated.json','latest-integrated-success.json',f'latest-integrated-{category}-success.json']:
        (ROOT/'assets/construction-room'/filename).write_text(json.dumps(manifest,indent=2))
    print('INTEGRATED_READY',json.dumps({'run':directory.name,'status':value['status'],'landing':value['landing'],
        'capture_pairs':len(value['images']),'model_tracks':len(result['findings']),
        'depth_replans':len(value.get('depth_replan_events',[]))}),flush=True)


if __name__=='__main__':
    main()
