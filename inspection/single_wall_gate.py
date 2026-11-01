"""Postflight ground-truth gate; never imported by the detector or pilot."""
import hashlib,json
from pathlib import Path

def implementation_sha(root):
    names=('association.py','frame_postprocess.py','wall_tracks.py','room_vision.py','room_live_model.py','wall_recheck.py','room_flight.py','available_view.py','mission_scope.py','adaptive_view.py','inspection_completeness.py','class_consensus.py','wall_support.py','model_masks.py','model_client.py')
    return hashlib.sha256(b''.join((Path(root)/'inspection'/name).read_bytes() for name in names)).hexdigest()

def make_gate(directory,room,analysis,evaluation):
    directory=Path(directory);metrics=evaluation['identity_metrics']
    reasons=[]
    for kind,values in metrics.items():
        for key in ('missed_visible_physical_ids','not_observed_by_route','extra_split_tracks','incorrect_merged_tracks','unmatched_candidate_tracks'):
            if values[key]:reasons.append(f'{kind}: {key}={values[key]}')
    root=Path(__file__).resolve().parents[1]
    result=dict(implementation_sha256=implementation_sha(root),adaptive_reinspection=bool(room.get('adaptive_reinspection')),passed=not reasons,reasons=reasons,scene_profile=room['scene_profile'],wall_id=room['inspection_wall_id'],
                model_sha256=analysis['model']['sha256'],source_truth_sha256=hashlib.sha256((directory/'truth/truth.json').read_bytes()).hexdigest(),
                frame_postprocess=room.get('frame_postprocess'),association_backend=room.get('association_backend'),identity_metrics=metrics,
                scope='Single-wall acceptance, not proof of all-wall or real-world accuracy')
    (directory/'single-wall-gate.json').write_text(json.dumps(result,indent=2));print('SINGLE_WALL_GATE',json.dumps(result),flush=True)
    return result

def require_passed(root,profile,adaptive_reinspection=False):
    root=Path(root);manifest=root/'assets/construction-room/latest-integrated-wall-S-success.json'
    if not manifest.exists():raise ValueError('Run the south-wall test first; a full-room wall-map test is gated on its result.')
    latest=json.loads(manifest.read_text());directory=root/'assets/construction-room/flights'/latest['run'];path=directory/'single-wall-gate.json'
    if not path.exists():raise ValueError('Latest wall test has no acceptance result')
    gate=json.loads(path.read_text())
    if bool(gate.get('adaptive_reinspection'))!=adaptive_reinspection:raise ValueError('Run a South-wall verification with the selected adaptive mode first')
    if not gate['passed'] or gate['scene_profile']!=profile or gate['association_backend']!='wall-footprint-map-v4' or gate['frame_postprocess']!='instance-mask-nms-v2':raise ValueError('Single-wall test has not passed for this scene and tracking version; full-room test blocked.')
    if gate.get('implementation_sha256')!=implementation_sha(root):raise ValueError('Tracking implementation changed since the wall test')
    if gate['model_sha256']!=hashlib.sha256((root/'weights/best.pt').read_bytes()).hexdigest():raise ValueError('Model changed since the wall test')
    source={'sparse-demo-v4':'scene'}.get(profile)
    if source is None or gate['source_truth_sha256']!=hashlib.sha256((root/'assets/construction-room'/source/'truth.json').read_bytes()).hexdigest():raise ValueError('Scene changed since the wall test')
    return gate
