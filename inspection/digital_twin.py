"""Persistent inspection history from model observations, independent of labels."""
import hashlib,json,math,sqlite3
from datetime import datetime,timezone
from contextlib import contextmanager
from pathlib import Path

REVIEW_STATES={'unreviewed','confirmed','follow_up','repair_reported'}
MEASUREMENT_METHOD='multi-view-footprint-extent-v2'

def support_score(left,right):
    a={tuple(p) for p in left};b={tuple(p) for p in right}
    if not a or not b:return 0.
    # Supports use a 10 mm wall grid; one cell is the registration allowance.
    expanded={(x+dx,y+dy) for x,y in b for dx in [-1,0,1] for dy in [-1,0,1]}
    return len(a&expanded)/len(a) if len(a)<=len(b) else support_score(right,left)

def match_visits(previous,current):
    matches={};ambiguous=set();candidates={}
    for i,new in enumerate(current):
        scores=sorted([(support_score(new['support'],old['support']),j) for j,old in enumerate(previous)
                       if old['kind']==new['kind']],reverse=True)
        scores=[(score,j) for score,j in scores if score>=.65]
        if not scores:continue
        if len(scores)>1 and scores[0][0]-scores[1][0]<.1:ambiguous.add(i);continue
        candidates[i]=scores[0]
    for i,(score,j) in candidates.items():
        competing=[k for k,(_,old_j) in candidates.items() if old_j==j]
        if len(competing)>1:ambiguous.update(competing)
        elif i not in ambiguous:matches[i]=j
    return matches,ambiguous

def change_event(previous,current):
    if current.get('edge_only') or previous.get('edge_only'):
        return dict(event='needs_verification',delta_m=None,threshold_m=None)
    delta=current['extent_m']-previous['extent_m']
    threshold=current['uncertainty_m']+previous['uncertainty_m']
    event='extent_increased' if delta>threshold else 'extent_decreased' if delta<-threshold else 'no_measurable_change'
    return dict(event=event,delta_m=delta,threshold_m=threshold)

class TwinStore:
    def __init__(self,path):self.path=Path(path)
    @contextmanager
    def connect(self):
        connection=sqlite3.connect(self.path,timeout=15);connection.row_factory=sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:yield connection
        finally:connection.close()
    def initialize(self,room):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, wall_id TEXT UNIQUE, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS inspections(id TEXT PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),series TEXT NOT NULL,day INTEGER NOT NULL,data TEXT NOT NULL,digest TEXT NOT NULL,UNIQUE(asset_id,series,day));
            CREATE TABLE IF NOT EXISTS defects(id TEXT PRIMARY KEY,asset_id TEXT NOT NULL REFERENCES assets(id),series TEXT NOT NULL,kind TEXT NOT NULL,first_inspection TEXT NOT NULL REFERENCES inspections(id));
            CREATE TABLE IF NOT EXISTS observations(inspection_id TEXT NOT NULL REFERENCES inspections(id),defect_id TEXT NOT NULL REFERENCES defects(id),data TEXT NOT NULL,PRIMARY KEY(inspection_id,defect_id));
            CREATE TABLE IF NOT EXISTS events(inspection_id TEXT NOT NULL REFERENCES inspections(id),defect_id TEXT NOT NULL REFERENCES defects(id),data TEXT NOT NULL,PRIMARY KEY(inspection_id,defect_id));
            CREATE TABLE IF NOT EXISTS reviews(defect_id TEXT PRIMARY KEY REFERENCES defects(id),status TEXT NOT NULL,note TEXT NOT NULL,updated_utc TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS review_events(id INTEGER PRIMARY KEY AUTOINCREMENT,defect_id TEXT NOT NULL REFERENCES defects(id),status TEXT NOT NULL,note TEXT NOT NULL,updated_utc TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS verification_evidence(defect_id TEXT NOT NULL REFERENCES defects(id),capture_url TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(defect_id,capture_url));
            ''')
            for wall in room['walls']:
                asset=dict(id='ROOM-001-WALL-'+wall['id'],wall_id=wall['id'],name=wall['name']+' wall',
                    width_m=wall['length'],height_m=wall['height'],surface='Simulated plaster/concrete appearance',
                    structural_role='Unspecified; surface appearance does not establish structural role',
                    material_properties_verified=False)
                db.execute('INSERT OR IGNORE INTO assets VALUES(?,?,?)',(asset['id'],wall['id'],json.dumps(asset)))
    def ingest(self,asset_id,series,day,metadata,observations):
        digest=hashlib.sha256(json.dumps([metadata,observations],sort_keys=True).encode()).hexdigest()
        visit_id=hashlib.sha256(f'{asset_id}|{series}|{day}'.encode()).hexdigest()[:16]
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM assets WHERE id=?',(asset_id,)).fetchone():raise ValueError('Unknown building element')
            existing=db.execute('SELECT digest FROM inspections WHERE id=?',(visit_id,)).fetchone()
            if existing:
                if existing['digest']!=digest:raise ValueError('Inspection already exists with different evidence; create a new inspection instead')
                return visit_id
            latest=db.execute('SELECT * FROM inspections WHERE asset_id=? AND series=? ORDER BY day DESC LIMIT 1',(asset_id,series)).fetchone()
            if latest and day<=latest['day']:raise ValueError('Append inspections in chronological scenario order')
            previous=[]
            if latest:
                for defect in db.execute('SELECT id FROM defects WHERE asset_id=? AND series=?',(asset_id,series)).fetchall():
                    row=db.execute('SELECT o.data FROM observations o JOIN inspections i ON i.id=o.inspection_id WHERE o.defect_id=? ORDER BY i.day DESC LIMIT 1',(defect['id'],)).fetchone()
                    if row:previous.append(dict(json.loads(row['data']),defect_id=defect['id']))
            matches,ambiguous=match_visits(previous,observations)
            db.execute('INSERT INTO inspections VALUES(?,?,?,?,?,?)',(visit_id,asset_id,series,day,json.dumps(metadata),digest))
            seen=set()
            for i,observation in enumerate(observations):
                if i in matches:
                    old=previous[matches[i]];defect_id=old['defect_id'];event=change_event(old,observation)
                else:
                    number=db.execute('SELECT COUNT(*) FROM defects WHERE asset_id=?',(asset_id,)).fetchone()[0]+1
                    defect_id=asset_id.rsplit('-',1)[-1]+'-D'+str(number).zfill(3)
                    db.execute('INSERT INTO defects VALUES(?,?,?,?,?)',(defect_id,asset_id,series,observation['kind'],visit_id))
                    event=dict(event='needs_verification' if i in ambiguous or observation.get('edge_only') else 'first_observed',delta_m=None,threshold_m=None)
                seen.add(defect_id)
                db.execute('INSERT INTO observations VALUES(?,?,?)',(visit_id,defect_id,json.dumps(observation)))
                db.execute('INSERT INTO events VALUES(?,?,?)',(visit_id,defect_id,json.dumps(event)))
            for old in previous:
                if old['defect_id'] not in seen:
                    event=dict(event='not_observed',delta_m=None,threshold_m=None,
                        note='Not detected in this visit. Absence alone does not confirm a repair.')
                    db.execute('INSERT INTO events VALUES(?,?,?)',(visit_id,old['defect_id'],json.dumps(event)))
        return visit_id
    def review(self,defect_id,status,note):
        if status not in REVIEW_STATES or not isinstance(note,str) or len(note)>2000:raise ValueError('Invalid review status or note')
        with self.connect() as db:
            if not db.execute('SELECT 1 FROM defects WHERE id=?',(defect_id,)).fetchone():raise ValueError('Unknown defect')
            updated=datetime.now(timezone.utc).isoformat()
            db.execute('INSERT INTO reviews VALUES(?,?,?,?) ON CONFLICT(defect_id) DO UPDATE SET status=excluded.status,note=excluded.note,updated_utc=excluded.updated_utc',
                (defect_id,status,note,updated))
            db.execute('INSERT INTO review_events(defect_id,status,note,updated_utc) VALUES(?,?,?,?)',(defect_id,status,note,updated))
    def snapshot(self):
        if not self.path.exists():return dict(ready=False,assets=[],inspections=[],defects=[])
        with self.connect() as db:
            assets=[json.loads(r['data']) for r in db.execute('SELECT * FROM assets ORDER BY wall_id')]
            visits=[]
            for row in db.execute('SELECT * FROM inspections ORDER BY day'):
                visit=dict(json.loads(row['data']),id=row['id'],asset_id=row['asset_id'],series=row['series'],scenario_day=row['day'])
                visit['observations']=[];visit['events']=[]
                for obs in db.execute('SELECT defect_id,data FROM observations WHERE inspection_id=?',(row['id'],)):
                    value=json.loads(obs['data']);value.pop('support',None)
                    visit['observations'].append(dict(value,defect_id=obs['defect_id']))
                for event in db.execute('SELECT defect_id,data FROM events WHERE inspection_id=?',(row['id'],)):
                    visit['events'].append(dict(json.loads(event['data']),defect_id=event['defect_id']))
                visits.append(visit)
            defects=[]
            for row in db.execute('SELECT * FROM defects ORDER BY id'):
                value=dict(row);review=db.execute('SELECT status,note,updated_utc FROM reviews WHERE defect_id=?',(row['id'],)).fetchone()
                value['review']=dict(review) if review else dict(status='unreviewed',note='')
                value['review_history']=[dict(r) for r in db.execute('SELECT status,note,updated_utc FROM review_events WHERE defect_id=? ORDER BY id',(row['id'],))]
                value['verification_history']=[json.loads(r['data']) for r in db.execute('SELECT data FROM verification_evidence WHERE defect_id=?',(row['id'],))]
                defects.append(value)
        return dict(ready=True,assets=assets,inspections=visits,defects=defects,
            measurement_scope='Visible major-axis extent and approximate mask area, not validated crack-opening width or structural severity.')

    def add_verification(self,defect_id,evidence):
        with self.connect() as db:
            if not db.execute('SELECT 1 FROM defects WHERE id=?',(defect_id,)).fetchone():raise ValueError('Unknown defect')
            db.execute('INSERT OR IGNORE INTO verification_evidence VALUES(?,?,?)',(defect_id,evidence['capture_url'],json.dumps(evidence)))

def model_observations(directory):
    import cv2,numpy as np
    from .room_vision import canonical_record
    from .association import project
    from .vision import rotation
    directory=Path(directory).resolve();root=Path(__file__).resolve().parents[1]
    room=json.loads((directory/'room.json').read_text());analysis=json.loads((directory/'model-analysis.json').read_text())
    records={r['file']:r for r in json.loads((directory/'flight.json').read_text())['images']}
    values=[];crops=directory/'twin-evidence';crops.mkdir(exist_ok=True)
    for track in sorted(analysis['findings'],key=lambda t:t['location']['along_wall_m']):
        wall=next(w for w in room['walls'] if w['id']==track['wall_id']);samples=[];support=set();point_cloud=[]
        for frame in analysis['frames']:
            for feature in frame['findings']:
                if feature['id']!=track['id']:continue
                record=records[frame['file']];config,canonical=canonical_record(room,wall,record)
                pixels=np.array([(u,v) for v,a,b in feature['mask_runs'] for u in range(a,b)],float).reshape(-1,2)
                points=project(config,canonical,pixels,rotation(canonical['camera_orientation']))
                if len(points)<2:continue
                point_cloud.append(points)
                support.update(map(tuple,np.rint(points/.01).astype(int)))
                x,y,w,h=feature['bbox'];margin=16/room.get('model_input_scale',1.)
                edge=x<=margin or y<=margin or x+w>=record['width']-margin or y+h>=record['height']-margin
                pitch=2*abs(canonical['camera_position']['x_val'])*math.tan(math.radians(record['fov_deg']/2))/record['width']
                centered=points-points.mean(axis=0);axis=np.linalg.svd(centered,full_matrices=False)[2][0]
                extent=float(np.ptp(points@axis));area=len(points)*pitch*pitch
                samples.append(dict(edge=edge,area=area,extent=extent,pitch=pitch,file=frame['file'],bbox=feature['bbox'],confidence=feature['confidence']))
        if not samples:continue
        full=[s for s in samples if not s['edge']];best=max(full or samples,key=lambda s:s['area'])
        combined=np.concatenate(point_cloud)
        axis=np.linalg.svd(combined-combined.mean(axis=0),full_matrices=False)[2][0]
        extent=float(np.ptp(combined@axis))
        image=cv2.imread(str(directory/best['file']));x,y,w,h=best['bbox'];name=track['id']+'.jpg'
        cv2.imwrite(str(crops/name),image[max(0,y-35):min(image.shape[0],y+h+35),max(0,x-35):min(image.shape[1],x+w+35)])
        cells=np.array(sorted(support));lower=cells.min(axis=0)*.01;upper=cells.max(axis=0)*.01
        values.append(dict(local_track_id=track['id'],wall_id=track['wall_id'],kind=track['kind'],u_m=track['location']['along_wall_m'],height_m=track['location']['height_m'],
            support=cells.tolist(),extent_m=extent,measurement_method=MEASUREMENT_METHOD,area_m2=best['area'],uncertainty_m=max(.01,2*best['pitch']),
            uncertainty_basis='Declared two-native-pixel or 10 mm minimum allowance; not a calibrated confidence interval',
            edge_only=not full,confidence=track['confidence'],observation_count=len(samples),class_resolution=track.get('class_resolution',[]),
            bbox_wall_m=[float(lower[0]),float(lower[1]),float(upper[0]-lower[0]),float(upper[1]-lower[1])],
            evidence_url='/'+(crops/name).relative_to(root).as_posix(),best_frame_url='/'+(directory/best['file']).relative_to(root).as_posix()))
    return values

def register_run(root,directory,source_run=None):
    """Append an analysed AirSim visit; never mix it with controlled scenarios."""
    root=Path(root).resolve();directory=Path(directory).resolve()
    analysis=json.loads((directory/'model-analysis.json').read_text());room=json.loads((directory/'room.json').read_text())
    original=root/'assets/construction-room/flights'/source_run if source_run else directory
    flight=json.loads((original/'flight.json').read_text());audit=json.loads((original/'evidence-audit.json').read_text())
    if flight['status']!='completed' or flight['landing']!='completed' or not audit['passed']:
        raise ValueError('Only completed and audited simulator captures can enter the recorded inspection history')
    from .room import make_room
    store=TwinStore(root/'.state/digital-twin.sqlite');store.initialize(make_room())
    observations=model_observations(directory);registered=[]
    run=original.name;order=int(''.join(c for c in run.split('-')[0]+run.split('-')[1] if c.isdigit()))
    for wall_id in sorted({r['wall_id'] for r in analysis['frames'] if r.get('wall_id')}):
        metadata=dict(label='Recorded AirSim · '+run,source_type='airsim_capture',source_run=run,
            measurement_method=MEASUREMENT_METHOD,
            scenario_time_is_virtual=False,created_utc=datetime.fromtimestamp((original/'flight.json').stat().st_mtime,timezone.utc).isoformat(),
            preview_url=None,report_url='/'+(directory/'model-report.html').relative_to(root).as_posix(),
            validation_url='/'+(directory/'count-verification/index.html').relative_to(root).as_posix(),
            source_flight_url='/'+(original/'report.html').relative_to(root).as_posix(),
            analysis_source=str(directory.relative_to(root)),measurement_scope='Model-derived projected surface extent/area, not validated structural measurements')
        visit=store.ingest('ROOM-001-WALL-'+wall_id,'airsim-'+room['scene_profile'],order,metadata,
            [o for o in observations if o['wall_id']==wall_id])
        registered.append(dict(wall_id=wall_id,inspection_id=visit))
    return dict(status='saved',inspections=registered)
