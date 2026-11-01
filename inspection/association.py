"""Conservative multi-view association on the known facade plane.

Uses image-derived component pixels and simulator calibration, never defect truth.
Unobserved gaps and ambiguous identities are deliberately not guessed across.
"""
import hashlib
import json
import math
import numpy as np

VERSION='wall-support-v1'
CELL_M=.005
TOLERANCE_CELLS=2  # 10 mm plus grid quantization; fixed, not enlarged to force matches.
MIN_COMMON_CELLS=12
MIN_COMMON_SPAN_M=.05
MIN_AGREEMENT=.65

LEGACY_POLICY={'version':VERSION,'grid_m':CELL_M,'tolerance_cells':TOLERANCE_CELLS,
               'min_common_cells':MIN_COMMON_CELLS,'min_common_span_m':MIN_COMMON_SPAN_M,'min_agreement':MIN_AGREEMENT}
SMALL_CRACK_POLICY={'version':'wall-support-fine-v2','grid_m':.002,'tolerance_cells':2,
                    'min_common_cells':12,'min_common_span_m':.02,'min_agreement':.65}


def small_crack_policy(config,records,model_input_scale=1.0):
    """Use a 2 mm support grid and account for native camera pixel sampling."""
    if not 0<model_input_scale<=1:raise ValueError('Invalid model input scale')
    profile=dict(SMALL_CRACK_POLICY)
    if model_input_scale!=1:
        profile.update(version='wall-support-model-resolution-v3',model_input_scale=model_input_scale)
    worst=0.
    for record in records:
        focal=record['width']*model_input_scale/(2*math.tan(math.radians(record['fov_deg'])/2))
        worst=max(worst,abs(config['wall_x_m']-record['camera_position']['x_val'])/focal)
    profile['tolerance_cells']=max(2,math.ceil(1.5*worst/profile['grid_m']))
    return profile


def mask_runs(component,x,y):
    """Lossless foreground runs in absolute image coordinates: row, start, end."""
    runs=[]
    for row,line in enumerate(component>0):
        changes=np.diff(np.r_[False,line,False].astype(np.int8))
        runs.extend([row+y,int(a)+x,int(b)+x] for a,b in
                    zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
    return runs


def project(config,record,pixels,rotation):
    pixels=np.asarray(pixels,float).reshape(-1,2)
    focal=record['width']/(2*math.tan(math.radians(record['fov_deg'])/2))
    rays=np.column_stack([np.ones(len(pixels)),(pixels[:,0]-(record['width']-1)/2)/focal,
                          (pixels[:,1]-(record['height']-1)/2)/focal])@rotation.T
    origin=np.array([record['camera_position'][k] for k in ['x_val','y_val','z_val']])
    distance=(config['wall_x_m']-origin[0])/np.maximum(rays[:,0],1e-12)
    world=origin+rays*distance[:,None]
    valid=(rays[:,0]>.01)&(distance>0)&np.isfinite(world).all(axis=1)
    return world[valid][:,[1,2]]*np.array([1,-1])


def observation(config,record,feature,roi,rotation,policy=None):
    profile=dict(policy or LEGACY_POLICY)
    runs=feature.get('mask_runs',[])
    pixels=np.array([(u,v) for v,a,b in runs for u in range(a,b)],float).reshape(-1,2)
    mapped=project(config,record,pixels,rotation)
    support=set(map(tuple,np.rint(mapped/profile['grid_m']).astype(int)))
    cells=[[int(x),int(y)] for x,y in sorted(support)]
    payload=[record['file'],feature['kind'],runs,cells]
    if profile['version']!=VERSION:payload.insert(0,profile['version'])
    key=hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
    return {'file':record['file'],'kind':feature['kind'],'feature':feature,'record':record,
            'roi':roi,'rotation':rotation,'support':support,'key':key,'wall_x_m':config['wall_x_m'],
            'policy':profile,'wall_id':record.get('wall_id')}


def visible(support,other):
    """Restrict support to the other view's depth-validated, eroded ROI."""
    if not support:
        return set()
    points=np.array(sorted(support))
    cell=other.get('policy',LEGACY_POLICY)['grid_m']
    world=np.column_stack([np.full(len(points),other['wall_x_m']),points[:,0]*cell,-points[:,1]*cell])
    rec=other['record']; origin=np.array([rec['camera_position'][k] for k in ['x_val','y_val','z_val']])
    camera=(world-origin)@other['rotation']
    focal=rec['width']/(2*math.tan(math.radians(rec['fov_deg'])/2))
    uv=np.rint(camera[:,1:]/np.maximum(camera[:,:1],1e-12)*focal+
                np.array([(rec['width']-1)/2,(rec['height']-1)/2])).astype(int)
    ok=(camera[:,0]>.01)&(uv[:,0]>=0)&(uv[:,0]<rec['width'])&(uv[:,1]>=0)&(uv[:,1]<rec['height'])
    indices=np.flatnonzero(ok)
    indices=indices[other['roi'][uv[indices,1],uv[indices,0]]>0]
    return set(map(tuple,points[indices]))


def expanded(points,tolerance=TOLERANCE_CELLS):
    return {(x+dx,y+dy) for x,y in points
            for dx in range(-tolerance,tolerance+1)
            for dy in range(-tolerance,tolerance+1)
            if dx*dx+dy*dy<=tolerance**2}


def span(points,cell_m=CELL_M):
    if not points:
        return 0
    a=np.array(list(points))
    return float(np.linalg.norm(np.ptp(a,axis=0))*cell_m)


def compare(a,b):
    if a.get('wall_id')!=b.get('wall_id'):return 'different_wall',0
    profile=a.get('policy',LEGACY_POLICY)
    if profile!=b.get('policy',LEGACY_POLICY):raise ValueError('Mixed association profiles are not comparable.')
    if a['kind']!=b['kind']:
        return 'different_type',0
    if a['file']==b['file']:
        return 'same_frame',0
    aa=visible(a['support'],b); bb=visible(b['support'],a)
    if min(len(aa),len(bb))<profile['min_common_cells'] or min(span(aa,profile['grid_m']),span(bb,profile['grid_m']))<profile['min_common_span_m']:
        return 'insufficient_overlap',0
    score=min(len(aa&expanded(bb,profile['tolerance_cells']))/len(aa),len(bb&expanded(aa,profile['tolerance_cells']))/len(bb))
    return ('match' if score>=profile['min_agreement'] else 'shape_conflict'),score


def associate(observations):
    """Batch graph association; same-frame and shape conflicts prohibit a union."""
    profile=observations[0].get('policy',LEGACY_POLICY) if observations else LEGACY_POLICY
    if any(o.get('policy',LEGACY_POLICY)!=profile for o in observations):raise ValueError('Mixed association profiles.')
    count=len(observations); groups=[{i} for i in range(count)]
    pairs={}; edges=[]; ambiguous=set(); conflicts=[]
    for i in range(count):
        for j in range(i+1,count):
            status,score=compare(observations[i],observations[j])
            pairs[i,j]=(status,score)
            if status=='match':
                edges.append((score,i,j))
    # A component matching multiple detections from one other image is ambiguous.
    for i in range(count):
        by_frame={}
        for _,a,b in edges:
            j=b if a==i else a if b==i else None
            if j is not None:
                by_frame.setdefault(observations[j]['file'],[]).append(j)
        for js in by_frame.values():
            if len(js)>1:
                ambiguous.update([i,*js])
                conflicts.append({'reason':'competing_matches','observations':[i,*js]})
    accepted=[]
    for score,i,j in sorted(edges,key=lambda e:(-e[0],observations[e[1]]['key'],observations[e[2]]['key'])):
        if i in ambiguous or j in ambiguous:
            continue
        ga=next(g for g in groups if i in g); gb=next(g for g in groups if j in g)
        if ga is gb:
            continue
        blocked=any(pairs[min(a,b),max(a,b)][0] in ['same_frame','shape_conflict','different_wall'] for a in ga for b in gb)
        if blocked:
            ambiguous.update(ga|gb)
            conflicts.append({'reason':'inconsistent_track','observations':sorted(ga|gb)})
            continue
        ga.update(gb); groups.remove(gb)
        accepted.append({'observations':[i,j],'agreement':round(score,3)})
    # Split any track subsequently found to be ambiguous; don't retain an early guess.
    final=[]
    for group in groups:
        if group&ambiguous:
            final.extend([{i} for i in sorted(group)])
        else:
            final.append(group)
    return sorted(final,key=lambda g:min(observations[i]['key'] for i in g)),ambiguous,{
        'version':profile['version'],'grid_m':profile['grid_m'],'match_tolerance_m':profile['grid_m']*profile['tolerance_cells'],
        'minimum_mutual_agreement':profile['min_agreement'],'minimum_common_span_m':profile['min_common_span_m'],
        'accepted_links':[e for e in accepted if not set(e['observations'])&ambiguous],
        'conflicts':conflicts,'observation_count':count,
        'unresolved_policy':'Separate candidates when overlap is absent or identity is ambiguous; totals are not verified physical defect counts.'}
