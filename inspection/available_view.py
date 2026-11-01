"""Choose a nearby feasible viewpoint using observed occupancy, never defect truth."""
import math
from .navigation import plan_transit
from .scene import segment_hits_box


def route_to_view(room,current,target,obstacles,wall_id=None,allow_adjust=True,planning_margin=.25):
    if not math.isfinite(planning_margin) or planning_margin<.25:raise ValueError('Planning margin cannot be reduced below 0.25 m')
    c=room['clearance_m']+planning_margin
    if not all(room['clearance_m']<=p<=d-room['clearance_m'] for p,d in zip(current,room['size_m'])):raise RuntimeError('Measured start violates required room clearance.')
    bounds=[[min(c,p) for p in current],[max(d-c,p) for p,d in zip(current,room['size_m'])]]
    blocked=lambda point:any(segment_hits_box(point,point,o['lower'],o['upper'],c) for o in obstacles) or not all(c<=p<=d-c for p,d in zip(point,room['size_m']))
    candidates=[list(target)]
    if blocked(target) and allow_adjust:
        if wall_id:
            wall=next(w for w in room['walls'] if w['id']==wall_id)
            directions=[wall['tangent'],[-v for v in wall['tangent']]]
        else:directions=[[math.cos(a*math.pi/4),math.sin(a*math.pi/4),0] for a in range(8)]
        for radius in [.5,.75,1.,1.25,1.5,2.,2.5]:
            candidates.extend([[target[k]+radius*d[k] for k in range(3)] for d in directions])
        candidates.sort(key=lambda p:math.dist(p,target)+.05*math.dist(p,current))
    for candidate in candidates:
        if blocked(candidate):continue
        try:
            route=plan_transit(current,candidate,obstacles,bounds,resolution=.25,clearance=c)
            return route,candidate
        except ValueError:continue
    raise RuntimeError('No reachable inspection viewpoint after depth update; movement stopped.')
