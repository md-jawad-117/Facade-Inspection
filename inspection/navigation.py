"""Offline 3D transit planning with known static AABBs and swept clearance.

This planner is not wired into flight control. No online sensing/replanning yet.
Bounds constrain the vehicle center; clearance inflates every obstacle.
"""
import heapq
import itertools
import math
from .scene import segment_hits_box


def plan_transit(start,goal,obstacles,bounds,resolution=.5,clearance=.8,max_expansions=200000):
    start=tuple(map(float,start));goal=tuple(map(float,goal))
    lower,upper=[tuple(map(float,p)) for p in bounds]
    if len(start)!=3 or len(goal)!=3 or len(lower)!=3 or len(upper)!=3:
        raise ValueError('Three coordinates are required.')
    if not math.isfinite(resolution) or resolution<=0 or not math.isfinite(clearance) or clearance<0:
        raise ValueError('Resolution must be positive and clearance nonnegative.')
    if not all(math.isfinite(v) for p in [start,goal,lower,upper] for v in p) or any(a>=b for a,b in zip(lower,upper)):
        raise ValueError('Invalid map bounds or coordinates.')
    boxes=[]
    for obstacle in obstacles:
        lo,hi=tuple(obstacle['lower']),tuple(obstacle['upper'])
        if len(lo)!=3 or len(hi)!=3 or not all(math.isfinite(v) for v in lo+hi) or any(a>b for a,b in zip(lo,hi)):
            raise ValueError('Invalid obstacle box.')
        boxes.append((lo,hi))
    inside=lambda p:all(a<=v<=b for v,a,b in zip(p,lower,upper))
    def free(a,b):
        return inside(a) and inside(b) and not any(segment_hits_box(a,b,lo,hi,clearance) for lo,hi in boxes)
    if not free(start,start) or not free(goal,goal):raise ValueError('Start or goal is occupied or outside bounds.')
    if free(start,goal):
        return {'path':[list(start),list(goal)],'path_length_m':math.dist(start,goal),'expanded_nodes':0,'clearance_m':clearance}
    counts=tuple(int(math.floor((b-a)/resolution))+1 for a,b in zip(lower,upper))
    if math.prod(counts)>2_000_000:raise ValueError('Grid exceeds two million nodes; use a coarser resolution or smaller bounds.')
    world=lambda n:tuple(a+i*resolution for a,i in zip(lower,n))
    def snap(p):
        node=tuple(int(round((v-a)/resolution)) for v,a in zip(p,lower))
        valid=lambda n:all(0<=i<count for i,count in zip(n,counts))
        if valid(node) and free(p,world(node)):return node
        # A free continuous endpoint can round into an inflated obstacle. Try
        # nearby grid cells, accepting only a fully swept-clear connector.
        nearby=[tuple(i+d for i,d in zip(node,delta)) for delta in itertools.product(range(-2,3),repeat=3)]
        for candidate in sorted((n for n in nearby if valid(n)),key=lambda n:(math.dist(p,world(n)),n)):
            if free(p,world(candidate)):return candidate
        raise ValueError('No clear nearby-grid connector; refine the grid or move the endpoint.')
    source,target=snap(start),snap(goal)
    neighbors=list(itertools.product([-1,0,1],repeat=3));neighbors.remove((0,0,0))
    queue=[];counter=itertools.count();costs={source:0.0};parents={};closed=set()
    heapq.heappush(queue,(math.dist(world(source),world(target)),next(counter),source))
    while queue:
        _,_,node=heapq.heappop(queue)
        if node in closed:continue
        if len(closed)>=max_expansions:raise RuntimeError('Static planner expansion limit reached.')
        closed.add(node)
        if node==target:break
        position=world(node)
        for delta in neighbors:
            neighbor=tuple(i+d for i,d in zip(node,delta))
            if neighbor in closed or any(i<0 or i>=count for i,count in zip(neighbor,counts)):continue
            other=world(neighbor)
            # Swept segment checks also prohibit diagonal corner cutting.
            if not free(position,other):continue
            cost=costs[node]+math.dist(position,other)
            if cost<costs.get(neighbor,float('inf')):
                costs[neighbor]=cost;parents[neighbor]=node
                heapq.heappush(queue,(cost+math.dist(other,world(target)),next(counter),neighbor))
    else:raise ValueError('No reachable route within the configured static map.')
    nodes=[target]
    while nodes[-1]!=source:nodes.append(parents[nodes[-1]])
    raw=[start]+[world(n) for n in reversed(nodes)]+[goal]
    # Farthest-visible shortcutting retains the same continuous clearance checks.
    path=[raw[0]];index=0
    while index<len(raw)-1:
        farthest=next(j for j in range(len(raw)-1,index,-1) if free(raw[index],raw[j]))
        if raw[farthest]!=path[-1]:path.append(raw[farthest])
        index=farthest
    return {'path':[list(p) for p in path],
            'path_length_m':sum(math.dist(a,b) for a,b in zip(path,path[1:])),
            'expanded_nodes':len(closed),'clearance_m':clearance,'resolution_m':resolution,
            'scope':'Known static geometry only; no live sensing or flight-control integration.'}
