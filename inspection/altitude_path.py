"""Convert collision-free legs into explicit altitude-hold motion stages."""
import math
from .scene import segment_hits_box


def stage_path(path,obstacles,clearance):
    output=[list(path[0])]
    for source,target in zip(path,path[1:]):
        top=max(source[2],target[2])
        stages=[[source[0],source[1],top],[target[0],target[1],top],list(target)]
        for point in stages:
            if math.dist(output[-1],point)<.001:continue
            if any(segment_hits_box(output[-1],point,o['lower'],o['upper'],clearance) for o in obstacles):
                raise RuntimeError('Altitude-hold staging is obstructed; movement stopped.')
            output.append(point)
    return output
