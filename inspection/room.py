"""Deterministic room fixture and multi-wall coverage, in local x/y/up metres.

Static known-map planning only. No live flight control or depth replanning.
"""
import math
import random
from .scene import segment_hits_box
from .navigation import plan_transit


def make_room(seed=42):
    rng=random.Random(seed)
    room={'version':'construction-room-v1','seed':seed,'size_m':[24,18,6],
          'coordinates':'local x/y/positive height; AirSim mapping [100+x,100+y,-height]',
          'airsim_origin_ned_m':[100,100,0],'wall_thickness_m':.25,
          'launch_m':[4,4,1.2],'clearance_m':.7,'standoff_m':2,'camera_forward_offset_m':.46,
          'camera':{'horizontal_fov_deg':89.90362548828125,'width':3840,'height':2160,
                    'scope':'Candidate camera for planning; live simulator camera unchanged.'},
          'walls':[
              {'id':'S','name':'South','origin':[0,0,0],'tangent':[1,0,0],'inward':[0,1,0],'length':24,'height':6,'yaw_deg':-90},
              {'id':'E','name':'East','origin':[24,0,0],'tangent':[0,1,0],'inward':[-1,0,0],'length':18,'height':6,'yaw_deg':0},
              {'id':'N','name':'North','origin':[24,18,0],'tangent':[-1,0,0],'inward':[0,-1,0],'length':24,'height':6,'yaw_deg':90},
              {'id':'W','name':'West','origin':[0,18,0],'tangent':[0,-1,0],'inward':[1,0,0],'length':18,'height':6,'yaw_deg':180}],
          'obstacles':[]}
    # This wall-adjacent stack intentionally tests blocked viewpoints and occlusion.
    room['obstacles'].append({'id':'block-stack-01','kind':'blocks','lower':[9,.25,0],'upper':[12,2.8,2.6]})
    kinds=['machine','pallets','scaffold','blocks','machine','pallets','scaffold']
    for index,kind in enumerate(kinds):
        for _ in range(500):
            dx,dy=rng.uniform(1.4,3),rng.uniform(1.2,2.4)
            x,y=rng.uniform(4,20-dx),rng.uniform(4,14-dy)
            height={'machine':2.4,'pallets':1.2,'scaffold':4.5,'blocks':1.7}[kind]
            item={'id':f'{kind}-{index+2:02d}','kind':kind,'lower':[x,y,0],'upper':[x+dx,y+dy,height]}
            # Deterministic random placement with non-overlap and launch exclusion.
            a,b=item['lower'],item['upper']
            if a[0]-1<4<b[0]+1 and a[1]-1<4<b[1]+1:continue
            if any(all(a[k]-.8<o['upper'][k] and b[k]+.8>o['lower'][k] for k in [0,1]) for o in room['obstacles']):continue
            room['obstacles'].append(item);break
        else:raise ValueError('Could not place construction objects with separation.')
    # Preserve IDs and seeded positions of other props when applying layout edits.
    room['obstacles']=[o for o in room['obstacles'] if o['id']!='blocks-05']
    if seed==42:
        for item in room['obstacles']:
            anchor={'machine-02':[4,5,0],'machine-06':[17,13.5,0]}.get(item['id'])
            if anchor is not None:
                size=[b-a for a,b in zip(item['lower'],item['upper'])]
                item['lower']=anchor
                item['upper']=[a+d for a,d in zip(anchor,size)]
        room['layout_revision']='spread-02: remove blocks-05; separate machines from scaffold-04'
    return room


def wall_point(wall,u,height,standoff=0):
    return [wall['origin'][0]+u*wall['tangent'][0]+standoff*wall['inward'][0],
            wall['origin'][1]+u*wall['tangent'][1]+standoff*wall['inward'][1],height]


def to_ned(room,point):
    origin=room['airsim_origin_ned_m']
    return [origin[0]+point[0],origin[1]+point[1],origin[2]-point[2]]


def in_free_space(room,point):
    c=room['clearance_m'];size=room['size_m']
    return (all(c<=p<=d-c for p,d in zip(point,size)) and
            not any(segment_hits_box(point,point,o['lower'],o['upper'],c) for o in room['obstacles']))


def spaced(lo,hi,max_step):
    count=max(1,math.ceil((hi-lo)/max_step))
    return [lo+(hi-lo)*i/count for i in range(count+1)] if hi>lo else [(lo+hi)/2]


def plan_room(room,overlap=.35,sample_step=.3):
    if not 0<=overlap<1 or sample_step<=0:raise ValueError('Invalid overlap or sample spacing.')
    camera=room['camera'];distance=room['standoff_m']
    half_w=distance*math.tan(math.radians(camera['horizontal_fov_deg']/2))
    half_h=half_w*camera['height']/camera['width']
    result={'scope':'Offline ideal frontal static-map plan; not flown or sensor-verified.',
            'overlap':overlap,'footprint_m':[half_w*2,half_h*2],
            'views':[],'skipped_views':[],'transits':[],'coverage':{},'sample_step_m':sample_step}
    previous=room['launch_m'];bounds=[[room['clearance_m']]*3,[d-room['clearance_m'] for d in room['size_m']]]
    for wall in room['walls']:
        us=spaced(half_w,wall['length']-half_w,2*half_w*(1-overlap))
        hs=spaced(half_h,wall['height']-half_h,2*half_h*(1-overlap))
        for row,height in enumerate(hs):
            for u in (us if row%2==0 else list(reversed(us))):
                point=wall_point(wall,u,height,distance+room.get('camera_forward_offset_m',0))
                view={'wall_id':wall['id'],'u_m':u,'height_m':height,'position':point,
                      'position_ned':to_ned(room,point),'yaw_deg':wall['yaw_deg']}
                if not in_free_space(room,point):
                    result['skipped_views'].append(dict(view,reason='Viewpoint violates static clearance'));continue
                try:
                    transit=plan_transit(previous,point,room['obstacles'],bounds,resolution=.5,clearance=room['clearance_m'])
                except (ValueError,RuntimeError) as error:
                    result['skipped_views'].append(dict(view,reason='Unreachable: '+str(error)));continue
                view['id']=f"{wall['id']}-{len(result['views'])+1:03d}"
                result['views'].append(view);result['transits'].append(transit)
                previous=point
        samples=[];views=[v for v in result['views'] if v['wall_id']==wall['id']]
        nu=math.ceil(wall['length']/sample_step);nh=math.ceil(wall['height']/sample_step)
        for j in range(nh):
            for i in range(nu):
                u=(i+.5)*wall['length']/nu;height=(j+.5)*wall['height']/nh
                point=wall_point(wall,u,height);framed=0;visible=False
                for view in views:
                    if abs(u-view['u_m'])>half_w or abs(height-view['height_m'])>half_h:continue
                    framed+=1
                    camera_origin=wall_point(wall,view['u_m'],view['height_m'],distance)
                    if not any(segment_hits_box(camera_origin,point,o['lower'],o['upper'],0) for o in room['obstacles']):
                        visible=True;break
                samples.append({'u':round(u,4),'height':round(height,4),
                                'status':'visible' if visible else 'occluded' if framed else 'unobserved'})
        counts={k:sum(s['status']==k for s in samples) for k in ['visible','occluded','unobserved']}
        result['coverage'][wall['id']]={'samples':samples,'counts':counts,
              'sampled_visible_percent':100*counts['visible']/len(samples),'wall_area_m2':wall['length']*wall['height']}
    result['transits'].append(plan_transit(previous,room['launch_m'],room['obstacles'],bounds,resolution=.5,clearance=room['clearance_m']))
    result['path_length_m']=sum(t['path_length_m'] for t in result['transits'])
    result['coverage_scope']='Known AABB occlusion + ideal frustum sample test; not actual camera/depth visibility or exact area.'
    return result
