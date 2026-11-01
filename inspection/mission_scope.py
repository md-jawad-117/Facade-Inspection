"""Worker-selected walls; never remove geometry from the collision map."""
import copy

WALL_IDS=('S','E','N','W')
WALL_NAMES={'S':'South','E':'East','N':'North','W':'West'}

def normalize_wall_ids(value=None):
    if value is None:return list(WALL_IDS)
    if not isinstance(value,list) or not value:raise ValueError('Select at least one wall.')
    if any(not isinstance(v,str) or v not in WALL_IDS for v in value):raise ValueError('Unknown inspection wall.')
    if len(value)!=len(set(value)):raise ValueError('Select each wall only once.')
    return [w for w in WALL_IDS if w in value]

def selection_from_values(values):
    if 'wall_ids' in values:
        if values.get('wall_id') is not None:raise ValueError('Use wall_ids or wall_id, not both.')
        if values['wall_ids'] is None:raise ValueError('Select at least one wall.')
        return normalize_wall_ids(values['wall_ids'])
    if values.get('wall_id') is not None:return normalize_wall_ids([values['wall_id']])
    return normalize_wall_ids()

def inspection_walls(room):
    ids=room.get('inspection_wall_ids')
    if ids is None and room.get('inspection_wall_id'):ids=[room['inspection_wall_id']]
    selected=normalize_wall_ids(ids)
    return [w for w in room['walls'] if w['id'] in selected]

def select_room_walls(room,ids):
    result=copy.deepcopy(room);result['inspection_wall_ids']=normalize_wall_ids(ids)
    result.pop('inspection_wall_id',None)
    if len(result['inspection_wall_ids'])==1:result['inspection_wall_id']=result['inspection_wall_ids'][0]
    return result

def plan_selected_walls(room,ids):
    from .room import plan_room
    selected=select_room_walls(room,ids)
    planning=copy.deepcopy(selected);planning['clearance_m']+=.25
    planning['walls']=inspection_walls(selected)
    return plan_room(planning)

def preview_selection(ids):
    from .room import make_room
    room=make_room();ids=normalize_wall_ids(ids);plan=plan_selected_walls(room,ids)
    return dict(wall_ids=ids,wall_names=[WALL_NAMES[w] for w in ids],planned_captures=len(plan['views']),
                wall_area_m2=sum(w['length']*w['height'] for w in room['walls'] if w['id'] in ids),
                path_length_m=round(plan['path_length_m'],1),coverage={k:v['sampled_visible_percent'] for k,v in plan['coverage'].items()},
                views=plan['views'],paths=[t['path'] for t in plan['transits']],
                size_m=room['size_m'],obstacles=room['obstacles'],launch_m=room['launch_m'],
                scope='Offline planned route. Optional rechecks can add captures; actual visibility is measured after flight.')
