"""Conservative same-frame duplicate suppression with retained evidence.

Does not join disconnected fragments. Two masks must coincide strongly before
the lower-confidence duplicate is suppressed. Nearby separate cracks remain.
"""
def intersection(a,b):
    rows={}
    for y,x,z in a['mask_runs']:rows.setdefault(y,[]).append((x,z))
    total=0
    for y,x,z in b['mask_runs']:
        total+=sum(max(0,min(z,zz)-max(x,xx)) for xx,zz in rows.get(y,[]))
    return total


def suppress(findings,iou_threshold=.65,containment_threshold=.90):
    result=[]
    for source in sorted(findings,key=lambda f:-f['confidence']):
        item=dict(source);duplicate=None
        for kept in result:
            if kept['kind']!=item['kind']:continue
            ax,ay,aw,ah=kept['bbox'];bx,by,bw,bh=item['bbox']
            if min(ax+aw,bx+bw)<=max(ax,bx) or min(ay+ah,by+bh)<=max(ay,by):continue
            common=intersection(kept,item);union=kept['pixel_area']+item['pixel_area']-common
            overlap=common/max(1,union);containment=common/max(1,min(kept['pixel_area'],item['pixel_area']))
            if overlap>=iou_threshold and containment>=containment_threshold:
                duplicate=kept
                duplicate.setdefault('suppressed_overlapping_predictions',[]).append({'bbox':item['bbox'],
                    'confidence':item['confidence'],'mask_iou':round(overlap,5),'containment':round(containment,5),
                    'mask_runs':item['mask_runs'],'reason':'Strong same-frame mask coincidence; retained for review.'})
                break
        if duplicate is None:result.append(item)
    return result
