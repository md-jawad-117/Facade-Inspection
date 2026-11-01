"""Require majority depth support before calling a mask a wall finding."""
VERSION='majority-depth-wall-v1'


def partition(findings,policy=None):
    if not policy:return findings,[]
    if policy!=VERSION:raise ValueError('Unknown wall support policy')
    accepted=[];excluded=[]
    for original in findings:
        if 'pre_wall_pixel_area' not in original:raise ValueError('Unclipped model-mask provenance is required')
        item=dict(original);ratio=item['wall_support_fraction']
        item['wall_support_policy']=VERSION
        if ratio>=.5:accepted.append(item)
        else:
            item['exclusion_reason']='Less than half of original model mask is supported by the depth-verified wall plane'
            item['disposition']='non_wall_or_occluded_mask_requires_review'
            excluded.append(item)
    return accepted,excluded
