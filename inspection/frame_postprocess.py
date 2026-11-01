"""Versioned image-only suppression of repeated instance-mask predictions."""
import copy
from .duplicate_masks import intersection

VERSION='instance-mask-nms-v2'


def apply(findings,version='recorded-v1'):
    if version=='recorded-v1':return findings
    if version!=VERSION:raise ValueError('Unknown frame postprocess version')
    result=[]
    for source in sorted(findings,key=lambda f:-f['confidence']):
        item=copy.deepcopy(source);duplicate=None
        for kept in result:
            if item['kind']!=kept['kind']:continue
            ax,ay,aw,ah=kept['bbox'];bx,by,bw,bh=item['bbox']
            if min(ax+aw,bx+bw)<=max(ax,bx) or min(ay+ah,by+bh)<=max(ay,by):continue
            common=intersection(kept,item)
            iou=common/max(1,kept['pixel_area']+item['pixel_area']-common)
            contained=common/max(1,item['pixel_area'])
            # Independent NMS criteria: a near-identical full mask OR a fragment
            # contained in the stronger mask. The old AND condition left both.
            # A larger mask extending well beyond a kept fragment is not removed
            # merely because the fragment lies inside it.
            if iou>=.65 or contained>=.90:
                duplicate=kept
                kept.setdefault('suppressed_instance_predictions',[]).append(dict(
                    bbox=item['bbox'],confidence=item['confidence'],mask_runs=item['mask_runs'],
                    mask_iou=round(iou,6),fraction_inside_kept=round(contained,6),
                    reason='Same-class full-mask overlap or contained lower-confidence fragment',
                    version=VERSION))
                break
        if duplicate is None:result.append(item)
    return result
