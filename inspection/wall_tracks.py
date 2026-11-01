"""Fuse overlapping defect footprints on each wall, with co-visibility guards.

Camera pose projects pixels into wall coordinates before this stage. No texture
labels or reference defect locations are used here.
"""
from . import association as base

VERSION='wall-footprint-map-v4.3'


def agreement(a,b):
    if a.get('wall_id')!=b.get('wall_id') or a['kind']!=b['kind']:return None
    policy=a.get('policy',base.LEGACY_POLICY)
    if policy!=b.get('policy',base.LEGACY_POLICY):raise ValueError('Mixed wall-map resolution policies')
    aa=base.visible(a['support'],b);bb=base.visible(b['support'],a)
    if min(len(aa),len(bb))<policy['min_common_cells']:return None
    if min(base.span(aa,policy['grid_m']),base.span(bb,policy['grid_m']))<policy['min_common_span_m']:return None
    ca=len(aa&base.expanded(bb,policy['tolerance_cells']))/len(aa)
    cb=len(bb&base.expanded(aa,policy['tolerance_cells']))/len(bb)
    smaller=ca if len(aa)<=len(bb) else cb
    return smaller,min(ca,cb)


def associate(observations):
    if not observations:return [],set(),{'version':VERSION,'observation_count':0,'conflicts':[],'accepted_links':[]}
    policy=observations[0].get('policy',base.LEGACY_POLICY);count=len(observations)
    if any(o.get('policy',base.LEGACY_POLICY)!=policy for o in observations):raise ValueError('Mixed wall-map policies')
    groups=[{i} for i in range(count)];scores={};redundant=set();edges=[]
    nested={};children={}
    for i in range(count):
        for j in range(i+1,count):
            score=agreement(observations[i],observations[j]);scores[i,j]=score
            if score is None:continue
            if observations[i]['file']==observations[j]['file']:
                # Two predictions from one image can share an identity only
                # with strong mutual wall-footprint agreement. Crossing or
                # merely neighbouring masks are separate co-visible instances.
                if score[1]>=policy['min_agreement']:
                    redundant.add((i,j));edges.append((score[1],i,j))
                else:
                    aa=base.visible(observations[i]['support'],observations[j])
                    bb=base.visible(observations[j]['support'],observations[i])
                    # Exact shared cells, with no dilation: a higher-confidence
                    # fragment may have survived image NMS ahead of its full mask.
                    fraction=len(aa&bb)/min(len(aa),len(bb))
                    if fraction>=.90:
                        nested[i,j]=fraction
                        parent,child=(j,i) if len(aa)<=len(bb) else (i,j)
                        children.setdefault(parent,[]).append(child)
            elif score[0]>=policy['min_agreement']:edges.append((score[0],i,j))
    ambiguous=set();conflicts=[]
    # A same-frame broad mask can contain two genuinely distinct instances.
    # Quarantine it rather than using nested pairs to join its children.
    potential=redundant|set(nested)
    for parent,js in children.items():
        if any((min(a,b),max(a,b)) not in potential for k,a in enumerate(js) for b in js[k+1:]):
            ambiguous.add(parent)
            conflicts.append({'reason':'same_frame_broad_mask_contains_distinct_instances','observations':[parent,*js]})
    accepted_nested=[]
    for (i,j),fraction in nested.items():
        if i in ambiguous or j in ambiguous:continue
        redundant.add((i,j));edges.append((fraction,i,j))
        accepted_nested.append({'observations':[i,j],'exact_containment_fraction':round(fraction,6)})
    # Quarantine a broad prediction that matches two distinct instances in one
    # other frame. Keep those reliable instances intact; do not let the broad
    # mask steal either identity or invalidate all their other observations.
    same_frame_quarantined=set(ambiguous)
    for i in range(count):
        if i in same_frame_quarantined:continue
        by_frame={}
        for _,a,b in edges:
            j=b if a==i else a if b==i else None
            if j is not None and j not in same_frame_quarantined and observations[j]['file']!=observations[i]['file']:
                by_frame.setdefault(observations[j]['file'],[]).append(j)
        for js in by_frame.values():
            if any((min(a,b),max(a,b)) not in redundant for k,a in enumerate(js) for b in js[k+1:]):
                ambiguous.add(i);conflicts.append({'reason':'broad_mask_spans_distinct_instances','observations':[i,*js]})
    accepted=[]
    for score,i,j in sorted(edges,key=lambda e:(-e[0],observations[e[1]]['key'],observations[e[2]]['key'])):
        if i in ambiguous or j in ambiguous:continue
        ga=next(g for g in groups if i in g);gb=next(g for g in groups if j in g)
        if ga is gb:continue
        blocked=any(observations[a]['file']==observations[b]['file'] and (min(a,b),max(a,b)) not in redundant for a in ga for b in gb)
        if blocked:
            conflicts.append({'reason':'distinct_same_frame_instances','observations':[i,j]});continue
        ga.update(gb);groups.remove(gb);accepted.append({'observations':[i,j],'agreement':round(score,4)})
    from .class_consensus import resolve
    groups,class_resolutions=resolve(observations,groups,ambiguous)
    # Disagreement between partial masks is not proof of a separate physical
    # defect. Keep positive-overlap components, without guessing across gaps.
    return sorted(groups,key=lambda g:min(observations[i]['key'] for i in g)),ambiguous,{
        'version':VERSION,'grid_m':policy['grid_m'],'match_tolerance_m':policy['grid_m']*policy['tolerance_cells'],
        'minimum_common_span_m':policy['min_common_span_m'],'minimum_covered_footprint_fraction':policy['min_agreement'],
        'observation_count':count,'accepted_links':accepted,'conflicts':conflicts,
        'same_frame_redundant_pairs':[list(pair) for pair in sorted(redundant)],
        'same_frame_nested_pairs':accepted_nested,'class_resolutions':class_resolutions,
        'unresolved_policy':'Disconnected footprints remain separate; co-visible distinct instances cannot share a track. Raw masks remain available for review.'}
