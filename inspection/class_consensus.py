"""Resolve an edge-only class conflict using explicit, repeated image evidence.

No reference labels or distance-only joins. Raw feature classes remain unchanged.
"""
from . import association as base

VERSION='linked-recheck-class-consensus-v2'


def edge_only(observation):
    feature=observation.get('feature',{});record=observation['record']
    if 'bbox' not in feature:return False
    x,y,w,h=feature['bbox'];margin=16/observation['policy'].get('model_input_scale',1.)
    return x<=margin or y<=margin or x+w>=record['width']-margin or y+h>=record['height']-margin


def shared_support(a,b):
    if a['wall_id']!=b['wall_id'] or a['policy']!=b['policy']:return None
    aa=base.visible(a['support'],b);bb=base.visible(b['support'],a);p=a['policy']
    if min(len(aa),len(bb))<p['min_common_cells']:return None
    if min(base.span(aa,p['grid_m']),base.span(bb,p['grid_m']))<p['min_common_span_m']:return None
    # Exact grid support, without the association dilation allowance.
    return dict(exact=min(len(aa&bb)/len(aa),len(aa&bb)/len(bb)),
        within_sampling=min(len(aa&base.expanded(bb,p['tolerance_cells']))/len(aa),len(bb&base.expanded(aa,p['tolerance_cells']))/len(bb)))


def resolve(observations,groups,ambiguous):
    original=[set(g) for g in groups];proposals=[]
    for source in original:
        if source&ambiguous or not all(edge_only(observations[i]) for i in source):continue
        kinds={observations[i]['kind'] for i in source}
        if len(kinds)!=1:continue
        eligible=[]
        for target in original:
            if source==target or target&ambiguous:continue
            target_kinds={observations[i]['kind'] for i in target}
            if len(target_kinds)!=1 or target_kinds==kinds:continue
            if len({observations[i]['wall_id'] for i in source|target})!=1:continue
            # Co-visible, separately predicted instances cannot be collapsed.
            if {observations[i]['file'] for i in source}&{observations[i]['file'] for i in target}:continue
            interior=[i for i in target if not edge_only(observations[i])]
            evidence=[]
            for i in source:
                a=observations[i];supporters=[];linked=[]
                for j in interior:
                    b=observations[j];fraction=shared_support(a,b)
                    if fraction is None or fraction['exact']<a['policy']['min_agreement'] or fraction['within_sampling']<.90:continue
                    if b['feature'].get('confidence',0)<=a['feature'].get('confidence',0):continue
                    supporters.append(j)
                    rec=b['record']
                    if rec.get('verification_parent_file')==a['file'] and rec.get('verification_parent_bbox')==a['feature']['bbox']:
                        linked.append(j)
                positions=[tuple(observations[j]['record']['camera_position'][k] for k in ['x_val','y_val','z_val']) for j in supporters]
                separated=any(sum((a-b)**2 for a,b in zip(pa,pb))>=.1**2 for n,pa in enumerate(positions) for pb in positions[n+1:])
                if len({observations[j]['file'] for j in supporters})<2 or not linked or not separated:break
                evidence.append(dict(source_key=a['key'],source_file=a['file'],original_kind=a['kind'],
                    supporting_keys=[observations[j]['key'] for j in supporters],
                    supporting_files=sorted({observations[j]['file'] for j in supporters}),
                    linked_recheck_files=[observations[j]['file'] for j in linked],
                    minimum_exact_common_support_agreement=min(shared_support(a,observations[j])['exact'] for j in supporters),
                    minimum_resolution_aware_common_support_agreement=min(shared_support(a,observations[j])['within_sampling'] for j in supporters),
                    model_sampling_tolerance_m=a['policy']['grid_m']*a['policy']['tolerance_cells']))
            else:
                eligible.append((target,evidence))
        if len(eligible)==1:proposals.append((source,*eligible[0]))
    accepted=[]
    for source,target,evidence in proposals:
        # Reject competing edge groups for one destination rather than joining
        # potentially distinct children through a broad target mask.
        if sum(t==target for _,t,_ in proposals)!=1:continue
        merged=source|target
        if source not in groups or target not in groups:continue
        groups.remove(source);groups.remove(target);groups.append(merged)
        accepted.append(dict(version=VERSION,resolved_kind=observations[next(iter(target))]['kind'],
            source_keys=sorted(observations[i]['key'] for i in source),
            target_keys=sorted(observations[i]['key'] for i in target),evidence=evidence,
            raw_classes_preserved=True,human_confirmed=False,
            scope='Provisional multi-view class consensus; not human review or proof of structural condition.'))
    return sorted(groups,key=lambda g:min(observations[i]['key'] for i in g)),accepted
