"""Worker-facing evidence dispositions. Never changes masks, IDs or raw scoring."""
import html
import json
import math
from collections import Counter
from pathlib import Path

VERSION='worker-evidence-review-v1'


def interior(feature,record,scale):
    x,y,w,h=feature['bbox'];margin=16/scale
    return x>margin and y>margin and x+w<record['width']-margin and y+h<record['height']-margin


def independent(items):
    positions=[tuple(record['camera_position'][k] for k in ('x_val','y_val','z_val')) for _,record in items]
    return any(math.dist(a,b)>=.1 for i,a in enumerate(positions) for b in positions[i+1:])


def support(feature):
    return {(x,y) for y,start,end in feature['mask_runs'] for x in range(start,end)}


def compile_review(analysis,flight):
    records={v['file']:v for v in flight['images']};tracks={v['id']:v for v in analysis['findings']}
    scale=analysis.get('model_input_scale',1.);by_track={k:[] for k in tracks}
    by_file={}
    for frame in analysis['frames']:
        record=records[frame['file']];by_file[frame['file']]=frame['findings']
        for feature in frame['findings']:
            if feature['id'] in tracks:by_track[feature['id']].append((feature,record))
    stable={};entries={}
    for key,track in tracks.items():
        matching=[v for v in by_track[key] if v[0]['kind']==track['kind']]
        full=[v for v in matching if interior(*v,scale)]
        qualifies=track.get('identity_status')!='ambiguous' and len({r['file'] for _,r in full})>=2 and independent(full)
        if qualifies:stable[key]=full
        best=max(matching or by_track[key],key=lambda v:v[0]['confidence'],default=None)
        entries[key]=dict(id=key,wall_id=track['wall_id'],kind=track['kind'],location=track['location'],
            status='image_supported' if qualifies else 'needs_review',
            score=track['confidence'],observation_count=len(matching),interior_view_count=len({r['file'] for _,r in full}),
            evidence_files=sorted({r['file'] for _,r in full}),
            crop_file=best[0].get('crop_file') if best else None,capture_file=best[1]['file'] if best else None,
            class_consensus=bool(track.get('class_resolution')),human_confirmation='not_recorded_in_this_report',
            reason='Repeated class observations in interior views from separated camera positions.' if qualifies else 'Independent interior observations do not establish this class, or identity is ambiguous.')
    for key,track in tracks.items():
        observations=by_track[key]
        # A repeated or strong co-visible class stays independent. This rule
        # only labels a weak, one-view minority mask as an alternate hypothesis.
        if key in stable or track.get('identity_status')=='ambiguous' or len(observations)!=1:continue
        source,record=observations[0]
        if source['confidence']>=.5:continue
        pixels=support(source)
        if not pixels:continue
        choices=[]
        for parent in by_file[record['file']]:
            target_id=parent['id']
            if target_id not in stable or parent['kind']==source['kind'] or tracks[target_id]['wall_id']!=track['wall_id']:continue
            if parent['confidence']<.5 or parent['confidence']<=source['confidence']:continue
            parent_pixels=support(parent)
            if len(parent_pixels)<4*len(pixels):continue
            fraction=len(pixels&parent_pixels)/len(pixels)
            if fraction<.9:continue
            other_views=[v for v in stable[target_id] if v[1]['file']!=record['file'] and v[0]['confidence']>source['confidence']]
            if len({r['file'] for _,r in other_views})<2 or not independent(other_views):continue
            linked=[(f,r) for f,r in other_views if r.get('verification_parent_file')==record['file'] and r.get('verification_parent_bbox')==source['bbox']]
            if not linked:continue
            choices.append(dict(related_finding_id=target_id,related_kind=parent['kind'],
                source_mask_containment_fraction=round(fraction,6),
                linked_recheck_files=sorted({r['file'] for _,r in linked}),
                independent_support_files=sorted({r['file'] for _,r in other_views})))
        if len(choices)==1:
            entries[key].update(status='alternative_class_hypothesis',**choices[0],
                reason='One weak, nested class prediction; a stronger same-image mask and independent linked recheck support the related finding. Retained for review, not added to the supported-defect total.')
    supported=[v for v in entries.values() if v['status']=='image_supported']
    uncertain=[v for v in entries.values() if v['status']!='image_supported']
    return dict(version=VERSION,raw_candidate_count=len(tracks),supported_count=len(supported),needs_review_count=len(uncertain),
        linked_alternative_count=sum(v['status']=='alternative_class_hypothesis' for v in uncertain),
        supported_by_class=dict(Counter(v['kind'] for v in supported)),supported_findings=supported,uncertain_candidates=uncertain,
        human_confirmation_scope='Image-supported is an automatic evidence status, not an engineer or human confirmation. See the building twin for recorded human reviews.',
        policy=dict(minimum_independent_views=2,minimum_camera_separation_m=.1,interior_margin_model_pixels=16,
            weak_class_score_below=.5,minimum_native_mask_containment=.9,minimum_parent_area_ratio=4,
            explicit_parent_file_and_bbox_recheck_required=True,
            note='Scores and these evidence rules are heuristic, not calibrated probabilities. No raw class, mask, ID or independent acceptance metric is changed.'),
        scope='Disposition of saved model hypotheses for worker review; not new inference, a new flight, structural safety certification or a passed raw-model acceptance gate.')


def write_report(directory):
    directory=Path(directory);analysis=json.loads((directory/'model-analysis.json').read_text());flight=json.loads((directory/'flight.json').read_text())
    review=compile_review(analysis,flight);review['source_run']=directory.name
    evaluation_path=directory/'count-verification/results.json'
    if evaluation_path.exists():
        evaluation=json.loads(evaluation_path.read_text());review['independent_validation']=dict(identity_metrics=evaluation['identity_metrics'],
            raw_acceptance_passed=all(not m[k] for m in evaluation['identity_metrics'].values() for k in ('missed_visible_physical_ids','not_observed_by_route','extra_split_tracks','incorrect_merged_tracks','unmatched_candidate_tracks')))
    (directory/'worker-review.json').write_text(json.dumps(review,indent=2))
    esc=lambda v:html.escape(str(v),quote=True)
    def evidence(entry):
        content=f'<a href="{esc(entry["capture_file"])}">Original capture</a>' if entry.get('capture_file') else ''
        if entry.get('crop_file'):content+=f'<a href="{esc(entry["crop_file"])}"><img loading="lazy" src="{esc(entry["crop_file"])}" alt="{esc(entry["kind"])} observation crop"></a>'
        return content
    rows=''.join(f'<tr><td>{esc(v["wall_id"])}</td><td>{esc(v["kind"])}</td><td><a href="#finding-{esc(v["id"])}">{esc(v["id"])}</a></td><td>{v["interior_view_count"]}</td></tr>' for v in review['supported_findings'])
    supported=''
    for entry in review['supported_findings']:
        linked=[u for u in review['uncertain_candidates'] if u.get('related_finding_id')==entry['id']]
        note=' · provisional class consensus' if entry['class_consensus'] else ''
        links=''.join(f'<p class="notice">An uncertain {esc(u["kind"])} label overlaps this finding. <a href="#candidate-{esc(u["id"])}">Review the alternative label</a>.</p>' for u in linked)
        supported+=f'<article id="finding-{esc(entry["id"])}"><h3>{esc(entry["wall_id"])} · {esc(entry["kind"])} · {esc(entry["id"])}</h3><p>{entry["interior_view_count"]} interior views{note}. No human confirmation is recorded in this capture report.</p>{links}{evidence(entry)}</article>'
    uncertain=''
    for entry in review['uncertain_candidates']:
        relation=''
        if entry.get('related_finding_id'):
            relation=f'<p><strong>Alternative label linked to <a href="#finding-{esc(entry["related_finding_id"])}">{esc(entry["related_kind"])} {esc(entry["related_finding_id"])}</a>.</strong> {100*entry["source_mask_containment_fraction"]:.2f}% native-mask containment. This label adds no second image-supported defect.</p>'
            relation+='<p>Linked verification: '+', '.join(f'<a href="{esc(name)}">{esc(name)}</a>' for name in entry['linked_recheck_files'])+'.</p>'
        uncertain+=f'<article class="uncertain" id="candidate-{esc(entry["id"])}"><h3>{esc(entry["wall_id"])} · {esc(entry["kind"])} hypothesis · {esc(entry["id"])}</h3><p>{esc(entry["reason"])}</p>{relation}<p>Original model score: {entry["score"]:.3f}. Original label, ID and mask are retained.</p>{evidence(entry)}</article>'
    validity=review.get('independent_validation',{}).get('raw_acceptance_passed')
    raw_note='The independent raw-model acceptance test still has a residual candidate and has not passed.' if validity is False else 'The independent count results remain separate from these worker evidence statuses.'
    coverage='<p>Visibility and image quality are separate from defect counts. A plain wall can be visible while its focus quality is uncertain. These statuses do not declare any wall defect-free.</p>'
    if (directory/'inspection-completeness.html').exists():coverage+='<p><a href="inspection-completeness.html">Wall visibility and quality map</a></p>'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Worker inspection report</title><style>body{{font:16px system-ui;background:#edf1eb;color:#20372d;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}a{{color:#125c45}}p{{line-height:1.65}}nav{{display:flex;gap:22px;flex-wrap:wrap}}section,article{{background:white;border-radius:12px;padding:24px;margin:22px 0}}article{{border:1px solid #d4dfd3;scroll-margin-top:20px}}.stats{{display:flex;gap:20px;flex-wrap:wrap}}.stat{{flex:1;min-width:180px;padding:20px;border-radius:12px;background:white;border-top:5px solid #277854}}.stat.review{{border-color:#ac6c16}}.number{{font-size:38px;font-weight:700}}.notice,.uncertain{{background:#fff3dd;border-left:5px solid #ad6e17;padding:18px}}img{{display:block;max-width:100%;max-height:330px;object-fit:contain;object-position:left;margin-top:16px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #d6ded4}}.scroll{{overflow:auto}}small{{display:block;line-height:1.55}}summary{{cursor:pointer;font-weight:600}}@media(max-width:650px){{main{{padding:16px}}section,article{{padding:18px}}}}</style><main><nav><a href="/assets/inspection-system/index.html">Inspection controls</a><a href="/assets/digital-twin/index.html">Building twin / human reviews</a><a href="count-verification/index.html">Independent validation</a></nav><p>SAVED INSPECTION · {esc(directory.name)}</p><h1>Findings and review queue</h1><div class="stats"><div class="stat"><div class="number">{review['supported_count']}</div><strong>Image-supported findings</strong><small>Repeated interior observations from separated camera positions.</small></div><div class="stat review"><div class="number">{review['needs_review_count']}</div><strong>Uncertain candidates</strong><small>{review['linked_alternative_count']} linked alternative class hypotheses; kept out of the supported-defect total.</small></div></div><p><strong>Image-supported does not mean human-confirmed.</strong> This report does not invent an engineer's approval.</p><section><h2>Review these first</h2>{uncertain or '<p>No uncertain candidates under the declared evidence rules.</p>'}</section><section><h2>Image-supported findings by wall</h2><div class="scroll"><table><tr><th>Wall</th><th>Class</th><th>Finding ID</th><th>Interior views</th></tr>{rows}</table></div><details><summary>View supporting crops and linked labels</summary>{supported}</details></section><section><h2>Wall coverage is a separate result</h2>{coverage}</section><section><h2>Raw model and validation history</h2><p>{review['raw_candidate_count']} raw candidate tracks remain in the original analysis. {raw_note} The worker summary does not change classes, masks, track IDs, reference comparisons or acceptance gates.</p><p>For this report, weak means below the existing 0.5 recheck score. A linked alternative additionally requires a much larger, stronger containing mask, a direct parent-file/bbox recheck and two independent supporting views. Scores are not calibrated probabilities.</p><p><a href="model-report.html">All raw annotations</a> · <a href="worker-review.json">Evidence rules and dispositions</a> · <a href="report.html">Flight and landing evidence</a></p></section></main></html>'''
    (directory/'worker-report.html').write_text(page,encoding='utf-8')
    return review
