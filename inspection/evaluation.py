"""Independent postflight scoring; reference masks never guide flight or detection."""
import collections
import html
import json
from pathlib import Path
import cv2
import numpy as np
from .scene_geometry import source_projection
from .room_vision import canonical_record
from .vision import wall_roi
KINDS=['crack','rust','spall']


def match(predictions,labels,refs):
    pairs=[];overlaps={};multi=[]
    for i,feature in enumerate(predictions):
        counts=collections.Counter()
        for y,a,b in feature['mask_runs']:counts.update(labels[y,a:b].tolist())
        substantial=[]
        for obj in refs:
            intersection=counts[obj['instance_id']]
            overlap=intersection/max(1,feature['pixel_area']+obj['area']-intersection)
            overlaps[i,obj['physical_id']]=overlap
            if feature['kind']==KINDS[obj['class_id']] and overlap>=.5:
                pairs.append((overlap,i,obj['physical_id']))
            if intersection/max(1,obj['area'])>=.25:substantial.append(obj['physical_id'])
        if len(substantial)>1:multi.append({'prediction':i,'covered_reference_ids':substantial})
    assigned={};used=set()
    for overlap,i,identity in sorted(pairs,reverse=True):
        if i not in assigned and identity not in used:
            assigned[i]=identity;used.add(identity)
    return assigned,overlaps,multi


def evaluate(directory):
    directory=Path(directory);flight=json.loads((directory/'flight.json').read_text())
    room=json.loads((directory/'room.json').read_text());truth=json.loads((directory/'truth/truth.json').read_text())
    from inspection.mission_scope import inspection_walls
    selected={w['id'] for w in inspection_walls(room)}
    truth['panels']=[p for p in truth['panels'] if p['wall_id'] in selected]
    analysis=json.loads((directory/'model-analysis.json').read_text());frames={f['file']:f for f in analysis['frames']}
    physical={o['physical_id']:{'kind':KINDS[o['class_id']],'wall_id':p['wall_id'],'observations':0,'tracks':set()} for p in truth['panels'] for o in p['objects']}
    tracks=collections.defaultdict(set);metrics={k:collections.Counter(tp=0,fp=0,fn=0) for k in KINDS};rows=[];before_support={k:collections.Counter(tp=0,fp=0,fn=0) for k in KINDS}
    evidence=directory/'count-verification';evidence.mkdir(exist_ok=True)
    for record in flight['images']:
        if record['file'] not in frames:continue
        frame=frames[record['file']];wall=next(w for w in room['walls'] if w['id']==record['wall_id'])
        config,canonical=canonical_record(room,wall,record)
        depth=np.fromfile(directory/record['depth_file'],np.float32).reshape(record['height'],record['width'])
        roi=wall_roi(config,canonical,depth)
        labels=np.zeros((record['height'],record['width']),np.uint16);refs=[]
        for pi,panel in enumerate(truth['panels']):
            if panel['wall_id']!=wall['id'] or panel['negative']:continue
            h=source_projection(room,panel,record)
            if h is None:continue
            ids=cv2.imread(str(directory/'truth'/panel['mask']),cv2.IMREAD_UNCHANGED)
            warped=cv2.warpPerspective(ids,h,(record['width'],record['height']),flags=cv2.INTER_NEAREST,borderValue=0)
            visible=(warped>0)&(roi>0);labels[visible]=warped[visible]+(pi+1)*1000
            for o in panel['objects']:
                identity=(pi+1)*1000+o['instance_id'];area=int(np.count_nonzero(labels==identity))
                if area>=12:
                    refs.append(dict(o,instance_id=identity,area=area));physical[o['physical_id']]['observations']+=1
        assigned,overlaps,multi=match(frame['findings'],labels,refs);used=set(assigned.values())
        for i,prediction in enumerate(frame['findings']):
            track=prediction['id']
            if i in assigned:
                identity=assigned[i];tracks[track].add(identity);physical[identity]['tracks'].add(track)
            else:tracks.setdefault(track,set())
        for kind in KINDS:
            metrics[kind].update(tp=sum(p['kind']==kind and i in assigned for i,p in enumerate(frame['findings'])),
                                 fp=sum(p['kind']==kind and i not in assigned for i,p in enumerate(frame['findings'])),
                                 fn=sum(KINDS[o['class_id']]==kind and o['physical_id'] not in used for o in refs))
        raw_predictions=frame['findings']+frame.get('excluded_findings',[])
        raw_assigned,_,_=match(raw_predictions,labels,refs);raw_used=set(raw_assigned.values())
        for kind in KINDS:
            before_support[kind].update(tp=sum(p['kind']==kind and i in raw_assigned for i,p in enumerate(raw_predictions)),
                fp=sum(p['kind']==kind and i not in raw_assigned for i,p in enumerate(raw_predictions)),
                fn=sum(KINDS[o['class_id']]==kind and o['physical_id'] not in raw_used for o in refs))
        row={'file':record['file'],'wall_id':wall['id'],'reference_instances':len(refs),'predicted_instances':len(frame['findings']),
             'missed_ids':[o['physical_id'] for o in refs if o['physical_id'] not in used],
             'unmatched_predictions':len(frame['findings'])-len(assigned),'multi_reference_overlap_predictions':multi}
        if refs or frame['findings']:
            raw=cv2.imread(str(directory/record['file']));reference=raw.copy()
            mask=labels>0;reference[mask]=(reference[mask]*.3+np.array([230,210,0])*.7).astype(np.uint8)
            annotated=cv2.imread(str(directory/frame['annotated_file']))
            canvas=np.concatenate([cv2.resize(reference,(1920,1080)),cv2.resize(annotated,(1920,1080))],axis=1)
            row['comparison']=record['file'].replace('.png','-reference-vs-model.png');cv2.imwrite(str(evidence/row['comparison']),canvas)
        rows.append(row)
    identity_metrics={}
    for kind in KINDS:
        objects=[v for v in physical.values() if v['kind']==kind];visible=[v for v in objects if v['observations']]
        identity_metrics[kind]={'ground_truth_physical_ids':len(objects),'visible_physical_ids':len(visible),
            'matched_physical_ids':sum(bool(v['tracks']) for v in visible),
            'missed_visible_physical_ids':sum(not v['tracks'] for v in visible),
            'not_observed_by_route':len(objects)-len(visible),
            'extra_split_tracks':sum(max(0,len(v['tracks'])-1) for v in visible),
            'incorrect_merged_tracks':sum(len(ids)>1 and any(physical[i]['kind']==kind for i in ids) for ids in tracks.values()),
            'unmatched_candidate_tracks':sum(not tracks.get(f['id']) and f['kind']==kind for f in analysis['findings'])}
    serial={key:dict(v,tracks=sorted(v['tracks'])) for key,v in physical.items()}
    result={'source':'Independent world-sized source masks projected with recorded global camera poses and tested panel geometry.',
            'matching':'Same-class one-to-one mask IoU >= 0.5; minimum 12 visible reference pixels. Occluded wall pixels excluded by recorded depth.',
            'model':analysis['model'],'observation_metrics':metrics,'before_wall_support_observation_metrics':before_support,'excluded_from_wall_identities':sum(len(f.get('excluded_findings',[])) for f in frames.values()),'identity_metrics':identity_metrics,
            'physical_ids':serial,'track_to_truth':{k:sorted(v) for k,v in tracks.items()},'frames':rows,
            'scope':'Actual AirSim flight with texture-only procedural defects. Scored independently after inference; not real-world structural accuracy.'}
    (evidence/'results.json').write_text(json.dumps(result,indent=2))
    table=''.join(f'<tr><th>{k}</th>'+''.join(f'<td>{m[x]}</td>' for x in ['ground_truth_physical_ids','visible_physical_ids','matched_physical_ids','missed_visible_physical_ids','extra_split_tracks','incorrect_merged_tracks','unmatched_candidate_tracks'])+'</tr>' for k,m in identity_metrics.items())
    objects=''.join(f'<tr><td>{html.escape(key)}</td><td>{v["kind"]}</td><td>{v["observations"]}</td><td>{len(v["tracks"])}</td><td>{html.escape(", ".join(v["tracks"]))}</td></tr>' for key,v in serial.items())
    comparisons=''.join(f'<details><summary>{r["file"]} · {r["reference_instances"]} references · {r["predicted_instances"]} predictions · {len(r["missed_ids"])} unmatched references</summary><img loading="lazy" src="{r["comparison"]}" alt="Reference masks left, actual trained masks right"></details>' for r in rows if 'comparison' in r)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Physical defect count verification</title><style>body{{font:16px system-ui;background:#eef2ed;color:#20372d;margin:0}}main{{max-width:1200px;margin:auto;padding:30px}}section{{background:white;padding:22px;border-radius:12px;margin:22px 0}}p{{line-height:1.6}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d5dfd4}}.scroll{{overflow:auto}}details{{padding:15px;border:1px solid #d4ded4;margin:12px 0}}summary{{cursor:pointer}}img{{width:100%}}a{{color:#1b6046}}</style><main><a href="../report.html">Flight evidence</a> · <a href="../model-report.html">Trained analysis</a><h1>Physical defect count verification</h1><p>{result['scope']}</p><section><h2>Known IDs versus model tracks</h2><p>Wall identities require at least half of each original mask to be supported by the depth-verified wall. Excluded masks and their pre-filter per-image errors remain in the model report and evaluation JSON.</p><p>Ground-truth IDs come from the independent scene generator. They are not supplied to the model or tracker. “Extra splits” means a known defect appears in more than one candidate track. “Wrong merges” means one track contains matched observations from multiple known defects. A candidate without a matched mask is shown separately; a failed mask overlap can also cause this.</p><div class="scroll"><table><thead><tr><th>Class</th><th>Known IDs</th><th>Visible IDs</th><th>Matched IDs</th><th>Missed visible IDs</th><th>Extra splits</th><th>Wrong merges</th><th>Unmatched tracks</th></tr></thead><tbody>{table}</tbody></table></div><p>{result['matching']}</p></section><section><h2>Every physical defect</h2><div class="scroll"><table><thead><tr><th>Independent ID</th><th>Class</th><th>Visible views</th><th>Candidate tracks</th><th>Assigned track IDs</th></tr></thead><tbody>{objects}</tbody></table></div></section><section><h2>Reference masks and actual predictions</h2><p>Left: projected reference masks in cyan. Right: model output and track IDs. This side-by-side preview is reduced to half native scale; original and annotated 4K frames are in the linked flight/model reports.</p>{comparisons}</section><p><a href="results.json">Complete evaluation JSON</a></p></main></html>'''
    (evidence/'index.html').write_text(page,encoding='utf-8')
    print('PHYSICAL_COUNT_CHECK',json.dumps(identity_metrics),flush=True)
    return result
