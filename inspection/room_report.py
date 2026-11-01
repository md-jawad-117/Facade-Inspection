"""Reviewable evidence for an actual room flight, separate from offline plans."""
import html
import json
from pathlib import Path
from .scene import segment_hits_box


def write_room_report(directory):
    directory=Path(directory);value=json.loads((directory/'flight.json').read_text())
    room=json.loads((directory/'room.json').read_text())
    rows=[json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()] if (directory/'telemetry.jsonl').exists() else []
    (directory/'telemetry.json').write_text(json.dumps(rows),encoding='utf-8')
    baseline=value.get('airborne_collision_baseline_timestamp')
    # Older first-run files can be audited against the first measured airborne timestamp.
    if baseline is None and rows:baseline=rows[0]['timestamp']
    flight_rows=[r for r in rows if r['phase'] not in ['takeoff-hover']]
    collisions=[r for r in flight_rows if r['collision_timestamp']>baseline] if baseline is not None else []
    sx=lambda x:45+x*27;sy=lambda y:535-y*27
    shapes=''
    for item in room['obstacles']:
        lo,hi=item['lower'],item['upper']
        shapes+=f'<rect x="{sx(lo[0])}" y="{sy(hi[1])}" width="{(hi[0]-lo[0])*27}" height="{(hi[1]-lo[1])*27}" fill="#b58348"><title>{html.escape(item["id"])}</title></rect>'
    observed=[]
    sensor_file=directory/'navigation-depth/observations.jsonl'
    if sensor_file.exists():
        lines=sensor_file.read_text().splitlines()
        if lines:observed=json.loads(lines[-1]).get('occupied_boxes',[])
    for item in observed:
        lo,hi=item['lower'],item['upper']
        shapes+=f'<rect x="{sx(lo[0])}" y="{sy(hi[1])}" width="{(hi[0]-lo[0])*27}" height="{(hi[1]-lo[1])*27}" fill="#7e3fb2" fill-opacity=".4" stroke="#6a239b" stroke-width="2"><title>Depth-observed occupancy</title></rect>'
    for event in value.get('depth_replan_events',[]):
        p=event['position'];x,y=sx(p[0]),sy(p[1])
        shapes+=f'<path d="M{x-4},{y-4}L{x+4},{y+4}M{x+4},{y-4}L{x-4},{y+4}" stroke="black" stroke-width="3"><title>Stopped and replanned</title></path>'
    for transit in value['executed_transits']:
        coords=' '.join(f'{sx(p[0])},{sy(p[1])}' for p in transit['path'])
        shapes+=f'<polyline points="{coords}" fill="none" class="route-planned" stroke="#d45a00" stroke-width="5" stroke-dasharray="14 9" opacity=".85"/>'
    coords=' '.join(f'{sx(r["position"][0])},{sy(r["position"][1])}' for r in rows)
    shapes+=f'<polyline points="{coords}" fill="none" class="route-actual" stroke="#005fcc" stroke-width="2"/>'
    panels=''
    for record in value['images']:
        wall=record.get('wall_id') or 'navigation checkpoint'
        panels+=f'<section><h3>{html.escape(record["target"]["id"])}</h3><img src="{record["file"]}" loading="lazy" alt="Actual room-flight capture"><p>Wall: {wall} · RGB {record["width"]} × {record["height"]} · paired planar depth saved.</p></section>'
    coverage='';maps=''
    for key,wall in value.get('coverage',{}).get('walls',{}).items():
        coverage+=f'<tr><td>{key}</td><td>{len(wall["views"])}</td><td>{wall["visible_samples"]}/{wall["sample_count"]}</td><td>{wall["sampled_visible_percent"]:.2f}%</td></tr>'
        surface=next(w for w in room['walls'] if w['id']==key)
        nx,ny=wall['grid_width'],wall['grid_height'];covered=set(wall['covered_indices']);marks=''
        for j in range(ny):
            for i in range(nx):
                color='#287e68' if j*nx+i in covered else '#d7dcd7'
                marks+=f'<rect x="{i*480/nx}" y="{(ny-1-j)*120/ny}" width="{480/nx+.1}" height="{120/ny+.1}" fill="{color}"/>'
        maps+=f'<section><h3>{surface["name"]} · {wall["sampled_visible_percent"]:.2f}%</h3><svg viewBox="0 0 480 125" role="img" aria-label="Measured visibility of {surface["name"]} wall">{marks}</svg></section>'
    minimum=value.get('minimum_sampled_center_clearance_m')
    distances=[r['measured_camera_standoff_m'] for r in value['images'] if 'measured_camera_standoff_m' in r]
    detours=sum(any(segment_hits_box(t['path'][0],t['path'][-1],o['lower'],o['upper'],room['clearance_m']+.25)
                    for o in room['obstacles']) for t in value['executed_transits'])
    metrics={'telemetry_samples':len(rows),'new_collision_samples':len(collisions),
             'minimum_sampled_center_clearance_m':minimum,'status':value['status'],'landing':value['landing'],
             'return_verified':value.get('return_verified',False),'image_pairs':len(value['images']),
             'static_obstacle_detours':detours,'depth_replan_events':len(value.get('depth_replan_events',[])),
             'online_depth_enabled':room.get('online_depth',False),'trained_inspection_enabled':room.get('trained_inspection',False),'camera_standoff_range_m':[min(distances),max(distances)] if distances else None,
             'scope':'Sampled poses and collision history; live-depth and trained-inspection enablement recorded explicitly. No continuous rotor-clearance guarantee.'}
    (directory/'verification.json').write_text(json.dumps(metrics,indent=2))
    error='<p class="error">'+html.escape(value.get('error',''))+'</p>' if value.get('error') else ''
    minimum_text=f'{minimum:.2f} m' if minimum is not None else 'unavailable'
    route_controls='<div style="display:flex;gap:24px;flex-wrap:wrap;margin:20px 0"><label><input type="checkbox" data-route="route-planned" checked> <span style="color:#a54200">━━</span> Planned · orange dashed</label><label><input type="checkbox" data-route="route-actual" checked> <span style="color:#005fcc">━━</span> Actual · blue solid</label></div><script>document.addEventListener(\'change\',function(e){const kind=e.target.getAttribute(\'data-route\');if(kind)document.querySelectorAll(\'.\'+kind).forEach(function(line){line.style.display=e.target.checked?\'\':\'none\';});});</script>'
    if room.get('online_depth'):
        integration_scope='Navigation starts from the room map, senses forward planar depth during movement, stops and replans around observed occupied geometry, and adjusts newly blocked viewpoints. Occupancy is conservative within this mission; this does not predict fast moving machinery.'
    else:integration_scope='Navigation uses the known static map. Depth was not used for online obstacle updates in this run.'
    if room.get('trained_inspection'):integration_scope+=' The trained segmentation model processes native camera images during capture and saves wall-labeled defect observations. Physical-count evaluation is independent afterward.'
    system_links='<p><a href="model-report.html">Trained wall results</a> · <a href="count-verification/index.html">Physical-count verification</a></p>' if (directory/'model-analysis.json').exists() else ''
    if (directory/'inspection-completeness.html').exists():system_links+='<p><a href="inspection-completeness.html">Wall inspection completeness and unresolved areas</a></p>'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Room flight evidence</title><style>body{{font:16px system-ui;background:#eef0ec;color:#243b35;margin:0}}main{{max-width:1050px;margin:auto;padding:30px}}p{{line-height:1.6}}section{{background:#fff;padding:18px;border-radius:10px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}svg,img{{max-width:100%;height:auto}}table{{width:100%;text-align:left;border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #ccc}}a{{color:#14634f}}.error{{color:#a13535}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style><main><p>ACTUAL SIMULATOR FLIGHT · {directory.name}</p><h1>Room flight verification</h1><p><b>Status: {html.escape(value['status'])} · landing: {html.escape(value['landing'])}</b></p>{error}{system_links}<p>{len(value['images'])} RGB/depth pairs · {len(rows)} pose samples · {len(collisions)} samples with a new collision timestamp · minimum sampled center distance to mapped surfaces: {minimum_text}.</p><h2>Measured path</h2><p>Blue solid: actual sampled positions. Orange dashed: planned transits. Use the switches to view either route alone. Purple boxes: depth-observed occupancy. Black crosses: stop/replan positions. Heights overlap in this top view.</p>{route_controls}<svg viewBox="0 0 745 585" aria-label="Actual room flight path"><rect x="45" y="49" width="648" height="486" fill="#fff" stroke="#7c8a81" stroke-width="5"/>{shapes}</svg><h2>Actual wall visibility</h2><p>Computed from saved camera pose and planar depth. Unverified samples remain visible as gaps.</p><table><thead><tr><th>Wall</th><th>Views</th><th>Visible samples</th><th>Sampled visibility</th></tr></thead><tbody>{coverage}</tbody></table><p>Green: verified visible sample. Gray: not verified in this run.</p><div class="grid">{maps}</div><h2>Captured images</h2><div class="grid">{panels}</div><h2>Scope</h2><p>The drone was placed on the room floor before arming. All subsequent airborne movement used the simulator flight controller. Ground setup is separate from flight. {integration_scope}</p><p>Clearance is measured from the vehicle center to conservative mapped geometry at sampled poses. It is not continuous rotor/body clearance. Zero new collision samples is evidence for this run, not proof for unseen layouts.</p><p><a href="flight.json">Flight data</a> · <a href="telemetry.json">Telemetry</a> · <a href="verification.json">Verification summary</a> · <a href="evidence-audit.json">Saved-data audit</a> · <a href="/assets/inspection-system/index.html">Inspection controls</a></p></main></html>'''
    (directory/'report.html').write_text(page,encoding='utf-8')
    parent=directory.parents[1]
    (parent/'latest-flight.json').write_text(json.dumps({'run':directory.name,'report':f'flights/{directory.name}/report.html','verification':metrics},indent=2))
    return metrics
