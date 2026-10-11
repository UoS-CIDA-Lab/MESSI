"""Render MatchReport as a standalone tactical HTML page."""

from __future__ import annotations

from html import escape
from pathlib import Path
from math import isfinite, sqrt
from base64 import b64encode

from messi._json import strict_json_dumps
from messi.analysis.causal import is_goal_directed_shot
from messi.rendering.network_identity import (
    network_passes_with_generations,
    network_shots_with_generations,
)


def _pass_height_class(row: dict) -> str:
    """Classify a pass from observed vertical signals, not a reconstructed path."""
    heights = []
    for key in ("source_position", "terminal_position"):
        position = row.get(key)
        if position is not None and len(position) >= 3:
            height = float(position[2])
            if isfinite(height):
                heights.append(height)
    velocity = row.get("release_velocity")
    vertical_speed = None
    if velocity is not None and len(velocity) >= 3:
        value = float(velocity[2])
        if isfinite(value):
            vertical_speed = value
    if vertical_speed is None and not heights:
        return "unknown"
    if (vertical_speed is not None and vertical_speed >= 4.0) or any(
        height >= 1.0 for height in heights
    ):
        return "elevated"
    return "low"


def _pass_marker_defs() -> str:
    colors = {
        "team-0": "#38bdf8",
        "team-1": "#fb7185",
        "elevated": "#facc15",
        "unknown": "#94a3b8",
    }
    return "<defs>" + "".join(
        f'<marker id="pass-arrow-{name}" markerUnits="userSpaceOnUse" '
        'viewBox="0 0 2 2" refX="2" refY="1" markerWidth="1.4" '
        f'markerHeight="1.4" orient="auto"><path d="M0 0 L2 1 L0 2 Z" fill="{color}"/></marker>'
        for name, color in colors.items()
    ) + "</defs>"

_TEMPLATE = r"""<!doctype html><html lang="en" class="no-js"><meta charset="utf-8"><title>__DOCUMENT_TITLE__</title>
<style>
:root{--bg:#07111d;--panel:#111d2b;--panel-2:#0b1628;--line:#334155;--text:#e5edf5;--muted:#91a2b6;--blue:#38bdf8;--red:#fb7185;--green:#4ade80;--gold:#facc15;--team:#38bdf8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px system-ui,sans-serif}.head{padding:22px 28px;background:#0d1927;border-bottom:1px solid var(--line)}h1{margin:0 0 8px}.kpis,.report-grid{display:grid;gap:14px}.kpis{grid-template-columns:repeat(5,minmax(0,1fr));margin-top:14px}.kpi,.card,.team-focus{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:13px}.kpi b{display:block;font-size:21px}.kpi span,.muted,.card p{color:var(--muted)}.report-grid{grid-template-columns:minmax(0,1fr) minmax(0,1fr);padding:14px 22px}.wide{grid-column:1/-1}.card h2{margin:0 0 5px}.card p{margin:0 0 10px}.section-label{margin:6px 0 0;color:#cbd5e1;font-size:12px;letter-spacing:.12em;text-transform:uppercase}.team-focus{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:16px 18px;border-color:color-mix(in srgb,var(--team) 55%,var(--line));box-shadow:inset 4px 0 0 var(--team)}.team-focus h2{margin:0}.team-tabs{display:flex;gap:8px}.team-tab{min-width:120px;border-radius:999px;padding:8px 14px;cursor:pointer}.team-tab[aria-selected=true]{border-color:var(--team);box-shadow:0 0 0 1px var(--team) inset;background:#17283a}.team-name{display:inline-flex;align-items:center;gap:7px}.team-swatch{width:10px;height:10px;border-radius:50%;background:var(--team-color)}svg{width:100%;background:#1d6b40;border-radius:4px}.pitch{fill:none;stroke:#fff;stroke-width:.25}.team-table,.event-table{width:100%;border-collapse:collapse}.team-table th,.team-table td,.event-table td{padding:6px;border-bottom:1px solid #263548;text-align:right}.team-table th:first-child,.team-table td:first-child,.event-table td:first-child{text-align:left}.bars{display:grid;gap:7px}.bar{display:grid;grid-template-columns:155px 1fr 105px;gap:8px;align-items:center}.bar i{height:8px;background:var(--team);display:block}.bar>span:nth-child(2){height:8px;background:#08131f;border-radius:2px;overflow:hidden}.legend{display:flex;gap:12px;color:var(--muted);margin:7px 0;align-items:center}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.line-key{display:inline-block;width:22px;border-top:2px solid var(--team);margin-right:5px;vertical-align:middle}.line-key.failed{border-top-style:dashed;opacity:.6}.network-controls{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;margin:8px 0}.network-controls input{width:100%}.network-node text{fill:white;font-size:3px;text-anchor:middle}.small{font-size:11px}select,button{background:#08131f;color:white;padding:5px;border:1px solid var(--line)}
.head,.report-grid{max-width:1600px;margin-left:auto;margin-right:auto}
.legend{flex-wrap:wrap}
.report-grid{grid-template-columns:repeat(3,minmax(0,1fr));align-items:start}.span2{grid-column:span 2}.card{min-width:0;overflow:hidden}
svg{display:block;width:100%;height:auto;background:#08131f;border-radius:5px}
#heat{background:#0b1628;aspect-ratio:105/68}
#passes,#shots,#network{background:#173f31;aspect-ratio:105/68}
#possession{background:#0b1628;aspect-ratio:600/240}
.pitch{fill:none;stroke:#f4f7fa;stroke-width:.28;opacity:.82;pointer-events:none}
.team-table th{color:#b8c7d6}.team-table tr:hover td,.event-table tr:hover td{background:#17283a}
.bar>span:nth-child(2){height:8px;background:#08131f;border-radius:2px;overflow:hidden}
.network-node text{font-size:2.2px}
.roster-layout{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.roster-side{min-width:0}.roster-side h3{margin:4px 0 8px}.roster-side svg{max-height:280px}.roster-table-wrap{overflow-x:auto}.roster-table{font-size:11px;white-space:nowrap}.roster-table th,.roster-table td{text-align:right}.roster-table th:nth-child(2),.roster-table td:nth-child(2){text-align:left}@media(max-width:900px){.roster-layout{grid-template-columns:1fr}}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.report-grid{grid-template-columns:minmax(0,1fr)}.wide,.team-focus{grid-column:1}.team-focus{align-items:flex-start;flex-direction:column}.team-tabs{width:100%}.team-tab{flex:1;min-width:0}}@media(max-width:700px){.head{padding:16px}.report-grid{padding:8px}.bar{grid-template-columns:110px 1fr 80px}}
@media(max-width:1100px){.report-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.wide,.team-focus{grid-column:1/-1}}@media(max-width:700px){.report-grid{grid-template-columns:minmax(0,1fr)}.wide,.span2,.team-focus{grid-column:1}}
.js .network-static,.no-js .network-js,.js .static-team-control,.js .network-video-enabled{display:none}.no-js .team-tab,.no-js #layer{display:none}.no-js #static-team-0:checked ~ main .team-detail-1,.no-js #static-team-1:checked ~ main .team-detail-0{display:none!important}.static-team-control{cursor:pointer;border:1px solid var(--line);border-radius:999px;padding:8px 14px}.no-js #static-team-0:checked ~ main label[for="static-team-0"],.no-js #static-team-1:checked ~ main label[for="static-team-1"]{border-color:var(--blue);box-shadow:0 0 0 1px var(--blue) inset}.network-video-enabled video{display:block;width:100%;max-height:650px;background:#08131f}.network-video-enabled p{margin:8px 0}.no-js .network-video-enabled ~ .network-static-timeline,.no-js .network-video-enabled ~ #network{display:none}.network-static-timeline{display:flex;align-items:center;gap:10px;margin:10px 0}.network-static-timeline>label{cursor:pointer;border:1px solid var(--line);border-radius:5px;padding:6px 10px}.network-track{position:relative;display:flex;flex:1;min-width:150px;height:24px;background:linear-gradient(#243c51,#243c51) center/100% 8px no-repeat}.network-track label{flex:1;cursor:pointer;z-index:2}.network-thumb{position:absolute;top:2px;left:0;width:4px;height:20px;border-radius:3px;background:var(--blue);pointer-events:none}.network-readout{min-width:100px;text-align:right}.network-playing{display:none}.network-snapshot,.network-time-label{display:none}__NETWORK_STATIC_CSS__
</style>
<div class="head"><h1>__HEADING__</h1><div id="subtitle" class="muted"></div><div id="kpis" class="kpis"></div></div><input class="network-static" type="radio" name="static-team" id="static-team-0" hidden checked><input class="network-static" type="radio" name="static-team" id="static-team-1" hidden><main class="report-grid">
<section class="card wide"><h2>Team tactical comparison</h2><p>A goal-directed shot is a SHOOT release whose trajectory projects into the goal mouth; clearances share the SHOOT intent.</p><table id="teams" class="team-table"></table><p id="reference"></p></section>
<section class="card wide"><h2>Starting formations and player attributes</h2><p id="roster-source"></p><div id="roster-content" class="roster-layout"></div></section>
<div class="section-label wide">Match overview</div>
<section class="card span2"><h2>Cumulative controlled possession</h2><svg id="possession" viewBox="0 0 600 240"></svg></section>
<section class="card"><h2>Event counts</h2><table id="events" class="event-table"></table></section>
<section class="team-focus"><div><div class="section-label">Team detail</div><h2 id="team-focus-title"><span class="team-detail-0">Team 0 analysis</span><span class="team-detail-1">Team 1 analysis</span></h2><p class="muted">The views below show the selected team.</p></div><div class="team-tabs" role="tablist" aria-label="Select team"><label class="static-team-control" for="static-team-0">Team 0</label><label class="static-team-control" for="static-team-1">Team 1</label><button class="team-tab" data-team="0" role="tab" aria-selected="true"><span class="team-name" style="--team-color:#38bdf8"><i class="team-swatch"></i>Team 0</span></button><button class="team-tab" data-team="1" role="tab" aria-selected="false"><span class="team-name" style="--team-color:#fb7185"><i class="team-swatch"></i>Team 1</span></button></div></section>
<section class="card"><h2>Spatial occupancy and player movement heatmap</h2><select id="layer"><option value="team">Selected team</option><option value="ball">Live ball</option></select><svg id="heat" viewBox="-52.5 -34 105 68"></svg></section>
<section class="card"><h2>Realized pass map</h2><div class="legend"><span>→ Direction</span><span><i class="line-key"></i>Low proxy</span><span><i class="line-key" style="border-color:#facc15"></i>Elevated proxy</span><span><i class="line-key" style="border-color:#94a3b8"></i>Unknown</span></div><div class="legend"><span><i class="line-key"></i>Completed</span><span><i class="line-key failed"></i>Uncompleted</span></div><p class="small">Elevated proxy: release vertical speed ≥ 4 m/s or either endpoint at least 1 m high. Colors do not show maximum flight height.</p><svg id="passes" viewBox="-52.5 -34 105 68"></svg></section>
<section class="card"><h2>Goal-directed shot map</h2><p>Shows valid attempts by the selected team. Hover over a marker for its outcome.</p><svg id="shots" viewBox="-52.5 -34 105 68"></svg></section>
<section class="card wide"><h2>Open-play pass completion by 15-minute window</h2><div id="windows" class="bars"></div></section>
<section class="card"><h2>Pass terminal causes</h2><div id="terminals" class="bars"></div></section><section class="card"><h2>Open-play passes by five directions</h2><div id="directions" class="bars"></div></section>
<section class="card wide"><h2>Player possession, position, and spacing</h2><p>Absolute positions are means in pitch coordinates; spacing uses simultaneously active teammates.</p><table id="players" class="team-table small"></table><h3>Mean absolute position by possession phase</h3><table id="player-phases" class="team-table small"></table></section>
<section class="card wide"><h2>Time-resolved passing and shooting network</h2><p>Nodes show mean pass-contact and shot-release position up to the selected time. Orange links connect a shooter node to the observed shot endpoint; they are not ball trajectories.</p><div class="legend"><span><i class="line-key"></i>Field-player pass</span><span><i class="line-key" style="border-color:#facc15"></i>Goalkeeper pass</span><span><i class="line-key" style="border-color:#fb923c"></i>Shot</span></div>__NETWORK_VIDEO____NETWORK_STATIC_INPUTS__<div class="network-static network-static-timeline"><label for="network-static-play">Play / pause</label><div class="network-track">__NETWORK_STATIC_LABELS__<span class="network-thumb"></span></div><span class="network-readout">__NETWORK_STATIC_READOUT__<span class="network-playing">Playing</span></span></div><div class="network-js network-controls"><button id="network-play">Play</button><input id="network-time" type="range" min="0" value="0" step="1"><span id="network-status"></span></div><svg id="network" viewBox="-52.5 -34 105 68"></svg></section>
</main><script>
const d=__PAYLOAD__,S=d.spatial,ns='http://www.w3.org/2000/svg',xmin=S.x_edges_m[0],xmax=S.x_edges_m[S.x_edges_m.length-1],ymin=S.y_edges_m[0],ymax=S.y_edges_m[S.y_edges_m.length-1],hx=Math.max(Math.abs(xmin),Math.abs(xmax)),hy=Math.max(Math.abs(ymin),Math.abs(ymax));function el(n,a,t){let x=document.createElementNS(ns,n);Object.entries(a||{}).forEach(([k,v])=>x.setAttribute(k,v));if(t!==undefined)x.textContent=t;return x}for(let id of ['heat','passes','shots','network'])document.querySelector('#'+id).setAttribute('viewBox',`${xmin} ${ymin} ${xmax-xmin} ${ymax-ymin}`);function pitch(svg){let boxY=Math.min(20.16,hy),boxD=Math.min(16.5,hx);svg.innerHTML=`<rect class="pitch" x="${xmin}" y="${ymin}" width="${xmax-xmin}" height="${ymax-ymin}"/><line class="pitch" x1="0" y1="${ymin}" x2="0" y2="${ymax}"/><circle class="pitch" cx="0" cy="0" r="${Math.min(9.15,hx,hy)}"/><rect class="pitch" x="${hx-boxD}" y="${-boxY}" width="${boxD}" height="${2*boxY}"/><rect class="pitch" x="${-hx}" y="${-boxY}" width="${boxD}" height="${2*boxY}"/>`}const pct=(n,z)=>z?`${(100*n/z).toFixed(1)}%`:'—',T=d.tactical_summary;document.querySelector('#subtitle').textContent=`${d.frames} frames · schema ${d.schema}`;
let k=[['Score',`${d.final_score[0]}–${d.final_score[1]}`],['Possession',`${pct(d.possession_frames[0],d.possession_frames[0]+d.possession_frames[1])} / ${pct(d.possession_frames[1],d.possession_frames[0]+d.possession_frames[1])}`],['Open-play pass',`${T[0].open_play_pass_completions}/${T[0].open_play_pass_attempts} · ${T[1].open_play_pass_completions}/${T[1].open_play_pass_attempts}`],['Shots',`${T[0].shot_attempts}–${T[1].shot_attempts}`],['On target',`${T[0].shots_on_target}–${T[1].shots_on_target}`]];document.querySelector('#kpis').innerHTML=k.map(x=>`<div class="kpi"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('');let ref=d.reference_context&&d.reference_context.dfl_attacking_third_pass,reference=document.querySelector('#reference');reference.textContent=ref?`DFL descriptive reference · attacking-third completion ${pct(ref.completion,1)} · direction backward/lateral/forward ${pct(ref.direction_share.backward,1)} / ${pct(ref.direction_share.lateral,1)} / ${pct(ref.direction_share.forward,1)} · ${ref.comparability}`:'DFL descriptive reference unavailable';
let rows=[['Metric','<span class="team-name" style="--team-color:#38bdf8"><i class="team-swatch"></i>Team 0</span>','<span class="team-name" style="--team-color:#fb7185"><i class="team-swatch"></i>Team 1</span>'],['Open-play pass completion',pct(T[0].open_play_pass_completions,T[0].open_play_pass_attempts),pct(T[1].open_play_pass_completions,T[1].open_play_pass_attempts)],['Restart pass',`${T[0].restart_pass_completions}/${T[0].restart_pass_attempts}`,`${T[1].restart_pass_completions}/${T[1].restart_pass_attempts}`],['Attacking-third completed',T[0].attacking_third_pass_completions,T[1].attacking_third_pass_completions],['Box entries completed',T[0].box_entry_pass_completions,T[1].box_entry_pass_completions],['Shot attempts',T[0].shot_attempts,T[1].shot_attempts],['Releases near goal',T[0].shot_releases_near_goal,T[1].shot_releases_near_goal],['Shots on target',T[0].shots_on_target,T[1].shots_on_target],['Mean shot distance',`${T[0].mean_shot_attempt_distance_m.toFixed(1)} m`,`${T[1].mean_shot_attempt_distance_m.toFixed(1)} m`]];document.querySelector('#teams').innerHTML=rows.map((r,i)=>`<tr>${r.map(v=>i?`<td>${v}</td>`:`<th>${v}</th>`).join('')}</tr>`).join('');
const H=d.team_shape_summary||[{},{}],shapeValue=x=>x===null||x===undefined?'—':x.toFixed(1)+' m';rows.push(['Mean team width',shapeValue(H[0].mean_width_m),shapeValue(H[1].mean_width_m)],['Mean team depth',shapeValue(H[0].mean_depth_m),shapeValue(H[1].mean_depth_m)],['Mean pairwise distance',shapeValue(H[0].mean_pairwise_distance_m),shapeValue(H[1].mean_pairwise_distance_m)],['Width · in possession',shapeValue(H[0].in_possession_mean_width_m),shapeValue(H[1].in_possession_mean_width_m)],['Width · out of possession',shapeValue(H[0].out_of_possession_mean_width_m),shapeValue(H[1].out_of_possession_mean_width_m)]);document.querySelector('#teams').innerHTML=rows.map((r,i)=>`<tr>${r.map(v=>i?`<td>${v}</td>`:`<th>${v}</th>`).join('')}</tr>`).join('');
let shapePoint=p=>p?'('+p[0].toFixed(1)+', '+p[1].toFixed(1)+') m':'—';rows.push(['Mean centroid',shapePoint(H[0].mean_centroid_m),shapePoint(H[1].mean_centroid_m)],['Depth · in possession',shapeValue(H[0].in_possession_mean_depth_m),shapeValue(H[1].in_possession_mean_depth_m)],['Depth · out of possession',shapeValue(H[0].out_of_possession_mean_depth_m),shapeValue(H[1].out_of_possession_mean_depth_m)]);document.querySelector('#teams').innerHTML=rows.map((r,i)=>`<tr>${r.map(v=>i?`<td>${v}</td>`:`<th>${v}</th>`).join('')}</tr>`).join('');
let shapeTriplet=h=>[h.pairwise_distance_p10_m,h.pairwise_distance_p50_m,h.pairwise_distance_p90_m].map(shapeValue).join(' / ');rows.push(['Stretch index',shapeValue(H[0].mean_stretch_index_m),shapeValue(H[1].mean_stretch_index_m)],['Longitudinal max gap',shapeValue(H[0].mean_longitudinal_max_gap_m),shapeValue(H[1].mean_longitudinal_max_gap_m)],['Pairwise P10 / P50 / P90',shapeTriplet(H[0]),shapeTriplet(H[1])],['Opponent centroid distance',shapeValue(H[0].mean_opponent_centroid_distance_m),shapeValue(H[1].mean_opponent_centroid_distance_m)],['Ball–centroid distance',shapeValue(H[0].mean_ball_centroid_distance_m),shapeValue(H[1].mean_ball_centroid_distance_m)]);document.querySelector('#teams').innerHTML=rows.map((r,i)=>`<tr>${r.map(v=>i?`<td>${v}</td>`:`<th>${v}</th>`).join('')}</tr>`).join('');
const teamColors=['#38bdf8','#fb7185'];let activeTeam=0;
const heat=document.querySelector('#heat'),layer=document.querySelector('#layer'),playerCount=S.player_occupancy_seconds.length,teamSize=Math.floor(playerCount/2);
function populateLayers(){let previous=layer.value;while(layer.options.length>2)layer.remove(2);let start=activeTeam*teamSize,end=activeTeam===0?teamSize:playerCount;for(let i=start;i<end;i++)layer.insertAdjacentHTML('beforeend',`<option value="player${i}">Player slot ${i}</option>`);if(previous.startsWith('player')&&(+previous.slice(6)<start||+previous.slice(6)>=end))previous='team';layer.value=previous}
function drawHeat(){pitch(heat);let q=layer.value,pi=q.startsWith('player')?+q.slice(6):-1,m=q==='ball'?S.ball_density_seconds:q==='team'?S.team_occupancy_seconds[activeTeam]:S.player_occupancy_seconds[pi],mx=0;m.forEach(row=>row.forEach(v=>{mx=Math.max(mx,v)}));m.forEach((row,y)=>row.forEach((v,x)=>{if(v<=0)return;let c=q==='ball'?'#facc15':teamColors[activeTeam];heat.appendChild(el('rect',{x:S.x_edges_m[x],y:S.y_edges_m[y],width:S.x_edges_m[x+1]-S.x_edges_m[x],height:S.y_edges_m[y+1]-S.y_edges_m[y],fill:c,opacity:(.12+.75*Math.sqrt(v/mx)).toFixed(3)}))}));raisePitchLines(heat)}layer.onchange=drawHeat;
let possession=document.querySelector('#possession'),timeline=d.possession_timeline,maxT=Math.max(1,...timeline.map(x=>x.time_s));possession.innerHTML='';possession.appendChild(el('rect',{x:40,y:15,width:540,height:190,fill:'none',stroke:'#94a3b8'}));[0,1].forEach(team=>{let pts=timeline.map(x=>(40+540*x.time_s/maxT)+','+(205-190*x.share[team])).join(' ');possession.appendChild(el('polyline',{points:pts,fill:'none',stroke:team?'#fb7185':'#38bdf8','stroke-width':'3'}))});
const point=(r,q)=>r[q]&&[r[q][0],r[q][1]],pass=document.querySelector('#passes'),shot=document.querySelector('#shots'),passMarkerDefs=__PASS_MARKERS_JS__;
function drawPasses(){
  pitch(pass);pass.insertAdjacentHTML('afterbegin',passMarkerDefs);
  for(const row of d.pass_map.filter(item=>item.team===activeTeam)){
    const a=point(row,'attack_normalized_source_position'),b=point(row,'attack_normalized_terminal_position');if(!a||!b)continue;
    const height=row._pass_height_class,color=height==='elevated'?'#facc15':height==='unknown'?'#94a3b8':teamColors[activeTeam];
    const marker=height==='low'?`team-${activeTeam}`:height;
    const mid=[(a[0]+b[0])/2,(a[1]+b[1])/2];
    const line=el('polyline',{points:`${a[0]},${a[1]} ${mid[0]},${mid[1]} ${b[0]},${b[1]}`,fill:'none',stroke:color,'stroke-width':row.completed?'.5':'.35','stroke-dasharray':row.completed?'':'1.2 1','marker-mid':`url(#pass-arrow-${marker})`,opacity:row.completed?'.78':'.42'});
    const vz=row.release_velocity&&row.release_velocity.length>2?`${row.release_velocity[2].toFixed(1)} m/s`:'unknown';
    line.appendChild(el('title',{},`${row.completed?'Completed':'Uncompleted'} pass · ${height} height proxy · release vertical speed ${vz}`));pass.appendChild(line);
  }
  raisePitchLines(pass);
}
const colors={goal:'#4ade80',saved:'#38bdf8',blocked:'#c084fc',woodwork:'#facc15',off_target:'#fb7185',unresolved:'#94a3b8'};function drawShots(){pitch(shot);d.shot_map.filter(r=>r._goal_directed&&r.team===activeTeam).forEach(r=>{let a=point(r,'attack_normalized_source_position');if(!a)return;let c=el('circle',{cx:a[0],cy:a[1],r:'1.15',fill:colors[r.outcome]||'#fff',stroke:teamColors[activeTeam],'stroke-width':'.25'});c.appendChild(el('title',{},`${r.outcome} · frame ${r.source_frame}`));shot.appendChild(c)});raisePitchLines(shot)}
function bars(id,rs){let mx=Math.max(1,...rs.map(x=>x[1]));document.querySelector(id).innerHTML=rs.map(x=>`<div class="bar"><span>${x[0]}</span><span><i style="width:${100*x[1]/mx}%"></i></span><b>${x[1]}</b></div>`).join('')}
function drawTeamStats(){let wr=[];for(let w=0;w<6;w++){let x=T[activeTeam].pass_completion_by_15m[w];wr.push([`${w*15}–${(w+1)*15}m`,x.attempts?`${x.completions}/${x.attempts} (${pct(x.completions,x.attempts)})`:'—',x.attempts?100*x.completions/x.attempts:0])}document.querySelector('#windows').innerHTML=wr.map(x=>`<div class="bar"><span>${x[0]}</span><span><i style="width:${x[2]}%"></i></span><b>${x[1]}</b></div>`).join('');bars('#terminals',Object.entries(T[activeTeam].pass_terminal_kinds));bars('#directions',Object.entries(T[activeTeam].open_play_direction_five));let players=d.player_action_credit.filter(x=>x.team===activeTeam).slice().sort((a,b)=>(b.controlled_possession_seconds||0)-(a.controlled_possession_seconds||0));let value=x=>x===null||x===undefined?'—':x.toFixed(1);document.querySelector('#players').innerHTML='<tr><th>Player</th><th>Active</th><th>Possession</th><th>Mean absolute position</th><th>Mean teammate distance</th><th>Nearest teammate</th><th>Pass</th><th>Shot</th><th>Distance</th></tr>'+players.map(x=>{let p=x.mean_absolute_position_m;return `<tr><td>#${x.player_id}${x.generation?' · G'+x.generation:''}</td><td>${value(x.active_seconds)} s</td><td>${value(x.controlled_possession_seconds)} s</td><td>${p?'('+p[0].toFixed(1)+', '+p[1].toFixed(1)+') m':'—'}</td><td>${value(x.mean_teammate_distance_m)} m</td><td>${value(x.mean_nearest_teammate_distance_m)} m</td><td>${x.pass_completions}/${x.pass_attempts}</td><td>${x.shot_attempts}</td><td>${(x.distance_m/1000).toFixed(2)} km</td></tr>`}).join('')}
let network=document.querySelector('#network'),networkTime=document.querySelector('#network-time'),networkStatus=document.querySelector('#network-status'),timer=null,networkStepFrames=Math.max(1,Math.round(30/d.control_dt_s)),drawNetwork;networkTime.max=d.frames;networkTime.value=d.frames;
function drawPlayerPhases(){let players=d.player_action_credit.filter(x=>x.team===activeTeam),position=p=>p?'('+p[0].toFixed(1)+', '+p[1].toFixed(1)+') m':'—';document.querySelector('#player-phases').innerHTML='<tr><th>Player</th><th>Team in possession</th><th>Opponent in possession</th><th>Position change</th></tr>'+players.map(x=>{let a=x.mean_in_possession_position_m,b=x.mean_out_of_possession_position_m,delta=a&&b?Math.hypot(a[0]-b[0],a[1]-b[1]).toFixed(1)+' m':'—';return `<tr><td>#${x.player_id}${x.generation?' · G'+x.generation:''}</td><td>${position(a)}</td><td>${position(b)}</td><td>${delta}</td></tr>`}).join('')}
function renderTeam(){document.documentElement.style.setProperty('--team',teamColors[activeTeam]);document.querySelector('#team-focus-title').textContent=`Team ${activeTeam} analysis`;document.querySelectorAll('.team-tab').forEach(button=>button.setAttribute('aria-selected',String(+button.dataset.team===activeTeam)));populateLayers();drawHeat();drawPasses();drawShots();drawTeamStats();drawPlayerPhases();drawNetwork()}
document.querySelectorAll('.team-tab').forEach(button=>button.onclick=()=>{activeTeam=+button.dataset.team;renderTeam()});document.querySelector('#network-play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}if(+networkTime.value>=d.frames)networkTime.value=0;timer=setInterval(()=>{networkTime.value=Math.min(d.frames,+networkTime.value+networkStepFrames);drawNetwork();if(+networkTime.value>=d.frames){clearInterval(timer);timer=null}},100)};document.querySelector('#events').innerHTML=Object.entries(d.event_counts).map(x=>`<tr><td>${x[0]}</td><td>${x[1]}</td></tr>`).join('');
</script>
<script>
const raisePitchLines=svg=>[...svg.querySelectorAll('.pitch')].forEach(node=>svg.appendChild(node));
const networkPlayers=new Map(d.player_generations.map(player=>[`${player.slot}:${player.generation}`,player]));
drawNetwork=()=>{
  const cut=+networkTime.value,nodes=new Map(),edges=new Map();
  const shown=d.pass_map.filter(row=>row.team===activeTeam&&row.completed&&row.terminal_frame<=cut&&row.actor!==row.terminal_actor&&row._network_source_generation!==undefined&&row._network_target_generation!==undefined&&point(row,'attack_normalized_source_position')&&point(row,'attack_normalized_terminal_position'));
  const shots=d.shot_map.filter(row=>row.team===activeTeam&&row.terminal_frame<=cut&&row._network_shooter_generation!==undefined);
  pitch(network);
  for(const row of shown){
    const source=`${row.actor}:${row._network_source_generation}`,target=`${row.terminal_actor}:${row._network_target_generation}`;
    const endpoints=[[source,point(row,'attack_normalized_source_position')],[target,point(row,'attack_normalized_terminal_position')]];
    for(const [key,position] of endpoints){const node=nodes.get(key)||{x:0,y:0,count:0};node.x+=position[0];node.y+=position[1];node.count++;nodes.set(key,node)}
    const edge=`${source}>${target}`;edges.set(edge,(edges.get(edge)||0)+1);
  }
  for(const row of shots){
    const key=`${row.actor}:${row._network_shooter_generation}`,position=point(row,'attack_normalized_source_position');
    const node=nodes.get(key)||{x:0,y:0,count:0};node.x+=position[0];node.y+=position[1];node.count++;nodes.set(key,node);
  }
  for(const [edge,count] of edges){
    const [source,target]=edge.split('>'),a=nodes.get(source),b=nodes.get(target),from=networkPlayers.get(source),to=networkPlayers.get(target);
    const keeper=from.is_goalkeeper||to.is_goalkeeper;
    const line=el('line',{x1:a.x/a.count,y1:a.y/a.count,x2:b.x/b.count,y2:b.y/b.count,stroke:keeper?'#facc15':teamColors[activeTeam],'stroke-width':Math.min(1.8,.25+.18*Math.sqrt(count))+(keeper ? .35 : 0),opacity:keeper?'.9':'.6'});
    line.appendChild(el('title',{},`#${from.player_id} (slot ${from.slot}, G${from.generation}) → #${to.player_id} (slot ${to.slot}, G${to.generation}): ${count} completed`));network.appendChild(line);
  }
  for(const row of shots){
    const key=`${row.actor}:${row._network_shooter_generation}`,node=nodes.get(key),end=point(row,'attack_normalized_terminal_position'),player=networkPlayers.get(key);
    const line=el('line',{class:'network-shot',x1:node.x/node.count,y1:node.y/node.count,x2:end[0],y2:end[1],stroke:'#fb923c','stroke-width':'.55',opacity:'.9'});
    line.appendChild(el('title',{},`Shot by #${player.player_id} · ${row.outcome} · frame ${row.terminal_frame}`));network.appendChild(line);
    network.appendChild(el('circle',{cx:end[0],cy:end[1],r:'.65',fill:'#fb923c'}));
  }
  for(const [key,node] of nodes){
    const player=networkPlayers.get(key),x=node.x/node.count,y=node.y/node.count,keeper=player.is_goalkeeper,group=el('g',{class:'network-node'});
    group.appendChild(el('title',{},`Player #${player.player_id} · slot ${player.slot} · G${player.generation}`));
    group.appendChild(el('circle',{cx:x,cy:y,r:keeper?'1.65':'1.35',fill:keeper?'#facc15':teamColors[activeTeam],stroke:'white','stroke-width':'.25'}));
    group.appendChild(el('text',{x,y:y+1},keeper?`GK ${player.player_id}`:String(player.player_id)));network.appendChild(group);
  }
  raisePitchLines(network);
  networkStatus.textContent=`Team ${activeTeam} · ${(cut*d.control_dt_s/60).toFixed(1)} min · ${shown.length} completed passes · ${shots.length} shots`;
};
networkTime.oninput=drawNetwork;
renderTeam();document.documentElement.classList.replace('no-js','js');
</script>
</html>"""


def _roster_html(report: dict) -> tuple[str, str]:
    roster = report.get("roster_summary")
    if not roster:
        return "Roster metadata was not captured for this match.", ""
    source = roster.get("provenance_note") or (
        "Starting formations and realized model attributes captured from the match roster. "
        "Formation anchors use the reference 105 × 68 m pitch. These values are model inputs, not observed player ratings."
    )
    profiles = roster["players"]
    formations = roster["formations"]
    sides = []
    for side in (0, 1):
        formation = next((row for row in formations if row["team"] == side), None)
        name = "Unspecified" if formation is None else str(formation["formation"])
        players = sorted(
            (row for row in profiles if row["team"] == side),
            key=lambda row: (row["start_frame"], row["slot"]),
        )
        colors = ("#38bdf8", "#fb7185")
        diagram = [
            '<rect x="-52.5" y="-34" width="105" height="68" fill="#173f31" stroke="white" stroke-width=".3"/>',
            '<line x1="0" y1="-34" x2="0" y2="34" stroke="white" stroke-width=".3"/>',
            '<circle cx="0" cy="0" r="9.15" fill="none" stroke="white" stroke-width=".3"/>',
        ]
        for player in players:
            if player["generation"] != 0:
                continue
            anchor = player["formation_anchor_reference_m"]
            x, y = float(anchor[0]), float(anchor[1])
            color = "#facc15" if player["is_goalkeeper"] else colors[side]
            label = escape(str(player["player_id"]))
            title = escape(str(player["role"]))
            diagram.append(
                f'<g><circle cx="{x}" cy="{y}" r="2.25" fill="{color}" stroke="white" stroke-width=".25"/>'
                f'<text x="{x}" y="{y+.65}" text-anchor="middle" fill="#08131f" font-size="1.8" font-weight="bold">{label}</text>'
                f'<title>Player {label} · {title}</title></g>'
            )
        header = (
            "<tr><th>ID</th><th>Role</th><th>Slot / gen</th><th>From frame</th>"
            "<th>Max speed<br>m/s</th><th>Height<br>m</th><th>Reach<br>m</th>"
            "<th>Ball control</th><th>Endurance</th></tr>"
        )
        rows = []
        for player in players:
            rows.append(
                "<tr>"
                f"<td>{int(player['player_id'])}</td><td>{escape(str(player['role']))}</td>"
                f"<td>{int(player['slot'])} / {int(player['generation'])}</td>"
                f"<td>{int(player['start_frame'])}</td>"
                f"<td>{float(player['max_speed_mps']):.2f}</td>"
                f"<td>{float(player['height_m']):.2f}</td>"
                f"<td>{float(player['reach_height_m']):.2f}</td>"
                f"<td>{float(player['ball_control']):.3f}</td>"
                f"<td>{float(player['endurance_factor']):.3f}</td></tr>"
            )
        sides.append(
            f'<div class="roster-side"><h3>Team {side} · {escape(name)}</h3>'
            f'<svg viewBox="-52.5 -34 105 68" role="img" aria-label="Team {side} starting formation">{"".join(diagram)}</svg>'
            f'<div class="roster-table-wrap"><table class="team-table roster-table">{header}{"".join(rows)}</table></div></div>'
        )
    return escape(str(source)), "".join(sides)


def _static_report_content(
    report: dict, *, include_network_snapshots: bool = True
) -> dict[str, str]:
    """Fill the first view so report readers do not need script execution."""
    team = report["tactical_summary"]
    spatial = report["spatial"]
    x_edges, y_edges = spatial["x_edges_m"], spatial["y_edges_m"]
    xmin, xmax = x_edges[0], x_edges[-1]
    ymin, ymax = y_edges[0], y_edges[-1]

    def fmt(value: object) -> str:
        return escape(str(value))

    def pct(num: float, den: float) -> str:
        return f"{100 * num / den:.1f}%" if den else "—"

    def position(value: list[float] | None) -> str:
        return f"({value[0]:.1f}, {value[1]:.1f}) m" if value else "—"

    def length(value: float | None) -> str:
        return f"{value:.1f} m" if value is not None else "—"

    def table_row(*values: object, header: bool = False) -> str:
        tag = "th" if header else "td"
        return "<tr>" + "".join(f"<{tag}>{fmt(v)}</{tag}>" for v in values) + "</tr>"

    score = report["final_score"]
    possession = report["possession_frames"]
    kpis = [
        ("Score", f"{score[0]}–{score[1]}"),
        ("Possession", f"{pct(possession[0], sum(possession))} / {pct(possession[1], sum(possession))}"),
        ("Open-play pass", f"{team[0]['open_play_pass_completions']}/{team[0]['open_play_pass_attempts']} · {team[1]['open_play_pass_completions']}/{team[1]['open_play_pass_attempts']}"),
        ("Shots", f"{team[0]['shot_attempts']}–{team[1]['shot_attempts']}"),
        ("On target", f"{team[0]['shots_on_target']}–{team[1]['shots_on_target']}"),
    ]
    kpi_html = "".join(f"<div class='kpi'><b>{fmt(value)}</b><span>{fmt(label)}</span></div>" for label, value in kpis)
    metrics = [
        ("Open-play pass completion", lambda t: pct(t["open_play_pass_completions"], t["open_play_pass_attempts"])),
        ("Restart pass", lambda t: f"{t['restart_pass_completions']}/{t['restart_pass_attempts']}"),
        ("Attacking-third completed", lambda t: t["attacking_third_pass_completions"]),
        ("Box entries completed", lambda t: t["box_entry_pass_completions"]),
        ("Shot attempts", lambda t: t["shot_attempts"]),
        ("Releases near goal", lambda t: t["shot_releases_near_goal"]),
        ("Shots on target", lambda t: t["shots_on_target"]),
        ("Mean shot distance", lambda t: length(t["mean_shot_attempt_distance_m"])),
    ]
    teams_html = table_row("Metric", "Team 0", "Team 1", header=True)
    teams_html += "".join(table_row(label, fn(team[0]), fn(team[1])) for label, fn in metrics)
    shapes = report.get("team_shape_summary") or [{}, {}]
    shape_metrics = [
        ("Mean team width", "mean_width_m"),
        ("Mean team depth", "mean_depth_m"),
        ("Mean pairwise distance", "mean_pairwise_distance_m"),
        ("Width · in possession", "in_possession_mean_width_m"),
        ("Width · out of possession", "out_of_possession_mean_width_m"),
        ("Mean centroid", "mean_centroid_m"),
        ("Depth · in possession", "in_possession_mean_depth_m"),
        ("Depth · out of possession", "out_of_possession_mean_depth_m"),
        ("Stretch index", "mean_stretch_index_m"),
        ("Longitudinal max gap", "mean_longitudinal_max_gap_m"),
        ("Opponent centroid distance", "mean_opponent_centroid_distance_m"),
        ("Ball–centroid distance", "mean_ball_centroid_distance_m"),
    ]
    for label, key in shape_metrics:
        value = position if key == "mean_centroid_m" else length
        teams_html += table_row(label, *(value(s.get(key)) for s in shapes))
    teams_html += table_row("Pairwise P10 / P50 / P90", *(" / ".join(length(s.get(f"pairwise_distance_{p}_m")) for p in ("p10", "p50", "p90")) for s in shapes))
    reference = report.get("reference_context", {}).get("dfl_attacking_third_pass")
    if reference:
        dirs = reference["direction_share"]
        reference_html = fmt(f"DFL descriptive reference · attacking-third completion {pct(reference['completion'], 1)} · direction backward/lateral/forward {pct(dirs['backward'], 1)} / {pct(dirs['lateral'], 1)} / {pct(dirs['forward'], 1)} · {reference['comparability']}")
    else:
        reference_html = "DFL descriptive reference unavailable"
    roster_source, roster_content = _roster_html(report)

    timeline = report["possession_timeline"]
    max_time = max((point["time_s"] for point in timeline), default=1) or 1
    possession_svg = '<rect x="40" y="15" width="540" height="190" fill="none" stroke="#94a3b8"/>'
    for side, color in enumerate(("#38bdf8", "#fb7185")):
        points = " ".join(f"{40 + 540 * point['time_s'] / max_time:.2f},{205 - 190 * point['share'][side]:.2f}" for point in timeline)
        possession_svg += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'

    def pitch() -> str:
        hx, hy = max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax))
        box_y, box_depth = min(20.16, hy), min(16.5, hx)
        return (f'<rect class="pitch" x="{xmin}" y="{ymin}" width="{xmax-xmin}" height="{ymax-ymin}"/>'
                f'<line class="pitch" x1="0" y1="{ymin}" x2="0" y2="{ymax}"/>'
                f'<circle class="pitch" cx="0" cy="0" r="{min(9.15,hx,hy)}"/>'
                f'<rect class="pitch" x="{hx-box_depth}" y="{-box_y}" width="{box_depth}" height="{2*box_y}"/>'
                f'<rect class="pitch" x="{-hx}" y="{-box_y}" width="{box_depth}" height="{2*box_y}"/>')

    heat_groups, pass_groups, shot_groups = [], [], []
    shot_colors = {"goal": "#4ade80", "saved": "#38bdf8", "blocked": "#c084fc", "woodwork": "#facc15", "off_target": "#fb7185", "unresolved": "#94a3b8"}
    for side, team_color in enumerate(("#38bdf8", "#fb7185")):
        density = spatial["team_occupancy_seconds"][side]
        peak = max((max(row) for row in density if row), default=0)
        heat = []
        for yi, row in enumerate(density):
            for xi, value in enumerate(row):
                if value > 0 and peak:
                    opacity = .12 + .75 * sqrt(value / peak)
                    heat.append(f'<rect x="{x_edges[xi]}" y="{y_edges[yi]}" width="{x_edges[xi+1]-x_edges[xi]}" height="{y_edges[yi+1]-y_edges[yi]}" fill="{team_color}" opacity="{opacity:.3f}"/>')
        heat_groups.append(f'<g class="team-detail-{side}">{"".join(heat)}</g>')
        pass_lines = []
        for row in report["pass_map"]:
            a, b = row.get("attack_normalized_source_position"), row.get("attack_normalized_terminal_position")
            if row.get("team") == side and a and b:
                dash = "" if row.get("completed") else ' stroke-dasharray="1.2 1"'
                height = row.get("_pass_height_class", "unknown")
                color = "#facc15" if height == "elevated" else "#94a3b8" if height == "unknown" else team_color
                marker = f"team-{side}" if height == "low" else height
                midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                pass_lines.append(f'<polyline points="{a[0]},{a[1]} {midpoint[0]},{midpoint[1]} {b[0]},{b[1]}" fill="none" stroke="{color}" stroke-width="{.5 if row.get("completed") else .35}" opacity="{.78 if row.get("completed") else .42}" marker-mid="url(#pass-arrow-{marker})"{dash}><title>{"Completed" if row.get("completed") else "Uncompleted"} pass · {fmt(height)} height proxy</title></polyline>')
        pass_groups.append(f'<g class="team-detail-{side}">{"".join(pass_lines)}</g>')
        shots = []
        for row in report["shot_map"]:
            point = row.get("attack_normalized_source_position")
            if row.get("team") == side and row.get("_goal_directed") and point:
                shots.append(f'<circle cx="{point[0]}" cy="{point[1]}" r="1.15" fill="{shot_colors.get(row.get("outcome"), "#fff")}" stroke="{team_color}" stroke-width=".25"><title>{fmt(row.get("outcome"))} · frame {fmt(row.get("source_frame"))}</title></circle>')
        shot_groups.append(f'<g class="team-detail-{side}">{"".join(shots)}</g>')

    network_players = {
        (int(player["slot"]), int(player["generation"])): player
        for player in report["player_generations"]
    }

    def network_snapshot(cut: int, index: int, side: int) -> tuple[str, int, int]:
        nodes: dict[tuple[int, int], list[float]] = {}
        edges: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        shot_links = []
        for row in report["pass_map"]:
            source, target = row.get("actor"), row.get("terminal_actor")
            a, b = row.get("attack_normalized_source_position"), row.get("attack_normalized_terminal_position")
            if row.get("team") != side or not row.get("completed") or row.get("terminal_frame", 0) > cut or source is None or target is None or target < 0 or source == target or not a or not b or "_network_source_generation" not in row or "_network_target_generation" not in row:
                continue
            source = int(source), int(row["_network_source_generation"])
            target = int(target), int(row["_network_target_generation"])
            for player, point in ((source, a), (target, b)):
                node = nodes.setdefault(player, [0.0, 0.0, 0])
                node[0] += point[0]
                node[1] += point[1]
                node[2] += 1
            key = (source, target)
            edges[key] = edges.get(key, 0) + 1
        for row in report["shot_map"]:
            if row.get("team") != side or row.get("terminal_frame", 0) > cut or "_network_shooter_generation" not in row:
                continue
            source = int(row["actor"]), int(row["_network_shooter_generation"])
            start = row["attack_normalized_source_position"]
            end = row["attack_normalized_terminal_position"]
            node = nodes.setdefault(source, [0.0, 0.0, 0])
            node[0] += start[0]
            node[1] += start[1]
            node[2] += 1
            shot_links.append((source, end, row))
        marks = []
        for (source, target), count in edges.items():
            a, b = nodes[source], nodes[target]
            keeper = bool(network_players[source]["is_goalkeeper"] or network_players[target]["is_goalkeeper"])
            color = "#facc15" if keeper else ("#38bdf8" if side == 0 else "#fb7185")
            marks.append(f'<line x1="{a[0]/a[2]}" y1="{a[1]/a[2]}" x2="{b[0]/b[2]}" y2="{b[1]/b[2]}" stroke="{color}" stroke-width="{min(1.8, .25 + .18 * sqrt(count))}" opacity=".7"><title>#{network_players[source]["player_id"]} (slot {source[0]}, G{source[1]}) → #{network_players[target]["player_id"]} (slot {target[0]}, G{target[1]}): {count} completed</title></line>')
        for source, end, row in shot_links:
            node = nodes[source]
            marks.append(f'<line class="network-shot" x1="{node[0]/node[2]}" y1="{node[1]/node[2]}" x2="{end[0]}" y2="{end[1]}" stroke="#fb923c" stroke-width=".55" opacity=".9"><title>Shot by #{network_players[source]["player_id"]} · {fmt(row["outcome"])} · frame {int(row["terminal_frame"])}</title></line><circle cx="{end[0]}" cy="{end[1]}" r=".65" fill="#fb923c"/>')
        for player, point in nodes.items():
            keeper = bool(network_players[player]["is_goalkeeper"])
            x, y = point[0] / point[2], point[1] / point[2]
            color = "#facc15" if keeper else ("#38bdf8" if side == 0 else "#fb7185")
            label = f"GK {network_players[player]['player_id']}" if keeper else network_players[player]["player_id"]
            marks.append(f'<g class="network-node"><title>Player #{network_players[player]["player_id"]} · slot {player[0]} · G{player[1]}</title><circle cx="{x}" cy="{y}" r="{1.65 if keeper else 1.35}" fill="{color}" stroke="white" stroke-width=".25"/><text x="{x}" y="{y+1}">{label}</text></g>')
        return f'<g class="network-snapshot snapshot-{index} team-detail-{side}">{"".join(marks)}</g>', sum(edges.values()), len(shot_links)

    dt = report["control_dt_s"]
    frames = report["frames"]
    snapshots = (
        [(f"{minute} min", round(minute * 60 / dt)) for minute in range(0, int(frames * dt // 60) + 1)]
        if include_network_snapshots else []
    )
    if include_network_snapshots and (not snapshots or snapshots[-1][1] != frames):
        snapshots.append((f"Full · {frames * dt / 60:.1f} min", frames))
    snapshot_markup = []
    snapshot_counts = [[], []]
    snapshot_shot_counts = [[], []]
    for index, (_, cut) in enumerate(snapshots):
        for side in (0, 1):
            markup, count, shot_count = network_snapshot(cut, index, side)
            snapshot_markup.append(markup)
            snapshot_counts[side].append(count)
            snapshot_shot_counts[side].append(shot_count)
    static_inputs = '<input class="network-static" type="checkbox" id="network-static-play" hidden>' if include_network_snapshots else ""
    static_inputs += "".join(f'<input class="network-static" type="radio" name="network-snapshot" id="network-static-{index}" hidden{" checked" if index == len(snapshots)-1 else ""}>' for index in range(len(snapshots)))
    static_labels = "".join(f'<label for="network-static-{index}" title="{fmt(label)}" aria-label="Show network at {fmt(label)}"></label>' for index, (label, _) in enumerate(snapshots))
    static_readout = "".join(f'<span class="network-time-label time-{index}">{fmt(label)}</span>' for index, (label, _) in enumerate(snapshots))
    fraction = 100 / len(snapshots) if snapshots else 0
    step_seconds = .16
    static_css = f'@keyframes network-cycle{{0%,{fraction - .001:.4f}%{{opacity:1}}{fraction:.4f}%,100%{{opacity:0}}}}@keyframes network-thumb-sweep{{from{{left:0}}to{{left:100%}}}}#network-static-play:checked ~ #network .network-snapshot{{display:inline;opacity:0;animation:network-cycle {len(snapshots) * step_seconds:.2f}s linear infinite}}#network-static-play:checked ~ .network-static-timeline .network-thumb{{animation:network-thumb-sweep {len(snapshots) * step_seconds:.2f}s linear infinite}}#network-static-play:checked ~ .network-static-timeline .network-time-label{{display:none}}#network-static-play:checked ~ .network-static-timeline .network-playing{{display:inline}}' if snapshots else ""
    static_css += "".join(f'#network-static-{index}:checked ~ #network .snapshot-{index}{{display:inline}}#network-static-{index}:checked ~ .network-static-timeline .time-{index}{{display:inline}}#network-static-{index}:checked ~ .network-static-timeline .network-thumb{{left:{100 * index / max(1, len(snapshots)-1):.4f}%}}#network-static-play:checked ~ #network .snapshot-{index}{{animation-delay:-{index * step_seconds:.2f}s}}' for index in range(len(snapshots)))

    def bars(items: list[tuple[str, float, str]]) -> str:
        peak_value = max((value for _, value, _ in items), default=1) or 1
        return "".join(f'<div class="bar"><span>{fmt(label)}</span><span><i style="width:{100 * value / peak_value:.2f}%"></i></span><b>{fmt(display)}</b></div>' for label, value, display in items)

    window_html, terminal_html, direction_html = [], [], []
    player_html, phase_html = [], []
    for side in (0, 1):
        windows = []
        for index, window in enumerate(team[side]["pass_completion_by_15m"]):
            attempts, complete = window["attempts"], window["completions"]
            windows.append((f"{index * 15}–{(index + 1) * 15}m", 100 * complete / attempts if attempts else 0, f"{complete}/{attempts} ({pct(complete, attempts)})" if attempts else "—"))
        window_html.append(f'<div class="team-detail-{side}">{bars(windows)}</div>')
        terminal_html.append(f'<div class="team-detail-{side}">{bars([(k, v, str(v)) for k, v in team[side]["pass_terminal_kinds"].items()])}</div>')
        direction_html.append(f'<div class="team-detail-{side}">{bars([(k, v, str(v)) for k, v in team[side]["open_play_direction_five"].items()])}</div>')
        players = sorted((p for p in report["player_action_credit"] if p["team"] == side), key=lambda p: -(p.get("controlled_possession_seconds") or 0))
        player_rows = table_row("Player", "Active", "Possession", "Mean absolute position", "Mean teammate distance", "Nearest teammate", "Pass", "Shot", "Distance", header=True)
        phase_rows = table_row("Player", "Team in possession", "Opponent in possession", "Position change", header=True)
        for player in players:
            label = f"#{player['player_id']}" + (f" · G{player['generation']}" if player.get("generation") else "")
            player_rows += table_row(label, f"{player['active_seconds']:.1f} s", f"{player['controlled_possession_seconds']:.1f} s", position(player.get("mean_absolute_position_m")), length(player.get("mean_teammate_distance_m")), length(player.get("mean_nearest_teammate_distance_m")), f"{player['pass_completions']}/{player['pass_attempts']}", player["shot_attempts"], f"{player['distance_m']/1000:.2f} km")
            a, b = player.get("mean_in_possession_position_m"), player.get("mean_out_of_possession_position_m")
            delta = length(sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)) if a and b else "—"
            phase_rows += table_row(label, position(a), position(b), delta)
        player_html.append(f'<tbody class="team-detail-{side}">{player_rows}</tbody>')
        phase_html.append(f'<tbody class="team-detail-{side}">{phase_rows}</tbody>')
    return {
        '<div id="subtitle" class="muted"></div>': f'<div id="subtitle" class="muted">{fmt(report["frames"])} frames · schema {fmt(report["schema"])}</div>',
        '<div id="kpis" class="kpis"></div>': f'<div id="kpis" class="kpis">{kpi_html}</div>',
        '<table id="teams" class="team-table"></table>': f'<table id="teams" class="team-table">{teams_html}</table>',
        '<p id="reference"></p>': f'<p id="reference">{reference_html}</p>',
        '<p id="roster-source"></p>': f'<p id="roster-source">{roster_source}</p>',
        '<div id="roster-content" class="roster-layout"></div>': f'<div id="roster-content" class="roster-layout">{roster_content}</div>',
        '<svg id="possession" viewBox="0 0 600 240"></svg>': f'<svg id="possession" viewBox="0 0 600 240">{possession_svg}</svg>',
        '<table id="events" class="event-table"></table>': '<table id="events" class="event-table">' + "".join(table_row(k, v) for k, v in report["event_counts"].items()) + '</table>',
        '<svg id="heat" viewBox="-52.5 -34 105 68"></svg>': f'<svg id="heat" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}">{"".join(heat_groups)}{pitch()}</svg>',
        '<svg id="passes" viewBox="-52.5 -34 105 68"></svg>': f'<svg id="passes" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}">{_pass_marker_defs()}{"".join(pass_groups)}{pitch()}</svg>',
        '<svg id="shots" viewBox="-52.5 -34 105 68"></svg>': f'<svg id="shots" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}">{"".join(shot_groups)}{pitch()}</svg>',
        '<div id="windows" class="bars"></div>': f'<div id="windows" class="bars">{"".join(window_html)}</div>',
        '<div id="terminals" class="bars"></div>': f'<div id="terminals" class="bars">{"".join(terminal_html)}</div>',
        '<div id="directions" class="bars"></div>': f'<div id="directions" class="bars">{"".join(direction_html)}</div>',
        '<table id="players" class="team-table small"></table>': f'<table id="players" class="team-table small">{"".join(player_html)}</table>',
        '<table id="player-phases" class="team-table small"></table>': f'<table id="player-phases" class="team-table small">{"".join(phase_html)}</table>',
        '<svg id="network" viewBox="-52.5 -34 105 68"></svg>': f'<svg id="network" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}">{pitch()}{"".join(snapshot_markup)}</svg>',
        '<span id="network-status"></span>': f'<span id="network-status">Team 0 · {frames * dt / 60:.1f} min · {snapshot_counts[0][-1]} completed passes · {snapshot_shot_counts[0][-1]} shots</span>' if snapshots else '<span id="network-status"></span>',
        '__NETWORK_STATIC_INPUTS__': static_inputs,
        '__NETWORK_STATIC_LABELS__': static_labels,
        '__NETWORK_STATIC_READOUT__': static_readout,
        '__NETWORK_STATIC_CSS__': static_css,
    }


def render_match_report_html(
    report,
    path: str | Path,
    *,
    title: str | None = None,
    network_videos: tuple[str | Path, str | Path] | None = None,
) -> Path:
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ValueError("title must be a non-empty string")
    document_title = "MESSI Match Report" if title is None else title.strip()
    heading = "MESSI Match Report" if title is None else title.strip()
    report_payload = report.as_dict()
    report_payload["pass_map"] = network_passes_with_generations(report_payload)
    for row in report_payload["pass_map"]:
        row["_pass_height_class"] = _pass_height_class(row)
    report_payload["shot_map"] = network_shots_with_generations(report_payload)
    shot_map = report_payload.get("shot_map", ())
    if shot_map:
        x_edges = report_payload["spatial"]["x_edges_m"]
        goal_line_x = float(x_edges[-1])
        for row in shot_map:
            row["_goal_directed"] = bool(
                is_goal_directed_shot(row, goal_line_x=goal_line_x)
            )
    payload = strict_json_dumps(
        report_payload, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = _TEMPLATE.replace("__DOCUMENT_TITLE__", escape(document_title)).replace(
        "__HEADING__", escape(heading)
    )
    html = html.replace(
        "__PASS_MARKERS_JS__", strict_json_dumps(_pass_marker_defs()), 1
    )
    for empty, filled in _static_report_content(
        report_payload, include_network_snapshots=network_videos is None
    ).items():
        html = html.replace(empty, filled, 1)
    video_html = ""
    if network_videos is not None:
        if len(network_videos) != 2:
            raise ValueError("network_videos must provide one MP4 for each team")
        videos = []
        for side, source in enumerate(network_videos):
            data = Path(source).read_bytes()
            if not data:
                raise ValueError("network video must not be empty")
            encoded = b64encode(data).decode("ascii")
            videos.append(f'<video class="team-detail-{side}" controls playsinline preload="metadata" aria-label="Team {side} passing network" src="data:video/mp4;base64,{encoded}"></video>')
        video_html = '<div class="network-video-enabled"><p>Drag the playback bar to seek. One video second represents one match minute.</p>' + "".join(videos) + '</div>'
    html = html.replace("__NETWORK_VIDEO__", video_html, 1)
    destination.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
    return destination
