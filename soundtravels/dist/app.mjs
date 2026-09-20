const DAY = 24 * 60 * 60 * 1000;
const palette = ['#dcff8f','#5bd4c1','#f5b55b','#ec7f78','#82aaff','#bd91e2','#ef78ad','#6ed17c','#ffd76b','#9bc5bb'];
const state = {data:null,videos:[],audio:null,mode:'moment',day:0,startDay:0,endDay:1,dayCount:0,start:0,end:0,hovered:null,pinned:null,screen:{hashtag:[],topic:[]}};
const $ = id => document.getElementById(id);
const canvas = {hashtag:$('hashtag-canvas'),topic:$('topic-canvas')};
const context = {hashtag:canvas.hashtag.getContext('2d'),topic:canvas.topic.getContext('2d')};
const mapFields = {hashtag:{x:'hx',y:'hy',family:'hf'},topic:{x:'tx',y:'ty',family:'tf'}};

function unassigned(id){return Number(id)<0}
function colorFor(id){if(unassigned(id))return '#b5bdba';let hash=0;for(const c of String(id))hash=((hash<<5)-hash+c.charCodeAt(0))|0;return palette[Math.abs(hash)%palette.length]}
function escapeHtml(value){return String(value).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function formatDate(ms,year=true){return new Intl.DateTimeFormat('en-US',{month:'short',day:'numeric',...(year?{year:'numeric'}:{}),timeZone:'UTC'}).format(new Date(ms))}
function dateAt(day){return state.start+day*DAY}
function resize(target,ctx){const ratio=Math.max(1,devicePixelRatio||1),rect=target.getBoundingClientRect();target.width=Math.round(rect.width*ratio);target.height=Math.round(rect.height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);return rect}
function extent(key){let min=Infinity,max=-Infinity;for(const v of state.videos){min=Math.min(min,v[key]);max=Math.max(max,v[key])}return min===max?[min-1,max+1]:[min,max]}
function scales(kind,rect){const f=mapFields[kind],[minX,maxX]=extent(f.x),[minY,maxY]=extent(f.y),pad=22;return{x:n=>pad+(n-minX)/(maxX-minX)*(rect.width-pad*2),y:n=>rect.height-pad-(n-minY)/(maxY-minY)*(rect.height-pad*2)}}
function selectedVideos(){return state.videos.filter(v=>v.music_id===state.audio)}
function inActiveTime(v){return state.mode==='moment'?v._day===state.day:v._day>=state.startDay&&v._day<=state.endDay}
function activeSelected(){return state.videos.filter(v=>v.music_id===state.audio&&inActiveTime(v))}

function drawMap(kind){
  const target=canvas[kind],ctx=context[kind],rect=resize(target,ctx),f=mapFields[kind],scale=scales(kind,rect),focus=state.pinned||state.hovered;
  ctx.clearRect(0,0,rect.width,rect.height);state.screen[kind]=[];
  ctx.fillStyle='rgba(178,194,189,.13)';
  for(const v of state.videos){ctx.beginPath();ctx.arc(scale.x(v[f.x]),scale.y(v[f.y]),1.25,0,Math.PI*2);ctx.fill()}
  ctx.fillStyle='rgba(190,207,201,.25)';
  for(const v of state.videos){if(!inActiveTime(v)||v.music_id===state.audio)continue;ctx.beginPath();ctx.arc(scale.x(v[f.x]),scale.y(v[f.y]),1.8,0,Math.PI*2);ctx.fill()}
  for(const v of activeSelected()){
    const x=scale.x(v[f.x]),y=scale.y(v[f.y]),isMuted=unassigned(v[f.family]);state.screen[kind].push({x,y,video:v});ctx.globalAlpha=isMuted ? .48 : .94;ctx.fillStyle=colorFor(v[f.family]);ctx.beginPath();ctx.arc(x,y,isMuted ? 4.2 : 5.1,0,Math.PI*2);ctx.fill();
    if(focus===v.video_id){ctx.globalAlpha=1;ctx.strokeStyle='#fff';ctx.lineWidth=2.2;ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.stroke()}
  }
  ctx.globalAlpha=1;
}

function dailyCounts(){const counts=Array(state.dayCount).fill(0);for(const v of state.videos)if(v.music_id===state.audio)counts[v._day]++;return counts}
function drawUsage(){
  const target=$('usage-chart'),ctx=target.getContext('2d'),rect=resize(target,ctx),counts=dailyCounts(),max=Math.max(1,...counts),left=7,right=7,top=9,bottom=7,w=(rect.width-left-right)/counts.length;
  ctx.clearRect(0,0,rect.width,rect.height);ctx.strokeStyle='rgba(151,171,164,.17)';ctx.lineWidth=1;for(let i=0;i<3;i++){const y=top+i*(rect.height-top-bottom)/2;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(rect.width-right,y);ctx.stroke()}
  counts.forEach((n,i)=>{const active=state.mode==='moment'?i===state.day:i>=state.startDay&&i<=state.endDay;const h=n/max*(rect.height-top-bottom);ctx.fillStyle=active?'#dcff8f':'#536c65';ctx.fillRect(left+i*w,rect.height-bottom-h,Math.max(1,w-.6),h)});
  const positions=state.mode==='moment'?[state.day]:[state.startDay,state.endDay];ctx.strokeStyle='#fff';ctx.lineWidth=1;for(const day of positions){const x=left+(day+.5)*w;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,rect.height);ctx.stroke()}
}

function familyName(kind,id){return state.data.families[kind][id]||'No assigned family'}
function familyCounts(){const rows=selectedVideos();return{uses:rows.length,hashtags:new Set(rows.filter(v=>!unassigned(v.hf)).map(v=>v.hf)).size,topics:new Set(rows.filter(v=>!unassigned(v.tf)).map(v=>v.tf)).size}}
function buildLegend(kind){
  const f=mapFields[kind],counts=new Map();for(const v of selectedVideos())if(!unassigned(v[f.family]))counts.set(v[f.family],(counts.get(v[f.family])||0)+1);
  const items=[...counts].sort((a,b)=>b[1]-a[1]).slice(0,4);let html=items.map(([id])=>`<span class="legend-item"><i style="background:${colorFor(id)}"></i>${escapeHtml(familyName(kind,id))}</span>`).join('');html+=`<span class="legend-item unassigned"><i style="background:#b5bdba"></i>Unassigned</span>`;$(kind+'-legend').innerHTML=html;
}
function updateText(){
  const stats=familyCounts();$('stat-uses').textContent=stats.uses.toLocaleString();$('stat-hashtags').textContent=stats.hashtags.toLocaleString();$('stat-topics').textContent=stats.topics.toLocaleString();
  $('date-label').textContent=state.mode==='moment'?formatDate(dateAt(state.day),true):`${formatDate(dateAt(state.startDay),false)} — ${formatDate(dateAt(state.endDay),true)}`;
}
function render(){drawMap('hashtag');drawMap('topic');drawUsage();updateText()}

function tooltipHtml(video){
  const rows=[`<b>Video ${escapeHtml(video.video_id.slice(-8))}</b>`,`<span>${formatDate(video.ts*1000,true)}</span>`];
  rows.push(unassigned(video.hf)?'<span class="muted-family">Hashtag: No assigned family</span>':`<span>Hashtag: ${escapeHtml(familyName('hashtag',video.hf))}</span>`);
  rows.push(unassigned(video.tf)?'<span class="muted-family">Topic: No assigned family</span>':`<span>Topic: ${escapeHtml(familyName('topic',video.tf))}</span>`);
  return rows.join('');
}
function bindMap(kind){
  const target=canvas[kind],tip=$(kind+'-tooltip');
  target.addEventListener('pointermove',event=>{const rect=target.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=null,distance=100;for(const point of state.screen[kind]){const d=(point.x-x)**2+(point.y-y)**2;if(d<distance){distance=d;best=point}}const next=best?.video.video_id||null;if(best){tip.hidden=false;tip.innerHTML=tooltipHtml(best.video);tip.style.left=Math.min(rect.width-250,x+12)+'px';tip.style.top=Math.max(8,y-18)+'px'}else tip.hidden=true;if(next!==state.hovered){state.hovered=next;drawMap('hashtag');drawMap('topic')}});
  target.addEventListener('pointerleave',()=>{state.hovered=null;tip.hidden=true;drawMap('hashtag');drawMap('topic')});
  target.addEventListener('click',()=>{state.pinned=state.hovered===state.pinned?null:state.hovered;render()});
}

function setMode(mode){state.mode=mode;state.hovered=null;state.pinned=null;document.querySelectorAll('.mode').forEach(button=>{const active=button.dataset.mode===mode;button.classList.toggle('active',active);button.setAttribute('aria-pressed',active)});$('moment-control').hidden=mode!=='moment';$('window-control').hidden=mode!=='window';$('scrub-instruction').textContent=mode==='moment'?'DRAG TO SCRUB OBSERVED DAYS':'MOVE BOTH ENDS OF THE OBSERVED WINDOW';render()}
function configureAudio(){
  const meta=state.data.audios.find(audio=>audio.music_id===state.audio),player=$('audio-player'),button=$('play-audio'),status=$('audio-status'),source=$('source-video');player.pause();try{player.currentTime=0}catch{}button.textContent='▶ Play Audio';
  if(meta.audio_file){player.src=meta.audio_file;button.disabled=false;status.textContent='Observed audio';button.title=`Play ${meta.audio_file}`}else{player.removeAttribute('src');player.load();button.disabled=true;status.textContent=`Add ${meta.expected_audio_file}`;button.title=`No playable file is present. Add ${meta.expected_audio_file} to enable playback.`}
    source.hidden=!meta.source_video_url;if(meta.source_video_url){source.href=meta.source_video_url;source.title=`Representative dataset video ${meta.source_video_id}`}else{source.removeAttribute('href')}
}
function chooseAudio(id,reset=true){
  state.audio=id;state.hovered=null;state.pinned=null;const rows=selectedVideos(),days=rows.map(v=>v._day).sort((a,b)=>a-b),counts=dailyCounts();if(reset){state.day=counts.indexOf(Math.max(...counts));state.startDay=days[0];state.endDay=days[days.length-1];$('time-slider').value=state.day;$('start-slider').value=state.startDay;$('end-slider').value=state.endDay}configureAudio();buildLegend('hashtag');buildLegend('topic');render();
}
function load(data){
  state.data=data;state.start=Date.parse(data.window.start);state.end=Date.parse(data.window.end);state.dayCount=Math.floor((state.end-state.start)/DAY)+1;state.videos=data.videos.map(v=>({...v,_day:Math.min(state.dayCount-1,Math.max(0,Math.floor((v.ts*1000-state.start)/DAY)))}));
  const select=$('audio-select');select.innerHTML=data.audios.map(a=>`<option value="${escapeHtml(a.music_id)}">${escapeHtml(a.label)}</option>`).join('');select.value=data.defaultAudioId;for(const id of ['time-slider','start-slider','end-slider'])$(id).max=state.dayCount-1;$('axis-start').textContent=formatDate(state.start,true).toUpperCase();$('axis-end').textContent=formatDate(state.end,true).toUpperCase();const windowText=`${formatDate(state.start,true).toUpperCase()} → ${formatDate(state.end,true).toUpperCase()}`;$('header-window').textContent=windowText;$('footer-window').textContent=windowText;chooseAudio(data.defaultAudioId);
}

document.querySelectorAll('.mode').forEach(button=>button.addEventListener('click',()=>setMode(button.dataset.mode)));
$('audio-select').addEventListener('change',event=>chooseAudio(event.target.value));
$('play-audio').addEventListener('click',async()=>{const player=$('audio-player'),button=$('play-audio'),status=$('audio-status');if(player.paused){try{await player.play();button.textContent='❚❚ Pause Audio'}catch{status.textContent='Unable to play this audio file'}}else{player.pause();button.textContent='▶ Play Audio'}});
$('audio-player').addEventListener('ended',()=>{$('play-audio').textContent='▶ Play Audio'});
$('time-slider').addEventListener('input',event=>{state.day=Number(event.target.value);state.hovered=null;state.pinned=null;render()});
$('start-slider').addEventListener('input',event=>{state.startDay=Math.min(Number(event.target.value),state.endDay-1);event.target.value=state.startDay;state.hovered=null;state.pinned=null;render()});
$('end-slider').addEventListener('input',event=>{state.endDay=Math.max(Number(event.target.value),state.startDay+1);event.target.value=state.endDay;state.hovered=null;state.pinned=null;render()});
bindMap('hashtag');bindMap('topic');addEventListener('resize',render);
const sections=[...document.querySelectorAll('main section[id]')],links=[...document.querySelectorAll('nav a')];addEventListener('scroll',()=>{let current=null;for(const section of sections)if(section.getBoundingClientRect().top<innerHeight*.45)current=section.id;for(const link of links)link.toggleAttribute('aria-current',link.getAttribute('href')===`#${current}`)},{passive:true});
fetch('./data/umap-data.json').then(response=>{if(!response.ok)throw new Error('Observed data unavailable');return response.json()}).then(load).catch(error=>document.querySelector('.atlas').insertAdjacentHTML('afterbegin',`<p class="load-error">${escapeHtml(error.message)}</p>`));
