// Results copied from the notebook outputs (confusion matrices, 1,354 test scans of 4,513 images).
const CLASSES=[{n:"COVID-19",c:"--covid"},{n:"Pneumonia",c:"--pneu"},{n:"Normal",c:"--norm"}];
const DATA={total:4513,models:{
  "SVM":{cm:[[505,0,0],[0,218,0],[0,0,631]],note:"Linear kernel."},
  "Random Forest":{cm:[[503,0,2],[0,216,2],[0,0,631]],note:"100 trees."},
  "Logistic Regression":{cm:[[505,0,0],[0,217,1],[0,0,631]],note:"max_iter=1000."}}};
const $=id=>document.getElementById(id),pct=x=>(x*100).toFixed(2)+"%";
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const col=l=>`var(${(CLASSES.find(c=>c.n===l)||CLASSES[0]).c})`;
const API=(window.CT_API||"").replace(/\/$/,"")||(location.protocol==="file:"?"http://localhost:5173":"");
const SPACE=(window.CT_SPACE||"").trim();
const GRADIO_CLIENT_URL="https://cdn.jsdelivr.net/npm/@gradio/client@2.7.1/dist/index.min.js";
let gradioClientPromise=null,gradioHandleFile=null;

function stats(cm){
  const tot=cm.flat().reduce((a,b)=>a+b,0),per=cm.map((r,i)=>{const tp=r[i],sup=r.reduce((a,b)=>a+b,0),pp=cm.reduce((a,x)=>a+x[i],0),p=pp?tp/pp:0,rc=sup?tp/sup:0;return{p,r:rc,f:p+rc?2*p*rc/(p+rc):0,sup}});
  const w=k=>per.reduce((a,x)=>a+x[k]*x.sup,0)/tot,ok=cm.reduce((a,r,i)=>a+r[i],0);
  return{per,tot,acc:ok/tot,p:w("p"),r:w("r"),f:w("f"),err:tot-ok};
}

/* ---------- routing ---------- */
const PAGES=["home","classify","insights","about"];let insightsDone=false;
function route(){
  const h=location.hash.replace("#/",""),p=PAGES.includes(h)?h:"home";
  PAGES.forEach(x=>$("p-"+x).hidden=x!==p);
  document.querySelectorAll("[data-nav]").forEach(a=>a.classList.toggle("on",a.dataset.nav===p));
  $("menu").classList.remove("open");$("menuBtn").setAttribute("aria-expanded","false");window.scrollTo(0,0);
  if(p==="classify")checkServer();
  if(p==="insights"&&!insightsDone){renderInsights();insightsDone=true}
}
window.addEventListener("hashchange",route);
$("menuBtn").onclick=()=>{const o=$("menu").classList.toggle("open");$("menuBtn").setAttribute("aria-expanded",o)};

/* ---------- classifier ---------- */
const MAX=10*1024*1024;let file=null,url=null;
const states=["empty","selected","loading","result"];
function show(s){states.forEach(x=>$("s-"+x).hidden=x!==s)}
function err(m){const e=$("err");e.textContent=m||"";e.hidden=!m}
async function checkServer(){
  const s=$("status");
  try{
    if(SPACE){s.textContent="Connecting to classifier…";s.className="pill";await getGradioClient()}
    else{const r=await fetch(API+"/health");if(!r.ok)throw new Error("Local classifier unavailable")}
    s.textContent="Classifier online";s.className="pill on"
  }
  catch{s.textContent="Classifier offline";s.className="pill off";s.title="Start it with: python server.py"}
}
async function getGradioClient(){
  if(!SPACE)throw new Error("Set window.CT_SPACE in frontend/config.js to your public Hugging Face Space (username/space-name).");
  if(!gradioClientPromise)gradioClientPromise=(async()=>{
    const {Client,handle_file}=await import(GRADIO_CLIENT_URL);
    gradioHandleFile=handle_file;
    return Client.connect(SPACE);
  })();
  try{return await gradioClientPromise}catch(e){gradioClientPromise=null;throw e}
}
function pick(f){
  err();
  if(!f)return err("No file was selected. Choose a CT image to continue.");
  if(f.size===0)return err("This file is empty. Choose a different image.");
  if(!/^image\/(jpeg|png)$/.test(f.type)||!/\.(jpe?g|png)$/i.test(f.name))return err("Unsupported file type. Please upload a JPG, JPEG or PNG image.");
  if(f.size>MAX)return err(`This file is too large (${(f.size/1048576).toFixed(1)} MB). The limit is 10 MB.`);
  const u=URL.createObjectURL(f),im=new Image();
  im.onerror=()=>{URL.revokeObjectURL(u);err("This file could not be read as an image. It may be corrupted.")};
  im.onload=()=>{
    if(url)URL.revokeObjectURL(url);file=f;url=u;$("prev").src=u;$("prev2").src=u;
    $("m-name").textContent=f.name;$("m-type").textContent=f.type.replace("image/","").toUpperCase();
    $("m-size").textContent=f.size>1048576?(f.size/1048576).toFixed(2)+" MB":Math.round(f.size/1024)+" KB";
    show("selected");
  };im.src=u;
}
function reset(){file=null;if(url)URL.revokeObjectURL(url);url=null;$("file").value="";err();show("empty")}
async function analyze(){
  if(!file)return;err();show("loading");
  const fd=new FormData();fd.append("image",file);
  let t;
  try{
    let j;
    if(SPACE){
      const timeout=new Promise((_,reject)=>{t=setTimeout(()=>reject(new Error("The classifier took too long to respond. The Space may be waking up; please retry.")),180000)});
      const result=await Promise.race([getGradioClient().then(client=>client.predict("/predict",[gradioHandleFile(file)])),timeout]);
      j=result&&result.data&&result.data[0];
    }else{
      const r=await fetch(API+"/predict",{method:"POST",body:fd});
      j=null;try{j=await r.json()}catch{}
      if(r.status>=400&&r.status<500)throw new Error(j&&j.error?j.error+". Please try a different image.":"The image was not accepted. Please try a different one.");
      if(!r.ok)throw new Error("The analysis service ran into a problem. Please try again in a moment.");
    }
    if(j&&j.error)throw new Error(j.error);
    if(!j||!Array.isArray(j.results)||!j.results.length)throw new Error("The analysis service returned an unexpected response. Please try again.");
    render(j.results);show("result");
  }catch(e){
    show("selected");
    err(e instanceof TypeError?"Could not reach the classification service. Check the Space name and connection, then try again.":e.message);
  }finally{if(t)clearTimeout(t)}
}
function render(res){
  const ok=res.filter(x=>x&&x.probs),avg={};
  CLASSES.forEach(c=>avg[c.n]=ok.length?ok.reduce((a,x)=>a+(x.probs[c.n]||0),0)/ok.length:null);
  const top=ok.length?CLASSES.map(c=>c.n).sort((a,b)=>avg[b]-avg[a])[0]:res[0].label;
  const agree=res.filter(x=>x.label===top).length,all=agree===res.length;
  $("res").innerHTML=`<div class="verdict" style="color:${col(top)}">${esc(top)}</div>
   <p class="muted">${res.length===1?"Result from "+esc(res[0].model):all?`All ${res.length} models agree`:`${agree} of ${res.length} models predict ${esc(top)}`}</p>
   ${ok.length?`<div class="conf">Confidence ${pct(avg[top])}${res.length>1?" (average of models)":""}</div>`:""}
   ${!all&&res.length>1?`<p class="warn">The models disagree on this scan. Treat the result as unreliable.</p>`:ok.length&&avg[top]<.7?`<p class="warn">Confidence is low. The image may not be a chest CT slice like the training data.</p>`:""}
   ${ok.length?CLASSES.map(c=>`<div class="pr"><span>${c.n}</span><div class="bar"><i style="width:${avg[c.n]*100}%;background:var(${c.c})"></i></div><span>${(avg[c.n]*100).toFixed(1)}%</span></div>`).join(""):""}`;
  $("cmp").innerHTML=`<tr><th>Model</th><th>Prediction</th><th>Confidence</th><th>Test accuracy</th></tr>`+res.map(x=>`<tr><td>${esc(x.model)}</td><td style="color:${col(x.label)};font-weight:600">${esc(x.label)}</td><td>${typeof x.confidence==="number"?pct(x.confidence):"Not provided"}</td><td>${typeof x.test_accuracy==="number"?pct(x.test_accuracy):"Not available"}</td></tr>`).join("");
  $("cmpnote").textContent=`${res.length} of 3 models returned a result. Confidence is how sure a model is about this image; test accuracy is its score on the held-out test scans.`;
}
const dz=$("drop");
$("file").onchange=e=>pick(e.target.files[0]);
dz.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();$("file").click()}};
["dragover","dragenter"].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.add("over")}));
["dragleave","drop"].forEach(t=>dz.addEventListener(t,e=>{e.preventDefault();dz.classList.remove("over")}));
dz.addEventListener("drop",e=>pick(e.dataTransfer.files[0]));
$("analyze").onclick=analyze;$("change").onclick=reset;$("again").onclick=reset;

/* ---------- insights ---------- */
let cur="SVM";
function renderInsights(){
  const M=Object.entries(DATA.models).map(([n,m])=>({n,cm:m.cm,...stats(m.cm)}));
  $("perf").innerHTML=M.map(m=>`<div class="card"><h3>${m.n}</h3><div class="kpi">${pct(m.acc)}</div><span class="muted small">Accuracy</span>
    <div class="mini"><span><b>${pct(m.p)}</b>Precision</span><span><b>${pct(m.r)}</b>Recall</span><span><b>${pct(m.f)}</b>F1</span></div></div>`).join("");
  const w=Math.max(...M.map(m=>m.err),1);
  $("bars").innerHTML=M.map(m=>`<div class="brow"><span>${m.n}</span><div class="bar"><i style="width:${m.err/w*100}%;background:var(--acc)"></i></div><span>${m.err} of ${m.tot} wrong · ${pct(m.acc)}</span></div>`).join("");
  const s=M[0],names=CLASSES.map(c=>c.n);
  $("ds").innerHTML=[["Total images",DATA.total.toLocaleString()],["Test scans",s.tot.toLocaleString()],["Classes","3"]].map(([a,b])=>`<div class="card"><div class="kpi">${b}</div><span class="muted small">${a}</span></div>`).join("")+
    `<div class="card" style="grid-column:1/-1"><b>Test scans per class</b><div class="mini" style="margin-top:8px">${s.per.map((x,i)=>`<span><b>${x.sup}</b>${names[i]}</span>`).join("")}</div></div>`;
  tabs();cm();
}
function tabs(){
  const t=$("tabs");t.innerHTML="";
  Object.keys(DATA.models).forEach(n=>{const b=document.createElement("button");b.textContent=n;b.setAttribute("role","tab");b.setAttribute("aria-selected",n===cur);b.onclick=()=>{cur=n;tabs();cm()};t.appendChild(b)});
}
function cm(){
  const m=DATA.models[cur],mx=Math.max(...m.cm.flat()),box=$("cm"),names=CLASSES.map(c=>c.n);
  box.innerHTML=`<div></div>`+names.map(n=>`<div class="lab">Predicted ${n}</div>`).join("");
  m.cm.forEach((row,i)=>{
    box.insertAdjacentHTML("beforeend",`<div class="lab">Actual ${names[i]}</div>`);
    row.forEach((v,j)=>{const d=document.createElement("div"),a=i===j?.12+.75*v/mx:v?.8:0;d.className="cell";d.textContent=v;d.tabIndex=0;
      d.style.background=v?`color-mix(in srgb, var(${i===j?"--acc":"--covid"}) ${a*100}%, white)`:"#fff";d.style.color=i===j&&a>.5?"#fff":"var(--ink)";
      d.title=i===j?`${v} ${names[i]} scans correct`:`${v} ${names[i]} scans predicted as ${names[j]}`;box.appendChild(d)});
  });
  $("cmnote").textContent=`${cur}: ${m.note} ${stats(m.cm).err} of ${stats(m.cm).tot} test scans misclassified.`;
}
show("empty");route();
