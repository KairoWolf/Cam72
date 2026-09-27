"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = ["#ff5050", "#ffc800", "#50dc50", "#1ea0ff", "#e65ae6", "#00dcdc", "#ff8c00", "#a064ff",
  "#bef03c", "#ff64a0", "#b4b400", "#ffbebe"];
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const S = {
  status: "unlabeled",
  list: [],
  current: null,
  meta: null,
  img: null,
  objects: [], // {cls, polygon, box, pred?, pending?, conf?}
  maybe: [], // low-confidence model guesses, click to accept
  selected: -1,
  cls: 0,
  boxOnly: false,
  hide: false,
  undo: [],
  dirty: false,
  expected: 9,
  view: { scale: 1, ox: 0, oy: 0 },
  drag: null,
  space: false,
  token: 0,
};

// ---------- helpers ----------
async function api(url, opts = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(detail);
  }
  return r.json();
}

function toast(text, ms = 2200) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), ms);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("image failed to load"));
    img.src = src;
  });
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const esc = (t) => String(t == null ? "" : t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const rectPoly = (b) => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
const toImg = (x, y) => [(x - S.view.ox) / S.view.scale, (y - S.view.oy) / S.view.scale];
const toScr = (x, y) => [S.view.ox + x * S.view.scale, S.view.oy + y * S.view.scale];

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function area(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s / 2);
}

function centroid(poly) {
  let x = 0, y = 0;
  for (const p of poly) { x += p[0]; y += p[1]; }
  return [x / poly.length, y / poly.length];
}

function inside(poly, x, y) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

function hit(list, x, y) {
  let best = -1, bestArea = Infinity;
  list.forEach((o, i) => {
    if (inside(o.polygon, x, y)) {
      const a = area(o.polygon);
      if (a < bestArea) { best = i; bestArea = a; }
    }
  });
  return best;
}

// ---------- drawing ----------
function stageSize() {
  const r = $("stage").getBoundingClientRect();
  return [r.width, r.height];
}

function resize() {
  const [w, h] = stageSize();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (S.img && !S.userZoomed) fit();
  draw();
}

function fit() {
  if (!S.img) return;
  const [w, h] = stageSize();
  const sc = Math.min(w / S.img.width, h / S.img.height);
  S.view = { scale: sc, ox: (w - S.img.width * sc) / 2, oy: (h - S.img.height * sc) / 2 };
  S.userZoomed = false;
}

function tracePath(poly) {
  ctx.beginPath();
  poly.forEach(([x, y], i) => {
    const [sx, sy] = toScr(x, y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  });
  ctx.closePath();
}

function badge(x, y, text, color) {
  ctx.font = "bold 14px system-ui, sans-serif";
  const w = Math.max(22, ctx.measureText(text).width + 12);
  ctx.fillStyle = "rgba(0,0,0,.75)";
  ctx.beginPath(); ctx.roundRect(x - w / 2 - 1, y - 12, w + 2, 24, 12); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.roundRect(x - w / 2 + 1, y - 10, w - 2, 20, 10); ctx.fill();
  ctx.fillStyle = "#000";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, x, y + 1);
}

function draw() {
  const [w, h] = stageSize();
  ctx.clearRect(0, 0, w, h);
  if (!S.img) return;
  ctx.imageSmoothingEnabled = S.view.scale < 2;
  ctx.drawImage(S.img, S.view.ox, S.view.oy, S.img.width * S.view.scale, S.img.height * S.view.scale);
  if (!S.hide) {
    for (const m of S.maybe) {
      tracePath(m.polygon);
      ctx.setLineDash([2, 5]); ctx.lineWidth = 2; ctx.strokeStyle = "rgba(255,230,120,.95)"; ctx.stroke();
      ctx.setLineDash([]);
      const [cx, cy] = toScr(...centroid(m.polygon));
      badge(cx, cy, `? ${Math.round((m.conf || 0) * 100)}%`, "#ffe678");
    }
    let n = 0;
    S.objects.forEach((o, i) => {
      const pup = o.cls === 0;
      if (pup) n += 1;
      const color = pup ? COLORS[(n - 1) % COLORS.length] : "#ffffff";
      tracePath(o.polygon);
      ctx.fillStyle = hexA(color, pup ? 0.3 : 0.12); ctx.fill();
      ctx.setLineDash(o.pred || o.pending ? [8, 5] : []);
      ctx.lineWidth = i === S.selected ? 4 : 2.5;
      ctx.strokeStyle = i === S.selected ? "#ffffff" : color;
      ctx.stroke();
      ctx.setLineDash([]);
      const [cx, cy] = toScr(...centroid(o.polygon));
      badge(cx, cy, o.pending ? "..." : (pup ? String(n) : "Mom"), color);
    });
  }
  if (S.drag && S.drag.mode === "box") {
    const [x1, y1] = S.drag.start, [x2, y2] = S.drag.cur;
    ctx.setLineDash([6, 4]); ctx.lineWidth = 2;
    ctx.strokeStyle = S.cls === 0 ? "#4fb3ff" : "#ffffff";
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
    ctx.setLineDash([]);
  }
}

function updateCounter() {
  const pups = S.objects.filter((o) => o.cls === 0).length;
  const moms = S.objects.filter((o) => o.cls === 1).length;
  const c = $("counter");
  c.textContent = S.current ? `Puppies ${pups} / ${S.expected}${moms ? " + mom" : ""}` : "";
  c.className = "counter " + (pups === S.expected ? "ok" : "low");
  $("clsPuppy").classList.toggle("on", S.cls === 0);
  $("clsMom").classList.toggle("on", S.cls === 1);
  $("boxOnly").classList.toggle("on", S.boxOnly);
  $("hide").classList.toggle("on", S.hide);
}

// ---------- editing ----------
function snapshot() {
  return JSON.stringify({ objects: S.objects, maybe: S.maybe });
}

function pushUndo() {
  S.undo.push(snapshot());
  if (S.undo.length > 50) S.undo.shift();
}

function undo() {
  const prev = S.undo.pop();
  if (!prev) return;
  const st = JSON.parse(prev);
  S.objects = st.objects.map((o) => ({ ...o, pending: false }));
  S.maybe = st.maybe;
  S.selected = -1;
  S.dirty = true;
  draw(); updateCounter();
}

function setCls(c) {
  S.cls = c;
  if (S.selected >= 0) {
    pushUndo();
    S.objects[S.selected].cls = c;
    S.dirty = true;
  }
  draw(); updateCounter();
}

function removeSelected() {
  if (S.selected < 0) return;
  pushUndo();
  S.objects.splice(S.selected, 1);
  S.selected = -1;
  S.dirty = true;
  draw(); updateCounter();
}

async function addBox(start, end) {
  const [ax, ay] = toImg(...start), [bx, by] = toImg(...end);
  const W = S.img.width, H = S.img.height;
  const box = [clamp(Math.min(ax, bx), 0, W - 1), clamp(Math.min(ay, by), 0, H - 1),
    clamp(Math.max(ax, bx), 0, W - 1), clamp(Math.max(ay, by), 0, H - 1)];
  if (box[2] - box[0] < 6 || box[3] - box[1] < 6) return;
  pushUndo();
  const obj = { cls: S.cls, box, polygon: rectPoly(box), pending: !S.boxOnly };
  S.objects.push(obj);
  S.selected = S.objects.length - 1;
  S.dirty = true;
  draw(); updateCounter();
  if (S.boxOnly) return;
  const token = S.token;
  const slow = setTimeout(() => toast("Loading the outline AI (only slow the first time)...", 8000), 2000);
  try {
    const r = await api(`/api/frames/${S.current}/segment`, { method: "POST", body: JSON.stringify({ box }) });
    if (token !== S.token) return;
    obj.polygon = r.polygon;
    if (!r.sam) toast("Kept the plain box (the AI could not find an outline)");
  } catch (e) {
    if (token === S.token) toast(`Outline failed (${e.message}); kept the box`);
  } finally {
    clearTimeout(slow);
  }
  obj.pending = false;
  draw();
}

function click(p) {
  const [x, y] = toImg(...p);
  const oi = hit(S.objects, x, y);
  const gi = hit(S.maybe, x, y);
  if (oi >= 0) {
    S.selected = oi;
  } else if (gi >= 0) {
    pushUndo();
    const m = S.maybe.splice(gi, 1)[0];
    S.objects.push({ cls: m.cls, polygon: m.polygon, box: m.box, pred: true, conf: m.conf });
    S.selected = S.objects.length - 1;
    S.dirty = true;
  } else {
    S.selected = -1;
  }
  draw(); updateCounter();
}

canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  if (!S.img) return;
  canvas.setPointerCapture(e.pointerId);
  const p = [e.offsetX, e.offsetY];
  if (e.button === 1 || e.button === 2 || S.space) {
    S.drag = { mode: "pan", start: p, view: { ...S.view } };
  } else if (e.button === 0) {
    S.drag = { mode: "press", start: p, cur: p };
  }
});
canvas.addEventListener("pointermove", (e) => {
  if (!S.drag) return;
  const p = [e.offsetX, e.offsetY];
  if (S.drag.mode === "pan") {
    S.view.ox = S.drag.view.ox + p[0] - S.drag.start[0];
    S.view.oy = S.drag.view.oy + p[1] - S.drag.start[1];
    S.userZoomed = true;
  } else {
    S.drag.cur = p;
    if (S.drag.mode === "press" && Math.hypot(p[0] - S.drag.start[0], p[1] - S.drag.start[1]) > 5) {
      S.drag.mode = "box";
    }
  }
  draw();
});
canvas.addEventListener("pointerup", (e) => {
  const d = S.drag;
  S.drag = null;
  if (!d) return;
  if (d.mode === "box") addBox(d.start, [e.offsetX, e.offsetY]);
  else if (d.mode === "press") click(d.start);
  draw();
});
canvas.addEventListener("wheel", (e) => {
  if (!S.img) return;
  e.preventDefault();
  const [ix, iy] = toImg(e.offsetX, e.offsetY);
  const [w, h] = stageSize();
  const minScale = Math.min(w / S.img.width, h / S.img.height) * 0.5;
  S.view.scale = clamp(S.view.scale * Math.pow(1.0015, -e.deltaY), minScale, 30);
  S.view.ox = e.offsetX - ix * S.view.scale;
  S.view.oy = e.offsetY - iy * S.view.scale;
  S.userZoomed = true;
  draw();
}, { passive: false });

document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  // A focused button already handles Enter/Space itself; don't run the shortcut as well.
  if (e.target.tagName === "BUTTON" && (e.key === "Enter" || e.key === " ")) return;
  if (e.key === " ") { S.space = true; e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { undo(); e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  switch (e.key) {
    case "1": setCls(0); break;
    case "2": setCls(1); break;
    case "Delete": case "Backspace": removeSelected(); e.preventDefault(); break;
    case "Escape": S.selected = -1; S.drag = null; draw(); break;
    case "Enter": save(); e.preventDefault(); break;
    case "ArrowRight": case "s": case "S": step(1); break;
    case "ArrowLeft": step(-1); break;
    case "h": case "H": S.hide = !S.hide; draw(); updateCounter(); break;
    case "b": case "B": S.boxOnly = !S.boxOnly; updateCounter(); break;
    case "r": case "R": fit(); draw(); break;
    default: return;
  }
});
document.addEventListener("keyup", (e) => { if (e.key === " ") S.space = false; });
window.addEventListener("resize", resize);
window.addEventListener("beforeunload", (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ""; } });

// ---------- frames ----------
function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function renderQueue() {
  const q = $("queue");
  if (!S.list.length) {
    q.innerHTML = `<div class="empty">${S.status === "unlabeled"
      ? "Nothing to label right now.<br><br>Frames are saved automatically from the cameras (every few minutes, and whenever the model is unsure). You can also click <b>Save frame for labeling</b> on the Live page, or upload photos and videos above."
      : "No frames here yet."}</div>`;
    return;
  }
  q.innerHTML = S.list.map((f) => {
    const reason = f.reason || "";
    const cls = reason.startsWith("uncertain") ? "uncertain" : "";
    const status = f.labeled ? `<span class="badge done">${f.puppies} puppies</span>` : `<span class="badge ${cls}">${esc(reason.split(":")[0])}</span>`;
    return `<div class="item ${f.id === S.current ? "on" : ""}" data-id="${esc(f.id)}">
      <div>${fmtTime(f.created)} &middot; ${esc(f.camera)} ${status}</div>
      <div class="r">${esc(reason.startsWith("uncertain") ? reason.slice(11) : f.id)}</div></div>`;
  }).join("");
  q.querySelectorAll(".item").forEach((el) => el.addEventListener("click", () => open(el.dataset.id)));
}

async function loadList(keepCurrent = true) {
  const data = await api(`/api/frames?status=${S.status}`);
  S.list = data.frames;
  S.expected = data.expected;
  const st = data.stats;
  $("stats").textContent = `${st.labeled} labeled (${st.puppies} puppies, ${st.moms} mom) · ${st.unlabeled} to label`;
  renderQueue();
  if (!keepCurrent || !S.current) {
    const wanted = new URLSearchParams(location.search).get("id");
    if (wanted && !S.current) await open(wanted, true);
    else if (S.list.length) await open(S.list[0].id, true);
    else showEmpty();
  }
}

function showEmpty() {
  S.current = null; S.img = null; S.objects = []; S.maybe = [];
  $("msg").style.display = "block";
  $("msg").textContent = "No frame selected";
  $("frameInfo").textContent = "";
  draw(); updateCounter();
}

async function open(id, force = false) {
  if (!force && S.dirty && !confirm("You have unsaved changes on this frame. Discard them?")) return false;
  const token = ++S.token;
  $("msg").style.display = "block";
  $("msg").textContent = "Loading (the model is pre-labeling this frame)...";
  try {
    const [meta, img] = await Promise.all([api(`/api/frames/${id}`), loadImage(`/api/frames/${id}/image.jpg`)]);
    if (token !== S.token) return false;
    S.current = id; S.meta = meta; S.img = img;
    S.undo = []; S.selected = -1; S.dirty = false;
    if (meta.labeled) {
      S.objects = meta.objects.map((o) => ({ cls: o.cls, polygon: o.polygon, box: o.box }));
      S.maybe = [];
    } else if (meta.suggestions) {
      S.objects = meta.suggestions.objects.map((o) => ({ cls: o.cls, polygon: o.polygon, box: o.box, pred: true, conf: o.conf }));
      S.maybe = meta.suggestions.maybe.map((o) => ({ cls: o.cls, polygon: o.polygon, box: o.box, conf: o.conf }));
    } else {
      S.objects = []; S.maybe = [];
    }
    $("msg").style.display = "none";
    const hint = meta.suggestions && meta.suggestions.model ? " · pre-labeled by the model, please check" : "";
    $("frameInfo").textContent = `${meta.camera || ""} · ${fmtTime(meta.created)} · ${meta.reason || ""}${meta.labeled ? " · labeled" : hint}`;
    history.replaceState(null, "", `/label?id=${encodeURIComponent(id)}`);
    fit(); resize(); updateCounter(); renderQueue();
    return true;
  } catch (e) {
    $("msg").textContent = `Could not open frame: ${e.message}`;
    return false;
  }
}

async function step(dir) {
  const i = S.list.findIndex((f) => f.id === S.current);
  const next = S.list[i + dir];
  if (next) await open(next.id);
  else toast(dir > 0 ? "That was the last frame in this list" : "This is the first frame");
}

async function save() {
  if (!S.current || S.saving) return;
  if (S.objects.some((o) => o.pending)) { toast("Wait a moment, the outline is still being traced"); return; }
  const pups = S.objects.filter((o) => o.cls === 0).length;
  if (pups === 0 && !confirm("Save this frame with no puppies in it?")) return;
  S.saving = true;
  try {
    await saveAndAdvance(pups);
  } finally {
    S.saving = false;
  }
}

async function saveAndAdvance(pups) {
  try {
    await api(`/api/frames/${S.current}`, {
      method: "PUT",
      body: JSON.stringify({ objects: S.objects.map((o) => ({ cls: o.cls, polygon: o.polygon })) }),
    });
  } catch (e) {
    toast(`Save failed: ${e.message}`);
    return;
  }
  S.dirty = false;
  toast(`Saved: ${pups} ${pups === 1 ? "puppy" : "puppies"}${pups !== S.expected ? ` (expected ${S.expected}; fine if some are hidden)` : ""}`);
  const i = S.list.findIndex((f) => f.id === S.current);
  if (S.status === "unlabeled") {
    S.list.splice(i, 1);
    const next = S.list[Math.min(i, S.list.length - 1)];
    await loadList(true);
    if (next) await open(next.id, true); else showEmpty();
  } else {
    await loadList(true);
    const next = S.list[i + 1];
    if (next) await open(next.id, true);
  }
}

async function removeFrame() {
  if (!S.current || !confirm("Delete this frame from the dataset?")) return;
  await api(`/api/frames/${S.current}`, { method: "DELETE" });
  const i = S.list.findIndex((f) => f.id === S.current);
  S.list.splice(i, 1);
  S.dirty = false;
  const next = S.list[Math.min(i, S.list.length - 1)];
  await loadList(true);
  if (next) await open(next.id, true); else showEmpty();
}

// ---------- wiring ----------
$("clsPuppy").onclick = () => setCls(0);
$("clsMom").onclick = () => setCls(1);
$("boxOnly").onclick = () => { S.boxOnly = !S.boxOnly; updateCounter(); };
$("hide").onclick = () => { S.hide = !S.hide; draw(); updateCounter(); };
$("undo").onclick = undo;
$("clear").onclick = () => { if (!S.objects.length) return; pushUndo(); S.objects = []; S.selected = -1; S.dirty = true; draw(); updateCounter(); };
$("prev").onclick = () => step(-1);
$("skip").onclick = () => step(1);
$("save").onclick = save;
$("del").onclick = removeFrame;
$("uploadBtn").onclick = () => $("upload").click();
// Buttons must not keep keyboard focus, or Enter would press them instead of saving.
document.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => b.blur()));
$("upload").onchange = async () => {
  const files = [...$("upload").files];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  toast(`Uploading ${files.length} file(s)...`, 60000);
  try {
    const r = await fetch("/api/frames/upload?camera=upload", { method: "POST", body: form });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    toast(`Added ${data.ids.length} frame(s) to label`);
    await loadList(true);
    if (!S.current && S.list.length) await open(S.list[0].id, true);
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
  }
  $("upload").value = "";
};
document.querySelectorAll(".tabs button[data-status]").forEach((b) => b.addEventListener("click", async () => {
  document.querySelectorAll(".tabs button[data-status]").forEach((x) => x.classList.toggle("on", x === b));
  S.status = b.dataset.status;
  await loadList(true);
}));

resize();
loadList(false).catch((e) => { $("msg").textContent = `Could not load frames: ${e.message}`; });
