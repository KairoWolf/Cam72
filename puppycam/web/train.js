"use strict";

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t == null ? "" : t).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (v) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);
let defaultsApplied = false;

async function api(url, opts = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(detail);
  }
  return r.json();
}

function renderDataset(d) {
  $("labeled").textContent = `${d.labeled} labeled frames`;
  const cams = Object.entries(d.cameras || {}).map(([c, v]) => `${c}: ${v.labeled}/${v.frames}`).join(" · ");
  $("labeledSub").textContent = `${d.puppies} puppies and ${d.moms} mom outlines · ${d.unlabeled} frames waiting · ${cams}`;
  const goal = d.labeled < 30 ? 30 : d.labeled < 150 ? 150 : 300;
  $("dataBar").style.width = `${Math.min(100, (d.labeled / goal) * 100)}%`;
  $("dataGoal").textContent = d.labeled >= 300 ? "Great dataset. Keep adding the frames the model gets wrong."
    : `${goal - d.labeled} more to reach ${goal}.`;
}

function renderTraining(t) {
  const st = t.state || {};
  $("start").disabled = t.running;
  $("stop").disabled = !t.running;
  if (!t.name) return;
  $("phase").textContent = `${t.name}: ${st.message || st.phase || ""}`;
  const frac = st.phase === "done" ? 1 : (st.epochs ? (st.epoch || 0) / st.epochs : 0);
  $("trainBar").style.width = `${frac * 100}%`;
  const m = st.metrics || {};
  const maskMap = m["metrics/mAP50(M)"], boxMap = m["metrics/mAP50(B)"];
  $("metrics").textContent = maskMap != null || boxMap != null
    ? `Validation during training: mAP50 boxes ${pct(boxMap)}, outlines ${pct(maskMap)}` : "";
  const log = $("log");
  log.hidden = !t.log;
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = t.log || "";
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function renderModels(models) {
  if (!models.length) return;
  $("models").innerHTML = models.map((m) => {
    const ev = m.eval || {};
    const ds = m.dataset || {};
    const prev = m.previous ? `<div class="muted">previous model on the same frames: ${pct(m.previous.count_accuracy)}</div>` : "";
    return `<tr>
      <td><b>${esc(m.name)}</b> ${m.current ? '<span class="pill">in use</span>' : ""}
        <div class="muted">${new Date((m.created || 0) * 1000).toLocaleString()} · ${esc(m.base || "")}</div></td>
      <td>${ds.train_frames ?? "-"} / ${ds.val_frames ?? "-"}${ds.val_overlaps_train ? " (too few frames: test = train)" : ""}</td>
      <td><b>${pct(ev.count_accuracy)}</b>${prev}</td>
      <td>${ev.count_mae ?? "-"}</td>
      <td>${pct(ev.precision)} / ${pct(ev.recall)}</td>
      <td>${m.conf != null ? Number(m.conf).toFixed(2) : "-"}</td>
      <td>${m.current ? "" : `<button data-activate="${esc(m.name)}">Use</button>`}
        <button data-mistakes="${esc(m.name)}">Mistakes</button></td>
    </tr>`;
  }).join("");
  document.querySelectorAll("[data-activate]").forEach((b) => b.onclick = async () => {
    await api(`/api/models/${b.dataset.activate}/activate`, { method: "POST" });
    refresh();
  });
  document.querySelectorAll("[data-mistakes]").forEach((b) => b.onclick = () => showMistakes(b.dataset.mistakes));
}

async function showMistakes(name) {
  const data = await api(`/api/models/${name}/mistakes`);
  const list = data.mistakes;
  $("mistakes").innerHTML = `<h3>Test frames ${esc(name)} counted wrong</h3>` + (list.length
    ? `<p class="muted">Open each one: fix the labels if they are wrong, otherwise label similar frames so the next model learns.</p><ul>`
      + list.map((x) => `<li><a href="/label?id=${encodeURIComponent(x.id)}">${esc(x.id)}</a>: labeled ${x.true}, model counted ${x.predicted}</li>`).join("")
      + "</ul>"
    : "<p>None - every test frame was counted exactly right.</p>");
}

async function refresh() {
  try {
    const t = await api("/api/train");
    if (!defaultsApplied && t.defaults) {
      $("base").value = t.defaults.base;
      $("imgsz").value = t.defaults.imgsz;
      defaultsApplied = true;
    }
    renderDataset(t.dataset);
    renderTraining(t);
    renderModels(t.models);
  } catch (e) { /* server restarting */ }
}

$("start").onclick = async () => {
  try {
    await api("/api/train", {
      method: "POST",
      body: JSON.stringify({ base: $("base").value, epochs: +$("epochs").value, imgsz: +$("imgsz").value }),
    });
  } catch (e) {
    alert(`Could not start training: ${e.message}`);
  }
  refresh();
};
$("stop").onclick = async () => {
  if (!confirm("Stop the training run?")) return;
  await api("/api/train/stop", { method: "POST" });
  refresh();
};

refresh();
setInterval(refresh, 3000);
