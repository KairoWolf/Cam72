"use strict";

const $ = (id) => document.getElementById(id);
let built = false;
let knownAlerts = new Set();

function ago(ts) {
  if (!ts) return "-";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return `${(s / 3600).toFixed(1)} h ago`;
}

function clock(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function esc(text) {
  const d = document.createElement("div");
  d.textContent = text == null ? "" : String(text);
  return d.innerHTML;
}

function buildCameras(cameras) {
  const box = $("cams");
  if (!cameras.length) {
    box.innerHTML = `<div class="card empty">No cameras configured.<br>
      Set <code>CAM1_URL</code> (and <code>CAM2_URL</code>) in your <code>.env</code> file, then restart.</div>`;
    return;
  }
  box.innerHTML = cameras.map((c) => `
    <div class="cam" id="cam-${c.slug}">
      <div class="bar">
        <span class="dot" id="dot-${c.slug}"></span>
        <span class="name">${esc(c.label)}</span>
        <span class="muted" id="meta-${c.slug}"></span>
        <span class="spacer"></span>
        <button data-cam="${c.slug}" class="capture">Save frame for labeling</button>
      </div>
      <img src="/api/cameras/${c.slug}/stream.mjpg" alt="${esc(c.label)} live view">
      <div class="chips" id="chips-${c.slug}"></div>
    </div>`).join("");
  box.querySelectorAll("button.capture").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    const r = await fetch(`/api/cameras/${b.dataset.cam}/capture`, { method: "POST" });
    b.textContent = r.ok ? "Saved - label it on the Label page" : "Could not save";
    setTimeout(() => { b.textContent = "Save frame for labeling"; b.disabled = false; }, 2500);
  }));
}

function beep() {
  if (!$("sound").checked) return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [0, 0.35, 0.7].forEach((t) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = 880; o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.25, ctx.currentTime + t);
      g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + t + 0.3);
      o.start(ctx.currentTime + t); o.stop(ctx.currentTime + t + 0.3);
    });
  } catch (e) { /* audio not available */ }
}

function update(s) {
  if (!built) { buildCameras(s.cameras); built = true; }
  $("expected").textContent = s.expected;
  $("count").textContent = s.count == null ? "-" : s.count;
  const card = $("countCard");
  card.classList.toggle("ok", s.count != null && s.count >= s.expected);
  card.classList.toggle("low", s.count != null && s.count < s.expected);
  const best = s.cameras.find((c) => c.slug === s.best_camera);
  $("best").textContent = best && s.cameras.length > 1 ? `(best view: ${best.label})` : "";
  $("mom").textContent = s.mom_visible == null ? "-" : (s.mom_visible ? "with the puppies" : "not visible");
  $("allSeen").textContent = s.model.name ? ago(s.alerts.last_all_visible) : "-";
  const ev = s.model.eval;
  $("model").innerHTML = s.model.name
    ? `${esc(s.model.name)}${ev ? ` <span class="muted">(exact count ${(ev.count_accuracy * 100).toFixed(0)}% on test frames)</span>` : ""}`
    : `none yet - <a href="/label">label frames</a>, then <a href="/train">train</a>`;
  if (s.model.error) $("model").innerHTML += `<br><span style="color:var(--bad)">${esc(s.model.error)}</span>`;
  $("fps").textContent = `${s.detect_fps} passes/s on ${s.model.device === "cpu" ? "CPU" : "GPU " + s.model.device}`;

  for (const c of s.cameras) {
    const dot = $(`dot-${c.slug}`);
    if (!dot) continue;
    dot.classList.toggle("on", c.connected);
    $(`meta-${c.slug}`).textContent = c.connected
      ? (c.count == null ? "" : `${c.count} puppies${c.mom ? " + mom" : ""}`)
      : (c.error ? `offline: ${c.error}` : "connecting...");
    $(`chips-${c.slug}`).innerHTML = c.puppies.map((p) =>
      `<span class="chip ${p.away ? "away" : ""}">#${p.number} ${(p.conf * 100).toFixed(0)}%${p.away ? " ALONE" : ""}</span>`).join("");
  }

  const active = s.alerts.active;
  $("alerts").innerHTML = active.map((a) =>
    `<div class="alert ${a.level}"><b>${esc(a.title)}</b>${esc(a.message)} <span class="muted">since ${clock(a.started)}</span></div>`).join("");
  const keys = new Set(active.map((a) => a.key));
  if ([...keys].some((k) => !knownAlerts.has(k)) && active.some((a) => a.level === "critical")) beep();
  knownAlerts = keys;

  const hist = s.alerts.history;
  $("history").innerHTML = hist.length ? hist.map((h) =>
    `<li><b>${esc(h.title)}</b> - ${clock(h.started)}${h.ended ? ` to ${clock(h.ended)}` : " (ongoing)"}</li>`).join("")
    : "<li>No alerts yet.</li>";
  $("notify").textContent = s.alerts.notifications ? "Phone notifications are on."
    : "Phone notifications are off (set NTFY_TOPIC in .env to get alerts on your phone).";
}

async function poll() {
  try {
    const r = await fetch("/api/status");
    if (r.ok) update(await r.json());
  } catch (e) { /* server restarting */ }
  setTimeout(poll, 1000);
}

try { $("sound").checked = localStorage.getItem("sound") === "1"; } catch (e) { /* storage blocked */ }
$("sound").addEventListener("change", () => {
  try { localStorage.setItem("sound", $("sound").checked ? "1" : "0"); } catch (e) { /* storage blocked */ }
  if ($("sound").checked) beep();
});
poll();
