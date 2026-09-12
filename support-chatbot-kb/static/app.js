const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

const sourceForm = document.getElementById("source-form");
const sourceType = document.getElementById("source-type");
const sourceLocation = document.getElementById("source-location");
const sourceInterval = document.getElementById("source-interval");
const sourceList = document.getElementById("source-list");
const chunkStat = document.getElementById("chunk-stat");
const refreshAllBtn = document.getElementById("refresh-all");

function addBubble(text, who, sources) {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.textContent = text;
  if (sources && sources.length) {
    const s = document.createElement("div");
    s.className = "sources";
    s.textContent = "Sources: " + sources.join(", ");
    div.appendChild(s);
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  addBubble(message, "user");
  chatInput.value = "";

  const thinking = document.createElement("div");
  thinking.className = "bubble bot";
  thinking.textContent = "…";
  messagesEl.appendChild(thinking);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    thinking.remove();
    addBubble(data.answer || "Sorry, something went wrong.", "bot", data.sources);
  } catch (err) {
    thinking.remove();
    addBubble("Network error talking to the server.", "bot");
  }
});

function fmtTime(ts) {
  if (!ts) return "never";
  return new Date(ts * 1000).toLocaleString();
}

async function loadSources() {
  const res = await fetch("/api/sources");
  const data = await res.json();
  chunkStat.textContent = `${data.stats.total_chunks} chunks indexed`;

  sourceList.innerHTML = "";
  data.sources.forEach((s) => {
    const li = document.createElement("li");
    li.className = "source-item";
    const statusClass = s.last_error ? "status-err" : "status-ok";
    const statusText = s.last_error ? `error: ${s.last_error}` : "healthy";
    li.innerHTML = `
      <div class="row1">
        <span class="id">${s.id}</span>
        <span class="${statusClass}">${statusText}</span>
      </div>
      <div class="loc">${s.type}: ${s.location.slice(0, 80)}</div>
      <div class="meta">
        <span>${s.chunk_count} chunks</span>
        <span>every ${s.update_interval_minutes}m</span>
        <span>updated ${fmtTime(s.last_updated_at)}</span>
      </div>
      <div class="item-actions">
        <button data-action="refresh" data-id="${s.id}">Refresh now</button>
        <button data-action="delete" data-id="${s.id}">Remove</button>
      </div>
    `;
    sourceList.appendChild(li);
  });
}

sourceList.addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  const id = btn.dataset.id;
  if (btn.dataset.action === "refresh") {
    btn.textContent = "Refreshing…";
    await fetch(`/api/sources/${encodeURIComponent(id)}/refresh`, { method: "POST" });
  } else if (btn.dataset.action === "delete") {
    await fetch(`/api/sources/${encodeURIComponent(id)}`, { method: "DELETE" });
  }
  loadSources();
});

sourceForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const type = sourceType.value;
  const location = sourceLocation.value.trim();
  const interval = parseInt(sourceInterval.value, 10) || 60;
  if (!location) return;

  const submitBtn = sourceForm.querySelector("button[type=submit]");
  submitBtn.disabled = true;
  submitBtn.textContent = "Ingesting…";
  try {
    await fetch("/api/sources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, location, update_interval_minutes: interval }),
    });
    sourceLocation.value = "";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Add & Ingest Now";
  }
  loadSources();
});

refreshAllBtn.addEventListener("click", async () => {
  refreshAllBtn.textContent = "Refreshing…";
  await fetch("/api/sources/refresh-all", { method: "POST" });
  refreshAllBtn.textContent = "Refresh all now";
  loadSources();
});

addBubble(
  "Hi! I'm your support assistant. Ask me anything covered by the knowledge base on the right — new sources get pulled in automatically as they're added or updated.",
  "bot"
);
loadSources();
setInterval(loadSources, 15000);
