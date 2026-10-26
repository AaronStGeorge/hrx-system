// Copyright 2026 The IREE Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const COLORS = {
  output: "#69d9da",
  input: "#f8aa74",
  active: "#aaa0e8",
  memory: "#afcf87",
};
let windowMs = 30000;
const decimalFormat = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 1,
});
const integerFormat = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 0,
});
const encoder = new TextEncoder();

// Fixed history, independent of server uptime, counter magnitude, and traffic.
class Ring {
  constructor(capacity = 512) {
    this.times = new Float64Array(capacity);
    this.values = new Float64Array(capacity);
    this.count = 0;
    this.next = 0;
  }
  push(time, value) {
    this.times[this.next] = time;
    this.values[this.next] = value;
    this.next = (this.next + 1) % this.times.length;
    this.count = Math.min(this.count + 1, this.times.length);
  }
  each(callback) {
    for (let i = 0; i < this.count; ++i) {
      const index =
        (this.next - this.count + i + this.times.length) % this.times.length;
      callback(this.times[index], this.values[index]);
    }
  }
}

// Difference exact counters before converting the small interval to a double.
function rate(current, previous, key, seconds) {
  if (!current || !previous || seconds <= 0) return NaN;
  const delta = BigInt(current[key]) - BigInt(previous[key]);
  return delta >= 0n ? Number(delta) / seconds : NaN;
}

const $ = (id) => document.getElementById(id);
const format = (value) =>
  Number.isFinite(value)
    ? (value < 10 ? decimalFormat : integerFormat).format(value)
    : "—";
const integer = (value) => integerFormat.format(BigInt(value));
const history = Object.fromEntries(
  ["output", "input", "requests", "active", "memory"].map((key) => [
    key,
    new Ring(),
  ]),
);
const models = new Map();
const pendingPlots = new Set();
let socket;
let paused = false;
let retry;
let retryDelay = 1000;
let serverIdentity;
let previousServer;
let latestTime = 0;
let receivedAt = 0;
let sequence = 0n;
let receivedBytes = 0;
let receivedFrames = 0;
let dirty = false;
let framePending = false;

function draw(canvas, series, now, grid = false) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const left = grid ? 35 : 0;
  const top = grid ? 10 : 3;
  const bottom = height - 3;
  let maximum = 1;
  for (const [ring] of series)
    ring.each((time, value) => {
      if (time >= now - windowMs && Number.isFinite(value))
        maximum = Math.max(maximum, value);
    });
  maximum *= 1.15;
  if (grid) {
    context.font = "9px ui-monospace, monospace";
    context.fillStyle = "#6f7c8d";
    context.strokeStyle = "#292f38";
    context.lineWidth = 0.7;
    for (let i = 0; i <= 4; ++i) {
      const y = top + ((bottom - top) * i) / 4;
      context.fillText(format(maximum * (1 - i / 4)), 0, y + 3);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(width, y);
      context.stroke();
    }
  }
  for (const [ring, color] of series) {
    const points = [];
    ring.each((time, value) => {
      if (time >= now - windowMs && time <= now) points.push([time, value]);
    });
    const x = (time) =>
      left + ((time - now + windowMs) / windowMs) * (width - left);
    const y = (value) => bottom - (value / maximum) * (bottom - top);
    const gradient = context.createLinearGradient(0, top, 0, bottom);
    gradient.addColorStop(0, `${color}35`);
    gradient.addColorStop(1, `${color}00`);
    // Missing observations create real gaps, not invented interpolated work.
    let segment = [];
    const paint = () => {
      if (!segment.length) return;
      context.beginPath();
      context.moveTo(x(segment[0][0]), bottom);
      for (const [time, value] of segment) context.lineTo(x(time), y(value));
      context.lineTo(x(segment.at(-1)[0]), bottom);
      context.closePath();
      context.fillStyle = gradient;
      context.fill();
      context.beginPath();
      segment.forEach(([time, value], index) =>
        index
          ? context.lineTo(x(time), y(value))
          : context.moveTo(x(time), y(value)),
      );
      context.lineWidth = grid ? 1.7 : 1.4;
      context.strokeStyle = color;
      context.stroke();
      const last = segment.at(-1);
      context.beginPath();
      context.arc(x(last[0]), y(last[1]), grid ? 2.8 : 2, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      segment = [];
    };
    for (const point of points) {
      if (!Number.isFinite(point[1])) {
        paint();
        continue;
      }
      if (segment.length && point[0] - segment.at(-1)[0] > 2000) paint();
      segment.push(point);
    }
    paint();
  }
}

function redraw(all = false) {
  dirty = true;
  if (all) for (const model of models.values()) pendingPlots.add(model);
  if (framePending || document.hidden) return;
  framePending = true;
  requestAnimationFrame(() => {
    framePending = false;
    if (!dirty) return;
    dirty = false;
    draw(
      $("flow"),
      [
        [history.input, COLORS.input],
        [history.output, COLORS.output],
      ],
      latestTime,
      true,
    );
    for (const [id, key, color] of [
      ["output-spark", "output", COLORS.output],
      ["request-spark", "requests", COLORS.input],
      ["active-spark", "active", COLORS.active],
      ["memory-spark", "memory", COLORS.memory],
    ])
      draw($(id), [[history[key], color]], latestTime);
    for (const model of pendingPlots)
      draw(model.canvas, [[model.ring, model.color]], latestTime);
    pendingPlots.clear();
  });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function createModel(model) {
  if (!models.size) $("model-grid").replaceChildren();
  const card = element("article", "panel model");
  const heading = element("div", "model-head");
  const name = element(
    "h3",
    "",
    model.name + (model.name_truncated ? "…" : ""),
  );
  const identity = element("div");
  identity.append(name, element("div", "kind", model.kind));
  const phase = element("span", "phase", model.phase);
  heading.append(
    element("div", "model-icon", model.kind === "image" ? "▧" : "≋"),
    identity,
    phase,
  );
  const metrics = element("div", "model-metrics");
  const numbers = [
    "ACTIVE",
    "QUEUED",
    model.kind === "image"
      ? "IMAGES"
      : model.kind === "text"
        ? "TOKENS / S"
        : "REQUESTS / S",
  ].map((label) => {
    const item = element("div");
    const value = element("strong", "", "—");
    item.append(element("label", "", label), value);
    metrics.append(item);
    return value;
  });
  const canvas = element("canvas");
  canvas.setAttribute("aria-label", `${model.name} completed work rate`);
  const pool = element("div");
  const poolHeading = element("div", "pool-heading");
  const poolValue = element("span", "", "—");
  poolHeading.append(element("span", "", "LOGICAL KV POSITIONS"), poolValue);
  const bar = element("div", "bar");
  const resident = element("span");
  const reserved = element("span", "reserved");
  bar.append(resident, reserved);
  pool.append(poolHeading, bar);
  pool.hidden = true;
  const details = element("details");
  const table = element("table");
  details.append(element("summary", "", "Source observations"), table);
  card.append(heading, metrics, canvas, pool, details);
  $("model-grid").append(card);
  return {
    id: model.incarnation,
    card,
    phase,
    numbers,
    canvas,
    pool,
    poolValue,
    resident,
    reserved,
    table,
    cells: new Map(),
    ring: new Ring(512),
    color: model.kind === "image" ? COLORS.input : COLORS.output,
  };
}

function updateModel(model, time, timeNs) {
  let state = models.get(model.slot);
  if (state && state.id !== model.incarnation) {
    pendingPlots.delete(state);
    state.card.remove();
    models.delete(model.slot);
    state = null;
  }
  if (!state) {
    state = createModel(model);
    models.set(model.slot, state);
  }
  const seconds = state.timeNs ? Number(timeNs - state.timeNs) / 1e9 : 0;
  const fields = new Map();
  for (const topic of model.topics)
    for (const field of topic.fields) {
      const key = `${topic.topic}.${field.name}`;
      fields.set(key, field);
      if (!state.cells.has(key)) {
        const row = element("tr");
        const cell = element("td");
        row.append(element("td", "", key.replaceAll("_", " ")), cell);
        state.table.append(row);
        state.cells.set(key, cell);
      }
      state.cells.get(key).textContent =
        `${integer(field.value)} ${field.unit}`;
    }
  for (const [key, cell] of state.cells) {
    if (!fields.has(key)) {
      cell.parentElement.remove();
      state.cells.delete(key);
    }
  }
  const value = (key) =>
    fields.has(key) ? Number(fields.get(key).value) : NaN;
  const imageTotal = fields.get("image.generated_images")?.value;
  const workRate =
    imageTotal !== undefined
      ? rate(
          { images: imageTotal },
          state.images === undefined ? null : { images: state.images },
          "images",
          seconds,
        )
      : rate(
          model.counters,
          state.counters,
          model.kind === "text" ? "output_tokens" : "completed",
          seconds,
        );
  state.ring.push(time, workRate);
  pendingPlots.add(state);
  state.images = imageTotal;
  state.counters = model.counters;
  state.timeNs = timeNs;
  state.model = model;
  state.fields = fields;
  state.phase.textContent = model.phase;
  state.numbers[0].textContent = format(model.active);
  state.numbers[1].textContent = format(model.queued + model.held);
  state.numbers[2].textContent =
    model.kind === "image"
      ? imageTotal === undefined
        ? "—"
        : integer(imageTotal)
      : format(workRate);
  const capacity = value("text.pool_capacity");
  state.pool.hidden = !(capacity > 0);
  if (capacity > 0) {
    const resident = value("text.pool_resident");
    const reserved = value("text.pool_reserved");
    state.poolValue.textContent = `${format(resident)} resident / ${format(capacity)}`;
    state.resident.style.width = `${(100 * resident) / capacity}%`;
    state.reserved.style.width = `${(100 * reserved) / capacity}%`;
  }
}

function summarize(time) {
  let active = 0,
    queued = 0,
    held = 0,
    memory = 0,
    hasMemory = false;
  const mix = new Map();
  for (const state of models.values()) {
    const model = state.model;
    active += model.active;
    queued += model.queued;
    held += model.held;
    const group = mix.get(model.kind) || { active: 0, queued: 0 };
    group.active += model.active;
    group.queued += model.queued + model.held;
    mix.set(model.kind, group);
    const field = state.fields.get("text.elastic_state_committed_bytes");
    if (field) {
      memory += Number(field.value);
      hasMemory = true;
    }
  }
  $("active").textContent = format(active);
  $("queued").textContent = format(queued);
  $("held").textContent = format(held);
  $("memory").textContent = hasMemory
    ? format(memory / (memory < 1048576 ? 1024 : 1048576))
    : "—";
  $("memory-unit").textContent = hasMemory && memory < 1048576 ? "KiB" : "MiB";
  $("model-count").textContent = models.size;
  history.active.push(time, active);
  history.memory.push(time, hasMemory ? memory : NaN);
  const rows = [];
  const largest = Math.max(
    1,
    ...[...mix.values()].map((item) => item.active + item.queued),
  );
  for (const [kind, values] of mix) {
    const row = element("div", "mix-row");
    const labels = element("div", "labels");
    labels.append(
      element("span", "", kind),
      element("span", "", `${values.active} active · ${values.queued} queued`),
    );
    const bar = element("div", "bar");
    const run = element("span");
    const wait = element("span", "waiting");
    run.style.width = `${(100 * values.active) / largest}%`;
    wait.style.width = `${(100 * values.queued) / largest}%`;
    bar.append(run, wait);
    row.append(labels, bar);
    rows.push(row);
  }
  $("mix").replaceChildren(
    ...(rows.length ? rows : [element("p", "empty", "No registered models.")]),
  );
  if (!models.size)
    $("model-grid").replaceChildren(
      element(
        "div",
        "panel empty",
        "No registered models. New registrations appear automatically.",
      ),
    );
}

function apply(frame) {
  if (frame.version !== 1 || BigInt(frame.sequence) !== sequence + 1n)
    throw Error("Unsupported or discontinuous observation stream");
  sequence = BigInt(frame.sequence);
  if (serverIdentity !== frame.server) {
    serverIdentity = frame.server;
    previousServer = null;
    pendingPlots.clear();
    models.clear();
    $("model-grid").replaceChildren();
    for (const key of Object.keys(history)) history[key] = new Ring();
  }
  const timeNs = BigInt(frame.time_ns);
  const time = Number(timeNs / 1000000n);
  latestTime = time;
  receivedAt = performance.now();
  const page = frame.page;
  if (page.type === "server") {
    const seconds = previousServer
      ? Number(timeNs - previousServer.timeNs) / 1e9
      : 0;
    const output = rate(
      page.counters,
      previousServer?.counters,
      "output_tokens",
      seconds,
    );
    const input = rate(
      page.counters,
      previousServer?.counters,
      "input_tokens",
      seconds,
    );
    const requests = rate(
      page.counters,
      previousServer?.counters,
      "completed",
      seconds,
    );
    history.output.push(time, output);
    history.input.push(time, input);
    history.requests.push(time, requests);
    $("output-rate").textContent = format(output);
    $("request-rate").textContent = format(requests);
    $("completed").textContent = page.counters
      ? integer(page.counters.completed)
      : "—";
    previousServer = { timeNs, counters: page.counters };
  } else if (page.type === "models") {
    // A live page establishes absence only for its explicitly scanned slots.
    for (let slot = page.first; slot < page.next; ++slot) {
      const model = models.get(slot);
      if (model && slot !== page.model?.slot) {
        model.card.remove();
        pendingPlots.delete(model);
        models.delete(slot);
      }
    }
    if (page.model) updateModel(page.model, time, timeNs);
    if (page.done) summarize(time);
  } else throw Error("Unknown observation page");
  $("delivery").textContent =
    `${receivedFrames} frames · ${(receivedBytes / 1024).toFixed(1)} KiB received · bounded history`;
  redraw();
}

function state(label, live = false) {
  $("connection").textContent = label;
  $("connection-dot").classList.toggle("live", live);
}

function connect() {
  if (paused || document.hidden) return;
  clearTimeout(retry);
  const url = new URL("../v1/observe", location.href);
  url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  $("endpoint").textContent = location.host;
  state("Connecting");
  const peer = new WebSocket(url, "loom.observe.v1");
  socket = peer;
  sequence = 0n;
  peer.onopen = () => {
    if (socket === peer) state("Live", true);
  };
  peer.onmessage = (event) => {
    if (socket !== peer) return;
    try {
      ++receivedFrames;
      receivedBytes += encoder.encode(event.data).length;
      const frame = JSON.parse(event.data);
      apply(frame);
      $("notice").hidden = true;
      retryDelay = 1000;
      // Applied facts and bounded rings, not socket delivery, return credit.
      peer.send(frame.sequence);
    } catch (error) {
      $("notice").textContent = `Observation error: ${error.message}`;
      $("notice").hidden = false;
      paused = true;
      $("pause").textContent = "Resume";
      peer.close(1002, "Invalid observation");
    }
  };
  peer.onclose = () => {
    if (socket !== peer) return;
    state(paused ? "Paused" : "Disconnected");
    if (!paused && !document.hidden) {
      $("notice").textContent =
        `Live feed disconnected. If this is the first connection, start loom-serve with --observation_origin=${location.origin}. Relays must forward WebSocket upgrades. Reconnecting…`;
      $("notice").hidden = false;
      retry = setTimeout(connect, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 30000);
    }
  };
}

$("pause").onclick = () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  clearTimeout(retry);
  if (paused) {
    state("Paused");
    socket?.close(1000, "Viewer paused");
  } else connect();
};
$("window").onclick = () => {
  windowMs = windowMs === 30000 ? 120000 : 30000;
  $("window").textContent = windowMs === 30000 ? "30 SEC" : "2 MIN";
  document.querySelectorAll(".chart-footer span").forEach((tick, index) => {
    tick.textContent =
      index === 4 ? "NOW" : `−${((windowMs / 1000) * (4 - index)) / 4}s`;
  });
  redraw(true);
};
document.addEventListener("visibilitychange", () => {
  clearTimeout(retry);
  if (document.hidden) socket?.close(1000, "Viewer hidden");
  else if (!paused) {
    connect();
    redraw();
  }
});
new ResizeObserver(() => redraw(true)).observe(document.querySelector("main"));
setInterval(() => {
  $("clock").textContent = new Date().toLocaleTimeString(undefined, {
    hour12: false,
  });
  if (receivedAt)
    $("freshness").textContent =
      `Last sample ${((performance.now() - receivedAt) / 1000).toFixed(1)}s ago`;
}, 1000);
connect();
