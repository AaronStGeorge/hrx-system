// Copyright 2026 The IREE Authors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const COLORS = {
  output: "#69d9da",
  input: "#f8aa74",
  active: "#aaa0e8",
  memory: "#afcf87",
  errors: "#f58c97",
};
const RATE_WINDOW_NS = 5000000000n;
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

// Retain exact counters, not an unweighted mean of interval rates. The oldest
// sample brackets the five-second boundary; warm-up uses only observed time.
// Each sample enters and leaves this fixed window once. A reset or an interval
// longer than the window starts a new baseline instead of inventing activity.
class CounterRate {
  constructor() {
    this.times = new BigUint64Array(64);
    this.values = new BigUint64Array(64);
    this.first = 0;
    this.count = 0;
  }
  reset() {
    this.count = 0;
  }
  push(time, raw) {
    if (raw === undefined || raw === null) {
      this.reset();
      return NaN;
    }
    const value = BigInt(raw);
    if (this.count) {
      const last = (this.first + this.count - 1) % this.times.length;
      if (
        time <= this.times[last] ||
        time - this.times[last] > RATE_WINDOW_NS ||
        value < this.values[last]
      )
        this.reset();
    }
    if (this.count === this.times.length) {
      this.first = (this.first + 1) % this.times.length;
      --this.count;
    }
    const index = (this.first + this.count++) % this.times.length;
    this.times[index] = time;
    this.values[index] = value;
    while (this.count > 1) {
      const next = (this.first + 1) % this.times.length;
      if (this.times[next] > time - RATE_WINDOW_NS) break;
      this.first = next;
      --this.count;
    }
    const elapsed = time - this.times[this.first];
    return elapsed > 0n
      ? Number(value - this.values[this.first]) / (Number(elapsed) / 1e9)
      : NaN;
  }
}

const $ = (id) => document.getElementById(id);
const format = (value) =>
  Number.isFinite(value)
    ? (value < 10 ? decimalFormat : integerFormat).format(value)
    : "—";
const integer = (value) => integerFormat.format(BigInt(value));
const scaled = (value, base, units) => {
  if (!Number.isFinite(value)) return "—";
  let unit = 0;
  while (value >= base && unit < units.length - 1) {
    value /= base;
    ++unit;
  }
  return `${format(value)} ${units[unit]}`;
};
const bytes = (value) => scaled(value, 1024, ["B", "KiB", "MiB", "GiB", "TiB"]);
const history = Object.fromEntries(
  ["output", "input", "requests", "active", "queued", "kv", "errors"].map(
    (key) => [key, new Ring()],
  ),
);
const models = new Map();
const pendingPlots = new Set();
const serverRates = Object.fromEntries(
  ["output_tokens", "input_tokens", "completed"].map((key) => [
    key,
    new CounterRate(),
  ]),
);
let socket;
let paused = false;
let retry;
let retryDelay = 1000;
let serverIdentity;
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
    draw(
      $("request-flow"),
      [[history.requests, COLORS.input]],
      latestTime,
      true,
    );
    draw(
      $("activity-flow"),
      [
        [history.active, COLORS.active],
        [history.queued, COLORS.input],
      ],
      latestTime,
      true,
    );
    for (const [id, key, color] of [
      ["output-spark", "output", COLORS.output],
      ["input-spark", "input", COLORS.input],
      ["request-spark", "requests", COLORS.input],
      ["active-spark", "active", COLORS.active],
      ["kv-spark", "kv", COLORS.memory],
      ["error-spark", "errors", COLORS.errors],
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
  heading.append(identity, phase);
  const metrics = element("div", "model-metrics");
  const workUnit =
    model.kind === "image"
      ? "Images"
      : model.kind === "text"
        ? "Tokens"
        : "Batches";
  const numbers = {};
  for (const [key, label] of Object.entries({
    active: "Active now",
    queued: "Queued + held",
    completed: "Completed",
    requests: "Requests/s · 5s",
    work: `${workUnit}/s · 5s`,
    total: `${workUnit} total`,
  })) {
    const item = element("div");
    const value = element("strong", "", "—");
    item.append(element("label", "", label), value);
    metrics.append(item);
    numbers[key] = value;
  }
  const canvas = element("canvas");
  canvas.setAttribute("aria-label", `${model.name} completed work rate`);
  const pool = element("div");
  const poolHeading = element("div", "pool-heading");
  const poolValue = element("span", "", "—");
  poolHeading.append(element("span", "", "KV positions"), poolValue);
  const bar = element("div", "bar");
  const resident = element("span");
  const reserved = element("span", "reserved");
  bar.append(resident, reserved);
  pool.append(poolHeading, bar);
  pool.hidden = true;
  const cache = element("div", "model-facts");
  const outcomes = element("div", "model-facts");
  const details = element("details");
  const table = element("table");
  details.append(element("summary", "", "All source counters"), table);
  card.append(heading, metrics, canvas, pool, cache, outcomes, details);
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
    cache,
    outcomes,
    table,
    cells: new Map(),
    ring: new Ring(512),
    requestRate: new CounterRate(),
    workRate: new CounterRate(),
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
      const cell = state.cells.get(key);
      cell.textContent =
        field.unit === "bytes"
          ? bytes(Number(field.value))
          : field.unit === "nanoseconds"
            ? scaled(Number(field.value), 1000, ["ns", "µs", "ms", "s"])
            : `${integer(field.value)} ${field.unit}`;
      cell.title = `${field.value} ${field.unit}`;
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
  const workTotal =
    model.kind === "image"
      ? imageTotal
      : model.counters?.[model.kind === "text" ? "output_tokens" : "batches"];
  const workRate = state.workRate.push(timeNs, workTotal);
  state.requests = state.requestRate.push(timeNs, model.counters?.completed);
  state.ring.push(time, workRate);
  pendingPlots.add(state);
  state.model = model;
  state.fields = fields;
  state.phase.textContent = model.phase;
  state.numbers.active.textContent = format(model.active);
  state.numbers.queued.textContent = format(model.queued + model.held);
  state.numbers.completed.textContent = model.counters
    ? integer(model.counters.completed)
    : "—";
  state.numbers.requests.textContent = format(state.requests);
  state.numbers.work.textContent = format(workRate);
  state.numbers.total.textContent =
    workTotal === undefined ? "—" : integer(workTotal);
  const capacity = value("text.pool_capacity");
  state.pool.hidden = !(capacity > 0);
  if (capacity > 0) {
    const resident = value("text.pool_resident");
    const reserved = value("text.pool_reserved");
    state.poolValue.textContent = `${format(resident)} resident + ${format(reserved)} reserved / ${format(capacity)}`;
    state.resident.style.width = `${(100 * resident) / capacity}%`;
    state.reserved.style.width = `${(100 * reserved) / capacity}%`;
  }
  const facts =
    model.kind === "text"
      ? [
          ["text.session_capacity", "rows"],
          ["text.session_cache_endpoints", "cached sessions"],
          ["text.prefix_cache_endpoints", "cached prefixes"],
        ]
      : model.kind === "image"
        ? [["image.generated_pixels", "pixels generated"]]
        : [];
  state.cache.textContent = facts
    .filter(([key]) => fields.has(key))
    .map(([key, label]) => `${integer(fields.get(key).value)} ${label}`)
    .join(" · ");
  state.outcomes.textContent = model.counters
    ? `${integer(model.counters.failed)} failed · ${integer(model.counters.rejected)} rejected · ${integer(model.counters.cancelled)} cancelled`
    : "";
}

function summarize(time) {
  let active = 0,
    queued = 0,
    held = 0,
    capacity = 0,
    resident = 0,
    reserved = 0,
    cachedSessions = 0,
    cachedPrefixes = 0,
    hasCache = false;
  const mix = new Map();
  for (const state of models.values()) {
    const model = state.model;
    active += model.active;
    queued += model.queued;
    held += model.held;
    const group = mix.get(model.kind) || {
      active: 0,
      queued: 0,
      rate: 0,
      available: false,
      completed: 0n,
      hasCounters: false,
    };
    group.active += model.active;
    group.queued += model.queued + model.held;
    if (Number.isFinite(state.requests)) {
      group.rate += state.requests;
      group.available = true;
    }
    if (model.counters) {
      group.completed += BigInt(model.counters.completed);
      group.hasCounters = true;
    }
    mix.set(model.kind, group);
    const value = (key) => Number(state.fields.get(`text.${key}`)?.value ?? 0);
    capacity += value("pool_capacity");
    resident += value("pool_resident");
    reserved += value("pool_reserved");
    cachedSessions += value("session_cache_endpoints");
    cachedPrefixes += value("prefix_cache_endpoints");
    hasCache ||= state.fields.has("text.session_cache_endpoints");
  }
  $("active").textContent = format(active);
  $("queued").textContent = format(queued);
  $("held").textContent = format(held);
  const occupancy = capacity ? (100 * resident) / capacity : NaN;
  $("kv").textContent = format(occupancy);
  $("kv-resident").textContent = capacity ? format(resident) : "—";
  $("kv-capacity").textContent = capacity ? format(capacity) : "—";
  $("kv-reserved").textContent = capacity ? format(reserved) : "—";
  $("cached-sessions").textContent = hasCache ? format(cachedSessions) : "—";
  $("cached-prefixes").textContent = hasCache ? format(cachedPrefixes) : "—";
  $("model-count").textContent = models.size;
  history.active.push(time, active);
  history.queued.push(time, queued + held);
  history.kv.push(time, occupancy);
  const rows = [];
  let totalRate = 0;
  for (const item of mix.values()) totalRate += item.rate;
  for (const [kind, values] of mix) {
    const row = element("div", "mix-row");
    const labels = element("div", "labels");
    labels.append(
      element("span", "", kind),
      element(
        "span",
        "",
        `${format(values.available ? values.rate : NaN)} req/s`,
      ),
    );
    const bar = element("div", "bar");
    const run = element("span");
    run.style.width = `${totalRate ? (100 * values.rate) / totalRate : 0}%`;
    bar.append(run);
    row.append(
      labels,
      bar,
      element(
        "small",
        "",
        `${values.hasCounters ? integer(values.completed) : "—"} completed · ${values.active} active · ${values.queued} queued`,
      ),
    );
    rows.push(row);
  }
  $("mix").replaceChildren(
    ...(rows.length ? rows : [element("p", "empty", "No registered models.")]),
  );
  if (!models.size)
    $("model-grid").replaceChildren(
      element("div", "panel empty", "No registered models."),
    );
}

function apply(frame) {
  if (frame.version !== 1 || BigInt(frame.sequence) !== sequence + 1n)
    throw Error("Unsupported or discontinuous observation stream");
  sequence = BigInt(frame.sequence);
  if (serverIdentity !== frame.server) {
    serverIdentity = frame.server;
    for (const rate of Object.values(serverRates)) rate.reset();
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
    const output = serverRates.output_tokens.push(
      timeNs,
      page.counters?.output_tokens,
    );
    const input = serverRates.input_tokens.push(
      timeNs,
      page.counters?.input_tokens,
    );
    const requests = serverRates.completed.push(
      timeNs,
      page.counters?.completed,
    );
    history.output.push(time, output);
    history.input.push(time, input);
    history.requests.push(time, requests);
    $("output-rate").textContent = format(output);
    $("input-rate").textContent = format(input);
    $("request-rate").textContent = format(requests);
    for (const [id, key] of [
      ["completed", "completed"],
      ["output-total", "output_tokens"],
      ["input-total", "input_tokens"],
      ["errors", "failed"],
      ["rejected", "rejected"],
      ["cancelled", "cancelled"],
      ["batches", "batches"],
    ]) {
      $(id).textContent = page.counters ? integer(page.counters[key]) : "—";
    }
    history.errors.push(
      time,
      page.counters ? Number(page.counters.failed) : NaN,
    );
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
    `${receivedFrames} frames · ${bytes(receivedBytes)}`;
  redraw();
}

function state(label, live = false) {
  $("connection").textContent = label;
  $("connection-dot").classList.toggle("live", live);
}

function connect() {
  if (paused || document.hidden) return;
  clearTimeout(retry);
  for (const rate of Object.values(serverRates)) rate.reset();
  for (const model of models.values()) {
    model.requestRate.reset();
    model.workRate.reset();
  }
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
  document.querySelectorAll(".chart-footer").forEach((footer) =>
    footer.querySelectorAll("span").forEach((tick, index) => {
      tick.textContent =
        index === 4 ? "NOW" : `−${((windowMs / 1000) * (4 - index)) / 4}s`;
    }),
  );
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
