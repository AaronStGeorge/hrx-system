# Embedded live observations

`index.html`, `style.css`, and `app.js` are ordinary static browser assets.
The existing C embed-data generator includes them in the server; there is no
JavaScript compiler, package installation, chart dependency, or runtime file
lookup. The router serves exact asset names with a same-origin content policy.

The [observation workflow](../../../../docs/src/serving/workflows/observe-model-server.md#watch-the-live-dashboard)
owns the user-facing launch and relay instructions.

## Data and ownership

Services expose synchronous numeric topics through
[`telemetry/sample.h`](../telemetry/sample.h). Field names, units, kinds, and
schema versions belong to the source, not the WebSocket collector. Providers
copy already-maintained owner facts; sampling never waits for a device. Both
the Prometheus exporter and current-state pages consume this interface. No
collector-side text/image switch or parsing of private service JSON is needed.

[`server/observation.c`](../server/observation.c) visits the existing registry
on its serialized owner. A sweep starts with common server counters, then
copies one model per page while scanning at most 64 registry positions per
model page. The server aggregate uses the registry's existing capacity scan.
Pages carry explicit scanned ranges: absence applies only to that range,
and an entire sweep is not an atomic inventory. Display names are capped at
256 bytes with an explicit truncation flag. Slot and incarnation identify a
model, independently of its name.

The HTTP adapter schedules sampling at four hertz while readers exist. Sources
are sampled/formatted once per sweep, independently of reader count. A shared
latest-state cache retains one copied payload per source; exact byte comparison
detects changes. Payload size is bounded by the fixed formatting extent; retained
buffers grow geometrically and are reused, including across slot retirement/reuse.
No per-token or per-reader snapshot allocation occurs. One shared 16 KiB
formatting buffer and a fixed reader table serve all clients. The existing
WebSocket receiver owns bounded message allocations for acknowledgements and
controls. A page is copied into
the carrier's ordinary send storage; no service,
parameter, session, or ring pointer survives that synchronous copy.

## WebSocket projection

`GET /v1/observe` uses the existing HTTP upgrade and negotiates
`loom.observe.v1` when offered. Browser origins must match application
configuration; native clients can omit Origin. A text frame contains:

```json
{"version":1,"server":"1790000000000000000:1","sequence":"1","page":{"type":"server","model_capacity":32,"counters":null},"time_ns":"1000000000","revision":"1","previous_revision":"0","previous_time_ns":"0"}
```

The client applies the page, then sends a text frame containing only its
decimal sequence, such as `1`. One application page can be outstanding.
Transport completion alone is not application acknowledgement: browsers can
buffer incoming messages while JavaScript is stalled. Wrong acknowledgements
close only that peer. Ping and close remain available while a page is pending.

Model pages have `type:"models"`, a half-open `[first,next)` slot range,
`done`, and either `model:null` or a model object with current activity,
common counters, and source-owned `topics`. All 64-bit counters, times, and
incarnations are decimal strings. The browser differences counters as `BigInt`
before converting an interval into a plotted rate.

A content change starts a sweep; unchanged snapshots do not produce traffic.
After a sweep, small `type:"heartbeat"` pages advance sampled unchanged state
once per second until the five-second rate window settles, then once per minute.
They use the same application credit as data. New changes remain visible at the
sampling cadence, independent of the idle heartbeat interval. A heartbeat is
sent only after the reader has applied a complete sweep with no newer revision.

Each source page carries its content `revision`, `previous_revision`, and the
last unchanged sample time of that previous revision. `time_ns` is monotonic
capture time, not wall time or send time. A consumer holding the same revision
can extend its known unchanged boundary; a directly preceding revision can be
extended through `previous_time_ns`. Skipped revisions cannot authorize such an
extension. Copies can have been sampled before a concurrent registration change;
the next sample replaces/removes them without borrowing the retired source.

Withheld credit skips intermediate samples; it does not queue snapshots or
overwrite bytes already accepted by the network. Reconnect starts a fresh
sweep. This feed is current state, not an event replay: independent log
interfaces retain their explicit history/gap semantics. Oversized output or
invalid client input fails that observer rather than silently truncating facts
or failing model execution.

The browser retains 512 samples per chart, discards retired model cards, and
uses the server identity and model incarnation to reset counter baselines.
Rates use exact counter differences across a five-second rolling window, retaining
64 counter/time pairs per rate. The first sample at or before the window boundary
anchors the delta; warm-up uses the available observed duration. Reconnect,
counter reset, non-increasing time, or an unobserved gap longer than the window
clears that baseline. Revision-qualified unchanged intervals bridge quiet periods
so the first burst after idleness remains measurable. Rates are never unweighted
means of differently sized intervals.
Serving mix groups recent successful-request rates by modality, independently
of current active/queued gauges. Short requests can finish between gauge samples;
a zero current count does not imply zero work over the rate window. Logical KV
occupancy is token-position accounting, not whole-device or process memory.
Missing observations produce gaps rather than interpolated work. Rendering
is sample-driven, pixel ratio is capped at two, and hiding the page releases
its subscription.
