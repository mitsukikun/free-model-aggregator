// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Reads the brief (provider directory, quality table, workload manifest), probes each
// provider's catalog route, builds one catalog, registers it with the env, then serves
// the client's 20 requests: each answered by the best free model that meets it, failing
// over to the next-best when a provider refuses, and exits 0 when the env says done.
//
// Routing: every free candidate that meets the request's requirements, ranked by the
// quality table's rating_milli (best first), ties broken by fewest calls made to that
// provider so far, then provider id, then model id. A provider that answers that its
// free-tier budget is spent, or that it is out, is dead for the rest of the trial; a
// scheduled refusal only costs this request's attempt. Exactly one provider call may
// succeed per request: the loop stops at the first 2xx.
//
// Wires: the client arrives as OpenAI Chat Completions or Anthropic Messages, in JSON
// or server-sent events. The provider is dialled in its own wire (openai, anthropic or
// gemini) and its answer translated back into the wire the client used, with the
// provider's text whole, its usage counts as that wire reports them, and its stop
// reason mapped end_turn to stop, max_tokens to length, tool_use to tool_calls and
// refusal to content_filter. A JSON-mode request on the OpenAI wire is never routed to
// an anthropic-wire provider: that wire cannot carry response_format and echoes it
// absent. Everything here is a function of the brief, the requests and the providers'
// answers: no clock, no unseeded randomness, fixed timestamps in every listing.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  anthropicToOpenAIStream,
  geminiToAnthropicResponse,
  geminiToAnthropicStream,
  geminiToOpenAIResponse,
  geminiToOpenAIStream,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Per-provider trial state: counted calls made, and whether it is dead (budget spent / out). */
const providerState = new Map<string, { calls: number; dead: boolean }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): { calls: number; dead: boolean } {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  const probed = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  // One retry for a probe that failed: the catalog is worth 5% and a missing provider
  // also loses every request only it could serve. The retry is fixed, never timed.
  const failed = probed.filter((p) => !p.ok);
  if (failed.length > 0) {
    const retried = await Promise.all(failed.map((p) => probeProvider(ENV_URL, p.provider, PROBE_TIMEOUT_MS)));
    for (const retry of retried) {
      const at = probed.findIndex((p) => p.provider.id === retry.provider.id);
      if (at !== -1 && retry.ok) probed[at] = retry;
    }
  }
  probes = probed;
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists, each id once. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best first. Ranked by the quality
 * table's rating; ties go to the provider called least so far, then provider id,
 * then model id, so the order is a function of the brief and the calls made, never
 * of the clock. An OpenAI JSON-mode request never goes to an anthropic-wire provider.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', clientJson: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (stateOf(probe.provider.id).dead) continue;
    if (clientApi === 'openai' && clientJson && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      stateOf(a.probe.provider.id).calls - stateOf(b.probe.provider.id).calls ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client's OpenAI request asks for JSON mode. */
function clientJsonMode(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The provider's error message, when its body carries one in any wire's envelope. */
function errorMessageOf(text: string): string | null {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value)) return null;
    const error = isRecord(value.error) ? value.error : null;
    if (error !== null && typeof error.message === 'string') return error.message;
    return null;
  } catch {
    return null;
  }
}

/**
 * Whether a refusal is permanent for this trial: the provider's own message says its
 * free-tier budget is spent, or that it is out. A scheduled refusal says only that it
 * refused this call, and the provider stays usable.
 */
function refusalIsPermanent(message: string | null): boolean {
  if (message === null) return false;
  return message.includes('budget is spent') || message.includes(' is out');
}

/** POST one request to an openai-wire provider's Chat Completions route. */
async function callOpenAIWire(probe: Probe, modelId: string, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  const upstream: Json = { ...body, model: modelId };
  if (body.stream === true) upstream.stream_options = { include_usage: true };
  return fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstream),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

/** POST one request to an anthropic-wire provider's Messages route. */
async function callAnthropicWire(probe: Probe, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1/messages`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

/** POST one request to a gemini-wire provider's generateContent route; the model rides in the path. */
async function callGeminiWire(probe: Probe, modelId: string, body: Json, stream: boolean): Promise<Response> {
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  return fetch(`${base}/v1beta/models/${encodeURIComponent(modelId)}:${action}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

/**
 * One provider call for one client request, answered in the client's wire.
 * Returns true when the request was answered (the response is already sent).
 */
async function tryCandidate(
  candidate: Candidate,
  api: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<boolean> {
  const probe = candidate.probe;
  const state = stateOf(probe.provider.id);
  const stream = body.stream === true;
  let upstream: Response;
  try {
    if (probe.provider.wire === 'openai') {
      const upstreamBody = api === 'anthropic' ? anthropicToOpenAI(body, candidate.model.id) : body;
      upstream = await callOpenAIWire(probe, candidate.model.id, upstreamBody);
    } else if (probe.provider.wire === 'anthropic') {
      const upstreamBody = api === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
      upstream = await callAnthropicWire(probe, upstreamBody);
    } else {
      const upstreamBody = api === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
      upstream = await callGeminiWire(probe, candidate.model.id, upstreamBody, stream);
    }
  } catch {
    return false;
  }
  state.calls += 1;
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    if (refusalIsPermanent(errorMessageOf(text))) state.dead = true;
    return false;
  }
  const modelId = candidate.model.id;
  const wire = probe.provider.wire;

  // JSON: the provider's answer in the client's wire. Same wire passes through whole;
  // a cross-wire answer is translated, text whole, usage as the provider counted it.
  if (!stream) {
    if (wire === api) {
      const text = await upstream.text();
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
      res.end(text);
      return true;
    }
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer)) {
      sendError(res, api, 502, 'api_error', 'the provider answered a body that is not JSON');
      return true;
    }
    if (wire === 'openai') sendJson(res, 200, openAIToAnthropic(answer, modelId));
    else if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(answer, modelId));
    else if (api === 'openai') sendJson(res, 200, geminiToOpenAIResponse(answer, modelId));
    else sendJson(res, 200, geminiToAnthropicResponse(answer, modelId));
    return true;
  }

  // Streaming, same wire: the provider's events are already the client's events; pipe them through.
  if (wire === api && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    const reader = upstream.body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
    return true;
  }
  // Streaming, OpenAI provider to an Anthropic client: translate the chunks as they arrive.
  if (wire === 'openai' && api === 'anthropic' && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const event of anthropicEventsFromOpenAI(upstream.body, modelId)) res.write(event);
    res.end();
    return true;
  }
  // The remaining cross-wire streams are translated whole: the provider's stream is a
  // complete body, so it is read first and its usage known before the first event is written.
  const sseText = await upstream.text();
  let translated: string;
  if (wire === 'anthropic') translated = anthropicToOpenAIStream(sseText, modelId);
  else if (api === 'openai') translated = geminiToOpenAIStream(sseText, modelId);
  else translated = geminiToAnthropicStream(sseText, modelId);
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(translated);
  return true;
}

/** One client request: the best free model that meets it, failing over while providers refuse. */
async function serve(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, api, clientJsonMode(api, body));
  for (const candidate of candidates) {
    if (stateOf(candidate.probe.provider.id).dead) continue;
    if (await tryCandidate(candidate, api, body, res)) return;
  }
  sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
