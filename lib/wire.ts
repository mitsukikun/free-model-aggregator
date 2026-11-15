// The client wires and the three upstream wires this entry speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. Each provider is dialled in its own
// wire (openai, anthropic or gemini) and its answer translated back into the wire the client used:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//   - the Gemini path rewrites either client body to generateContent (system instruction, contents,
//     function declarations, generation config) and rewrites its answer into either client wire, JSON
//     or SSE, with the usage from its usageMetadata and STOP mapped to stop / end_turn.
//   - an OpenAI client body is rewritten to an Anthropic Messages body for an anthropic-wire provider,
//     and an Anthropic answer is rewritten to a Chat Completions answer or SSE stream in return.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Upstream request builders: the client's request in a provider's own wire.
// ---------------------------------------------------------------------------

/** The text of one OpenAI message: a string, or its text blocks joined. */
function openAITextOf(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  if (!Array.isArray(message.content)) return '';
  const texts: string[] = [];
  for (const block of message.content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts.join('');
}

/** The system text of an OpenAI body: its system messages' text, joined with a newline. */
function openAISystemText(body: Json): string {
  const texts: string[] = [];
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message) || message.role !== 'system') continue;
      const text = openAITextOf(message);
      if (text !== '') texts.push(text);
    }
  }
  return texts.join('\n');
}

/** One OpenAI image_url block as the Anthropic image block it means, or null when it is not a data URI. */
function openAIImageToAnthropic(block: Json): Json | null {
  const url = isRecord(block.image_url) ? block.image_url.url : undefined;
  if (typeof url !== 'string') return null;
  const match = /^data:([^;,]+);base64,(.+)$/.exec(url);
  if (match === null) return null;
  return { type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } };
}

/** One OpenAI tool as the Anthropic tool it means, or null when it is not a function tool. */
function anthropicToolOf(tool: unknown): Json | null {
  if (!isRecord(tool) || !isRecord(tool.function)) return null;
  const fn = tool.function;
  if (typeof fn.name !== 'string') return null;
  const out: Json = { name: fn.name, input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } };
  if (typeof fn.description === 'string') out.description = fn.description;
  return out;
}

/** An OpenAI Chat Completions body as an Anthropic Messages body, for one model. */
export function openAIToAnthropicRequest(body: Json, modelId: string): Json {
  const system = openAISystemText(body);
  const messages: Json[] = [];
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message)) continue;
      if (message.role === 'system') continue;
      if (message.role === 'user') {
        const content: Json[] = [];
        if (typeof message.content === 'string') content.push({ type: 'text', text: message.content });
        else if (Array.isArray(message.content)) {
          for (const block of message.content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') content.push({ type: 'text', text: block.text });
            else if (block.type === 'image_url') {
              const image = openAIImageToAnthropic(block);
              if (image !== null) content.push(image);
            }
          }
        }
        if (content.length > 0) messages.push({ role: 'user', content });
        continue;
      }
      if (message.role === 'assistant') {
        const content: Json[] = [];
        const text = openAITextOf(message);
        if (text !== '') content.push({ type: 'text', text });
        if (Array.isArray(message.tool_calls)) {
          for (const call of message.tool_calls) {
            if (!isRecord(call) || !isRecord(call.function)) continue;
            let input: unknown = {};
            if (typeof call.function.arguments === 'string') {
              try {
                input = JSON.parse(call.function.arguments) as unknown;
              } catch {
                input = {};
              }
            }
            content.push({
              type: 'tool_use',
              id: typeof call.id === 'string' ? call.id : '',
              name: typeof call.function.name === 'string' ? call.function.name : '',
              input,
            });
          }
        }
        if (content.length > 0) messages.push({ role: 'assistant', content });
        continue;
      }
      if (message.role === 'tool') {
        messages.push({
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
              content: openAITextOf(message),
            },
          ],
        });
      }
    }
  }
  const request: Json = {
    model: modelId,
    max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024,
    messages,
  };
  if (system !== '') request.system = system;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.map((tool) => anthropicToolOf(tool)).filter((t): t is Json => t !== null);
    if (tools.length > 0) request.tools = tools;
  }
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') request.tool_choice = { type: 'auto' };
    else if (choice.type === 'none') request.tool_choice = { type: 'none' };
    else if (choice.type === 'required') request.tool_choice = { type: 'any' };
    else if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      request.tool_choice = { type: 'tool', name: choice.function.name };
    }
  }
  if (typeof body.stop === 'string') request.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) request.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (body.stream === true) request.stream = true;
  if (typeof body.temperature === 'number') request.temperature = body.temperature;
  if (typeof body.top_p === 'number') request.top_p = body.top_p;
  return request;
}

/** One OpenAI image_url block as a Gemini inlineData part, or null when it is not a data URI. */
function openAIImageToGemini(block: Json): Json | null {
  const url = isRecord(block.image_url) ? block.image_url.url : undefined;
  if (typeof url !== 'string') return null;
  const match = /^data:([^;,]+);base64,(.+)$/.exec(url);
  if (match === null) return null;
  return { inlineData: { mimeType: match[1], data: match[2] } };
}

/** One OpenAI message as Gemini parts: text, images, and function calls. */
function openAIMessageParts(message: Json): Json[] {
  const parts: Json[] = [];
  const text = openAITextOf(message);
  if (text !== '') parts.push({ text });
  if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image_url') {
        const image = openAIImageToGemini(block);
        if (image !== null) parts.push(image);
      }
    }
  }
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let args: Json = {};
      if (typeof call.function.arguments === 'string') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) args = parsed;
        } catch {
          args = {};
        }
      }
      parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
    }
  }
  return parts;
}

/** One OpenAI tool as the Gemini function declaration it means, or null when it is not a function tool. */
function geminiFunctionOf(tool: unknown): Json | null {
  if (!isRecord(tool) || !isRecord(tool.function)) return null;
  const fn = tool.function;
  if (typeof fn.name !== 'string') return null;
  const out: Json = { name: fn.name };
  if (typeof fn.description === 'string') out.description = fn.description;
  if (isRecord(fn.parameters)) out.parameters = fn.parameters;
  return out;
}

/** An OpenAI Chat Completions body as a Gemini generateContent body. */
export function openAIToGeminiRequest(body: Json): Json {
  const system = openAISystemText(body);
  const contents: Json[] = [];
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isRecord(message)) continue;
      if (message.role === 'system') continue;
      if (message.role === 'user') {
        const parts = openAIMessageParts(message);
        if (parts.length > 0) contents.push({ role: 'user', parts });
        continue;
      }
      if (message.role === 'assistant') {
        const parts = openAIMessageParts(message);
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (message.role === 'tool') {
        contents.push({
          role: 'model',
          parts: [
            {
              functionResponse: {
                name: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
                response: { content: openAITextOf(message) },
              },
            },
          ],
        });
      }
    }
  }
  const request: Json = { contents };
  if (system !== '') request.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.map((tool) => geminiFunctionOf(tool)).filter((t): t is Json => t !== null);
    if (declarations.length > 0) request.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop)) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) request.generationConfig = config;
  return request;
}

/** The system text of an Anthropic body: a string, or its text blocks joined. */
function anthropicSystemText(body: Json): string {
  if (typeof body.system === 'string') return body.system;
  if (!Array.isArray(body.system)) return '';
  const texts: string[] = [];
  for (const block of body.system) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts.join('');
}

/** One Anthropic image block as a Gemini inlineData part, or null when it is not base64. */
function anthropicImageToGemini(block: Json): Json | null {
  if (!isRecord(block.source) || block.source.type !== 'base64') return null;
  if (typeof block.source.media_type !== 'string' || typeof block.source.data !== 'string') return null;
  return { inlineData: { mimeType: block.source.media_type, data: block.source.data } };
}

/**
 * One Anthropic message as Gemini contents: a user turn's text and images go in a user
 * content, a user turn's tool results go in a following model content as function responses,
 * and an assistant turn's text and tool uses go in a model content.
 */
function anthropicMessageContents(message: Json): Json[] {
  const out: Json[] = [];
  if (!Array.isArray(message.content)) return out;
  if (message.role === 'user') {
    const parts: Json[] = [];
    const responses: Json[] = [];
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image') {
        const image = anthropicImageToGemini(block);
        if (image !== null) parts.push(image);
      } else if (block.type === 'tool_result') {
        responses.push({
          functionResponse: {
            name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            response: { content: typeof block.content === 'string' ? block.content : '' },
          },
        });
      }
    }
    if (parts.length > 0) out.push({ role: 'user', parts });
    if (responses.length > 0) out.push({ role: 'model', parts: responses });
    return out;
  }
  if (message.role === 'assistant') {
    const parts: Json[] = [];
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'tool_use') {
        parts.push({
          functionCall: {
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          },
        });
      }
    }
    if (parts.length > 0) out.push({ role: 'model', parts });
    return out;
  }
  return out;
}

/** An Anthropic Messages body as a Gemini generateContent body. */
export function anthropicToGeminiRequest(body: Json): Json {
  const system = anthropicSystemText(body);
  const contents: Json[] = [];
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (isRecord(message)) contents.push(...anthropicMessageContents(message));
    }
  }
  const request: Json = { contents };
  if (system !== '') request.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations: Json[] = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      const declaration: Json = { name: tool.name };
      if (typeof tool.description === 'string') declaration.description = tool.description;
      if (isRecord(tool.input_schema)) declaration.parameters = tool.input_schema;
      declarations.push(declaration);
    }
    if (declarations.length > 0) request.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences)) config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) request.generationConfig = config;
  return request;
}

// ---------------------------------------------------------------------------
// Response translators: a provider's answer in the wire the client used.
// ---------------------------------------------------------------------------

/** The text of a Gemini answer: its candidate's parts' text, joined. */
function geminiTextOf(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  const texts: string[] = [];
  for (const part of content.parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** A Gemini answer's usage counts, as { prompt, completion }. */
function geminiUsageOf(answer: Json): { prompt: number; completion: number } {
  const metadata = isRecord(answer.usageMetadata) ? answer.usageMetadata : null;
  return {
    prompt: metadata !== null ? numberOrZero(metadata.promptTokenCount) : 0,
    completion: metadata !== null ? numberOrZero(metadata.candidatesTokenCount) : 0,
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions answer. */
export function geminiToOpenAIResponse(answer: Json, modelId: string): Json {
  const usage = geminiUsageOf(answer);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(answer) }, finish_reason: 'stop' }],
    usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages answer. */
export function geminiToAnthropicResponse(answer: Json, modelId: string): Json {
  const usage = geminiUsageOf(answer);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model: modelId,
    content: [{ type: 'text', text: geminiTextOf(answer) }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: usage.prompt, output_tokens: usage.completion },
  };
}

/** The text of an Anthropic answer: its text blocks joined. */
function anthropicTextOf(answer: Json): string {
  if (!Array.isArray(answer.content)) return '';
  const texts: string[] = [];
  for (const block of answer.content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return texts.join('');
}

/** An Anthropic stop reason as the OpenAI finish reason it means. */
function openAIFinishOf(stopReason: unknown): string {
  if (stopReason === 'max_tokens') return 'length';
  if (stopReason === 'tool_use') return 'tool_calls';
  if (stopReason === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function anthropicToOpenAIResponse(answer: Json, modelId: string): Json {
  const usage = isRecord(answer.usage) ? answer.usage : null;
  const prompt = usage !== null ? numberOrZero(usage.input_tokens) : 0;
  const completion = usage !== null ? numberOrZero(usage.output_tokens) : 0;
  const message: Json = { role: 'assistant', content: anthropicTextOf(answer) };
  if (Array.isArray(answer.content)) {
    const toolCalls: Json[] = [];
    for (const block of answer.content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
  }
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: openAIFinishOf(answer.stop_reason) }],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

// ---------------------------------------------------------------------------
// Stream translators: a provider's SSE text, whole, in the client's wire.
// The provider's stream is a complete body, so it is read first, then re-emitted;
// the usage counts are therefore known before the first event is written.
// ---------------------------------------------------------------------------

/** The literal terminator of an OpenAI SSE stream. */
export const OPENAI_DONE = 'data: [DONE]\n\n';

/** One SSE text as frames: each frame's event name (or null) and its data payload. */
export function parseSSEFrames(text: string): { event: string | null; data: unknown }[] {
  const frames: { event: string | null; data: unknown }[] = [];
  for (const frame of text.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    let payload: string | null = null;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('event:')) event = trimmed.slice(6).trim();
      else if (trimmed.startsWith('data:')) payload = trimmed.slice(5).trim();
    }
    if (payload === null) continue;
    if (payload === '[DONE]') {
      frames.push({ event, data: '[DONE]' });
      continue;
    }
    try {
      frames.push({ event, data: JSON.parse(payload) as unknown });
    } catch {
      // A frame whose data is not JSON carries nothing this entry can translate.
    }
  }
  return frames;
}

/** One OpenAI chunk: the role chunk first, then one chunk per text piece, then the finish and the usage. */
function openAIStreamOf(modelId: string, pieces: string[], usage: { prompt: number; completion: number }): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const out: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of pieces) out.push(sseEvent(null, chunk({ content: piece }, null)));
  out.push(sseEvent(null, chunk({}, 'stop')));
  out.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model: modelId,
      choices: [],
      usage: { prompt_tokens: usage.prompt, completion_tokens: usage.completion, total_tokens: usage.prompt + usage.completion },
    }),
  );
  out.push(OPENAI_DONE);
  return out.join('');
}

/** A Gemini stream as an OpenAI SSE stream: its frames' text pieces, then STOP, usage, [DONE]. */
export function geminiToOpenAIStream(text: string, modelId: string): string {
  const pieces: string[] = [];
  let usage = { prompt: 0, completion: 0 };
  for (const frame of parseSSEFrames(text)) {
    if (!isRecord(frame.data)) continue;
    const candidate = Array.isArray(frame.data.candidates) && isRecord(frame.data.candidates[0]) ? frame.data.candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : null;
      if (content !== null && Array.isArray(content.parts)) {
        for (const part of content.parts) {
          if (isRecord(part) && typeof part.text === 'string' && part.text !== '') pieces.push(part.text);
        }
      }
    }
    if (isRecord(frame.data.usageMetadata)) usage = geminiUsageOf(frame.data);
  }
  return openAIStreamOf(modelId, pieces, usage);
}

/** A Gemini stream as an Anthropic SSE stream: message_start with the real input count, then the text deltas, then message_delta and message_stop. */
export function geminiToAnthropicStream(text: string, modelId: string): string {
  const pieces: string[] = [];
  let usage = { prompt: 0, completion: 0 };
  for (const frame of parseSSEFrames(text)) {
    if (!isRecord(frame.data)) continue;
    const candidate = Array.isArray(frame.data.candidates) && isRecord(frame.data.candidates[0]) ? frame.data.candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : null;
      if (content !== null && Array.isArray(content.parts)) {
        for (const part of content.parts) {
          if (isRecord(part) && typeof part.text === 'string' && part.text !== '') pieces.push(part.text);
        }
      }
    }
    if (isRecord(frame.data.usageMetadata)) usage = geminiUsageOf(frame.data);
  }
  const out: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of pieces) {
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  out.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.completion } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return out.join('');
}

/** An Anthropic stream as an OpenAI SSE stream: the role chunk, the text deltas, then STOP, usage, [DONE]. */
export function anthropicToOpenAIStream(text: string, modelId: string): string {
  const pieces: string[] = [];
  let input = 0;
  let output = 0;
  let stopReason: string | null = null;
  for (const frame of parseSSEFrames(text)) {
    if (!isRecord(frame.data)) continue;
    if (frame.data.type === 'message_start' && isRecord(frame.data.message) && isRecord(frame.data.message.usage)) {
      input = numberOrZero(frame.data.message.usage.input_tokens);
    }
    if (frame.data.type === 'content_block_delta' && isRecord(frame.data.delta) && frame.data.delta.type === 'text_delta' && typeof frame.data.delta.text === 'string') {
      pieces.push(frame.data.delta.text);
    }
    if (frame.data.type === 'message_delta') {
      if (isRecord(frame.data.usage)) output = numberOrZero(frame.data.usage.output_tokens);
      if (isRecord(frame.data.delta) && typeof frame.data.delta.stop_reason === 'string') stopReason = frame.data.delta.stop_reason;
    }
  }
  const finish = openAIFinishOf(stopReason ?? 'end_turn');
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  const out: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of pieces) out.push(sseEvent(null, chunk({ content: piece }, null)));
  out.push(sseEvent(null, chunk({}, finish)));
  out.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model: modelId,
      choices: [],
      usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
    }),
  );
  out.push(OPENAI_DONE);
  return out.join('');
}
