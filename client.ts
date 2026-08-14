import { KieResponseError } from "./errors.ts";
import { parseKieResponse, type ParsedKieResponse } from "./response.ts";
import { getKieModelSpec, kieProvider, type GeminiOptions, type GrokOptions, type KieChatMessage, type KieContentPart, type KieModelName, type KieModelOptionsByModel } from "./provider.ts";

export type KieClientInput<TModel extends KieModelName> = {
  model: TModel;
  messages: KieChatMessage[];
  options?: KieModelOptionsByModel[TModel];
  apiKey?: string;
  baseUrl?: string;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
};

function parseDataUrl(url: string): { mimeType: string; data: string } {
  const match = /^data:([^;]+);base64,(.*)$/.exec(url);
  if (match) {
    return { mimeType: match[1], data: match[2] };
  }
  return { mimeType: "image/*", data: url };
}

function mapOpenAIContent(content: string | KieContentPart[]): string | Record<string, unknown>[] {
  if (typeof content === "string") return content;
  return content.map((part) =>
    part.type === "text"
      ? { type: "text", text: part.text }
      : { type: "image_url", image_url: { url: part.image_url.url } },
  );
}

function mapResponseInputContent(content: string | KieContentPart[]): string | Record<string, unknown>[] {
  if (typeof content === "string") return content;
  return content.map((part) =>
    part.type === "text"
      ? { type: "input_text", text: part.text }
      : { type: "input_image", image_url: part.image_url.url },
  );
}

function mapClaudeContent(content: string | KieContentPart[]): string | Record<string, unknown>[] {
  if (typeof content === "string") return content;
  return content.map((part) =>
    part.type === "text"
      ? { type: "text", text: part.text }
      : { type: "image_url", image_url: { url: part.image_url.url } },
  );
}

function mapGeminiNativeParts(content: string | KieContentPart[]): Array<Record<string, unknown>> {
  if (typeof content === "string") return [{ text: content }];
  return content.map((part) => {
    if (part.type === "text") return { text: part.text };
    const parsed = parseDataUrl(part.image_url.url);
    return { inline_data: { mime_type: parsed.mimeType, data: parsed.data } };
  });
}

function mapGrokContent(content: string | KieContentPart[]): Array<Record<string, unknown>> {
  if (typeof content === "string") return [{ type: "input_text", text: content }];
  return content.map((part) =>
    part.type === "text"
      ? { type: "input_text", text: part.text }
      : { type: "input_image", image_url: part.image_url.url },
  );
}

function mapOpenAIChatMessages(messages: KieChatMessage[]): Record<string, unknown>[] {
  return messages.map((message) => ({
    role: message.role === "developer" ? "system" : message.role,
    content: mapOpenAIContent(message.content),
  }));
}

function mapOpenAIResponsesInput(messages: KieChatMessage[]): Record<string, unknown>[] {
  return messages.map((message) => ({
    role: message.role === "system" ? "developer" : message.role,
    content: mapResponseInputContent(message.content),
  }));
}

function mapClaudeMessages(messages: KieChatMessage[]): Record<string, unknown>[] {
  return messages.map((message) => ({
    role: message.role === "developer" ? "system" : message.role,
    content: mapClaudeContent(message.content),
  }));
}

function mapGeminiNativeContents(messages: KieChatMessage[]): Record<string, unknown>[] {
  return messages.map((message) => ({
    role: message.role,
    parts: mapGeminiNativeParts(message.content),
  }));
}

function mapGeminiOpenAIMessages(messages: KieChatMessage[]): Record<string, unknown>[] {
  return messages.map((message) => ({
    role: message.role === "developer" ? "system" : message.role,
    content: mapOpenAIContent(message.content),
  }));
}

function mapGrokMessages(messages: KieChatMessage[]): Record<string, unknown>[] {
  return messages.map((message) => ({
    role: message.role === "developer" ? "system" : message.role,
    content: mapGrokContent(message.content),
  }));
}

function coerceReasoningEffort(value: unknown): "low" | "medium" | "high" | "xhigh" | undefined {
  if (typeof value === "string") {
    if (value === "minimal" || value === "low") return "low";
    if (value === "medium") return "medium";
    if (value === "high") return "high";
    if (value === "xhigh" || value === "max") return "xhigh";
  }

  if (value && typeof value === "object" && "effort" in value && typeof (value as { effort?: unknown }).effort === "string") {
    const effort = (value as { effort: string }).effort;
    if (effort === "low" || effort === "medium" || effort === "high" || effort === "xhigh") {
      return effort;
    }
  }

  return undefined;
}

function buildRequestBody<TModel extends KieModelName>(input: KieClientInput<TModel>): Record<string, unknown> {
  const spec = getKieModelSpec(input.model);
  const options = (input.options ?? {}) as Record<string, unknown>;

  switch (spec.requestStyle) {
    case "openai-chat": {
      const body: Record<string, unknown> = {
        model: input.model,
        messages: mapOpenAIChatMessages(input.messages),
        stream: false,
      };
      const reasoningEffort = coerceReasoningEffort(options.reasoning_effort ?? options.reasoning);
      if (reasoningEffort) body.reasoning_effort = reasoningEffort === "low" ? "low" : "high";
      if (options.tools) body.tools = options.tools;
      if (options.tool_choice) body.tool_choice = options.tool_choice;
      return body;
    }
    case "openai-responses": {
      const body: Record<string, unknown> = {
        model: input.model,
        input: mapOpenAIResponsesInput(input.messages),
        stream: false,
      };
      const reasoning = coerceReasoningEffort(options.reasoning);
      if (reasoning) body.reasoning = { effort: reasoning };
      if (options.tools) body.tools = options.tools;
      if (options.tool_choice) body.tool_choice = options.tool_choice;
      return body;
    }
    case "anthropic-messages": {
      const body: Record<string, unknown> = {
        model: input.model,
        messages: mapClaudeMessages(input.messages),
        stream: false,
      };
      if (typeof options.thinkingFlag === "boolean") body.thinkingFlag = options.thinkingFlag;
      if (typeof options.max_tokens === "number") body.max_tokens = options.max_tokens;
      if (options.tools) body.tools = options.tools;
      return body;
    }
    case "gemini-native": {
      const gemini = options as GeminiOptions;
      const body: Record<string, unknown> = {
        contents: mapGeminiNativeContents(input.messages),
        stream: false,
      };
      const reasoningEffort = coerceReasoningEffort(options.reasoning_effort ?? options.reasoning);
      if (typeof gemini.include_thoughts === "boolean") body.include_thoughts = gemini.include_thoughts;
      else if (reasoningEffort) body.include_thoughts = true;
      if (gemini.reasoning_effort) body.reasoning_effort = gemini.reasoning_effort;
      else if (reasoningEffort) body.reasoning_effort = reasoningEffort === "low" ? "low" : "high";
      if (gemini.tools) body.tools = gemini.tools;
      return body;
    }
    case "gemini-openai": {
      const gemini = options as GeminiOptions;
      const body: Record<string, unknown> = {
        model: input.model,
        messages: mapGeminiOpenAIMessages(input.messages),
        stream: false,
      };
      const reasoningEffort = coerceReasoningEffort(options.reasoning_effort ?? options.reasoning);
      if (typeof gemini.include_thoughts === "boolean") body.include_thoughts = gemini.include_thoughts;
      else if (reasoningEffort) body.include_thoughts = true;
      if (gemini.reasoning_effort) body.reasoning_effort = gemini.reasoning_effort;
      else if (reasoningEffort) body.reasoning_effort = reasoningEffort === "low" ? "low" : "high";
      if (gemini.tools) body.tools = gemini.tools;
      return body;
    }
    case "grok-responses": {
      const grok = options as GrokOptions;
      const body: Record<string, unknown> = {
        model: input.model,
        input: mapGrokMessages(input.messages),
        stream: false,
      };
      const reasoning = coerceReasoningEffort(grok.reasoning);
      if (reasoning) body.reasoning = { effort: reasoning };
      if (grok.tools) body.tools = grok.tools;
      if (grok.tool_choice) body.tool_choice = grok.tool_choice;
      if (grok.test) body.test = grok.test;
      return body;
    }
    default: {
      const _never: never = spec.requestStyle;
      return _never;
    }
  }
}

export function buildKieRequest<TModel extends KieModelName>(input: KieClientInput<TModel>): {
  url: string;
  body: Record<string, unknown>;
  headers: Record<string, string>;
} {
  const spec = getKieModelSpec(input.model);
  const baseUrl = input.baseUrl ?? kieProvider.baseUrl;
  const apiKey = input.apiKey ?? process.env[kieProvider.apiKeyEnv ?? "KIE_API_KEY"] ?? process.env.KIE_API_KEY;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  return {
    url: `${baseUrl}${spec.endpoint}`,
    body: buildRequestBody(input),
    headers,
  };
}

export async function callKieChat<TModel extends KieModelName>(input: KieClientInput<TModel>): Promise<ParsedKieResponse> {
  const { url, body, headers } = buildKieRequest(input);
  const fetchImpl = input.fetchImpl ?? fetch;
  const response = await fetchImpl(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: input.signal,
  });

  const rawText = await response.text();
  let raw: unknown;
  try {
    raw = rawText ? JSON.parse(rawText) : {};
  } catch {
    raw = rawText;
  }

  if (!response.ok) {
    throw new KieResponseError(
      typeof raw === "string"
        ? raw
        : (raw as Record<string, unknown>)?.error && typeof (raw as Record<string, unknown>).error === "object"
          ? String((raw as Record<string, unknown>).error)
          : `KIE request failed with HTTP ${response.status}`,
      {
        retryable: response.status === 429 || response.status === 408 || response.status === 455 || response.status === 500 || response.status === 501,
        reason: response.status === 429 ? "rate_limited" : response.status === 408 || response.status === 455 || response.status === 500 || response.status === 501 ? "temporary_upstream" : "malformed_response",
        status: response.status,
        raw,
      },
    );
  }

  return parseKieResponse(raw);
}
