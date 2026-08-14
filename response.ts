import { KieResponseError, normalizeKieError } from "./errors.ts";
import type { KieModelName } from "./provider.ts";

export type ParsedKieResponse = {
  text: string;
  raw: unknown;
  model?: KieModelName;
};

function asObject(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function textFromString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function firstNonEmpty(values: Array<string | undefined>): string | undefined {
  return values.find((v) => typeof v === "string" && v.trim())?.trim();
}

export function extractKieText(raw: unknown): string | undefined {
  const obj = asObject(raw);
  if (!obj) return undefined;

  const choices = Array.isArray(obj.choices) ? obj.choices : undefined;
  if (choices?.length) {
    const choice0 = asObject(choices[0]);
    const message = asObject(choice0?.message);
    const content = message?.content;
    const text = firstNonEmpty([
      textFromString(content),
      textFromString(choice0?.text),
    ]);
    if (text) return text;

    if (Array.isArray(content)) {
      const parts = content
        .map((part) => asObject(part))
        .flatMap((part) => {
          if (!part) return [];
          if (typeof part.text === "string") return [part.text];
          if (typeof part.output_text === "string") return [part.output_text];
          return [];
        });
      const joined = parts.map((p) => p.trim()).filter(Boolean).join("\n").trim();
      if (joined) return joined;
    }
  }

  const content = Array.isArray(obj.content) ? obj.content : undefined;
  if (content?.length) {
    const textBlocks = content
      .map((part) => asObject(part))
      .flatMap((part) => {
        if (!part) return [];
        if (part.type === "text" && typeof part.text === "string") return [part.text];
        if (part.type === "output_text" && typeof part.text === "string") return [part.text];
        if (part.type === "message" && typeof part.text === "string") return [part.text];
        return [];
      });
    const joined = textBlocks.map((p) => p.trim()).filter(Boolean).join("\n").trim();
    if (joined) return joined;
  }

  const candidates = Array.isArray(obj.candidates) ? obj.candidates : undefined;
  if (candidates?.length) {
    const candidate0 = asObject(candidates[0]);
    const parts = Array.isArray(asObject(candidate0?.content)?.parts)
      ? (asObject(candidate0?.content)?.parts as unknown[])
      : undefined;
    if (parts?.length) {
      const texts = parts
        .map((part) => asObject(part))
        .flatMap((part) => {
          if (!part) return [];
          if (typeof part.text === "string") return [part.text];
          return [];
        });
      const joined = texts.map((p) => p.trim()).filter(Boolean).join("\n").trim();
      if (joined) return joined;
    }
  }

  const output = Array.isArray(obj.output) ? obj.output : undefined;
  if (output?.length) {
    const messages = output
      .map((item) => asObject(item))
      .flatMap((item) => {
        if (!item) return [];
        const contentParts = Array.isArray(item.content) ? item.content : undefined;
        if (!contentParts) return [];
        return contentParts
          .map((part) => asObject(part))
          .flatMap((part) => {
            if (!part) return [];
            if (typeof part.text === "string") return [part.text];
            return [];
          });
      });
    const joined = messages.map((p) => p.trim()).filter(Boolean).join("\n").trim();
    if (joined) return joined;
  }

  const generic = firstNonEmpty([
    textFromString(obj.text),
    textFromString(obj.message),
    textFromString(obj.output_text),
    textFromString(asObject(obj.error)?.message),
    textFromString(obj.msg),
  ]);
  return generic;
}

export function parseKieResponse(raw: unknown): ParsedKieResponse {
  const obj = asObject(raw);
  if (!obj) {
    throw new KieResponseError("KIE response was not an object", {
      retryable: true,
      reason: "malformed_response",
      raw,
    });
  }

  const error = obj.error ?? raw;
  if (obj.error || obj.code === 401 || obj.code === 429 || obj.code === 408 || obj.code === 455 || obj.code === 500 || obj.code === 501) {
    throw normalizeKieError(error, typeof obj.code === "number" ? obj.code : undefined);
  }

  const text = extractKieText(raw);
  if (!text) {
    throw new KieResponseError("KIE returned no text", {
      retryable: true,
      reason: "empty_response",
      raw,
    });
  }

  return {
    text,
    raw,
    model: typeof obj.model === "string" ? (obj.model as KieModelName) : undefined,
  };
}

export function responseLooksComplete(raw: unknown): boolean {
  try {
    const parsed = parseKieResponse(raw);
    return typeof parsed.text === "string" && parsed.text.trim().length > 0;
  } catch {
    return false;
  }
}
