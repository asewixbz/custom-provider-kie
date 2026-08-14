export * from "./provider.ts";
export * from "./errors.ts";
export * from "./response.ts";
export * from "./retry.ts";
export * from "./client.ts";

import type { AssistantMessage, AssistantMessageEventStream, Context, ImageContent, Model, SimpleStreamOptions, TextContent } from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { callKieChat } from "./client.ts";
import { getKieModelSpec, kieProvider, listKieModels, type KieChatMessage, type KieModelName } from "./provider.ts";
import { retryKie } from "./retry.ts";

function humanizeModelName(model: string): string {
  return model
    .replace(/[-.]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function imageToUrl(image: ImageContent): string {
  return `data:${image.mimeType};base64,${image.data}`;
}

function toKieContent(value: string | (TextContent | ImageContent)[]): string | Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }> {
  if (typeof value === "string") {
    return value;
  }

  return value.map((part) => {
    if (part.type === "text") {
      return { type: "text" as const, text: part.text };
    }
    return { type: "image_url" as const, image_url: { url: imageToUrl(part) } };
  });
}

function toKieMessages(context: Context): KieChatMessage[] {
  const messages: KieChatMessage[] = [];

  if (context.systemPrompt?.trim()) {
    messages.push({ role: "developer", content: context.systemPrompt.trim() });
  }

  for (const message of context.messages) {
    if (message.role === "user") {
      messages.push({ role: "user", content: toKieContent(message.content) });
      continue;
    }

    if (message.role === "assistant") {
      const text = message.content
        .map((part) => {
          if (part.type === "text") return part.text;
          if (part.type === "thinking") return part.thinking;
          if (part.type === "toolCall") return `[tool:${part.name}]`;
          return "";
        })
        .filter(Boolean)
        .join("\n")
        .trim();

      if (text) {
        messages.push({ role: "assistant", content: text });
      }
      continue;
    }

    if (message.role === "toolResult") {
      const text = message.content
        .map((part) => (part.type === "text" ? part.text : imageToUrl(part)))
        .filter(Boolean)
        .join("\n")
        .trim();

      messages.push({
        role: "tool",
        content: text ? `${message.toolName}: ${text}` : message.toolName,
      });
    }
  }

  return messages;
}

function modelName(model: KieModelName): string {
  return humanizeModelName(model);
}

function apiForModel(model: KieModelName): string {
  const spec = getKieModelSpec(model);
  switch (spec.requestStyle) {
    case "openai-chat":
      return "openai-completions";
    case "openai-responses":
    case "grok-responses":
      return "openai-responses";
    case "anthropic-messages":
      return "anthropic-messages";
    case "gemini-native":
      return "google-generative-ai";
    case "gemini-openai":
      return "openai-completions";
    default: {
      const _never: never = spec.requestStyle;
      return _never;
    }
  }
}

export function registerKieProvider(pi: ExtensionAPI) {
  pi.registerProvider("kie", {
    name: "KIE",
    baseUrl: kieProvider.baseUrl,
    apiKey: kieProvider.apiKeyEnv,
    api: "kie-api",
    models: listKieModels().map((id) => ({
      id,
      name: modelName(id),
      api: apiForModel(id),
      baseUrl: kieProvider.baseUrl,
      reasoning: true,
      input: ["text", "image"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: getKieModelSpec(id).family === "claude" ? 200000 : 128000,
      maxTokens: getKieModelSpec(id).family === "claude" ? 32000 : 16384,
      featured: kieProvider.defaultModel === id,
    })),
    streamSimple: function streamKie(model: Model<"kie-api">, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
      const stream = createAssistantMessageEventStream();

      (async () => {
        const output: AssistantMessage = {
          role: "assistant",
          content: [],
          api: model.api,
          provider: model.provider,
          model: model.id,
          usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: "stop",
          timestamp: Date.now(),
        };

        stream.push({ type: "start", partial: output });

        try {
          const parsed = await retryKie(
            () =>
              callKieChat({
                model: model.id as KieModelName,
                messages: toKieMessages(context),
                options: options as never,
                apiKey: options?.apiKey,
                baseUrl: model.baseUrl,
                signal: options?.signal,
              }),
          );

          if (parsed.model) {
            output.responseModel = parsed.model;
          }

          const text = parsed.text.trim();
          output.content.push({ type: "text", text: "" });
          stream.push({ type: "text_start", contentIndex: 0, partial: output });
          if (text) {
            (output.content[0] as TextContent).text = text;
            stream.push({ type: "text_delta", contentIndex: 0, delta: text, partial: output });
          }
          stream.push({ type: "text_end", contentIndex: 0, content: text, partial: output });
          stream.push({ type: "done", reason: "stop", message: output });
          stream.end();
        } catch (error) {
          output.stopReason = options?.signal?.aborted ? "aborted" : "error";
          output.errorMessage = error instanceof Error ? error.message : String(error);
          stream.push({ type: "error", reason: output.stopReason, error: output });
          stream.end();
        }
      })();

      return stream;
    },
  });
}

export default function (pi: ExtensionAPI) {
  registerKieProvider(pi);
}
