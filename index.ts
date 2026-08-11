/**
 * KIE GPT provider extension for Prime Agent.
 *
 * First-pass support:
 * - GPT 5.2 -> chat/completions
 * - GPT 5.4 / 5.5 / 5.6 -> responses
 *
 * Usage:
 *   export KIE_API_KEY=...
 *   prime-agent -e ./packages/coding-agent/examples/extensions/custom-provider-kie
 */

import {
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	calculateCost,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StopReason,
	type Tool,
	type ToolResultMessage,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const KIE_BASE_URL = "https://api.kie.ai";
const KIE_CHAT_PATH = "/gpt-5-2/v1/chat/completions";
const KIE_RESPONSES_PATH = "/codex/v1/responses";

function isResponsesModel(modelId: string): boolean {
	const id = modelId.toLowerCase();
	return (
		id.startsWith("gpt-5.4") ||
		id.startsWith("gpt-5.5") ||
		id.startsWith("gpt-5.6") ||
		id.includes("response")
	);
}

function normalizeText(value: unknown): string {
	if (typeof value === "string") return value.trim();
	if (value == null) return "";
	return String(value).trim();
}

function contentToText(content: any): string {
	if (typeof content === "string") return content.trim();
	if (!Array.isArray(content)) return "";

	const parts: string[] = [];
	for (const block of content) {
		if (typeof block === "string") {
			parts.push(block);
			continue;
		}

		if (block?.type === "text" && typeof block.text === "string") {
			parts.push(block.text);
			continue;
		}

		if (block?.type === "thinking" && typeof block.thinking === "string") {
			parts.push(block.thinking);
			continue;
		}

		if (block?.type === "image") {
			parts.push("[image omitted]");
		}
	}

	return parts.join("\n").trim();
}

function convertTools(tools?: Tool[]): any[] | undefined {
	if (!tools?.length) return undefined;

	return tools.map((tool) => ({
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters: (tool.parameters as any) ?? {
				type: "object",
				properties: {},
				required: [],
			},
		},
	}));
}

function convertMessages(context: Context): any[] {
	const messages: any[] = [];

	if (context.systemPrompt?.trim()) {
		messages.push({
			role: "system",
			content: context.systemPrompt.trim(),
		});
	}

	for (const message of context.messages) {
		if (message.role === "user") {
			const text = contentToText(message.content);
			if (text) {
				messages.push({ role: "user", content: text });
			}
			continue;
		}

		if (message.role === "assistant") {
			const blocks = Array.isArray(message.content) ? (message.content as any[]) : [];
			const textParts: string[] = [];
			const toolCalls: any[] = [];

			for (const block of blocks) {
				if (block?.type === "text" && typeof block.text === "string" && block.text.trim()) {
					textParts.push(block.text);
					continue;
				}

				if (block?.type === "toolCall") {
					const argumentsText =
						typeof block.arguments === "string"
							? block.arguments
							: JSON.stringify(block.arguments ?? {});
					toolCalls.push({
						id: block.id ?? `tool_${toolCalls.length}`,
						type: "function",
						function: {
							name: block.name,
							arguments: argumentsText,
						},
					});
				}
			}

			const assistantMessage: any = { role: "assistant" };
			if (textParts.length) {
				assistantMessage.content = textParts.join("\n");
			}
			if (toolCalls.length) {
				assistantMessage.tool_calls = toolCalls;
			}
			if (assistantMessage.content || assistantMessage.tool_calls) {
				messages.push(assistantMessage);
			}
			continue;
		}

		if (message.role === "toolResult") {
			const toolResult = message as ToolResultMessage;
			messages.push({
				role: "tool",
				tool_call_id: toolResult.toolCallId,
				content: contentToText(toolResult.content) || (toolResult.isError ? "Tool error" : ""),
			});
		}
	}

	return messages;
}

function renderTranscript(context: Context): string {
	const lines: string[] = [];

	if (context.systemPrompt?.trim()) {
		lines.push(`System:\n${context.systemPrompt.trim()}`);
	}

	for (const message of context.messages) {
		if (message.role === "user") {
			const text = contentToText(message.content);
			if (text) lines.push(`User:\n${text}`);
			continue;
		}

		if (message.role === "assistant") {
			const blocks = Array.isArray(message.content) ? (message.content as any[]) : [];
			const textParts: string[] = [];
			for (const block of blocks) {
				if (block?.type === "text" && typeof block.text === "string" && block.text.trim()) {
					textParts.push(block.text);
				} else if (block?.type === "toolCall") {
					textParts.push(
						`[tool call: ${block.name} ${JSON.stringify(block.arguments ?? {})}]`,
					);
				}
			}
			if (textParts.length) lines.push(`Assistant:\n${textParts.join("\n")}`);
			continue;
		}

		if (message.role === "toolResult") {
			const toolResult = message as ToolResultMessage;
			lines.push(
				`ToolResult (${toolResult.toolCallId}${toolResult.isError ? ", error" : ""}):\n${contentToText(toolResult.content)}`,
			);
		}
	}

	return lines.join("\n\n");
}

function toNumber(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function updateUsageFromAny(output: AssistantMessage, payload: any, model: Model<any>): void {
	const usage = payload?.usage ?? payload?.data?.usage;
	if (!usage || typeof usage !== "object") return;

	const input =
		toNumber(usage.input_tokens) ||
		toNumber(usage.prompt_tokens) ||
		toNumber(usage.input) ||
		toNumber(usage.cache_read_input_tokens);

	const outputTokens =
		toNumber(usage.output_tokens) ||
		toNumber(usage.completion_tokens) ||
		toNumber(usage.output);

	const cacheRead =
		toNumber(usage.cache_read_input_tokens) ||
		toNumber(usage.cacheRead) ||
		0;

	const cacheWrite =
		toNumber(usage.cache_creation_input_tokens) ||
		toNumber(usage.cache_write_tokens) ||
		toNumber(usage.cacheWrite) ||
		0;

	output.usage.input = input;
	output.usage.output = outputTokens;
	output.usage.cacheRead = cacheRead;
	output.usage.cacheWrite = cacheWrite;
	output.usage.totalTokens =
		toNumber(usage.total_tokens) || input + outputTokens + cacheRead + cacheWrite;

	calculateCost(model, output.usage);
}

function mapStopReason(reason: unknown): StopReason {
	const r = normalizeText(reason).toLowerCase();

	switch (r) {
		case "stop":
		case "end_turn":
		case "completed":
			return "stop";
		case "length":
		case "max_tokens":
			return "length";
		case "tool_calls":
		case "tool_use":
		case "function_call":
			return "toolUse";
		case "abort":
		case "aborted":
			return "aborted" as StopReason;
		default:
			return "error";
	}
}

function extractTextLike(payload: any): string | null {
	if (payload == null) return null;

	if (typeof payload === "string") {
		const trimmed = payload.trim();
		return trimmed && trimmed !== "[DONE]" ? trimmed : null;
	}

	if (Array.isArray(payload)) {
		const parts = payload
			.map((item) => extractTextLike(item))
			.filter((item): item is string => Boolean(item));
		return parts.length ? parts.join("") : null;
	}

	if (typeof payload !== "object") return null;

	if (typeof payload.text === "string" && payload.text.trim()) {
		return payload.text.trim();
	}

	if (typeof payload.content === "string" && payload.content.trim()) {
		return payload.content.trim();
	}

	if (payload.response) {
		const nested = extractTextLike(payload.response);
		if (nested) return nested;
	}

	if (payload.message) {
		const nested = extractTextLike(payload.message);
		if (nested) return nested;
	}

	if (payload.data) {
		const nested = extractTextLike(payload.data);
		if (nested) return nested;
	}

	if (typeof payload.resultJson === "string" && payload.resultJson.trim()) {
		try {
			const parsed = JSON.parse(payload.resultJson);
			const nested = extractTextLike(parsed);
			if (nested) return nested;
		} catch {
			return payload.resultJson.trim();
		}
	}

	if (Array.isArray(payload.choices) && payload.choices.length > 0) {
		const choice = payload.choices[0];
		const delta = choice?.delta ?? {};

		if (typeof delta.content === "string" && delta.content.trim()) {
			return delta.content;
		}
		if (typeof choice?.message?.content === "string" && choice.message.content.trim()) {
			return choice.message.content;
		}
		if (typeof choice?.text === "string" && choice.text.trim()) {
			return choice.text;
		}
		if (choice?.message?.content) {
			const nested = extractTextLike(choice.message.content);
			if (nested) return nested;
		}
	}

	if (Array.isArray(payload.output) && payload.output.length > 0) {
		const parts: string[] = [];
		for (const item of payload.output) {
			if (typeof item?.text === "string" && item.text.trim()) {
				parts.push(item.text);
			}
			if (typeof item?.content === "string" && item.content.trim()) {
				parts.push(item.content);
			}
			if (Array.isArray(item?.content)) {
				const nested = extractTextLike(item.content);
				if (nested) parts.push(nested);
			}
		}
		if (parts.length) return parts.join("");
	}

	if (Array.isArray(payload.candidates) && payload.candidates.length > 0) {
		const candidate = payload.candidates[0];
		const content = candidate?.content;
		if (typeof candidate?.text === "string" && candidate.text.trim()) {
			return candidate.text;
		}
		if (content) {
			const nested = extractTextLike(content);
			if (nested) return nested;
		}
	}

	return null;
}

function extractChunk(payload: any): {
	text?: string;
	toolCalls?: Array<{ index: number; id?: string; name?: string; arguments?: string }>;
	final?: boolean;
	stopReason?: StopReason;
	usage?: any;
} {
	const result: {
		text?: string;
		toolCalls?: Array<{ index: number; id?: string; name?: string; arguments?: string }>;
		final?: boolean;
		stopReason?: StopReason;
		usage?: any;
	} = {};

	if (!payload || typeof payload !== "object") {
		const text = extractTextLike(payload);
		if (text) result.text = text;
		return result;
	}

	result.usage = payload.usage ?? payload.data?.usage;

	if (typeof payload.type === "string") {
		if (payload.type.includes("output_text.delta") && typeof payload.delta === "string") {
			result.text = payload.delta;
		}
		if (payload.type.includes("output_text.done")) {
			result.final = true;
		}
		if (payload.type.includes("completed") || payload.type.endsWith("done")) {
			result.final = true;
		}
	}

	if (Array.isArray(payload.choices) && payload.choices.length > 0) {
		const choice = payload.choices[0];
		const delta = choice?.delta ?? {};

		if (typeof delta.content === "string" && delta.content) {
			result.text = delta.content;
		}

		if (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0) {
			result.toolCalls = delta.tool_calls.map((call: any, index: number) => ({
				index: typeof call.index === "number" ? call.index : index,
				id: call.id,
				name: call.function?.name ?? call.name,
				arguments: call.function?.arguments ?? call.arguments ?? "",
			}));
		}

		if (typeof choice?.finish_reason !== "undefined") {
			result.final = true;
			result.stopReason = mapStopReason(choice.finish_reason);
		}
	}

	if (Array.isArray(payload.output) && payload.output.length > 0) {
		for (let i = 0; i < payload.output.length; i++) {
			const item = payload.output[i];
			if (typeof item?.text === "string" && item.text.trim()) {
				result.text = item.text;
			}

			if (typeof item?.content === "string" && item.content.trim()) {
				result.text = item.content;
			}

			if (item?.type && String(item.type).includes("tool")) {
				const toolCall = {
					index: i,
					id: item.id,
					name: item.name ?? item.function?.name,
					arguments: item.arguments ?? item.input ?? "",
				};
				result.toolCalls = result.toolCalls ?? [];
				result.toolCalls.push(toolCall);
			}
		}
	}

	if (typeof payload.delta === "string" && payload.delta.trim()) {
		result.text = payload.delta;
	}

	if (typeof payload.text === "string" && payload.text.trim()) {
		result.text = payload.text;
	}

	if (typeof payload.content === "string" && payload.content.trim()) {
		result.text = payload.content;
	}

	if (payload.response) {
		const nested = extractChunk(payload.response);
		if (nested.text) result.text = nested.text;
		if (nested.toolCalls?.length) {
			result.toolCalls = (result.toolCalls ?? []).concat(nested.toolCalls);
		}
		if (nested.final) result.final = true;
		if (nested.stopReason) result.stopReason = nested.stopReason;
	}

	return result;
}

async function* readSseEvents(response: Response): AsyncGenerator<{ event?: string; data: string }> {
	const reader = response.body?.getReader();
	if (!reader) return;

	const decoder = new TextDecoder();
	let buffer = "";

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;

		buffer += decoder.decode(value, { stream: true });

		let boundary = buffer.indexOf("\n\n");
		while (boundary !== -1) {
			const rawEvent = buffer.slice(0, boundary).trim();
			buffer = buffer.slice(boundary + 2);

			if (rawEvent) {
				let eventName: string | undefined;
				const dataLines: string[] = [];

				for (const line of rawEvent.split(/\r?\n/)) {
					if (line.startsWith("event:")) {
						eventName = line.slice(6).trim();
					} else if (line.startsWith("data:")) {
						dataLines.push(line.slice(5).trimStart());
					} else {
						dataLines.push(line);
					}
				}

				yield { event: eventName, data: dataLines.join("\n").trim() };
			}

			boundary = buffer.indexOf("\n\n");
		}
	}

	const tail = buffer.trim();
	if (tail) {
		let eventName: string | undefined;
		const dataLines: string[] = [];

		for (const line of tail.split(/\r?\n/)) {
			if (line.startsWith("event:")) {
				eventName = line.slice(6).trim();
			} else if (line.startsWith("data:")) {
				dataLines.push(line.slice(5).trimStart());
			} else {
				dataLines.push(line);
			}
		}

		yield { event: eventName, data: dataLines.join("\n").trim() };
	}
}

function streamKieGpt(
	model: Model<any>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
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

		let finalPayload: any = null;
		let textBlockIndex: number | null = null;
		const toolStateByIndex = new Map<number, { contentIndex: number; partialJson: string; ended: boolean }>();

		const pushText = (text: string) => {
			if (textBlockIndex === null) {
				output.content.push({ type: "text", text: "", index: output.content.length } as any);
				textBlockIndex = output.content.length - 1;
				stream.push({ type: "text_start", contentIndex: textBlockIndex, partial: output });
			}

			const block: any = output.content[textBlockIndex];
			block.text += text;
			stream.push({ type: "text_delta", contentIndex: textBlockIndex, delta: text, partial: output });
		};

		const pushToolCall = (toolCall: { index: number; id?: string; name?: string; arguments?: string }) => {
			let state = toolStateByIndex.get(toolCall.index);

			if (!state) {
				output.content.push({
					type: "toolCall",
					id: toolCall.id ?? `tool_${toolCall.index}`,
					name: toolCall.name ?? "",
					arguments: {},
					partialJson: "",
					index: output.content.length,
				} as any);

				state = {
					contentIndex: output.content.length - 1,
					partialJson: "",
					ended: false,
				};
				toolStateByIndex.set(toolCall.index, state);
				stream.push({ type: "toolcall_start", contentIndex: state.contentIndex, partial: output });
			}

			const block: any = output.content[state.contentIndex];
			if (toolCall.id) block.id = toolCall.id;
			if (toolCall.name) block.name = toolCall.name;

			if (typeof toolCall.arguments === "string" && toolCall.arguments.length > 0) {
				state.partialJson += toolCall.arguments;
				block.partialJson = state.partialJson;
				try {
					block.arguments = JSON.parse(state.partialJson);
				} catch {
					// partial JSON, keep accumulating
				}
				stream.push({
					type: "toolcall_delta",
					contentIndex: state.contentIndex,
					delta: toolCall.arguments,
					partial: output,
				});
			}
		};

		const finalizeOpenToolCalls = () => {
			for (const state of toolStateByIndex.values()) {
				if (state.ended) continue;
				const block: any = output.content[state.contentIndex];
				if (block?.partialJson) {
					try {
						block.arguments = JSON.parse(block.partialJson);
					} catch {
						// leave whatever we have
					}
				}
				delete block.partialJson;
				delete block.index;
				state.ended = true;
				stream.push({
					type: "toolcall_end",
					contentIndex: state.contentIndex,
					toolCall: block,
					partial: output,
				});
			}
		};

		const endpoint = isResponsesModel(model.id)
			? `${KIE_BASE_URL}${KIE_RESPONSES_PATH}`
			: `${KIE_BASE_URL}${KIE_CHAT_PATH}`;

		const requestBody: Record<string, unknown> = {
			model: model.id,
			stream: true,
		};

		if (isResponsesModel(model.id)) {
			requestBody.input = renderTranscript(context);
			requestBody.max_output_tokens = options?.maxTokens ?? model.maxTokens;
		} else {
			requestBody.messages = convertMessages(context);
			requestBody.max_tokens = options?.maxTokens ?? model.maxTokens;
		}

		const tools = convertTools(context.tools);
		if (tools) {
			requestBody.tools = tools;
		}

		const headers: Record<string, string> = {
			Authorization: `Bearer ${options?.apiKey ?? ""}`,
			"Content-Type": "application/json",
			Accept: "text/event-stream, application/json",
		};

		try {
			const apiKey = options?.apiKey ?? "";
			if (!apiKey.trim()) {
				throw new Error("KIE_API_KEY is missing");
			}

			const response = await fetch(endpoint, {
				method: "POST",
				headers,
				body: JSON.stringify(requestBody),
				signal: options?.signal,
			});

			if (!response.ok) {
				const text = await response.text();
				throw new Error(`KIE request failed (${response.status}): ${text.slice(0, 500)}`);
			}

			stream.push({ type: "start", partial: output });

			const contentType = response.headers.get("content-type") ?? "";
			if (contentType.includes("text/event-stream")) {
				for await (const evt of readSseEvents(response)) {
					if (!evt.data || evt.data === "[DONE]") continue;

					const parsed = (() => {
						try {
							return JSON.parse(evt.data);
						} catch {
							return evt.data;
						}
					})();

					const chunk = extractChunk(parsed);
					updateUsageFromAny(output, parsed, model);

					if (chunk.text) {
						pushText(chunk.text);
					}

					if (chunk.toolCalls?.length) {
						for (const toolCall of chunk.toolCalls) {
							pushToolCall(toolCall);
						}
					}

					if (chunk.final) {
						finalPayload = parsed;
						if (chunk.stopReason) {
							output.stopReason = chunk.stopReason;
						}
					}
				}
			} else {
				const json = await response.json().catch(async () => ({ text: await response.text() }));
				finalPayload = json;
				updateUsageFromAny(output, json, model);

				const finalText = extractTextLike(json);
				if (finalText) {
					pushText(finalText);
				}

				const chunk = extractChunk(json);
				if (chunk.stopReason) {
					output.stopReason = chunk.stopReason;
				}
			}

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			finalizeOpenToolCalls();

			if (textBlockIndex !== null) {
				const block: any = output.content[textBlockIndex];
				stream.push({
					type: "text_end",
					contentIndex: textBlockIndex,
					content: block.text,
					partial: output,
				});
				delete block.index;
				textBlockIndex = null;
			}

			if (output.content.length === 0) {
				const fallbackText = extractTextLike(finalPayload);
				if (fallbackText) {
					output.content.push({ type: "text", text: fallbackText } as any);
				}
			}

			stream.push({
				type: "done",
				reason: output.stopReason as "stop" | "length" | "toolUse",
				message: output,
			});
			stream.end();
		} catch (error) {
			for (const block of output.content as any[]) {
				delete block.index;
				delete block.partialJson;
			}

			output.stopReason = options?.signal?.aborted ? ("aborted" as StopReason) : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);

			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
}

export default function (pi: ExtensionAPI) {
	pi.registerProvider("kie-gpt", {
		name: "KIE GPT",
		baseUrl: KIE_BASE_URL,
		apiKey: "KIE_API_KEY",
		authHeader: true,
		api: "kie-gpt-api",
		streamSimple: streamKieGpt,

		models: [
			{
				id: "gpt-5.2",
				name: "GPT 5.2 (KIE)",
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 16384,
			},
			{
				id: "gpt-5.4",
				name: "GPT 5.4 (KIE)",
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 32768,
			},
			{
				id: "gpt-5.5",
				name: "GPT 5.5 (KIE)",
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 32768,
			},
			{
				id: "gpt-5.6-luna",
				name: "GPT 5.6 Luna (KIE)",
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 32768,
			},
			{
				id: "gpt-5.6-terra",
				name: "GPT 5.6 Terra (KIE)",
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 32768,
			},
			{
				id: "gpt-5.6-sol",
				name: "GPT 5.6 Sol (KIE)",
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 128000,
				maxTokens: 32768,
			},
		],
	});
}
