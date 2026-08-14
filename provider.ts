export type KieModelFamily = "openai" | "claude" | "gemini" | "grok";
export type KieRequestStyle = "openai-chat" | "openai-responses" | "anthropic-messages" | "gemini-native" | "gemini-openai" | "grok-responses";
export type KieChatMessageRole = "developer" | "system" | "user" | "assistant" | "tool";

export type KieTextPart = { type: "text"; text: string };
export type KieImagePart = { type: "image_url"; image_url: { url: string } };
export type KieContentPart = KieTextPart | KieImagePart;

export type KieChatMessage = {
  role: KieChatMessageRole;
  content: string | KieContentPart[];
};

export type KieToolFunction = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
};

export type KieToolWebSearch = { type: "web_search" };
export type KieTool = KieToolFunction | KieToolWebSearch;

export type KieReasoningEffort = "low" | "medium" | "high" | "xhigh";
export type KieThinkingLevel = "low" | "high";

export type OpenAIStyleOptions = {
  stream?: boolean;
  tools?: KieTool[];
  reasoning_effort?: "low" | "high";
  tool_choice?: "auto";
};

export type OpenAIResponsesOptions = {
  stream?: boolean;
  tools?: KieTool[];
  reasoning?: { effort?: KieReasoningEffort };
  tool_choice?: "auto";
};

export type ClaudeOptions = {
  thinkingFlag?: boolean;
  stream?: boolean;
  max_tokens?: number;
  tools?: Array<{
    name: string;
    description?: string;
    input_schema: Record<string, unknown>;
  }>;
};

export type GeminiOptions = {
  stream?: boolean;
  tools?: Array<
    | { googleSearch?: Record<string, never> }
    | {
        functionDeclarations: Array<{
          name: string;
          description?: string;
          parameters?: Record<string, unknown>;
        }>;
      }
  >;
  include_thoughts?: boolean;
  reasoning_effort?: KieThinkingLevel;
};

export type GrokOptions = {
  stream?: boolean;
  reasoning?: { effort?: KieReasoningEffort };
  tools?: KieTool[];
  tool_choice?: "auto";
  test?: Record<string, unknown>;
};

export type KieModelOptionsByModel = {
  "gpt-5-2": OpenAIStyleOptions;
  "gpt-5-4": OpenAIResponsesOptions;
  "gpt-5-5": OpenAIResponsesOptions;
  "gpt-5-6-luna": OpenAIResponsesOptions;
  "gpt-5-6-terra": OpenAIResponsesOptions;
  "gpt-5-6-sol": OpenAIResponsesOptions;
  "gpt-5-codex": OpenAIResponsesOptions;
  "gpt-5.1-codex": OpenAIResponsesOptions;
  "gpt-5.2-codex": OpenAIResponsesOptions;
  "gpt-5.3-codex": OpenAIResponsesOptions;
  "gpt-5.4-codex": OpenAIResponsesOptions;
  "claude-opus-4-5": ClaudeOptions;
  "claude-opus-4-6": ClaudeOptions;
  "claude-opus-4-7": ClaudeOptions;
  "claude-opus-4-8": ClaudeOptions;
  "claude-opus-5": ClaudeOptions;
  "claude-sonnet-4-5": ClaudeOptions;
  "claude-sonnet-4-6": ClaudeOptions;
  "claude-sonnet-5": ClaudeOptions;
  "claude-haiku-4-5": ClaudeOptions;
  "claude-fable-5": ClaudeOptions;
  "gemini-2-5-pro": GeminiOptions;
  "gemini-2-5-flash": GeminiOptions;
  "gemini-3-pro": GeminiOptions;
  "gemini-3-flash": GeminiOptions;
  "gemini-3-1-pro": GeminiOptions;
  "gemini-3-5-flash": GeminiOptions;
  "gemini-3-5-flash-openai": GeminiOptions;
  "gemini-3-6-flash": GeminiOptions;
  "gemini-3-6-flash-openai": GeminiOptions;
  "gemini-3-flash-v1beta": GeminiOptions;
  "grok-4-3": GrokOptions;
  "grok-4-5": GrokOptions;
};

export type KieModelName = keyof KieModelOptionsByModel;

export type KieModelSpec = {
  family: KieModelFamily;
  model: KieModelName;
  endpoint: string;
  requestStyle: KieRequestStyle;
};

const OPENAI_CHAT_MODELS = ["gpt-5-2"] as const satisfies readonly KieModelName[];

const OPENAI_RESPONSES_MODELS = [
  "gpt-5-4",
  "gpt-5-5",
  "gpt-5-6-luna",
  "gpt-5-6-terra",
  "gpt-5-6-sol",
  "gpt-5-codex",
  "gpt-5.1-codex",
  "gpt-5.2-codex",
  "gpt-5.3-codex",
  "gpt-5.4-codex",
] as const satisfies readonly KieModelName[];

const CLAUDE_MODELS = [
  "claude-opus-4-5",
  "claude-opus-4-6",
  "claude-opus-4-7",
  "claude-opus-4-8",
  "claude-opus-5",
  "claude-sonnet-4-5",
  "claude-sonnet-4-6",
  "claude-sonnet-5",
  "claude-haiku-4-5",
  "claude-fable-5",
] as const satisfies readonly KieModelName[];

const GEMINI_NATIVE_MODELS = [
  "gemini-2-5-pro",
  "gemini-2-5-flash",
  "gemini-3-pro",
  "gemini-3-flash",
  "gemini-3-1-pro",
  "gemini-3-5-flash",
  "gemini-3-6-flash",
  "gemini-3-flash-v1beta",
] as const satisfies readonly KieModelName[];

const GEMINI_OPENAI_MODELS = ["gemini-3-5-flash-openai", "gemini-3-6-flash-openai"] as const satisfies readonly KieModelName[];

const GROK_MODELS = ["grok-4-3", "grok-4-5"] as const satisfies readonly KieModelName[];

export const kieModelRegistry: Record<KieModelName, KieModelSpec> = Object.fromEntries(
  [
    ...OPENAI_CHAT_MODELS.map((model) => [
      model,
      {
        family: "openai",
        model,
        endpoint: `/${model}/v1/chat/completions`,
        requestStyle: "openai-chat",
      },
    ]),
    ...OPENAI_RESPONSES_MODELS.map((model) => [
      model,
      {
        family: "openai",
        model,
        endpoint: model.includes("codex") ? "/api/v1/responses" : "/codex/v1/responses",
        requestStyle: "openai-responses",
      },
    ]),
    ...CLAUDE_MODELS.map((model) => [
      model,
      {
        family: "claude",
        model,
        endpoint: "/claude/v1/messages",
        requestStyle: "anthropic-messages",
      },
    ]),
    ...GEMINI_NATIVE_MODELS.map((model) => [
      model,
      {
        family: "gemini",
        model,
        endpoint: `/gemini/v1/models/${model}:streamGenerateContent`,
        requestStyle: "gemini-native",
      },
    ]),
    ...GEMINI_OPENAI_MODELS.map((model) => [
      model,
      {
        family: "gemini",
        model,
        endpoint: `/${model}/v1/chat/completions`,
        requestStyle: "gemini-openai",
      },
    ]),
    ...GROK_MODELS.map((model) => [
      model,
      {
        family: "grok",
        model,
        endpoint: "/grok/v1/responses",
        requestStyle: "grok-responses",
      },
    ]),
  ] as const,
) as Record<KieModelName, KieModelSpec>;

export type KieProviderConfig = {
  name: "kie";
  baseUrl: string;
  apiKeyEnv?: string;
  defaultModel?: KieModelName;
  models: Partial<{ [K in KieModelName]: KieModelOptionsByModel[K] }>;
};

export const kieProvider: KieProviderConfig = {
  name: "kie",
  baseUrl: "https://api.kie.ai",
  apiKeyEnv: "KIE_API_KEY",
  defaultModel: "gpt-5-2",
  models: {},
};

export function getKieModelSpec(model: KieModelName): KieModelSpec {
  return kieModelRegistry[model];
}

export function isKieModelName(value: string): value is KieModelName {
  return value in kieModelRegistry;
}

export function listKieModels(): KieModelName[] {
  return Object.keys(kieModelRegistry) as KieModelName[];
}

export function familyForModel(model: KieModelName): KieModelFamily {
  return kieModelRegistry[model].family;
}
