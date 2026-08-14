export type KieApiError = {
  code?: number;
  msg?: string;
  error?: {
    message?: string;
    type?: string;
  };
};

export type KieRetryableReason = "empty_response" | "rate_limited" | "temporary_upstream" | "malformed_response";

export class KieResponseError extends Error {
  readonly retryable: boolean;
  readonly reason: KieRetryableReason;
  readonly status?: number;
  readonly raw?: unknown;

  constructor(message: string, opts: { retryable?: boolean; reason: KieRetryableReason; status?: number; raw?: unknown }) {
    super(message);
    this.name = "KieResponseError";
    this.retryable = opts.retryable ?? false;
    this.reason = opts.reason;
    this.status = opts.status;
    this.raw = opts.raw;
  }
}

export function isKieErrorEnvelope(value: unknown): value is KieApiError {
  if (!value || typeof value !== "object") return false;
  const obj = value as Record<string, unknown>;
  return "error" in obj || "code" in obj || "msg" in obj;
}

export function normalizeKieError(value: unknown, status?: number): KieResponseError {
  const raw = value;

  if (isKieErrorEnvelope(value)) {
    const code = typeof value.code === "number" ? value.code : status;
    const msg = typeof value.msg === "string"
      ? value.msg
      : typeof value.error?.message === "string"
        ? value.error.message
        : `KIE request failed${status ? ` (${status})` : ""}`;

    const retryable = code === 429 || code === 408 || code === 455 || code === 500 || code === 501;
    const reason: KieRetryableReason = code === 429
      ? "rate_limited"
      : code === 408 || code === 455 || code === 500 || code === 501
        ? "temporary_upstream"
        : "malformed_response";

    return new KieResponseError(msg, { retryable, reason, status: code, raw });
  }

  return new KieResponseError(
    typeof value === "string" ? value : `KIE request failed${status ? ` (${status})` : ""}`,
    {
      retryable: status === 429 || status === 408 || status === 455 || status === 500 || status === 501,
      reason: status === 429 ? "rate_limited" : "malformed_response",
      status,
      raw,
    },
  );
}

export function isRetryableKieError(error: unknown): error is KieResponseError {
  return error instanceof KieResponseError && error.retryable;
}
