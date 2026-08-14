import { isRetryableKieError, KieResponseError } from "./errors.ts";

export type RetryOptions = {
  maxAttempts?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function jitter(ms: number): number {
  const spread = Math.max(1, Math.round(ms * 0.25));
  return Math.max(0, ms + Math.floor((Math.random() * spread * 2) - spread));
}

export async function retryKie<T>(fn: () => Promise<T>, opts: RetryOptions = {}): Promise<T> {
  const maxAttempts = Math.max(1, opts.maxAttempts ?? 3);
  const baseDelayMs = Math.max(0, opts.baseDelayMs ?? 750);
  const maxDelayMs = Math.max(baseDelayMs, opts.maxDelayMs ?? 5000);

  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const retryable = isRetryableKieError(error) || (error instanceof KieResponseError && error.retryable);
      if (!retryable || attempt >= maxAttempts) throw error;

      const delay = Math.min(maxDelayMs, baseDelayMs * (2 ** (attempt - 1)));
      await sleep(jitter(delay));
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}
