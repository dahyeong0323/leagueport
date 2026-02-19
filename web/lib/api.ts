const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

if (!API_BASE_URL) {
  throw new Error("NEXT_PUBLIC_API_BASE_URL is not defined");
}

type RequestOptions = {
  method?: string;
  body?: unknown;
  timeoutMs?: number;
};

export type CreateReportBody = {
  riot_id: string;
  region: "EUW" | "KR";
  tone: "funny" | "roast" | "sweet";
  language: string;
};

export type ReportStatusResponse = {
  status: "queued" | "processing" | "done" | "failed";
  progress: number;
  error?: string;
};

export type ReportResponse = {
  status: "done";
  report_id: string;
  sections: Array<{ title: string; content_markdown: string }>;
  generation_warning?: string | null;
  meta: {
    riot_id: string;
    region: string;
    games_analyzed: number;
    created_at: string;
    tone: string;
    language: string;
    data_source: "riot" | "fallback";
    riot_error: string | null;
    matches_fetched: number;
    puuid: string | null;
  };
};

async function fetchWithTimeout<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? 10000;
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: options.method ?? "GET",
      headers: {
        "Content-Type": "application/json"
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
      cache: "no-store"
    });

    if (!response.ok) {
      let message = `API error (${response.status})`;
      try {
        const data = await response.json();
        if (data.detail) {
          message = String(data.detail);
        }
      } catch {
        // noop
      }
      throw new Error(message);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("Request timed out. Please retry.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function createReport(body: CreateReportBody): Promise<{ report_id: string; status: string }> {
  return fetchWithTimeout("/create-report", { method: "POST", body, timeoutMs: 15000 });
}

export async function getReportStatus(reportId: string): Promise<ReportStatusResponse> {
  const query = new URLSearchParams({ report_id: reportId }).toString();
  return fetchWithTimeout(`/report-status?${query}`, { timeoutMs: 10000 });
}

export async function getReport(reportId: string): Promise<ReportResponse> {
  const query = new URLSearchParams({ report_id: reportId }).toString();
  return fetchWithTimeout(`/report?${query}`, { timeoutMs: 10000 });
}
