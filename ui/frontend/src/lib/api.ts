/**
 * API client for communicating with the FastAPI backend.
 */

import {
  EvaluateRequest,
  EvaluateResponse,
  EvalSummary,
  Example,
  Checklist,
  ScoreResult,
  Settings,
  ConnectionTestResult,
  RegistryInfo,
  StreamRequest,
  GenerateRequest,
  GenerateResponse,
  ScoreRequest,
  ScoreResponse,
  BatchSummary,
  BatchUploadResponse,
  BatchResultsPage,
  SavedChecklist,
  ChecklistSummary,
  PromptTemplate,
  PromptTemplateSummary,
  PipelineSummary,
  PipelineConfig,
  DimensionGenerateRequest,
  DimensionGenerateResponse,
  ScorerConfig,
} from "./types";

/**
 * Dynamically determine the API base URL based on the current window location.
 * When accessed via network IP, we need to use the same host but different port.
 */
function getApiBaseUrl(): string {
  // Allow explicit override via environment variable
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }

  // In browser, use the same host as the current page but on port 7771
  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:7771`;
  }

  // Server-side fallback
  return "http://localhost:7771";
}

class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${getApiBaseUrl()}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      errorData.detail || `HTTP error ${response.status}`
    );
  }

  return response.json();
}

// ---- Evaluation APIs ----

export async function evaluate(
  request: EvaluateRequest
): Promise<EvaluateResponse> {
  return fetchApi<EvaluateResponse>("/api/evaluate", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export interface StreamCallbacks {
  onChecklist: (method: string, checklist: Checklist) => void;
  onScore: (
    method: string,
    score: ScoreResult,
    items: Array<{ passed?: boolean | null; confidence?: number | null; reasoning?: string | null }>
  ) => void;
  onError: (method: string, phase: string, error: string) => void;
  onDone: () => void;
}

export async function evaluateStream(
  request: EvaluateRequest | StreamRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const url = `${getApiBaseUrl()}/api/evaluate/stream`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP error ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete events from buffer
    const lines = buffer.split("\n");
    buffer = lines.pop() || ""; // Keep incomplete line in buffer

    let currentEvent = "";
    let currentData = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7);
      } else if (line.startsWith("data: ")) {
        currentData = line.slice(6);
      } else if (line === "" && currentEvent && currentData) {
        // End of event, process it
        try {
          const data = JSON.parse(currentData);

          switch (currentEvent) {
            case "checklist":
              callbacks.onChecklist(data.method, data.checklist);
              break;
            case "score":
              callbacks.onScore(data.method, data.score, data.items);
              break;
            case "error":
              callbacks.onError(data.method, data.phase, data.error);
              break;
            case "done":
              callbacks.onDone();
              break;
          }
        } catch (e) {
          console.error("Failed to parse SSE data:", e);
        }

        currentEvent = "";
        currentData = "";
      }
    }
  }
}

// ---- Async Eval APIs ----

export async function startEvalAsync(
  request: EvaluateRequest | StreamRequest
): Promise<{ eval_id: string }> {
  return fetchApi<{ eval_id: string }>("/api/evaluate/async", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getEvalStatus(evalId: string): Promise<EvalSummary> {
  return fetchApi<EvalSummary>(`/api/evaluate/${evalId}`);
}

export async function getRecentEvals(
  limit = 10
): Promise<EvalSummary[]> {
  return fetchApi<EvalSummary[]>(`/api/evaluate/recent?limit=${limit}`);
}

export async function deleteEvaluation(
  evalId: string
): Promise<{ status: string; id: string }> {
  return fetchApi(`/api/evaluate/${evalId}`, { method: "DELETE" });
}

export async function cancelEvaluation(
  evalId: string
): Promise<{ status: string; id: string }> {
  return fetchApi(`/api/evaluate/${evalId}/cancel`, { method: "POST" });
}

// ---- Generate Only API ----

export async function evaluateGenerate(
  request: GenerateRequest,
  options?: { signal?: AbortSignal }
): Promise<GenerateResponse> {
  return fetchApi<GenerateResponse>("/api/evaluate/generate", {
    method: "POST",
    body: JSON.stringify(request),
    signal: options?.signal,
  });
}

// ---- Score Only API ----

export async function evaluateScore(
  request: ScoreRequest
): Promise<ScoreResponse> {
  return fetchApi<ScoreResponse>("/api/evaluate/score", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

// ---- Examples API ----

export async function getExamples(): Promise<Example[]> {
  return fetchApi<Example[]>("/api/examples");
}

// ---- Health API ----

export async function healthCheck(): Promise<{ status: string }> {
  return fetchApi<{ status: string }>("/api/health");
}

// ---- Settings APIs ----

export async function getSettings(): Promise<Settings> {
  return fetchApi<Settings>("/api/settings");
}

export async function updateSettings(
  settings: Partial<Settings>
): Promise<Settings> {
  return fetchApi<Settings>("/api/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export async function testConnection(
  provider: string,
  model: string,
  apiKey?: string,
  baseUrl?: string
): Promise<ConnectionTestResult> {
  return fetchApi<ConnectionTestResult>("/api/settings/test", {
    method: "POST",
    body: JSON.stringify({
      provider,
      model,
      api_key: apiKey,
      base_url: baseUrl,
    }),
  });
}

// ---- Registry APIs ----

export async function getRegistry(): Promise<RegistryInfo> {
  return fetchApi<RegistryInfo>("/api/registry");
}

export async function getPresetPrompt(
  presetName: string
): Promise<{ preset: string; prompt_text: string; generator_class: string }> {
  return fetchApi(`/api/registry/preset-prompt/${presetName}`);
}

export async function getScorerPrompt(
  scorerName: string
): Promise<{ scorer: string; prompt_text: string }> {
  return fetchApi(`/api/registry/scorer-prompt/${scorerName}`);
}

export async function getFormat(
  formatName: string
): Promise<{ format: string; text: string }> {
  return fetchApi(`/api/registry/format/${formatName}`);
}

// ---- Batch APIs ----

export async function uploadBatch(
  file: File,
  method: string,
  scorer?: string,
  pipelineMode?: string,
  checklistId?: string,
  customPrompt?: string,
  pipelineId?: string,
  scorerConfig?: ScorerConfig
): Promise<BatchUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("method", method);
  if (scorer) formData.append("scorer", scorer);
  if (pipelineMode) formData.append("pipeline_mode", pipelineMode);
  if (checklistId) formData.append("checklist_id", checklistId);
  if (customPrompt) formData.append("custom_prompt", customPrompt);
  if (pipelineId) formData.append("pipeline_id", pipelineId);
  if (scorerConfig) formData.append("scorer_config", JSON.stringify(scorerConfig));

  const url = `${getApiBaseUrl()}/api/batch/upload`;
  const response = await fetch(url, { method: "POST", body: formData });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(response.status, errorData.detail || "Upload failed");
  }
  return response.json();
}

export async function uploadBatchFromPath(
  filePath: string,
  method: string,
  scorer?: string,
  pipelineMode?: string,
  checklistId?: string,
  customPrompt?: string,
  pipelineId?: string,
  scorerConfig?: ScorerConfig
): Promise<BatchUploadResponse> {
  return fetchApi<BatchUploadResponse>("/api/batch/upload-path", {
    method: "POST",
    body: JSON.stringify({
      file_path: filePath,
      method,
      scorer,
      pipeline_mode: pipelineMode,
      checklist_id: checklistId,
      custom_prompt: customPrompt,
      pipeline_id: pipelineId,
      scorer_config: scorerConfig,
    }),
  });
}

export async function startBatch(
  batchId: string,
  provider?: string,
  model?: string,
  generatorModel?: string,
  scorerModel?: string,
  candidateModel?: string
): Promise<{ status: string }> {
  return fetchApi<{ status: string }>(`/api/batch/${batchId}/start`, {
    method: "POST",
    body: JSON.stringify({
      provider,
      model,
      generator_model: generatorModel,
      scorer_model: scorerModel,
      candidate_model: candidateModel,
    }),
  });
}

export async function listBatches(): Promise<BatchSummary[]> {
  return fetchApi<BatchSummary[]>("/api/batch");
}

export async function getBatchStatus(
  batchId: string
): Promise<BatchSummary> {
  return fetchApi<BatchSummary>(`/api/batch/${batchId}`);
}

export async function getBatchResults(
  batchId: string,
  offset = 0,
  limit = 100
): Promise<BatchResultsPage> {
  return fetchApi<BatchResultsPage>(
    `/api/batch/${batchId}/results?offset=${offset}&limit=${limit}`
  );
}

export async function cancelBatch(
  batchId: string
): Promise<{ status: string }> {
  return fetchApi<{ status: string }>(`/api/batch/${batchId}/cancel`, {
    method: "POST",
  });
}

export async function deleteBatch(
  batchId: string
): Promise<{ status: string; batch_id: string }> {
  return fetchApi(`/api/batch/${batchId}`, { method: "DELETE" });
}

export function getBatchExportUrl(
  batchId: string,
  format: "json" | "csv" = "json"
): string {
  return `${getApiBaseUrl()}/api/batch/${batchId}/export?format=${format}`;
}

// ---- Checklist Library APIs ----

export async function listChecklists(): Promise<ChecklistSummary[]> {
  return fetchApi<ChecklistSummary[]>("/api/checklists");
}

export async function getChecklist(id: string): Promise<SavedChecklist> {
  return fetchApi<SavedChecklist>(`/api/checklists/${id}`);
}

export async function createChecklist(data: {
  name: string;
  method?: string;
  level?: string;
  model?: string;
  items?: Array<{ question: string; weight: number }>;
  metadata?: Record<string, unknown>;
}): Promise<SavedChecklist> {
  return fetchApi<SavedChecklist>("/api/checklists", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateChecklist(
  id: string,
  data: {
    name?: string;
    items?: Array<{ question: string; weight: number }>;
    metadata?: Record<string, unknown>;
  }
): Promise<SavedChecklist> {
  return fetchApi<SavedChecklist>(`/api/checklists/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deleteChecklist(
  id: string
): Promise<{ status: string; id: string }> {
  return fetchApi<{ status: string; id: string }>(`/api/checklists/${id}`, {
    method: "DELETE",
  });
}

export async function importChecklist(
  file: File,
  name: string
): Promise<SavedChecklist> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", name);

  const url = `${getApiBaseUrl()}/api/checklists/import`;
  const response = await fetch(url, { method: "POST", body: formData });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(response.status, errorData.detail || "Import failed");
  }
  return response.json();
}

// ---- Prompt Template APIs ----

export async function listPromptTemplates(): Promise<PromptTemplateSummary[]> {
  return fetchApi<PromptTemplateSummary[]>("/api/prompt-templates");
}

export async function getPromptTemplate(id: string): Promise<PromptTemplate> {
  return fetchApi<PromptTemplate>(`/api/prompt-templates/${id}`);
}

export async function createPromptTemplate(data: {
  name: string;
  prompt_text: string;
  description?: string;
  metadata?: Record<string, unknown>;
}): Promise<PromptTemplate> {
  return fetchApi<PromptTemplate>("/api/prompt-templates", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updatePromptTemplate(
  id: string,
  data: {
    name?: string;
    prompt_text?: string;
    description?: string;
    metadata?: Record<string, unknown>;
  }
): Promise<PromptTemplate> {
  return fetchApi<PromptTemplate>(`/api/prompt-templates/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deletePromptTemplate(
  id: string
): Promise<{ status: string; id: string }> {
  return fetchApi<{ status: string; id: string }>(`/api/prompt-templates/${id}`, {
    method: "DELETE",
  });
}

export async function seedDefaultPrompts(): Promise<{ seeded: string[]; total: number }> {
  return fetchApi<{ seeded: string[]; total: number }>("/api/prompt-templates/seed-defaults", {
    method: "POST",
  });
}

// ---- Pipeline APIs ----

export async function listPipelines(): Promise<PipelineSummary[]> {
  return fetchApi<PipelineSummary[]>("/api/pipelines");
}

export async function getPipeline(id: string): Promise<PipelineConfig> {
  return fetchApi<PipelineConfig>(`/api/pipelines/${id}`);
}

export async function createPipeline(data: {
  name: string;
  description?: string;
  generator_class: string;
  generator_prompt: string;
  scorer_class?: string;
  scorer_prompt?: string;
  output_format?: string;
  scorer_config?: ScorerConfig;
}): Promise<PipelineConfig> {
  return fetchApi<PipelineConfig>("/api/pipelines", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updatePipeline(
  id: string,
  data: {
    name?: string;
    description?: string;
    generator_class?: string;
    generator_prompt?: string;
    scorer_class?: string;
    scorer_prompt?: string;
    output_format?: string;
    scorer_config?: ScorerConfig;
  }
): Promise<PipelineConfig> {
  return fetchApi<PipelineConfig>(`/api/pipelines/${id}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function deletePipeline(
  id: string
): Promise<{ status: string; id: string }> {
  return fetchApi<{ status: string; id: string }>(`/api/pipelines/${id}`, {
    method: "DELETE",
  });
}

// ---- Clear Data API ----

export async function clearAllData(): Promise<{ status: string; details: Array<{ directory: string; files_deleted: number }> }> {
  return fetchApi("/api/settings/data", { method: "DELETE" });
}

// ---- Checklist Builder APIs ----

export async function generateDimensionChecklist(
  request: DimensionGenerateRequest
): Promise<DimensionGenerateResponse> {
  return fetchApi<DimensionGenerateResponse>("/api/checklist-builder/dimension", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export { ApiError };
