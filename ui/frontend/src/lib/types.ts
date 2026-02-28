/**
 * TypeScript interfaces for the Auto-Checklists UI.
 */

export interface ChecklistItem {
  question: string;
  weight: number;
  passed?: boolean | null;
  confidence?: number | null;
  reasoning?: string | null;
}

// Available models for evaluation
export const AVAILABLE_MODELS = [
  { value: "openai/gpt-5-mini", label: "GPT-5 Mini (Recommended)" },
  { value: "openai/gpt-4o-mini", label: "GPT-4o Mini" },
  { value: "openai/gpt-4o", label: "GPT-4o" },
  { value: "anthropic/claude-sonnet-4", label: "Claude Sonnet 4" },
  { value: "anthropic/claude-3.5-sonnet", label: "Claude 3.5 Sonnet" },
] as const;

export const DEFAULT_MODEL = "openai/gpt-5-mini";

export interface Checklist {
  items: ChecklistItem[];
  source_method: string;
  metadata: Record<string, unknown>;
}

export interface ScoreResult {
  score: number;
  max_score: number;
  percentage: number;
  primary_metric?: string;  // "pass", "weighted", "normalized"
  item_results: Array<{
    question: string;
    passed?: boolean | null;
    confidence?: number | null;
    reasoning?: string | null;
  }>;
}

/** Merge score item_results (passed, confidence, reasoning) into checklist items. */
export function mergeScoreIntoItems(
  items: ChecklistItem[],
  score: ScoreResult | undefined | null,
): ChecklistItem[] {
  if (!score?.item_results?.length) return items;
  return items.map((item, i) => {
    const sr = score.item_results[i];
    if (!sr) return item;
    return {
      ...item,
      passed: sr.passed ?? item.passed,
      confidence: sr.confidence ?? item.confidence,
      reasoning: sr.reasoning ?? item.reasoning,
    };
  });
}

export interface MethodResult {
  checklist: Checklist;
  score: ScoreResult;
}

export interface EvaluateRequest {
  input: string;
  target: string;
  reference?: string | null;
  model?: string;
  generator_provider?: string;
  generator_model?: string;
  scorer_provider?: string;
  scorer_model?: string;
  custom_prompt?: string | null;
  custom_scorer_prompt?: string | null;
  pipeline_ids?: string[];
}

export interface EvaluateResponse {
  results: Record<string, MethodResult>;
}

export interface Example {
  name: string;
  input: string;
  target: string;
  reference: string | null;
}

export type MethodName =
  | "tick"
  | "rlcf_direct"
  | "rlcf_candidates_only"
  | "rlcf_candidate"
  | "rocketeval";

export const METHOD_LABELS: Record<MethodName, string> = {
  tick: "TICK",
  rlcf_direct: "RLCF Direct",
  rlcf_candidates_only: "RLCF Candidates Only",
  rlcf_candidate: "RLCF Candidate",
  rocketeval: "RocketEval",
};

export const METHOD_DESCRIPTIONS: Record<MethodName, string> = {
  tick: "Instruction-only checklist generation",
  rlcf_direct: "Direct checklist from instruction + reference",
  rlcf_candidates_only: "Auto-generates candidates, compares without reference",
  rlcf_candidate: "Auto-generates candidates, compares against reference",
  rocketeval: "Confidence-aware checklist with logprobs",
};

export const REQUIRES_REFERENCE: Record<MethodName, boolean> = {
  tick: false,
  rlcf_direct: true,
  rlcf_candidates_only: false,
  rlcf_candidate: true,
  rocketeval: true,
};

export type GeneratorClass =
  | "direct"
  | "contrastive"
  | "inductive"
  | "deductive"
  | "interactive";

export const GENERATOR_CLASSES: Record<
  GeneratorClass,
  { label: string; level: "instance" | "corpus"; description: string; methods: string[] }
> = {
  direct: {
    label: "Direct",
    level: "instance",
    description: "Generate checklist directly from input",
    methods: ["tick", "rlcf_direct", "rocketeval"],
  },
  contrastive: {
    label: "Contrastive",
    level: "instance",
    description: "Auto-generate candidates, then derive checklist by contrasting",
    methods: ["rlcf_candidate", "rlcf_candidates_only"],
  },
  inductive: {
    label: "Inductive",
    level: "corpus",
    description: "Observations to criteria (bottom-up)",
    methods: ["feedback"],
  },
  deductive: {
    label: "Deductive",
    level: "corpus",
    description: "Dimensions to criteria (top-down, deductive)",
    methods: ["checkeval"],
  },
  interactive: {
    label: "Interactive",
    level: "corpus",
    description: "Evaluation sessions to criteria (protocol analysis)",
    methods: ["interacteval"],
  },
};

/** Map preset method names to their generator class labels for display.
 *  e.g. "feedback" → "Inductive", "checkeval" → "Dimension" */
export const METHOD_TO_CLASS_LABEL: Record<string, string> = Object.fromEntries(
  Object.values(GENERATOR_CLASSES).flatMap((cls) =>
    cls.methods.map((m) => [m, cls.label])
  )
);

// ---- Scorer Config types ----

export interface ScorerConfig {
  mode: "batch" | "item";
  primary_metric: "pass" | "weighted" | "normalized";
  capture_reasoning: boolean;
}

export const DEFAULT_SCORER_CONFIG: ScorerConfig = {
  mode: "batch",
  primary_metric: "pass",
  capture_reasoning: false,
};

/** Maps scorer config → format file name for output format display */
export function scorerConfigToFormatName(config: ScorerConfig): string {
  if (config.mode === "batch") {
    return config.capture_reasoning ? "batch_scoring_reasoned" : "batch_scoring";
  }
  return config.capture_reasoning ? "item_scoring_reasoned" : "item_scoring";
}

/** Maps scorer config → scorer prompt name for loading prompt templates */
export function scorerConfigToPromptName(config: ScorerConfig): string {
  return config.mode;  // "batch" or "item" — rlcf/rocketeval handled by custom_scorer_prompt
}

/** Instance-level classes only (used by Compare form) */
export const INSTANCE_GENERATOR_CLASSES = {
  direct: GENERATOR_CLASSES.direct,
  contrastive: GENERATOR_CLASSES.contrastive,
} as const;

// ---- Stream/Generate/Score request types ----

export interface StreamRequest {
  input: string;
  target: string;
  reference?: string | null;
  methods?: string[];
  scorer?: string;
  scorer_config?: ScorerConfig;
  provider?: string;
  model?: string;
  generator_provider?: string;
  generator_model?: string;
  scorer_provider?: string;
  scorer_model?: string;
  custom_prompt?: string | null;
  custom_scorer_prompt?: string | null;
  candidate_model?: string | null;
  pipeline_ids?: string[];
}

export interface GenerateRequest {
  input: string;
  target?: string | null;
  reference?: string | null;
  method: string;
  provider?: string;
  model?: string;
  custom_prompt?: string | null;
  candidate_model?: string | null;
}

export interface GenerateResponse {
  method: string;
  checklist: Checklist;
}

export interface ScoreRequest {
  input?: string;
  target: string;
  checklist_items: Array<{ question: string; weight?: number }>;
  scorer?: string;
  scorer_config?: ScorerConfig;
  provider?: string;
  model?: string;
  custom_scorer_prompt?: string | null;
}

export interface ScoreResponse {
  score: ScoreResult;
}

// ---- Batch types ----

export type BatchStatus = "pending" | "running" | "completed" | "failed" | "cancelled";
export type PipelineMode = "full" | "generate_only" | "score_only";

export interface BatchSummary {
  batch_id: string;
  status: BatchStatus;
  total: number;
  completed: number;
  failed: number;
  mean_score: number | null;
  macro_pass_rate: number | null;
  micro_pass_rate: number | null;
  started_at: string | null;
  completed_at: string | null;
  error?: string;
  config?: {
    method?: string;
    scorer?: string;
    filename?: string;
    total_items?: number;
    pipeline_mode?: PipelineMode;
    checklist_id?: string;
    checklist_name?: string;
    checklist_level?: string;
    checklist_method?: string;
    pipeline_id?: string;
    pipeline_name?: string;
    provider?: string;
    model?: string;
    generator_model?: string;
    scorer_model?: string;
    candidate_model?: string;
  };
}

export interface BatchResult {
  index: number;
  input: string;
  target: string;
  pass_rate: number | null;
  primary_score: number | null;
  primary_metric?: string | null;
  total_score: number | null;
  item_scores?: Array<{
    item_id: string;
    question?: string;
    answer: string;
    reasoning?: string;
  }>;
  checklist_items?: Array<{ question: string; weight: number }>;
}

export interface BatchUploadResponse {
  batch_id: string;
  total_items: number;
  filename: string;
}

export interface BatchResultsPage {
  results: BatchResult[];
  offset: number;
  limit: number;
}

// ---- Saved Checklist types ----

export interface SavedChecklist {
  id: string;
  name: string;
  method: string;
  level: string;
  model: string;
  items: Array<{ question: string; weight: number }>;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface ChecklistSummary {
  id: string;
  name: string;
  method: string;
  level: string;
  item_count: number;
  created_at: string;
}

// ---- Async eval types ----

export interface EvalSummary {
  eval_id: string;
  status: "running" | "completed" | "failed" | "cancelled";
  methods: string[];
  completed_methods: string[];
  results: Record<string, {
    checklist?: Checklist;
    score?: ScoreResult;
    error?: string;
  }>;
  request: {
    input: string;
    target: string;
    reference?: string | null;
    methods: string[];
    model: string;
    provider: string;
    generator_model?: string;
    generator_provider?: string;
    scorer_model?: string;
    scorer_provider?: string;
  };
  started_at: string;
  completed_at?: string;
  error?: string;
}

// Settings types
export interface Settings {
  openrouter_api_key: string;
  openai_api_key: string;
  default_provider: string;
  default_model: string;
  vllm_base_url: string;
}

export interface ConnectionTestResult {
  success: boolean;
  message: string;
  provider: string;
  model?: string;
  latency_ms?: number;
}

export interface GeneratorInfo {
  name: string;
  level: "instance" | "corpus";
  generator_class?: GeneratorClass;
  description: string;
  detail?: string;
  requires_reference?: boolean;
  recommended_scorer?: string;
}

export interface ScorerInfo {
  name: string;
  method?: string;
  description: string;
  detail?: string;
}

export interface RegistryInfo {
  generators: GeneratorInfo[];
  scorers: ScorerInfo[];
  default_scorers?: Record<string, ScorerConfig | string>;
}

// ---- Pipeline types ----

export interface PipelineSummary {
  id: string;
  name: string;
  description: string;
  generator_class: "direct" | "contrastive";
  scorer_class: string;
  scorer_config?: ScorerConfig;
  created_at: string;
  updated_at: string;
}

export interface PipelineConfig {
  id: string;
  name: string;
  description: string;
  generator_class: "direct" | "contrastive";
  generator_prompt: string;
  scorer_class: string;
  scorer_config?: ScorerConfig;
  scorer_prompt: string;
  output_format: string;
  created_at: string;
  updated_at: string;
}

// ---- Prompt Template types ----

export interface PromptTemplate {
  id: string;
  name: string;
  prompt_text: string;
  description: string;
  placeholders: string[];
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface PromptTemplateSummary {
  id: string;
  name: string;
  description: string;
  placeholders: string[];
  created_at: string;
  updated_at: string;
  metadata?: Record<string, string>;
}

// ---- Checklist Builder types ----

export interface DimensionInput {
  name: string;
  definition: string;
  sub_dimensions: string[];
}

export interface DimensionGenerateRequest {
  dimensions: DimensionInput[];
  task_type?: string;
  augmentation_mode?: string;
  provider?: string;
  model?: string;
}

export interface DimensionGenerateResponse {
  checklist: Checklist;
}
