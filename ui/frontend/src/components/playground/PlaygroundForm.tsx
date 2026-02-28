"use client";

import { useState, useEffect, useRef } from "react";
import { useQueryClient, useQuery } from "@tanstack/react-query";
import {
  Checklist,
  ScoreResult,
  StreamRequest,
  GenerateRequest,
  ScorerConfig,
  DEFAULT_SCORER_CONFIG,
  scorerConfigToFormatName,
  scorerConfigToPromptName,
  mergeScoreIntoItems,
} from "@/lib/types";
import {
  evaluateGenerate,
  startEvalAsync,
  deleteEvaluation,
  cancelEvaluation,
  createChecklist,
  createPromptTemplate,
  createPipeline,
  getPromptTemplate,

  getScorerPrompt,
  getFormat,
} from "@/lib/api";
import { EXAMPLES } from "@/lib/examples";
import { useDefaultModel, usePromptTemplates, useEvalStatus } from "@/lib/hooks";
import { Button } from "@/components/ui/Button";
import { Textarea } from "@/components/ui/Textarea";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";

import { Dialog, DialogContent } from "@/components/ui/Dialog";
import { ChecklistDisplay } from "@/components/ChecklistDisplay";
import { ModelSelector } from "@/components/layout/ModelSelector";
import { ExampleLoader } from "@/components/ExampleLoader";
import { PromptEditor } from "./PromptEditor";
import { ScorerConfigPanel } from "./ScorerConfigPanel";

type PromptTab = "generator" | "scorer";

const EVAL_STORAGE_KEY = "playground_eval_id";
const EDITOR_STORAGE_KEY = "playground_editor";

interface PlaygroundResult {
  checklist?: Checklist;
  score?: ScoreResult;
  phase?: "generating" | "scoring";
  error?: string;
}

/* ── PlaceholderInfo ── click-toggle popover for placeholder badges ── */

function PlaceholderInfo({
  items,
}: {
  items: { placeholder: string; label: string; color: string }[];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node))
        setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-bold leading-none"
        style={{
          backgroundColor: open ? "var(--accent-primary)" : "var(--surface-elevated)",
          color: open ? "white" : "var(--text-tertiary)",
          border: open ? "none" : "1px solid var(--border-strong)",
        }}
        title="Available placeholders"
      >
        ?
      </button>
      {open && (
        <div
          className="absolute left-0 top-6 z-50 rounded-md p-2.5 flex flex-col gap-1.5 whitespace-nowrap"
          style={{
            backgroundColor: "white",
            border: "1px solid var(--border)",
            boxShadow: "var(--shadow-sm)",
          }}
        >
          {items.map((item) => (
            <div key={item.placeholder} className="flex items-center gap-2 text-xs">
              <code
                className="px-1 py-0.5 rounded"
                style={{
                  backgroundColor: `${item.color}15`,
                  color: item.color,
                  fontFamily: "var(--font-mono)",
                }}
              >
                {item.placeholder}
              </code>
              <span style={{ color: "var(--text-tertiary)" }}>{item.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Default prompts ── */

const DEFAULT_DIRECT_PROMPT = `Generate a list of 2-5 yes/no checklist questions to evaluate the quality of a response to a given input. Each question should be framed such that a "yes" answer corresponds to a desirable quality.

Input: 
{input}`;

const DEFAULT_CONTRASTIVE_PROMPT = `Generate a list of 2-5 yes/no checklist questions to evaluate the quality of a response to a given input. Each question should be framed such that a "yes" answer corresponds to a desirable quality.

You will be given a set of candidate responses as context. Use them to identify ways in which potential responses may deviate from the input and generate checklist questions to flag such cases.

Input: 
{input}

Candidates: 
{candidates}`;

/* ── PlaygroundForm ── */

export function PlaygroundForm() {
  // Prompt tabs
  const [promptTab, setPromptTab] = useState<PromptTab>("generator");
  const [generatorClass, setGeneratorClass] = useState<"direct" | "contrastive">("direct");
  const [generatorPrompt, setGeneratorPrompt] = useState(DEFAULT_DIRECT_PROMPT);
  const [scorerPrompt, setScorerPrompt] = useState("");
  const [scorerConfig, setScorerConfig] = useState<ScorerConfig>(DEFAULT_SCORER_CONFIG);
  const defaultScorerPromptRef = useRef("");
  const [selectedGenTemplate, setSelectedGenTemplate] = useState("");
  const [selectedScorerTemplate, setSelectedScorerTemplate] = useState("");
  const [savePromptOpen, setSavePromptOpen] = useState(false);
  const [promptName, setPromptName] = useState("");
  const [promptDesc, setPromptDesc] = useState("");
  const [savePipelineOpen, setSavePipelineOpen] = useState(false);
  const [pipelineName, setPipelineName] = useState("");
  const [pipelineDesc, setPipelineDesc] = useState("");
  const [savePromptFeedback, setSavePromptFeedback] = useState<string | null>(null);
  const [savePipelineFeedback, setSavePipelineFeedback] = useState<string | null>(null);

  // Candidate model for contrastive generators
  const {
    provider: candidateProvider,
    setProvider: setCandidateProvider,
    model: candidateModel,
    setModel: setCandidateModel,
  } = useDefaultModel();

  const [genFormatName, setGenFormatName] = useState("checklist");
  const [showGenFormatter, setShowGenFormatter] = useState(false);
  const [showScorerFormatter, setShowScorerFormatter] = useState(false);

  // Input fields
  const [input, setInput] = useState("");
  const [target, setTarget] = useState("");
  const [reference, setReference] = useState("");
  const [showReference, setShowReference] = useState(false);
  const [selectedExample, setSelectedExample] = useState("");

  // Model selectors
  const {
    provider: genProvider,
    setProvider: setGenProvider,
    model: genModel,
    setModel: setGenModel,
  } = useDefaultModel();
  const {
    provider: scoreProvider,
    setProvider: setScoreProvider,
    model: scoreModel,
    setModel: setScoreModel,
  } = useDefaultModel();

  // Result state
  const [evalId, setEvalId] = useState<string | null>(null);
  const [result, setResult] = useState<PlaygroundResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const abortRef = useRef<AbortController | null>(null);

  // Data hooks
  const { data: promptTemplates } = usePromptTemplates();
  const { data: evalData } = useEvalStatus(evalId);
  const queryClient = useQueryClient();

  // Hydrate evalId from sessionStorage on mount
  useEffect(() => {
    const stored = sessionStorage.getItem(EVAL_STORAGE_KEY);
    if (stored) setEvalId(stored);
  }, []);

  // Hydrate editor state from sessionStorage on mount
  const [editorHydrated, setEditorHydrated] = useState(false);
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(EDITOR_STORAGE_KEY);
      if (raw) {
        const s = JSON.parse(raw);
        if (s.generatorClass) setGeneratorClass(s.generatorClass);
        if (s.generatorPrompt) setGeneratorPrompt(s.generatorPrompt);
        if (s.scorerConfig) setScorerConfig(s.scorerConfig);
        if (s.scorerPrompt) setScorerPrompt(s.scorerPrompt);
        if (s.genFormatName) setGenFormatName(s.genFormatName);
        if (s.input) setInput(s.input);
        if (s.target) setTarget(s.target);
        if (s.reference) { setReference(s.reference); setShowReference(true); }
      }
    } catch { /* ignore */ }
    setEditorHydrated(true);
  }, []);

  // Persist editor state to sessionStorage on changes
  useEffect(() => {
    if (!editorHydrated) return;
    sessionStorage.setItem(EDITOR_STORAGE_KEY, JSON.stringify({
      generatorClass, generatorPrompt, scorerConfig, scorerPrompt,
      genFormatName, input, target, reference,
    }));
  }, [editorHydrated, generatorClass, generatorPrompt, scorerConfig, scorerPrompt, genFormatName, input, target, reference]);

  // Sync polled eval data → local result state
  useEffect(() => {
    if (!evalData?.results) return;
    // Find the first (and only) method key
    const methodKey = Object.keys(evalData.results)[0];
    if (!methodKey) return;
    const data = evalData.results[methodKey] as {
      checklist?: Checklist;
      score?: ScoreResult;
      error?: string;
    };
    if (data.error) {
      setResult({ error: data.error });
      setLoading(false);
      return;
    }
    const isCompleted = evalData.completed_methods?.includes(methodKey);
    const hasChecklist = !!data.checklist;
    const hasScore = !!data.score;
    setResult({
      checklist: data.checklist,
      score: data.score,
      phase: !isCompleted ? (hasChecklist ? "scoring" : "generating") : undefined,
      error: undefined,
    });
    if (isCompleted && hasChecklist && hasScore) {
      setLoading(false);
    }
    // Restore input fields from the saved request
    if (evalData.request) {
      if (evalData.request.input && !input) setInput(evalData.request.input);
      if (evalData.request.target && !target) setTarget(evalData.request.target);
      if (evalData.request.reference && !reference) {
        setReference(evalData.request.reference);
        setShowReference(true);
      }
    }
    if (evalData.status === "completed" || evalData.status === "failed" || evalData.status === "cancelled") {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally sync only when evalData changes; including input/target/reference would overwrite user edits
  }, [evalData]);

  // Cache all generator state per class so switching back restores everything
  interface GenClassState {
    prompt: string;
    templateId: string;
    formatName: string;
  }
  const prevClassRef = useRef<string | null>(null);
  const classCacheRef = useRef<Record<string, GenClassState>>({
    direct: { prompt: DEFAULT_DIRECT_PROMPT, templateId: "", formatName: "checklist" },
    contrastive: { prompt: DEFAULT_CONTRASTIVE_PROMPT, templateId: "", formatName: "checklist" },
  });
  useEffect(() => {
    if (!editorHydrated) return;
    if (prevClassRef.current === null) {
      // First run after hydration — record class + cache current state
      prevClassRef.current = generatorClass;
      classCacheRef.current[generatorClass] = {
        prompt: generatorPrompt,
        templateId: selectedGenTemplate,
        formatName: genFormatName,
      };
      return;
    }
    if (prevClassRef.current === generatorClass) return;
    // Save current state for the class we're leaving
    classCacheRef.current[prevClassRef.current] = {
      prompt: generatorPrompt,
      templateId: selectedGenTemplate,
      formatName: genFormatName,
    };
    prevClassRef.current = generatorClass;
    // Restore cached state for the class we're entering
    const cached = classCacheRef.current[generatorClass];
    if (cached) {
      setGeneratorPrompt(cached.prompt);
      setSelectedGenTemplate(cached.templateId);
      setGenFormatName(cached.formatName);
    } else {
      setGeneratorPrompt(
        generatorClass === "contrastive"
          ? DEFAULT_CONTRASTIVE_PROMPT
          : DEFAULT_DIRECT_PROMPT
      );
      setSelectedGenTemplate("");
      setGenFormatName("checklist");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- only trigger on class change; refs track state internally
  }, [editorHydrated, generatorClass]);

  // Load scorer prompt when scorer mode changes (skip initial if hydrated)
  const prevScorerModeRef = useRef<string | null>(null);
  useEffect(() => {
    if (!editorHydrated) return;
    const promptName = scorerConfigToPromptName(scorerConfig);
    if (prevScorerModeRef.current === null) {
      // First run after hydration — record, only fetch if no prompt hydrated
      prevScorerModeRef.current = promptName;
      if (scorerPrompt) return; // already hydrated
    }
    if (prevScorerModeRef.current === promptName && scorerPrompt) return;
    prevScorerModeRef.current = promptName;
    getScorerPrompt(promptName)
      .then((resp) => {
        setScorerPrompt(resp.prompt_text);
        defaultScorerPromptRef.current = resp.prompt_text;
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps -- only trigger on scorer config change; ref tracks previous value
  }, [editorHydrated, scorerConfig]);

  // Fetch generator output format (changes with generator preset)
  const { data: genFormat } = useQuery({
    queryKey: ["format", genFormatName],
    queryFn: () => getFormat(genFormatName),
    staleTime: Infinity,
  });

  // Fetch scorer output format (changes with scorer config)
  const scorerFormatName = scorerConfigToFormatName(scorerConfig);
  const { data: scorerFormat } = useQuery({
    queryKey: ["format", scorerFormatName],
    queryFn: () => getFormat(scorerFormatName),
    staleTime: Infinity,
  });

  // Filter templates by type for each tab, and by generator class for generators
  const generatorTemplates = promptTemplates?.filter(
    (t) => {
      if (t.metadata?.type && t.metadata.type !== "generator") return false;
      // If template has a generator_class, it must match the current selection
      if (t.metadata?.generator_class) {
        return t.metadata.generator_class === generatorClass;
      }
      // Templates without generator_class (user-created) show for all
      return true;
    }
  );
  const scorerTemplates = promptTemplates?.filter(
    (t) => t.metadata?.type === "scorer"
  );

  const promptNeedsReference = generatorPrompt.includes("{reference}");

  const validate = () => {
    const errs: Record<string, string> = {};
    if (!input.trim()) errs.input = "Input is required";
    if (!target.trim()) errs.target = "Target is required";
    if (!generatorPrompt.trim()) errs.prompt = "Generator prompt is required";
    if (promptNeedsReference && !reference.trim()) {
      errs.reference = "This prompt uses {reference} — a reference is required";
      setShowReference(true);
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    if (evalId) {
      cancelEvaluation(evalId).catch(() => {});
    }
    setLoading(false);
    setResult((prev) => prev ? { ...prev, phase: undefined } : null);
  };

  const handleClear = () => {
    if (evalId) {
      deleteEvaluation(evalId).catch(() => {});
      sessionStorage.removeItem(EVAL_STORAGE_KEY);
    }
    setEvalId(null);
    setResult(null);
    setInput("");
    setTarget("");
    setReference("");
    setSelectedExample("");
  };

  const handleGenerateAndScore = async () => {
    if (!validate()) return;

    setLoading(true);
    setResult({ phase: "generating" });

    const baseMethod = generatorClass === "direct" ? "tick" : "rlcf_candidates_only";
    const request: StreamRequest = {
      input: input.trim(),
      target: target.trim(),
      reference: reference.trim() || null,
      methods: [baseMethod],
      scorer_config: scorerConfig,
      generator_provider: genProvider,
      generator_model: genModel,
      scorer_provider: scoreProvider,
      scorer_model: scoreModel,
      custom_prompt: generatorPrompt,
      custom_scorer_prompt: scorerPrompt.trim() || null,
      candidate_model: generatorClass === "contrastive" ? candidateModel : null,
    };

    try {
      const resp = await startEvalAsync(request);
      setEvalId(resp.eval_id);
      sessionStorage.setItem(EVAL_STORAGE_KEY, resp.eval_id);
    } catch (err) {
      setResult({
        error: err instanceof Error ? err.message : "Failed to start evaluation",
        phase: undefined,
      });
      setLoading(false);
    }
  };

  const handleGenerateOnly = async () => {
    if (!validate()) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setResult({ phase: "generating" });

    const request: GenerateRequest = {
      input: input.trim(),
      target: target.trim() || null,
      reference: reference.trim() || null,
      method: generatorClass === "direct" ? "tick" : "rlcf_candidates_only",
      provider: genProvider,
      model: genModel,
      custom_prompt: generatorPrompt,
      candidate_model: generatorClass === "contrastive" ? candidateModel : null,
    };

    try {
      const resp = await evaluateGenerate(request, { signal: controller.signal });
      setResult({
        checklist: resp.checklist,
        phase: undefined,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setResult({
        error: err instanceof Error ? err.message : "Generation failed",
        phase: undefined,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSaveChecklist = async () => {
    if (!result?.checklist) return;
    try {
      await createChecklist({
        name: `Playground ${new Date().toLocaleString()}`,
        method: "custom",
        level: "instance",
        items: result.checklist.items.map((i) => ({
          question: i.question,
          weight: i.weight,
        })),
      });
      queryClient.invalidateQueries({ queryKey: ["checklists"] });
    } catch {
      // ignore
    }
  };

  const handleSavePrompt = async () => {
    if (!promptName.trim()) return;
    const isGenerator = promptTab === "generator";
    const promptText = isGenerator ? generatorPrompt : scorerPrompt;
    if (!promptText.trim()) return;
    try {
      await createPromptTemplate({
        name: promptName.trim(),
        prompt_text: promptText,
        description: promptDesc.trim() || undefined,
        metadata: isGenerator
          ? { type: "generator", generator_class: generatorClass }
          : { type: "scorer" },
      });
      queryClient.invalidateQueries({ queryKey: ["prompt-templates"] });
      setSavePromptOpen(false);
      setPromptName("");
      setPromptDesc("");
      setSavePromptFeedback("Saved!");
      setTimeout(() => setSavePromptFeedback(null), 2000);
    } catch {
      setSavePromptFeedback("Failed to save");
      setTimeout(() => setSavePromptFeedback(null), 2000);
    }
  };

  const handleSavePipeline = async () => {
    if (!pipelineName.trim()) return;
    try {
      await createPipeline({
        name: pipelineName.trim(),
        description: pipelineDesc.trim(),
        generator_class: generatorClass,
        generator_prompt: generatorPrompt,
        scorer_config: scorerConfig,
        scorer_prompt: scorerPrompt,
        output_format: genFormatName,
      });
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      setSavePipelineOpen(false);
      setPipelineName("");
      setPipelineDesc("");
      setSavePipelineFeedback("Saved!");
      setTimeout(() => setSavePipelineFeedback(null), 2000);
    } catch {
      setSavePipelineFeedback("Failed to save");
      setTimeout(() => setSavePipelineFeedback(null), 2000);
    }
  };

  const handleLoadTemplate = (templateId: string) => {
    if (!templateId) return;
    if (promptTab === "generator") {
      setSelectedGenTemplate(templateId);
    } else {
      setSelectedScorerTemplate(templateId);
    }
    getPromptTemplate(templateId).then((full) => {
      if (promptTab === "generator") {
        setGeneratorPrompt(full.prompt_text);
        // Auto-expand reference field if the template uses {reference}
        if (full.prompt_text.includes("{reference}")) setShowReference(true);
        // Auto-set output format from template metadata (e.g. "checklist" or "weighted_checklist")
        const formatName = full.metadata?.format_name as string | undefined;
        if (formatName) setGenFormatName(formatName);
        // Auto-set scorer config to match the preset's default
        const defaultScorer = full.metadata?.default_scorer as ScorerConfig | string | undefined;
        if (defaultScorer && typeof defaultScorer === "object") {
          setScorerConfig(defaultScorer);
        }
        // Auto-switch generator class based on template metadata
        const genClass = full.metadata?.generator_class as string | undefined;
        const targetClass = (genClass === "direct" || genClass === "contrastive") ? genClass : generatorClass;
        // Keep per-class cache in sync with loaded template (cache under target class)
        const effectiveFormat = formatName || genFormatName;
        classCacheRef.current[targetClass] = {
          prompt: full.prompt_text,
          templateId,
          formatName: effectiveFormat,
        };
        if (targetClass !== generatorClass) {
          setGeneratorClass(targetClass);
        }
      } else {
        setScorerPrompt(full.prompt_text);
        // Auto-set scorer config from template metadata
        const templateConfig = full.metadata?.scorer_config as ScorerConfig | undefined;
        if (templateConfig && typeof templateConfig === "object") {
          setScorerConfig({
            mode: templateConfig.mode ?? "batch",
            primary_metric: templateConfig.primary_metric ?? "pass",
            capture_reasoning: templateConfig.capture_reasoning ?? false,
          });
        } else {
          // Legacy fallback: parse "scorer:batch" or "scorer:item" from preset field
          const preset = full.metadata?.preset as string | undefined;
          if (preset?.startsWith("scorer:")) {
            const scorerMode = preset.slice("scorer:".length);
            if (scorerMode === "batch" || scorerMode === "item") {
              setScorerConfig((prev) => ({ ...prev, mode: scorerMode }));
            }
          }
        }
      }
    });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* LEFT PANEL — Prompt Configuration */}
      <div className="space-y-5">
        {/* Generator / Scorer Prompt Tab Toggle */}
        <div className="flex items-center gap-3">
          <span
            className="text-sm font-semibold"
            style={{ color: "var(--text-primary)", fontFamily: "var(--font-serif)" }}
          >
            Checklist
          </span>
          <div
            className="inline-flex rounded-md p-0.5"
            style={{ backgroundColor: "var(--surface-elevated)" }}
          >
            <button
              onClick={() => setPromptTab("generator")}
              className="px-4 py-1.5 text-sm font-medium rounded transition-all"
              style={{
                backgroundColor:
                  promptTab === "generator" ? "white" : "transparent",
                color:
                  promptTab === "generator"
                    ? "var(--accent-primary)"
                    : "var(--text-secondary)",
                boxShadow:
                  promptTab === "generator"
                    ? "0 1px 3px rgba(0,0,0,0.1)"
                    : "none",
              }}
            >
              Generator
            </button>
            <button
              onClick={() => setPromptTab("scorer")}
              className="px-4 py-1.5 text-sm font-medium rounded transition-all"
              style={{
                backgroundColor:
                  promptTab === "scorer" ? "white" : "transparent",
                color:
                  promptTab === "scorer"
                    ? "var(--accent-primary)"
                    : "var(--text-secondary)",
                boxShadow:
                  promptTab === "scorer"
                    ? "0 1px 3px rgba(0,0,0,0.1)"
                    : "none",
              }}
            >
              Scorer
            </button>
          </div>
        </div>

        {/* Generator Prompt Tab */}
        {promptTab === "generator" && (
          <div className="space-y-3">
            {/* Generator class selector */}
            <div className="flex items-center gap-3">
              <label
                className="text-xs font-medium"
                style={{ color: "var(--text-tertiary)" }}
              >
                Generator Class
              </label>
              <div
                className="inline-flex rounded-md p-0.5"
                style={{ backgroundColor: "var(--surface-elevated)" }}
              >
                {(["direct", "contrastive"] as const).map((cls) => (
                  <button
                    key={cls}
                    type="button"
                    onClick={() => setGeneratorClass(cls)}
                    disabled={!!result || loading}
                    className="px-3 py-1 text-xs font-medium rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      backgroundColor: generatorClass === cls ? "white" : "transparent",
                      color: generatorClass === cls ? "var(--accent-primary)" : "var(--text-secondary)",
                      boxShadow: generatorClass === cls ? "0 1px 3px rgba(0,0,0,0.1)" : "none",
                    }}
                  >
                    {cls === "direct" ? "Direct" : "Contrastive"}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <label
                  className="text-sm font-medium"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Prompt Editor
                </label>
                <PlaceholderInfo
                  items={[
                    { placeholder: "{input}", label: "instruction", color: "#1a7f37" },
                    { placeholder: "{target}", label: "response", color: "#0969da" },
                    { placeholder: "{reference}", label: "gold ref", color: "#8250df" },
                    ...(generatorClass === "contrastive"
                      ? [{ placeholder: "{candidates}", label: "auto-generated alternative responses from candidate model", color: "#cf222e" }]
                      : []),
                  ]}
                />
              </div>
              <div className="flex items-center gap-2">
                {generatorPrompt !== (generatorClass === "contrastive" ? DEFAULT_CONTRASTIVE_PROMPT : DEFAULT_DIRECT_PROMPT) && (
                  <button
                    type="button"
                    onClick={() => {
                      setGeneratorPrompt(generatorClass === "contrastive" ? DEFAULT_CONTRASTIVE_PROMPT : DEFAULT_DIRECT_PROMPT);
                      setSelectedGenTemplate("");
                    }}
                    className="text-xs font-medium hover:underline"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Reset
                  </button>
                )}
                {generatorTemplates && generatorTemplates.length > 0 && (
                  <select
                    value={selectedGenTemplate}
                    onChange={(e) => handleLoadTemplate(e.target.value)}
                    className="px-2 py-1 text-xs rounded-md border bg-white"
                    style={{
                      borderColor: "var(--border-strong)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <option value="" disabled>
                      Load from Library...
                    </option>
                    {generatorTemplates.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </div>
            <PromptEditor
              value={generatorPrompt}
              onChange={setGeneratorPrompt}
              minHeight={340}
              readOnly={!!result}
            />
            {errors.prompt && (
              <p className="text-xs" style={{ color: "var(--error)" }}>
                {errors.prompt}
              </p>
            )}
            {/* Show Output Formatter */}
            <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: "var(--text-tertiary)" }}>
              <input
                type="checkbox"
                checked={showGenFormatter}
                onChange={(e) => setShowGenFormatter(e.target.checked)}
                className="rounded"
                style={{ accentColor: "var(--accent-primary)" }}
              />
              Show Output Formatter
            </label>
            {showGenFormatter && genFormat && (
              <div
                className="rounded-md p-3 text-xs leading-relaxed whitespace-pre-wrap"
                style={{
                  backgroundColor: "var(--surface-elevated)",
                  border: "1px solid var(--border)",
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {genFormat.text}
              </div>
            )}
          </div>
        )}

        {/* Scorer Prompt Tab */}
        {promptTab === "scorer" && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <label
                  className="text-sm font-medium"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Prompt Editor
                </label>
                <PlaceholderInfo
                  items={[
                    { placeholder: "{checklist}", label: "checklist questions", color: "#8250df" },
                    { placeholder: "{target}", label: "response", color: "#0969da" },
                    { placeholder: "{input}", label: "instruction", color: "#1a7f37" },
                  ]}
                />
              </div>
              <div className="flex items-center gap-2">
                {defaultScorerPromptRef.current && scorerPrompt !== defaultScorerPromptRef.current && (
                  <button
                    type="button"
                    onClick={() => {
                      setScorerPrompt(defaultScorerPromptRef.current);
                      setSelectedScorerTemplate("");
                    }}
                    className="text-xs font-medium hover:underline"
                    style={{ color: "var(--text-tertiary)" }}
                  >
                    Reset
                  </button>
                )}
                {scorerTemplates && scorerTemplates.length > 0 && (
                  <select
                    value={selectedScorerTemplate}
                    onChange={(e) => handleLoadTemplate(e.target.value)}
                    className="px-2 py-1 text-xs rounded-md border bg-white"
                    style={{
                      borderColor: "var(--border-strong)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <option value="" disabled>
                      Load from Library...
                    </option>
                    {scorerTemplates.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            </div>
            <PromptEditor
              value={scorerPrompt}
              onChange={setScorerPrompt}
              minHeight={280}
              readOnly={!!result}
            />
            {/* Show Output Formatter */}
            <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: "var(--text-tertiary)" }}>
              <input
                type="checkbox"
                checked={showScorerFormatter}
                onChange={(e) => setShowScorerFormatter(e.target.checked)}
                className="rounded"
                style={{ accentColor: "var(--accent-primary)" }}
              />
              Show Output Formatter
            </label>
            {showScorerFormatter && scorerFormat && (
              <div
                className="rounded-md p-3 text-xs leading-relaxed whitespace-pre-wrap"
                style={{
                  backgroundColor: "var(--surface-elevated)",
                  border: "1px solid var(--border)",
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {scorerFormat.text}
              </div>
            )}
            {/* Scorer config */}
            <div>
              <label
                className="block text-sm font-medium mb-2"
                style={{ color: "var(--text-secondary)" }}
              >
                Scorer Config
              </label>
              <ScorerConfigPanel
                value={scorerConfig}
                onChange={setScorerConfig}
                disabled={!!result}
              />
            </div>
          </div>
        )}

        {/* Save actions — always available */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSavePromptOpen(true)}
            disabled={!(promptTab === "generator" ? generatorPrompt.trim() : scorerPrompt.trim())}
          >
            {savePromptFeedback ?? "Save Prompt"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSavePipelineOpen(true)}
            disabled={!generatorPrompt.trim()}
          >
            {savePipelineFeedback ?? "Save as Pipeline"}
          </Button>
        </div>

        {/* Model selectors — collapsed by default */}
        <hr style={{ borderColor: "var(--border)" }} />
        <details className="group">
          <summary
            className="flex items-center gap-1.5 text-xs font-medium cursor-pointer select-none list-none"
            style={{ color: "var(--text-tertiary)" }}
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="transition-transform group-open:rotate-90"
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
            Model Settings
          </summary>
          <div className="space-y-4 pt-3">
            <div className="space-y-2">
              <label
                className="block text-sm font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                Generator Model
              </label>
              <ModelSelector
                provider={genProvider}
                model={genModel}
                onProviderChange={setGenProvider}
                onModelChange={setGenModel}
              />
            </div>
            {generatorClass === "contrastive" && (
              <div className="space-y-2">
                <label
                  className="block text-sm font-medium"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Candidate Model
                </label>
                <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                  Generates alternative responses to contrast against. Used by the contrastive generator to derive checklist items.
                </p>
                <ModelSelector
                  provider={candidateProvider}
                  model={candidateModel}
                  onProviderChange={setCandidateProvider}
                  onModelChange={setCandidateModel}
                />
              </div>
            )}
            <div className="space-y-2">
              <label
                className="block text-sm font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                Scorer Model
              </label>
              <ModelSelector
                provider={scoreProvider}
                model={scoreModel}
                onProviderChange={setScoreProvider}
                onModelChange={setScoreModel}
              />
            </div>
          </div>
        </details>
      </div>

      {/* RIGHT PANEL — Input + Results */}
      <div className="space-y-5">
        {/* Input fields */}
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span
                className="text-sm font-medium whitespace-nowrap"
                style={{ color: "var(--text-secondary)" }}
              >
                Load Example
              </span>
              {(input || target || reference) && (
                <button
                  type="button"
                  onClick={() => { setInput(""); setTarget(""); setReference(""); setSelectedExample(""); }}
                  className="text-xs font-medium hover:underline"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Clear
                </button>
              )}
            </div>
            <ExampleLoader
              examples={EXAMPLES}
              value={selectedExample}
              onSelect={(ex) => {
                if (!ex) {
                  setSelectedExample("");
                  return;
                }
                setSelectedExample(ex.name);
                setInput(ex.input);
                setTarget(ex.target);
                if (ex.reference) {
                  setReference(ex.reference);
                  setShowReference(true);
                } else {
                  setReference("");
                }
              }}
            />
          </div>

          <Textarea
            label="Input (instruction/query)"
            required
            placeholder="Enter the instruction or query to evaluate..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            error={errors.input}
            rows={3}
          />
          <Textarea
            label="Target (response to evaluate)"
            required
            placeholder="Enter the response to evaluate..."
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            error={errors.target}
            rows={4}
          />
          {showReference ? (
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                  Reference (optional)
                </label>
                <button
                  type="button"
                  onClick={() => { setShowReference(false); setReference(""); }}
                  className="text-xs font-medium hover:underline"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Remove
                </button>
              </div>
              <Textarea
                placeholder="Optional gold reference target..."
                value={reference}
                onChange={(e) => setReference(e.target.value)}
                error={errors.reference}
                rows={3}
              />
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setShowReference(true)}
              className="text-xs font-medium hover:underline"
              style={{ color: "var(--text-tertiary)" }}
            >
              + Add Reference
            </button>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-3">
          <Button
            onClick={handleGenerateAndScore}
            disabled={loading}
          >
            Generate & Score
          </Button>
          <Button
            variant="outline"
            onClick={handleGenerateOnly}
            disabled={loading}
          >
            Generate Only
          </Button>
          {loading && (
            <Button
              variant="ghost"
              onClick={handleCancel}
              className="text-[var(--error)]"
            >
              Cancel
            </Button>
          )}
          {result && !loading && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              className="text-[var(--text-tertiary)]"
            >
              Clear
            </Button>
          )}
        </div>

        {/* Results */}
        {result && (
          <div
            className="rounded-lg p-5 space-y-4"
            style={{
              backgroundColor: "white",
              border: "1px solid var(--border)",
            }}
          >
            {/* Phase indicator */}
            {result.phase && (
              <div className="flex items-center gap-2">
                <svg
                  className="animate-spin h-4 w-4"
                  style={{ color: "var(--accent-primary)" }}
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                <span
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {result.phase === "generating"
                    ? "Generating checklist..."
                    : "Scoring response..."}
                </span>
              </div>
            )}

            {/* Error */}
            {result.error && (
              <div
                className="p-3 rounded-md text-sm"
                style={{
                  backgroundColor: "var(--error-light)",
                  color: "var(--error)",
                }}
              >
                {result.error}
              </div>
            )}

            {/* Score summary */}
            {result.score && (
              <div>
                <div
                  className="flex items-center justify-between p-3 rounded-md"
                  style={{ backgroundColor: "var(--surface-elevated)" }}
                >
                  <span
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Score
                  </span>
                  <span
                    className="text-xl font-semibold"
                    style={{
                      fontFamily: "var(--font-mono)",
                      color: "var(--accent-primary)",
                    }}
                  >
                    {Math.round(result.score.score)}/{Math.round(result.score.max_score)}
                  </span>
                </div>
                {result.score.primary_metric && result.score.primary_metric !== "pass" && (
                  <p
                    className="text-xs mt-1 text-right"
                    style={{
                      fontFamily: "var(--font-mono)",
                      color: "var(--text-tertiary)",
                    }}
                  >
                    {result.score.primary_metric}: {result.score.percentage.toFixed(1)}%
                  </p>
                )}
              </div>
            )}

            {/* Checklist items */}
            {result.checklist && (
              <div>
                <h3
                  className="text-sm font-medium mb-3"
                  style={{ color: "var(--text-primary)" }}
                >
                  Checklist
                </h3>
                <ChecklistDisplay
                  items={mergeScoreIntoItems(result.checklist.items, result.score)}
                  showPassFail={!!result.score}
                />
              </div>
            )}

            {/* Save buttons */}
            {result.checklist && !result.phase && (
              <div
                className="flex items-center gap-2 pt-3"
                style={{ borderTop: "1px solid var(--border)" }}
              >
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSaveChecklist}
                >
                  Save Checklist
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Save Prompt dialog */}
      <Dialog open={savePromptOpen} onOpenChange={setSavePromptOpen}>
        <DialogContent title="Save Prompt" description="Save the current prompt to the library.">
          <div className="space-y-3">
            <Input
              label="Prompt Name"
              required
              placeholder="My Custom Prompt"
              value={promptName}
              onChange={(e) => setPromptName(e.target.value)}
            />
            <Textarea
              label="Description (optional)"
              placeholder="What this prompt does..."
              value={promptDesc}
              onChange={(e) => setPromptDesc(e.target.value)}
              rows={2}
            />
            <div
              className="rounded-md p-3 text-xs space-y-1.5"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            >
              <div className="flex justify-between">
                <span style={{ color: "var(--text-tertiary)" }}>Type</span>
                <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {promptTab === "generator"
                    ? generatorClass === "contrastive" ? "ContrastiveGenerator" : "DirectGenerator"
                    : "Scorer"}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-tertiary)" }}>Length</span>
                <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {(promptTab === "generator" ? generatorPrompt : scorerPrompt).length} chars
                </span>
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={() => setSavePromptOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSavePrompt} disabled={!promptName.trim()}>
                Save Prompt
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Save as Pipeline dialog */}
      <Dialog open={savePipelineOpen} onOpenChange={setSavePipelineOpen}>
        <DialogContent title="Save as Pipeline" description="Save the current configuration as a reusable pipeline.">
          <div className="space-y-3">
            <Input
              label="Pipeline Name"
              required
              placeholder="My Custom Pipeline"
              value={pipelineName}
              onChange={(e) => setPipelineName(e.target.value)}
            />
            <Textarea
              label="Description (optional)"
              placeholder="What this pipeline evaluates..."
              value={pipelineDesc}
              onChange={(e) => setPipelineDesc(e.target.value)}
              rows={2}
            />
            {/* Pipeline summary */}
            <div
              className="rounded-md p-3 text-xs space-y-1.5"
              style={{ backgroundColor: "var(--surface-elevated)" }}
            >
              <div className="font-medium text-xs uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
                Pipeline Components
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-tertiary)" }}>Generator</span>
                <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {generatorClass === "contrastive" ? "ContrastiveGenerator" : "DirectGenerator"}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-tertiary)" }}>Scorer</span>
                <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {scorerConfig.mode} / {scorerConfig.primary_metric}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-tertiary)" }}>Output Format</span>
                <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {genFormatName}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-tertiary)" }}>Generator Prompt</span>
                <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {generatorPrompt.length > 40
                    ? generatorPrompt.slice(0, 40) + "…"
                    : generatorPrompt || "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-tertiary)" }}>Scorer Prompt</span>
                <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
                  {scorerPrompt.length > 40
                    ? scorerPrompt.slice(0, 40) + "…"
                    : scorerPrompt || "—"}
                </span>
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={() => setSavePipelineOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSavePipeline} disabled={!pipelineName.trim()}>
                Save Pipeline
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
