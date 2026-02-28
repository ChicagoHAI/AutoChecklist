"use client";

import { useState, useEffect, useMemo } from "react";
import {
  MethodName,
  METHOD_LABELS,
  METHOD_DESCRIPTIONS,
  REQUIRES_REFERENCE,
  INSTANCE_GENERATOR_CLASSES,
  Checklist,
  ScoreResult,
} from "@/lib/types";
import { EXAMPLES } from "@/lib/examples";
import { startEvalAsync, deleteEvaluation, cancelEvaluation } from "@/lib/api";
import { useEvalStatus, useDefaultModel, usePipelines } from "@/lib/hooks";
import { Button } from "@/components/ui/Button";
import { Textarea } from "@/components/ui/Textarea";
import { ExampleLoader } from "@/components/ExampleLoader";
import { ModelSelector } from "@/components/layout/ModelSelector";
import { MethodCard } from "@/components/MethodCard";
import { Tooltip } from "@/components/ui/Tooltip";

const EVAL_STORAGE_KEY = "compareEvalId";
const METHODS_STORAGE_KEY = "compareSelectedMethods";
const PIPELINES_STORAGE_KEY = "compareSelectedPipelines";

interface PartialMethodResult {
  checklist?: Checklist;
  score?: ScoreResult;
  loading?: "checklist" | "score";
  error?: string;
}

type PartialResults = Record<string, PartialMethodResult>;

export function CompareForm() {
  const [input, setInput] = useState("");
  const [target, setTarget] = useState("");
  const [reference, setReference] = useState("");
  const [showReference, setShowReference] = useState(false);
  const [selectedExample, setSelectedExample] = useState("");
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
  const {
    provider: candidateProvider,
    setProvider: setCandidateProvider,
    model: candidateModel,
    setModel: setCandidateModel,
  } = useDefaultModel();
  const [selectedMethods, setSelectedMethods] = useState<Set<MethodName>>(
    new Set()
  );
  const [selectedPipelines, setSelectedPipelines] = useState<Set<string>>(new Set());
  const { data: pipelines } = usePipelines();
  const [evalId, setEvalId] = useState<string | null>(null);
  const [results, setResults] = useState<PartialResults>({});
  const [submitting, setSubmitting] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [cancelledId, setCancelledId] = useState<string | null>(null);
  const { data: evalData } = useEvalStatus(evalId);

  // Hydrate from sessionStorage
  useEffect(() => {
    const stored = sessionStorage.getItem(EVAL_STORAGE_KEY);
    if (stored) setEvalId(stored);
    const storedMethods = sessionStorage.getItem(METHODS_STORAGE_KEY);
    if (storedMethods) {
      try {
        const arr = JSON.parse(storedMethods) as MethodName[];
        if (arr.length > 0) setSelectedMethods(new Set(arr));
      } catch { /* ignore */ }
    }
    const storedPipelines = sessionStorage.getItem(PIPELINES_STORAGE_KEY);
    if (storedPipelines) {
      try {
        const arr = JSON.parse(storedPipelines) as string[];
        if (arr.length > 0) setSelectedPipelines(new Set(arr));
      } catch { /* ignore */ }
    }
  }, []);

  // Persist selected methods/pipelines to sessionStorage
  useEffect(() => {
    sessionStorage.setItem(METHODS_STORAGE_KEY, JSON.stringify(Array.from(selectedMethods)));
  }, [selectedMethods]);
  useEffect(() => {
    sessionStorage.setItem(PIPELINES_STORAGE_KEY, JSON.stringify(Array.from(selectedPipelines)));
  }, [selectedPipelines]);

  // Sync poll data → local results
  useEffect(() => {
    if (!evalData?.results) return;
    const isStopped = evalData.status === "cancelled" || evalData.status === "completed" || evalData.status === "failed" || evalId === cancelledId;
    const newResults: PartialResults = {};
    for (const [method, data] of Object.entries(evalData.results)) {
      if ((data as { error?: string }).error) {
        newResults[method] = { error: (data as { error: string }).error };
      } else {
        const d = data as { checklist?: Checklist; score?: ScoreResult };
        const isCompleted = evalData.completed_methods?.includes(method);
        newResults[method] = {
          checklist: d.checklist,
          score: d.score || undefined,
          loading: isStopped ? undefined : (!isCompleted
            ? d.checklist
              ? "score"
              : "checklist"
            : undefined),
        };
      }
    }
    setResults(newResults);

    if (evalData.request) {
      if (evalData.request.input && !input)
        setInput(evalData.request.input);
      if (evalData.request.target && !target)
        setTarget(evalData.request.target);
      if (evalData.request.reference && !reference) {
        setReference(evalData.request.reference);
        setShowReference(true);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally sync only when evalData changes; including input/target/reference would overwrite user edits
  }, [evalData]);

  const isRunning = evalId !== null && evalData?.status === "running" && evalId !== cancelledId;

  const toggleMethod = (method: MethodName) => {
    setSelectedMethods((prev) => {
      const next = new Set(prev);
      if (next.has(method)) {
        next.delete(method);
      } else {
        next.add(method);
      }
      return next;
    });
  };

  const togglePipeline = (id: string) => {
    setSelectedPipelines((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Build a map of pipeline_id → pipeline_name for display
  const pipelineNameMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const p of pipelines ?? []) {
      map[p.id] = p.name;
    }
    return map;
  }, [pipelines]);

  // Group pipelines by generator_class so they appear under Direct/Contrastive headings
  const pipelinesByClass = useMemo(() => {
    const map: Record<string, typeof pipelines> = {};
    for (const p of pipelines ?? []) {
      const cls = p.generator_class || "direct";
      (map[cls] ??= []).push(p);
    }
    return map;
  }, [pipelines]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !target.trim()) return;
    if (selectedMethods.size === 0 && selectedPipelines.size === 0) return;

    setSubmitting(true);
    setCancelledId(null);
    setGlobalError(null);
    setResults({});

    // Init loading states
    const initial: PartialResults = {};
    for (const m of selectedMethods) {
      initial[m] = { loading: "checklist" };
    }
    for (const pid of selectedPipelines) {
      initial[`pipeline:${pid}`] = { loading: "checklist" };
    }
    setResults(initial);

    try {
      const pipelineIds = Array.from(selectedPipelines);
      const resp = await startEvalAsync({
        input: input.trim(),
        target: target.trim(),
        reference: reference.trim() || null,
        methods: Array.from(selectedMethods),
        pipeline_ids: pipelineIds.length > 0 ? pipelineIds : undefined,
        generator_provider: genProvider,
        generator_model: genModel,
        scorer_provider: scoreProvider,
        scorer_model: scoreModel,
        candidate_model: hasContrastive ? candidateModel : null,
      });
      setEvalId(resp.eval_id);
      sessionStorage.setItem(EVAL_STORAGE_KEY, resp.eval_id);
    } catch (err) {
      setGlobalError(
        err instanceof Error ? err.message : "Failed to start evaluation"
      );
      setResults({});
    } finally {
      setSubmitting(false);
    }
  };

  const handleClear = () => {
    if (evalId) {
      deleteEvaluation(evalId).catch(() => {});
      sessionStorage.removeItem(EVAL_STORAGE_KEY);
    }
    sessionStorage.removeItem(METHODS_STORAGE_KEY);
    sessionStorage.removeItem(PIPELINES_STORAGE_KEY);
    setCancelledId(null);
    setEvalId(null);
    setResults({});
    setGlobalError(null);
    setInput("");
    setTarget("");
    setReference("");
    setSelectedMethods(new Set());
    setSelectedPipelines(new Set());
    setSelectedExample("");
  };

  const hasContrastive =
    INSTANCE_GENERATOR_CLASSES.contrastive.methods.some(
      (m) => selectedMethods.has(m as MethodName)
    ) ||
    (pipelinesByClass["contrastive"] ?? []).some((p) =>
      selectedPipelines.has(p.id)
    );
  const hasResults = Object.keys(results).length > 0;

  return (
    <div className="space-y-6">
      {/* Input section */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Example loader + heading */}
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

        {/* Stacked textareas */}
        <Textarea
          label="Input (instruction/query)"
          required
          placeholder="Enter the instruction or query..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={3}
        />
        <Textarea
          label="Target (response to evaluate)"
          required
          placeholder="Enter the response to evaluate..."
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          rows={3}
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
              rows={2}
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

        {/* Comparison Settings — collapsed expander */}
        <details className="group">
          <summary
            className="flex items-center gap-1.5 text-sm font-medium cursor-pointer select-none list-none"
            style={{ color: "var(--text-secondary)" }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="transition-transform group-open:rotate-90"
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
            Comparison Settings
          </summary>
          <div className="space-y-4 pt-3">
          <div className="space-y-3">
            <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              <span className="opacity-60">*</span> requires a reference target
              <span className="mx-1.5">·</span>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="inline-block -mt-px opacity-50"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
              {" "}custom pipeline
            </p>
            {Object.entries(INSTANCE_GENERATOR_CLASSES).map(
              ([cls, info]) => {
                const classPipelines = pipelinesByClass[cls] ?? [];
                return (
                <div key={cls}>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span
                      className="text-xs font-semibold uppercase tracking-wider"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {info.label}
                    </span>
                    <span
                      className="text-xs"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      — {info.description}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {(info.methods as MethodName[]).map((m) => {
                      const active = selectedMethods.has(m);
                      const disabled =
                        REQUIRES_REFERENCE[m] && !reference.trim();
                      return (
                        <Tooltip
                          key={m}
                          content={
                            disabled
                              ? "Requires a reference target"
                              : METHOD_DESCRIPTIONS[m]
                          }
                        >
                          <button
                            type="button"
                            onClick={() => !disabled && toggleMethod(m)}
                            disabled={disabled}
                            className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-all ${
                              disabled
                                ? "opacity-40 cursor-not-allowed"
                                : "cursor-pointer"
                            }`}
                            style={{
                              backgroundColor: active
                                ? "var(--accent-primary)"
                                : "white",
                              color: active
                                ? "white"
                                : "var(--text-secondary)",
                              borderColor: active
                                ? "var(--accent-primary)"
                                : "var(--border-strong)",
                            }}
                          >
                            {METHOD_LABELS[m]}
                            {REQUIRES_REFERENCE[m] && (
                              <span className="ml-1 opacity-60">*</span>
                            )}
                          </button>
                        </Tooltip>
                      );
                    })}
                    {classPipelines.map((p) => {
                      const active = selectedPipelines.has(p.id);
                      return (
                        <Tooltip key={p.id} content={p.description || `Custom ${p.generator_class} pipeline, ${p.scorer_class} scorer`}>
                          <button
                            type="button"
                            onClick={() => togglePipeline(p.id)}
                            className="px-3 py-1.5 rounded-md text-xs font-medium border transition-all cursor-pointer"
                            style={{
                              backgroundColor: active
                                ? "var(--accent-primary)"
                                : "white",
                              color: active
                                ? "white"
                                : "var(--text-secondary)",
                              borderColor: active
                                ? "var(--accent-primary)"
                                : "var(--border-strong)",
                            }}
                          >
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="inline-block mr-1.5 -mt-px opacity-50"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                            {p.name}
                          </button>
                        </Tooltip>
                      );
                    })}
                  </div>
                  {cls === "contrastive" && hasContrastive && (
                    <div className="mt-2">
                      <label
                        className="block text-xs font-medium mb-1"
                        style={{ color: "var(--text-tertiary)" }}
                      >
                        Candidate Model
                      </label>
                      <ModelSelector
                        provider={candidateProvider}
                        model={candidateModel}
                        onProviderChange={setCandidateProvider}
                        onModelChange={setCandidateModel}
                      />
                    </div>
                  )}
                </div>
                );
              })
            }
          </div>

        {/* Model selectors */}
        <div className="flex items-center gap-4">
            <div>
              <label
                className="block text-xs font-medium mb-1"
                style={{ color: "var(--text-tertiary)" }}
              >
                Generator
              </label>
              <ModelSelector
                provider={genProvider}
                model={genModel}
                onProviderChange={setGenProvider}
                onModelChange={setGenModel}
              />
            </div>
            <div>
              <label
                className="block text-xs font-medium mb-1"
                style={{ color: "var(--text-tertiary)" }}
              >
                Scorer
              </label>
              <ModelSelector
                provider={scoreProvider}
                model={scoreModel}
                onProviderChange={setScoreProvider}
                onModelChange={setScoreModel}
              />
            </div>
          </div>
          </div>
        </details>

        {/* Action buttons */}
        <div className="flex items-center gap-2">
            <Button
              type="submit"
              loading={submitting || isRunning}
              disabled={
                !input.trim() ||
                !target.trim() ||
                (selectedMethods.size === 0 && selectedPipelines.size === 0) ||
                isRunning
              }
            >
              {isRunning ? "Evaluating..." : "Compare"}
            </Button>
            {isRunning && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (evalId) {
                    setCancelledId(evalId);
                    cancelEvaluation(evalId).catch(() => {});
                  }
                }}
              >
                Cancel
              </Button>
            )}
            {hasResults && !isRunning && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleClear}
              >
                Clear
              </Button>
            )}
          </div>
      </form>

      {/* Global error */}
      {globalError && (
        <div
          className="p-3 rounded-md text-sm"
          style={{
            backgroundColor: "var(--error-light)",
            color: "var(--error)",
          }}
        >
          {globalError}
        </div>
      )}

      {/* Results — Horizontal scrollable cards */}
      {hasResults && (
        <div>
          <h3
            className="text-sm font-medium mb-4"
            style={{ color: "var(--text-secondary)" }}
          >
            Results
          </h3>
          <div
            className="flex gap-4 overflow-x-auto pb-4"
            style={{ scrollSnapType: "x mandatory" }}
          >
            {Array.from(selectedMethods).map((m, i) => (
              <MethodCard
                key={m}
                method={m}
                partialResult={results[m]}
                hasReference={!!reference.trim()}
                loading={results[m]?.loading === "checklist"}
                scoringLoading={results[m]?.loading === "score"}
                figureNumber={i + 1}
              />
            ))}
            {Array.from(selectedPipelines).map((pid, i) => {
              const key = `pipeline:${pid}`;
              const pResult = results[key];
              return (
                <MethodCard
                  key={key}
                  method={key}
                  customLabel={pipelineNameMap[pid] || pid}
                  partialResult={pResult}
                  hasReference={!!reference.trim()}
                  loading={pResult?.loading === "checklist"}
                  scoringLoading={pResult?.loading === "score"}
                  figureNumber={selectedMethods.size + i + 1}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
